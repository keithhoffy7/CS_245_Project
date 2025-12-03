import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, GeminiLLM
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a


# ---- BPR Model loading -------------------------------------

BPR_MODEL_PATH = "/srv/CS_245_Project/example/bpr_model.pkl"
_bpr_model_cache: Optional[Dict] = None


def load_bpr_model() -> Optional[Dict]:
    """Lazy-load BPR model if available."""
    global _bpr_model_cache
    if _bpr_model_cache is not None:
        return _bpr_model_cache
    
    if os.path.exists(BPR_MODEL_PATH):
        try:
            with open(BPR_MODEL_PATH, "rb") as f:
                _bpr_model_cache = pickle.load(f)
            logging.info("Loaded BPR model (optimized for ranking).")
            return _bpr_model_cache
        except Exception as e:
            logging.warning("Failed to load BPR model: %s", e)
    
    logging.warning("BPR model not found. Using LLM-only ranking.")
    _bpr_model_cache = None
    return None


def get_bpr_ranking(user_id: str, candidate_ids: List[str]) -> Optional[List[str]]:
    """
    Get BPR ranking for candidates.
    Returns ranking list or None if model/user not available.
    """
    model = load_bpr_model()
    if model is None:
        return None

    user_factors = model.get("user_factors")
    item_factors = model.get("item_factors")
    user_id_to_idx = model.get("user_id_to_idx", {})
    item_id_to_idx = model.get("item_id_to_idx", {})

    if user_id not in user_id_to_idx:
        return None

    u_idx = user_id_to_idx[user_id]
    if u_idx < 0 or u_idx >= user_factors.shape[0]:
        return None

    # Collect indices for candidates that exist in the BPR model
    existing = []
    for cid in candidate_ids:
        idx = item_id_to_idx.get(cid)
        if idx is not None and 0 <= idx < item_factors.shape[0]:
            existing.append((cid, idx))

    if not existing:
        return None

    # Compute dot-product scores between user and candidate item factors
    u_vec = user_factors[u_idx]  # shape [k]
    cand_indices = np.array([idx for _, idx in existing], dtype=np.int64)
    cand_vecs = item_factors[cand_indices]  # [num_cand, k]
    scores = cand_vecs @ u_vec

    # Build ranking
    scored = list(zip([cid for cid, _ in existing], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = [cid for cid, _ in scored]

    # Append missing candidates at the end
    for cid in candidate_ids:
        if cid not in ranked:
            ranked.append(cid)

    return ranked


class RecReasoning(ReasoningBase):
    """
    Custom reasoning module specialized for Amazon-style recommendations.
    Similar structure to RecPlanning - very specific and structured.
    """

    def __init__(self, profile_type_prompt, llm, memory=None):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)

    def create_prompt(self, task_description: str, merged_reviews: str, 
                     readable_item_list: str, candidate_ids: List[str],
                     bpr_guidance: str = "") -> str:
        """
        Create a structured reasoning prompt for Amazon-style product recommendations.
        Similar to RecPlanning.create_prompt - very specific and emphasizes USER/ITEM/REVIEW.
        """
        prompt = '''You are a reasoning agent on an Amazon-style online shopping platform.
Your task is to rank products for a user based on their historical preferences and product information.

You have access to three key types of information:
1. USER + REVIEW: The user's historical product reviews and star ratings
2. ITEM: Detailed metadata for candidate products (titles, categories, ratings, descriptions, etc.)
3. COLLABORATIVE FILTERING: BPR model predictions based on millions of user-item interactions

Your reasoning process should follow these steps:

STEP 1: Analyze USER + REVIEW information
- Examine the user's past reviews and star ratings
- Identify patterns in what the user liked (4-5 stars) vs disliked (1-2 stars)
- Extract specific preferences mentioned in review text (e.g., "durable", "comfortable", "good sound quality")
- Note categories, brands, or product types the user frequently interacts with

STEP 2: Analyze ITEM information for each candidate
- For each candidate product, examine its metadata (title, description, average rating, review count)
- Identify key attributes, categories, and features
- Note the overall quality signals (star ratings, number of reviews)

STEP 3: Match candidates to USER preferences (USER + ITEM + REVIEW)
- Compare each candidate's attributes to items the user rated highly (4-5 stars)
- Compare each candidate's attributes to items the user rated poorly (1-2 stars)
- Identify candidates that match the user's explicit preferences from review text
- Identify candidates that are similar to liked items in category, features, or style

STEP 4: Incorporate COLLABORATIVE FILTERING signals
- Consider BPR model predictions as additional evidence
- The BPR model learned patterns from millions of interactions
- Use BPR suggestions as hints, but prioritize your analysis of the user's explicit review history

STEP 5: Synthesize and rank
- Combine all signals: review history, item attributes, similarity to liked/disliked items, BPR predictions
- Rank candidates from most preferred to least preferred
- Items matching highly-rated past purchases should rank higher
- Items similar to poorly-rated past purchases should rank lower
- Items with high BPR scores that also match review history should rank highest

CRITICAL RULES:
- You must rank ALL {num_candidates} candidate items
- Each item ID must appear exactly once in your ranking
- Do NOT invent new item IDs
- Do NOT omit any candidate items
- Your output must be ONLY a Python list, no other text

OUTPUT FORMAT:
Return ONLY a Python-style list of strings in this exact format:
['item_id_1', 'item_id_2', 'item_id_3', ..., 'item_id_{num_candidates}']

Now, here is the actual task:

USER + REVIEW Information (your historical reviews and ratings):
{merged_reviews}

ITEM Information (candidate products to rank):
{readable_item_list}

{bpr_guidance}

CANDIDATE LIST (you must rank all of these):
{candidate_ids}

Remember: Your output must be ONLY the ranked list, nothing else.
'''
        prompt = prompt.format(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids,
            num_candidates=len(candidate_ids),
            bpr_guidance=bpr_guidance
        )
        return prompt

    def __call__(self, task_description: str, merged_reviews: str,
                 readable_item_list: str, candidate_ids: List[str],
                 bpr_guidance: str = ""):
        """
        Execute reasoning with structured prompt.
        """
        prompt = self.create_prompt(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids,
            bpr_guidance=bpr_guidance
        )

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=4000
        )

        return reasoning_result


class MyRecommendationAgent(RecommendationAgent):
    """
    Combined agent: Keith's multi-step reasoning + Structured reasoning + BPR model
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm, memory=None)

    def workflow(self):
        """
        Simulate user behavior using Keith's multi-step reasoning + Structured reasoning + BPR model
        Returns:
            list: Sorted list of item IDs
        """
        # Step 1: Get candidate item information
        item_list = []
        for n_bus in range(len(self.task['candidate_list'])):
            item = self.interaction_tool.get_item(
                item_id=self.task['candidate_list'][n_bus])
            keys_to_extract = ['item_id', 'title',
                               'average_rating', 'rating_number', 'description']
            filtered_item = {key: item[key]
                             for key in keys_to_extract if key in item}
            item_list.append(filtered_item)

        # Step 2: Get user's review history
        history_review_dict = self.interaction_tool.get_reviews(
            user_id=self.task['user_id'])

        keys_to_keep = ['stars', 'title', 'text', 'item_id']
        history_review_dict = [
            {key: review.get(key) for key in keys_to_keep if key in review}
            for review in history_review_dict
        ]

        history_review = str(history_review_dict)
        input_tokens = num_tokens_from_string(history_review)
        if input_tokens > 12000:
            encoding = tiktoken.get_encoding("cl100k_base")
            history_review = encoding.decode(
                encoding.encode(history_review)[:12000])

        # Step 3: Get item information for reviewed items
        reviewed_items = []
        for review in history_review_dict:
            item_id = review.get('item_id')
            if item_id:
                item = self.interaction_tool.get_item(item_id=item_id)
                keys_to_extract = ['item_id', 'title',
                                   'average_rating', 'rating_number', 'description']
                filtered_item = {key: item[key]
                                 for key in keys_to_extract if key in item}
                reviewed_items.append(filtered_item)

        # Step 4: Merge reviews with item information (Keith's approach)
        # Use simple reasoning for this step
        merged_reviews_prompt = f'''
        Below is a list of some amazon items and their reviews, as well as another list with some information on the items in those reviews. 
        Please merge this information into a readable and easy to understand list of text that shows each review and information about the item being reviewed.
        Each review should show the number of stars, the title, the text, and the item id in text format.
        The item information shown for each review should include the title, item id, average rating, rating number, and description in text format.
        Make sure that you match the correct item id in the review to the item id in the item information.
        
        Amazon item reviews: 
        
        {history_review}

        Amazon item information:
        
        {reviewed_items}

        Your final output should only be this list of ratings and product information, DO NOT introduce any other text! Please number the ratings as well.
        '''

        # Simple LLM call for merging reviews
        messages_merge = [{"role": "user", "content": merged_reviews_prompt}]
        merged_reviews = self.llm(messages=messages_merge, temperature=0.1, max_tokens=4000)

        # Step 5: Make candidate item list readable (Keith's approach)
        readable_item_list_prompt = f'''
        Below is a list of some information about certain candidate products. Please make this information into text that is readable and easy to understand.
        The information shown for each item should include the title, item id, average rating, rating number, and description in text format.
        
        {item_list}

        Your final output should only be this list of item information, DO NOT introduce any other text! Please number the items as well.
        '''

        # Simple LLM call for making items readable
        messages_readable = [{"role": "user", "content": readable_item_list_prompt}]
        readable_item_list = self.llm(messages=messages_readable, temperature=0.1, max_tokens=4000)

        # Step 6: Get BPR model ranking
        candidate_ids = self.task['candidate_list']
        bpr_ranking = get_bpr_ranking(self.task['user_id'], candidate_ids)
        if bpr_ranking:
            print(f"BPR ranking (top 10): {bpr_ranking[:10]}")
            # Get top 5 from BPR for guidance
            bpr_top_5 = bpr_ranking[:5]
            bpr_guidance = f"""
NOTE: A collaborative filtering model (BPR) trained on millions of user-item interactions
suggests these items might be relevant: {', '.join(bpr_top_5)}.
However, prioritize your analysis of the user's explicit review history and preferences when ranking.
The BPR model is just a hint - your detailed analysis of USER + ITEM + REVIEW information is primary.
"""
        else:
            bpr_guidance = ""
            bpr_ranking = None

        # Step 7: Final ranking task using structured reasoning
        task_description = (
            "Rank the candidate products for this user based on their review history, "
            "item attributes, and collaborative filtering signals."
        )
        
        result = self.reasoning(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids,
            bpr_guidance=bpr_guidance
        )

        # Parse the LLM ranking
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
                return ['']
            
            parsed = eval(result)
            if not isinstance(parsed, list):
                print("Parsed output is not a list.")
                return ['']
            
            parsed = [str(x) for x in parsed]
            
            # Filter to valid candidate_ids and preserve order
            candidate_set = set(candidate_ids)
            cleaned = []
            for cid in parsed:
                if cid in candidate_set and cid not in cleaned:
                    cleaned.append(cid)

            # Append any missing candidates at the end
            for cid in candidate_ids:
                if cid not in cleaned:
                    cleaned.append(cid)

            # Combine BPR and LLM rankings (rank fusion)
            # Use LLM as primary (it's better), BPR as light guidance
            final_ranking = cleaned
            if bpr_ranking:
                try:
                    rank_l = {cid: i for i, cid in enumerate(cleaned)}
                    rank_b = {cid: i for i, cid in enumerate(bpr_ranking)}
                    max_rank = len(candidate_ids)
                    combined = []
                    for cid in candidate_ids:
                        rl = rank_l.get(cid, max_rank)
                        rb = rank_b.get(cid, max_rank)
                        # 90% LLM (structured reasoning works better), 10% BPR (light nudge)
                        score = -(0.85 * rl + 0.15 * rb)
                        combined.append((cid, score))
                    combined.sort(key=lambda x: x[1], reverse=True)
                    final_ranking = [cid for cid, _ in combined]
                except Exception as e:
                    print(f"Rank fusion failed, falling back to LLM ranking: {e}")
                    final_ranking = cleaned

            print('Processed Output:', final_ranking)
            return final_ranking

        except Exception as e:
            print(f'format error: {e}')
            return ['']


if __name__ == "__main__":
    task_set = "amazon"  # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(
        data_dir="/srv/output/data1/output", device="auto", cache=False)

    # Load scenarios
    simulator.set_task_and_groundtruth(
        task_dir=f"/srv/CS_245_Project/example/track2/{task_set}/tasks",
        groundtruth_dir=f"/srv/CS_245_Project/example/track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    simulator.set_llm(GeminiLLM(api_key=gemini_api_key))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(
        number_of_tasks=100, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'/srv/CS_245_Project/example/gemini_keith_structured_reasoning_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")

