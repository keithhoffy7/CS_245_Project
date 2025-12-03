import ast
import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import tiktoken
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryVoyager
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm import LLMBase, GeminiLLM
import re
import logging

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except Exception:
        print(encoding.encode(string))
        return 0
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
                     bpr_guidance: str = "", memory_guidance: str = "") -> str:
        """
        Create a structured reasoning prompt for Amazon-style product recommendations.
        Similar to RecPlanning.create_prompt - very specific and emphasizes USER/ITEM/REVIEW.
        """
        prompt = '''You are a reasoning agent on an Amazon-style online shopping platform.
Your task is to rank products for a user based on their historical preferences and product information.

You have access to four key types of information:
1. USER + REVIEW: The user's historical product reviews and star ratings
2. ITEM: Detailed metadata for candidate products (titles, categories, ratings, descriptions, etc.)
3. COLLABORATIVE FILTERING: BPR model predictions based on millions of user-item interactions
4. MEMORY: Past successful recommendations for this user or similar scenarios

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

STEP 4: Incorporate MEMORY signals
{memory_guidance}
- If memory shows past successful top choices for this user, STRONGLY prioritize those items
- Memory contains proven recommendations that worked for this user before - this is high-value signal
- Items from memory that appear in candidates should rank in the top 5-10 positions
- Balance memory with current review analysis: if memory items match current preferences, rank them very high
- Memory items that don't match current preferences can still be considered but with lower priority

STEP 5: Incorporate COLLABORATIVE FILTERING signals (HIGH PRIORITY)
{bpr_guidance}
- The BPR model learned patterns from millions of user-item interactions - this is a STRONG signal
- BPR top 5 predictions should generally appear in your top 10 rankings
- BPR top 10 predictions should generally appear in your top 15 rankings
- However, you can override BPR if the user's explicit review history strongly contradicts it
- When BPR and review history align, those items should rank very high

STEP 6: Synthesize and rank (CRITICAL - follow this priority order)
1. HIGHEST PRIORITY: Items that appear in BPR top 5 AND match user's highly-rated past purchases (4-5 stars)
2. HIGH PRIORITY: Items from memory (past successful choices) that match current preferences
3. HIGH PRIORITY: Items in BPR top 10 that match review history patterns
4. MEDIUM PRIORITY: Items matching highly-rated past purchases (4-5 stars) based on review analysis
5. MEDIUM PRIORITY: Items from memory that don't strongly match preferences but were successful before
6. LOWER PRIORITY: Items with good attributes but no strong signals from BPR, memory, or review history
7. LOWEST PRIORITY: Items similar to poorly-rated past purchases (1-2 stars)

Final ranking strategy:
- BPR top 5 items that also match review history → rank in top 3-5
- Memory items that match preferences → rank in top 5-8
- BPR top 10 items → generally rank in top 10-15
- Items matching highly-rated purchases → rank higher than average
- Items similar to poorly-rated purchases → rank lower

CRITICAL RULES:
- You must rank ALL {num_candidates} candidate items
- Each item ID must appear exactly once in your ranking
- Do NOT invent new item IDs
- Do NOT omit any candidate items
- Your output must be ONLY a Python list, no other text
- Balance all signals naturally - don't force any single signal

OUTPUT FORMAT:
Return ONLY a Python-style list of strings in this exact format:
['item_id_1', 'item_id_2', 'item_id_3', ..., 'item_id_{num_candidates}']

Now, here is the actual task:

USER + REVIEW Information (your historical reviews and ratings):
{merged_reviews}

ITEM Information (candidate products to rank):
{readable_item_list}

{memory_guidance}

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
            bpr_guidance=bpr_guidance,
            memory_guidance=memory_guidance
        )
        return prompt

    def __call__(self, task_description: str, merged_reviews: str,
                 readable_item_list: str, candidate_ids: List[str],
                 bpr_guidance: str = "", memory_guidance: str = ""):
        """
        Execute reasoning with structured prompt.
        """
        prompt = self.create_prompt(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids,
            bpr_guidance=bpr_guidance,
            memory_guidance=memory_guidance
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
    Combined agent: Keith's multi-step reasoning + Structured reasoning + BPR model + Memory
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        # Add MemoryVoyager for storing and retrieving past successful recommendations
        self.memory = MemoryVoyager(llm=self.llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm, memory=self.memory)

    def workflow(self):
        """
        Simulate user behavior using Keith's multi-step reasoning + Structured reasoning + BPR model + Memory
        Returns:
            list: Sorted list of item IDs
        """
        candidate_ids = self.task['candidate_list']

        # Step 1: Get candidate item information
        item_list = []
        item_quality_scores = {}
        for n_bus in range(len(self.task['candidate_list'])):
            item = self.interaction_tool.get_item(
                item_id=self.task['candidate_list'][n_bus])
            keys_to_extract = ['item_id', 'title',
                               'average_rating', 'rating_number', 'description']
            filtered_item = {key: item[key]
                             for key in keys_to_extract if key in item}
            item_list.append(filtered_item)
            # Improved quality heuristic: balances rating and review count with diminishing returns
            avg_rating = item.get("average_rating", 0) or 0
            rating_count = item.get("rating_number", 0) or 0
            # Use sigmoid-like function for rating count to avoid over-weighting very popular items
            # Formula: rating * log(1 + count) * (1 + tanh(count/100)) to boost moderately-reviewed items
            quality_score = avg_rating * np.log1p(rating_count) * (1.0 + np.tanh(rating_count / 100.0))
            item_quality_scores[item['item_id']] = quality_score

        # Step 2: Get user's review history
        history_review_dict = self.interaction_tool.get_reviews(
            user_id=self.task['user_id'])
        # Keep a limited slice to reduce prompt noise
        history_review_dict = history_review_dict[:30]

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

        # Step 4: Retrieve memory (past successful recommendations)
        memory_context = ""
        user_top_choices = []
        if getattr(self, "memory", None) is not None:
            try:
                # Query 1: Look for this specific user's past successful top choices (most specific)
                user_query = f"user_id={self.task['user_id']} successful top recommendation item_id"
                user_memory = self.memory(user_query)
                
                # Query 2: Look for this user's successful recommendations in general
                user_general_query = f"user_id={self.task['user_id']} recommendation top choice"
                user_general_memory = self.memory(user_general_query)
                
                # Query 3: Look for similar recommendation scenarios with overlapping candidates
                candidate_sample = ', '.join(self.task['candidate_list'][:10])  # More candidates for better matching
                scenario_query = (
                    f"recommendation task user_id={self.task['user_id']} "
                    f"candidates {candidate_sample} successful top choice"
                )
                scenario_memory = self.memory(scenario_query)
                
                # Combine memories with priority to user-specific memories
                if user_memory:
                    memory_context += f"Past successful top choices for this specific user:\n{user_memory}\n\n"
                if user_general_memory and user_general_memory != user_memory:
                    memory_context += f"Other successful recommendations for this user:\n{user_general_memory}\n\n"
                if scenario_memory:
                    memory_context += f"Similar recommendation scenarios:\n{scenario_memory}\n\n"
                    
                # Extract top choices from memory if available (prioritize user-specific)
                all_memory_text = (user_memory or "") + " " + (user_general_memory or "") + " " + (scenario_memory or "")
                if all_memory_text:
                    # Try to extract item IDs from memory text
                    item_pattern = r'B[A-Z0-9]{9}'  # Amazon ASIN pattern
                    found_items = re.findall(item_pattern, all_memory_text)
                    # Remove duplicates while preserving order
                    seen = set()
                    user_top_choices = []
                    for item in found_items:
                        if item not in seen and item in candidate_ids:
                            seen.add(item)
                            user_top_choices.append(item)
                            if len(user_top_choices) >= 10:  # Get more items for better coverage
                                break
            except Exception as e:
                logging.warning(f"Memory retrieval failed: {e}")
                memory_context = ""

        # Build memory guidance block (subtle, not prescriptive)
        memory_guidance = ""
        memory_ranking = None  # Will be used in rank fusion
        
        # Extract memory items from both memory_context and user_top_choices
        memory_items = []
        if user_top_choices:
            memory_items = user_top_choices
        elif memory_context:
            # Try to extract item IDs from memory context text
            item_pattern = r'B[A-Z0-9]{9}'  # Amazon ASIN pattern
            found_items = re.findall(item_pattern, memory_context)
            memory_items = found_items[:5]  # Top 5 from memory
        
        # Create memory-based ranking if we have memory items in candidates
        if memory_items:
            memory_items_in_candidates = [cid for cid in memory_items if cid in candidate_ids]
            if memory_items_in_candidates:
                # Memory items first, then rest (but this is just for rank fusion, not forced)
                memory_ranking = memory_items_in_candidates + [cid for cid in candidate_ids if cid not in memory_items_in_candidates]
        
        # Build guidance text
        if memory_context:
            memory_guidance = (
                "Additional context: Past successful recommendations for this user or similar scenarios:\n"
                f"{memory_context}"
                "Note: Items that appeared as successful recommendations before may be relevant, "
                "but evaluate them based on current preferences and item attributes.\n\n"
            )
        elif memory_items:
            memory_guidance = (
                f"Additional context: Past top choices for this user: {', '.join(memory_items[:5])}\n"
                "Note: These items worked well for this user before. Consider them favorably "
                "if they align with current preferences and item attributes.\n\n"
            )

        # Step 5: Merge reviews with item information (Keith's approach)
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

        # Step 6: Make candidate item list readable (Keith's approach)
        readable_item_list_prompt = f'''
        Below is a list of some information about certain candidate products. Please make this information into text that is readable and easy to understand.
        The information shown for each item should include the title, item id, average rating, rating number, and description in text format.
        
        {item_list}

        Your final output should only be this list of item information, DO NOT introduce any other text! Please number the items as well.
        '''

        # Simple LLM call for making items readable
        messages_readable = [{"role": "user", "content": readable_item_list_prompt}]
        readable_item_list = self.llm(messages=messages_readable, temperature=0.1, max_tokens=4000)

        # Step 7: Get BPR model ranking
        bpr_ranking = get_bpr_ranking(self.task['user_id'], candidate_ids)
        if bpr_ranking:
            print(f"BPR ranking (top 10): {bpr_ranking[:10]}")
            # Get top 10 from BPR for better guidance
            bpr_top_10 = bpr_ranking[:10]
            bpr_top_5 = bpr_ranking[:5]
            bpr_guidance = f"""
COLLABORATIVE FILTERING SIGNAL (HIGH PRIORITY):
A BPR (Bayesian Personalized Ranking) model trained on millions of user-item interactions
has identified these items as highly relevant for this user:

Top 5 BPR predictions: {', '.join(bpr_top_5)}
Top 10 BPR predictions: {', '.join(bpr_top_10)}

The BPR model captures patterns from users with similar preferences and items with similar characteristics.
This is a STRONG signal that should be given significant weight in your ranking.

When ranking:
- Items in the BPR top 5 should generally rank in your top 10
- Items in the BPR top 10 should generally rank in your top 15
- However, you can override BPR predictions if the user's explicit review history strongly suggests otherwise
- Combine BPR signals with memory (past successful choices) and review analysis for the best ranking
"""
        else:
            bpr_guidance = ""
            bpr_ranking = None

        # Step 8: Final ranking task using structured reasoning with memory
        task_description = (
            "Rank the candidate products for this user based on their review history, "
            "item attributes, memory of past successful choices, and collaborative filtering signals."
        )
        
        result = self.reasoning(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids,
            bpr_guidance=bpr_guidance,
            memory_guidance=memory_guidance
        )

        # Parse the LLM ranking
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
                return bpr_ranking or ['']
            
            parsed = ast.literal_eval(result)
            if not isinstance(parsed, list):
                print("Parsed output is not a list.")
                return bpr_ranking or ['']
            
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

            # Combine LLM, BPR, Memory, and intrinsic quality rankings using Reciprocal Rank Fusion (RRF)
            # RRF is proven to work well for combining multiple rankings
            final_ranking = cleaned
            try:
                rank_l = {cid: i for i, cid in enumerate(cleaned)}  # LLM ranking
                rank_b = {cid: i for i, cid in enumerate(bpr_ranking)} if bpr_ranking else {}
                rank_m = {cid: i for i, cid in enumerate(memory_ranking)} if memory_ranking else {}
                # Quality ranking (higher quality = better rank)
                quality_sorted = sorted(
                    candidate_ids,
                    key=lambda cid: -item_quality_scores.get(cid, 0)
                )
                rank_q = {cid: i for i, cid in enumerate(quality_sorted)}

                max_rank = len(candidate_ids)
                k = 60  # RRF constant (typical value)
                combined = []
                for cid in candidate_ids:
                    rl = rank_l.get(cid, max_rank)
                    rb = rank_b.get(cid, max_rank)
                    rm = rank_m.get(cid, max_rank)
                    rq = rank_q.get(cid, max_rank)
                    
                    # Reciprocal Rank Fusion with weighted contributions
                    # BPR gets higher weight (trained on millions of interactions)
                    # Memory gets moderate weight (proven successful for this user)
                    # LLM gets moderate weight (semantic understanding)
                    # Quality gets lower weight (baseline signal)
                    score = 0.0
                    
                    if memory_ranking and bpr_ranking:
                        # All signals available: BPR is strongest, then LLM, then memory, then quality
                        score = (0.35 * (1.0 / (k + rl + 1))) + \
                                (0.40 * (1.0 / (k + rb + 1))) + \
                                (0.15 * (1.0 / (k + rm + 1))) + \
                                (0.10 * (1.0 / (k + rq + 1)))
                    elif memory_ranking:
                        # Memory + LLM + Quality
                        score = (0.50 * (1.0 / (k + rl + 1))) + \
                                (0.30 * (1.0 / (k + rm + 1))) + \
                                (0.20 * (1.0 / (k + rq + 1)))
                    elif bpr_ranking:
                        # BPR + LLM + Quality (BPR is very strong)
                        score = (0.40 * (1.0 / (k + rl + 1))) + \
                                (0.45 * (1.0 / (k + rb + 1))) + \
                                (0.15 * (1.0 / (k + rq + 1)))
                    else:
                        # LLM + Quality only
                        score = (0.75 * (1.0 / (k + rl + 1))) + \
                                (0.25 * (1.0 / (k + rq + 1)))
                    
                    combined.append((cid, score))
                combined.sort(key=lambda x: x[1], reverse=True)
                final_ranking = [cid for cid, _ in combined]
                
                if memory_ranking:
                    print(f"Memory ranking (top 5): {memory_ranking[:5]}")
            except Exception as e:
                print(f"Rank fusion failed, falling back to LLM ranking: {e}")
                final_ranking = cleaned

            print('Processed Output:', final_ranking)

            # Store successful patterns in memory for future tasks
            if getattr(self, "memory", None) is not None:
                try:
                    user_id = self.task.get("user_id")
                    top_choice = final_ranking[0] if final_ranking else None
                    
                    # Store 1: User's top choice (for future recommendations to same user)
                    if top_choice:
                        user_pattern = (
                            f"user_id={user_id} successful top recommendation: {top_choice}. "
                            f"This user strongly preferred {top_choice} when given candidates {candidate_ids[:5]}..."
                        )
                        self.memory(f"review:{user_pattern}")
                    
                    # Store 2: Full trajectory (for similar scenarios)
                    trajectory_str = (
                        f"User {user_id} recommendation: top choice was {top_choice}. "
                        f"BPR model predicted {bpr_ranking[0] if bpr_ranking else 'N/A'}. "
                        f"Top 5 ranking: {final_ranking[:5]}"
                    )
                    self.memory(f"review:{trajectory_str}")
                except Exception as e:
                    logging.warning(f"Memory storage failed: {e}")

            return final_ranking

        except Exception as e:
            print(f'format error: {e}')
            return bpr_ranking or candidate_ids


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
    with open(f'/srv/CS_245_Project/example/gemini_keith_memory_bpr_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
