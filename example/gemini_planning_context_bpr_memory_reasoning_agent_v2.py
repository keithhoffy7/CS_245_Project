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

    def create_prompt(
        self,
        task_description: str,
        merged_reviews: str,
        readable_item_list: str,
        candidate_ids: List[str],
        user_pref_summary: str = "",
        bpr_guidance: str = "",
        memory_guidance: str = "",
    ) -> str:
        """
        Create a structured reasoning prompt for Amazon-style product recommendations.
        Includes a short user preference summary and clearer guidance on how to use memory/BPR.
        """
        prompt = '''You are a reasoning agent on an Amazon-style online shopping platform.
Your task is to rank products for a user based on their historical preferences and product information.

You have access to four key types of information:
1. USER + REVIEW: The user's historical product reviews and star ratings
2. USER PREFERENCE SUMMARY: A short summary of what this user tends to like and dislike
3. ITEM: Detailed metadata for candidate products (titles, categories, ratings, descriptions, etc.)
4. MEMORY + COLLABORATIVE FILTERING: Past successful recommendation trajectories and BPR model hints

USER PREFERENCE SUMMARY (very important):
{user_pref_summary}

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
- Memory summarizes past successful recommendation trajectories (for this or similar users)
- If memory highlights specific items that worked well in similar scenarios AND they appear in candidates,
  consider them favorably, but still check them against this user's preferences

STEP 5: Incorporate COLLABORATIVE FILTERING (BPR) signals
{bpr_guidance}
- BPR provides a ranking learned from many users; treat it as a useful hint, not as ground truth
- When BPR suggestions align with the user's preferences and memory signals, you should rank those items higher

STEP 6: Synthesize and rank
- Combine all signals: review history (primary), user preference summary, item attributes, memory (if available),
  similarity to liked/disliked items, and BPR predictions (if available)
- Rank candidates from most preferred to least preferred
- Items matching highly-rated past purchases AND fitting the preference summary should rank higher
- Items similar to poorly-rated past purchases should rank lower
- Items with strong support from memory and BPR that also match preferences should rank very high

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

MEMORY + PAST SUCCESSFUL TRAJECTORIES (if any):
{memory_guidance}

COLLABORATIVE FILTERING HINTS (if any):
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
            memory_guidance=memory_guidance,
            user_pref_summary=user_pref_summary,
        )
        return prompt

    def __call__(
        self,
        task_description: str,
        merged_reviews: str,
        readable_item_list: str,
        candidate_ids: List[str],
        user_pref_summary: str = "",
        bpr_guidance: str = "",
        memory_guidance: str = "",
    ):
        """
        Execute reasoning with structured prompt.
        """
        prompt = self.create_prompt(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids,
            user_pref_summary=user_pref_summary,
            bpr_guidance=bpr_guidance,
            memory_guidance=memory_guidance,
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
    Improved agent: Keith's multi-step reasoning + Structured reasoning + BPR model + Enhanced Memory
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        # Add MemoryVoyager for storing and retrieving past successful recommendations
        self.memory = MemoryVoyager(llm=self.llm)
        self.reasoning = RecReasoning(
            profile_type_prompt='', llm=self.llm, memory=self.memory)

    def _summarize_user_preferences(self, history_review: str) -> str:
        """Use LLM to summarize user preferences from raw review dict text."""
        pref_prompt = f"""
You are given a list of this user's past reviews (stars, titles, texts, item ids).
Summarize their preferences in 3-5 short bullet-style phrases, focusing on:
- product categories/genres they like
- attributes they value (e.g., durable, comfortable, long battery life, genre, writing style)
- clear dislikes or deal-breakers

Reviews:
{history_review}

Your output should be a concise paragraph or bullet list (no more than 4 sentences).
"""
        messages = [{"role": "user", "content": pref_prompt}]
        summary = self.llm(messages=messages, temperature=0.2, max_tokens=400)
        return summary.strip()

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
            # Simple quality heuristic to favor strong, well-reviewed items
            avg_rating = item.get("average_rating", 0) or 0
            rating_count = item.get("rating_number", 0) or 0
            item_quality_scores[item['item_id']
                                ] = avg_rating * np.log1p(rating_count)

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

        # Step 2.1: Summarize user preferences (used both in reasoning and memory)
        user_pref_summary = self._summarize_user_preferences(history_review)

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

        # Step 4: Retrieve memory (past successful recommendation trajectories)
        memory_context = ""
        memory_items = []
        memory_ranking = None

        if getattr(self, "memory", None) is not None:
            try:
                # Build a richer scenario description for memory retrieval
                scenario_descriptor = (
                    f"user_pref: {user_pref_summary}\n"
                    f"user_id: {self.task['user_id']}\n"
                    f"candidates_sample: {candidate_ids[:8]}"
                )

                # Retrieve the most similar past trajectory
                raw_memory = self.memory(scenario_descriptor) or ""

                if raw_memory:
                    memory_context += f"Most similar past recommendation trajectory:\n{raw_memory}\n\n"

                    # Extract item IDs that appeared as successful top choices or top-ranked items
                    item_pattern = r'B[A-Z0-9]{9}'  # Amazon ASIN pattern
                    found_items = re.findall(item_pattern, raw_memory)
                    # Deduplicate while preserving order
                    seen = set()
                    for it in found_items:
                        if it not in seen:
                            seen.add(it)
                            memory_items.append(it)

            except Exception as e:
                logging.warning(f"Memory retrieval failed: {e}")
                memory_context = ""
                memory_items = []

        # Create memory-based ranking if we have memory items in candidates
        if memory_items:
            memory_items_in_candidates = [
                cid for cid in memory_items if cid in candidate_ids]
            if memory_items_in_candidates:
                # Memory items first, then rest (for rank fusion)
                memory_ranking = memory_items_in_candidates + [
                    cid for cid in candidate_ids if cid not in memory_items_in_candidates
                ]

        # Build guidance text
        memory_guidance = ""
        if memory_context:
            memory_guidance = (
                "Additional context from similar past recommendation tasks:\n"
                f"{memory_context}"
                "Items that repeatedly appeared as successful top choices in similar scenarios "
                "should be considered strong candidates IF they also match this user's preferences.\n\n"
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
        merged_reviews = self.llm(
            messages=messages_merge, temperature=0.1, max_tokens=4000)

        # Step 6: Make candidate item list readable (Keith's approach)
        readable_item_list_prompt = f'''
        Below is a list of some information about certain candidate products. Please make this information into text that is readable and easy to understand.
        The information shown for each item should include the title, item id, average rating, rating number, and description in text format.
        
        {item_list}

        Your final output should only be this list of item information, DO NOT introduce any other text! Please number the items as well.
        '''

        # Simple LLM call for making items readable
        messages_readable = [
            {"role": "user", "content": readable_item_list_prompt}]
        readable_item_list = self.llm(
            messages=messages_readable, temperature=0.1, max_tokens=4000)

        # Step 7: Get BPR model ranking
        bpr_ranking = get_bpr_ranking(self.task['user_id'], candidate_ids)
        if bpr_ranking:
            print(f"BPR ranking (top 10): {bpr_ranking[:10]}")
            # Get top 5 from BPR for guidance
            bpr_top_5 = bpr_ranking[:5]
            bpr_guidance = f"""
NOTE: A collaborative filtering model (BPR) trained on many user-item interactions
suggests these items might be relevant: {', '.join(bpr_top_5)}.
Treat this as a helpful prior: items in this BPR top-5 that also match the user's preferences
and memory signals should be ranked especially high.
"""
        else:
            bpr_guidance = ""
            bpr_ranking = None

        # Step 8: Final ranking task using structured reasoning with enhanced memory
        task_description = (
            "Rank the candidate products for this user based on their review history, "
            "summarized preferences, item attributes, memory of past similar successful trajectories, "
            "and collaborative filtering signals."
        )

        result = self.reasoning(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids,
            user_pref_summary=user_pref_summary,
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

            # Combine LLM, BPR, Memory, and intrinsic quality rankings (rank fusion)
            final_ranking = cleaned
            try:
                rank_l = {cid: i for i, cid in enumerate(
                    cleaned)}  # LLM ranking
                rank_b = {cid: i for i, cid in enumerate(
                    bpr_ranking)} if bpr_ranking else {}
                rank_m = {cid: i for i, cid in enumerate(
                    memory_ranking)} if memory_ranking else {}
                # Quality ranking (higher quality = better rank)
                quality_sorted = sorted(
                    candidate_ids,
                    key=lambda cid: -item_quality_scores.get(cid, 0)
                )
                rank_q = {cid: i for i, cid in enumerate(quality_sorted)}

                max_rank = len(candidate_ids)
                combined = []
                for cid in candidate_ids:
                    rl = rank_l.get(cid, max_rank)
                    rb = rank_b.get(cid, max_rank)
                    rm = rank_m.get(cid, max_rank)
                    rq = rank_q.get(cid, max_rank)

                    # Weighted combination:
                    # - LLM reasoning + user preference summary remains primary
                    # - Memory provides a strong secondary signal when present
                    # - BPR and intrinsic quality provide additional signals
                    if memory_ranking and bpr_ranking:
                        score = -(0.55 * rl + 0.20 * rm +
                                  0.15 * rb + 0.10 * rq)
                    elif memory_ranking:
                        score = -(0.65 * rl + 0.25 * rm + 0.10 * rq)
                    elif bpr_ranking:
                        score = -(0.70 * rl + 0.20 * rb + 0.10 * rq)
                    else:
                        score = -(0.80 * rl + 0.20 * rq)

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

                    # Scenario summary for memory (more structured, cross-user useful)
                    scenario_memory_str = (
                        f"user_pref: {user_pref_summary} | "
                        f"user_id: {user_id} | "
                        f"candidates: {candidate_ids} | "
                        f"top_choice: {top_choice} | "
                        f"top_5_ranking: {final_ranking[:5]} | "
                        f"bpr_top_1: {bpr_ranking[0] if bpr_ranking else 'N/A'}"
                    )

                    # Store: full trajectory summary for this scenario
                    self.memory(f"review:{scenario_memory_str}")
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
        number_of_tasks=None, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'/srv/CS_245_Project/example/gemini_planning_context_bpr_memory_reasoning_agent_v2_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
