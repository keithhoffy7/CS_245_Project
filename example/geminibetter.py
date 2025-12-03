import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import (
    PlanningBase,
)
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningBase,
    ReasoningCOT,
)
from websocietysimulator.agent.modules.memory_modules import MemoryVoyager
import re
import logging

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except Exception:
        print(encoding.encode(string))
    return a


# ---- Recommender model loading (WARP > BPR > MF) -------------------------------------

WARP_MODEL_PATH = "/srv/CS_245_Project/example/warp_model.pkl"
BPR_MODEL_PATH = "/srv/CS_245_Project/example/bpr_model.pkl"
MF_MODEL_PATH = "/srv/CS_245_Project/example/mf_model.pkl"
_recommender_model_cache: Optional[Dict] = None


def load_recommender_model() -> Optional[Dict]:
    """Lazy-load WARP model (best), BPR model, or MF model (fallback) if available."""
    global _recommender_model_cache
    if _recommender_model_cache is not None:
        return _recommender_model_cache
    
    # Try WARP first (best for ranking with explicit negatives)
    if os.path.exists(WARP_MODEL_PATH):
        try:
            with open(WARP_MODEL_PATH, "rb") as f:
                _recommender_model_cache = pickle.load(f)
            logging.info("Loaded WARP model (optimized for ranking with explicit negatives).")
            return _recommender_model_cache
        except Exception as e:
            logging.warning("Failed to load WARP model: %s", e)
    
    # Try BPR second (better for ranking)
    if os.path.exists(BPR_MODEL_PATH):
        try:
            with open(BPR_MODEL_PATH, "rb") as f:
                _recommender_model_cache = pickle.load(f)
            logging.info("Loaded BPR model (optimized for ranking).")
            return _recommender_model_cache
        except Exception as e:
            logging.warning("Failed to load BPR model: %s", e)
    
    # Fall back to MF/ALS
    if os.path.exists(MF_MODEL_PATH):
        try:
            with open(MF_MODEL_PATH, "rb") as f:
                _recommender_model_cache = pickle.load(f)
            logging.info("Loaded MF/ALS model (fallback).")
            return _recommender_model_cache
        except Exception as e:
            logging.warning("Failed to load MF model: %s", e)
    
    logging.warning("No recommender model found. Using LLM-only ranking.")
    _recommender_model_cache = None
    return None


def load_mf_model() -> Optional[Dict]:
    """Legacy function for backward compatibility."""
    return load_recommender_model()


def get_mf_scores_and_ranking(user_id: str, candidate_ids: List[str]) -> Optional[tuple]:
    """
    Get MF scores and ranking for candidates.
    Returns (scores_dict, ranking_list) or None if model/user not available.
    scores_dict maps item_id -> float score (higher = better match).
    """
    model = load_mf_model()
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

    # Collect indices for candidates that exist in the MF model
    existing = []
    missing = []
    for cid in candidate_ids:
        idx = item_id_to_idx.get(cid)
        if idx is None or idx < 0 or idx >= item_factors.shape[0]:
            missing.append(cid)
        else:
            existing.append((cid, idx))

    if not existing:
        return None

    # Compute dot-product scores between user and candidate item factors
    u_vec = user_factors[u_idx]  # shape [k]
    cand_indices = np.array([idx for _, idx in existing], dtype=np.int64)
    cand_vecs = item_factors[cand_indices]  # [num_cand, k]
    scores = cand_vecs @ u_vec

    # Build scores dict (normalize to 0-1 range for readability)
    scores_dict = {}
    if len(scores) > 0:
        min_score = float(scores.min())
        max_score = float(scores.max())
        score_range = max_score - min_score if max_score > min_score else 1.0
        for (cid, _), raw_score in zip(existing, scores):
            # Normalize to 0-1, then scale to 0-10 for easier interpretation
            normalized = (float(raw_score) - min_score) / score_range if score_range > 0 else 0.5
            scores_dict[cid] = normalized * 10.0  # Scale to 0-10

    # Build ranking
    scored = list(zip([cid for cid, _ in existing], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = [cid for cid, _ in scored]

    # Append missing candidates at the end
    for cid in candidate_ids:
        if cid not in ranked:
            ranked.append(cid)
            scores_dict[cid] = 0.0  # Unknown items get 0 score

    return scores_dict, ranked


def rank_candidates_by_mf(user_id: str, candidate_ids: List[str]) -> Optional[List[str]]:
    """Legacy function for backward compatibility. Returns just ranking."""
    result = get_mf_scores_and_ranking(user_id, candidate_ids)
    if result is None:
        return None
    _, ranking = result
    return ranking


def rank_candidates_by_popularity(candidate_ids, item_list):
    """
    Simple non-LLM baseline: rank candidates by item popularity/quality
    using average_rating/stars and ratings_count/review_count.
    """
    # Build lookup from item_id to its metadata
    id_to_item = {}
    for item in item_list:
        item_id = item.get("item_id")
        if item_id:
            id_to_item[item_id] = item

    scores = []
    for cid in candidate_ids:
        item = id_to_item.get(cid, {})
        rating = item.get("average_rating", item.get("stars", 0.0))
        count = item.get("ratings_count", item.get("review_count", 0.0))
        try:
            rating = float(rating)
        except Exception:
            rating = 0.0
        try:
            count = float(count)
        except Exception:
            count = 0.0

        # Popularity-weighted quality score
        score = rating * (1.0 + count ** 0.5)
        scores.append((cid, score))

    # Sort by score descending; fall back to original order if ties
    scores.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scores]


class RecPlanning(PlanningBase):
    """Custom planning module specialized for Amazon-style recommendations."""

    def __init__(self, llm):
        super().__init__(llm=llm)

    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Make planning explicitly about USER / ITEM / REVIEW information."""
        if feedback == '':
            prompt = '''You are a planner who divides an {task_type} on an Amazon-style
online shopping platform into several concrete subtasks.

The overall goal is to recommend products to a user based on:
- the user's historical product reviews and star ratings,
- detailed metadata for a fixed list of candidate items (titles, categories,
  star ratings, review counts, attributes, descriptions, etc.),
- and any additional user profile information if needed.

For each subtask, you must:
- provide a short "description" that clearly indicates whether you are using
  USER information, ITEM information, or REVIEW information (or a combination),
- provide a brief "reasoning instruction" explaining why this step helps make
  better product recommendations.

Your output format should follow the example below:
Task: I need to find some information to complete a recommendation task on an
Amazon-style platform.
sub-task 1: {{"description": "First I need to retrieve the target user's historical product reviews and ratings (USER + REVIEW)", "reasoning instruction": "Understand the user's past preferences from their reviews and stars."}}
sub-task 2: {{"description": "Next, I need to fetch detailed metadata for each candidate product (ITEM)", "reasoning instruction": "Compare product attributes, categories, and ratings to the user's history."}}
sub-task 3: {{"description": "Then, I need to analyze how similar each candidate product is to the user's highly rated past products (USER + ITEM + REVIEW)", "reasoning instruction": "Recommend items similar to those the user liked and dissimilar to those they disliked."}}

Task: {task_description}
'''
            prompt = prompt.format(
                task_description=task_description,
                task_type=task_type,
            )
        else:
            prompt = '''You are a planner who divides an {task_type} on an Amazon-style
online shopping platform into several concrete subtasks, using feedback to refine
your plan.

The overall goal is to recommend products to a user based on:
- the user's historical product reviews and star ratings,
- detailed metadata for a fixed list of candidate items (titles, categories,
  star ratings, review counts, attributes, descriptions, etc.),
- and any additional user profile information if needed.

For each subtask, you must:
- provide a short "description" that clearly indicates whether you are using
  USER information, ITEM information, or REVIEW information (or a combination),
- provide a brief "reasoning instruction" explaining why this step helps make
  better product recommendations.

The following are some examples:
Task: I need to find some information to complete a recommendation task on an
Amazon-style platform.
sub-task 1: {{"description": "First I need to retrieve the target user's historical product reviews and ratings (USER + REVIEW)", "reasoning instruction": "Understand the user's past preferences from their reviews and stars."}}
sub-task 2: {{"description": "Next, I need to fetch detailed metadata for each candidate product (ITEM)", "reasoning instruction": "Compare product attributes, categories, and ratings to the user's history."}}
sub-task 3: {{"description": "Then, I need to analyze how similar each candidate product is to the user's highly rated past products (USER + ITEM + REVIEW)", "reasoning instruction": "Recommend items similar to those the user liked and dissimilar to those they disliked."}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(
                example=few_shot,
                task_description=task_description,
                task_type=task_type,
                feedback=feedback,
            )
        return prompt


class RecReasoning(ReasoningBase):
    """Simple reasoning wrapper (currently not used directly by the agent)."""

    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        prompt = '''
{task_description}
'''
        prompt = prompt.format(task_description=task_description)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        return reasoning_result


class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent for recommendation tasks.
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        # Voyager memory + custom planning + COT reasoning (empirically better)
        self.memory = MemoryVoyager(llm=self.llm)
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = ReasoningCOT(
            profile_type_prompt='',
            memory=self.memory,
            llm=self.llm
        )

    def workflow(self):
        """
        Simulate user behavior.
        Returns:
            list: Sorted list of item IDs (most preferred -> least preferred)
        """
        # Fallback static plan
        default_plan = [
            {'description': 'First I need to find user information'},
            {'description': 'Next, I need to find item information'},
            {'description': 'Next, I need to find review information'},
        ]

        # Use the custom planning module
        try:
            task_description = (
                "Plan how to recommend products on an Amazon-style platform. You have access to: "
                "(1) USER historical reviews and ratings, (2) ITEM metadata for a fixed candidate list, "
                "(3) REVIEW texts, (4) a trained collaborative filtering model (BPR/MF) that provides "
                "match scores for each candidate, and (5) memory of past successful recommendations. "
                "Decompose the steps needed to gather this information and combine MF model predictions "
                "with user history to rank the candidates."
            )
            plan = self.planning(
                task_type='Recommendation Task',
                task_description=task_description,
                feedback='',
                few_shot=''
            )
            if not plan:
                print("Planner returned empty plan. Using fallback static plan.")
                plan = default_plan
            print(f"The generated plan is: {plan}")
        except Exception as e:
            print(f"Planning failed: {e}. Using fallback static plan.")
            plan = default_plan

        user = ''
        item_list = []
        history_review = ''

        # Execute subtasks (case-insensitive matching on description)
        for sub_task in plan:
            desc = sub_task.get('description', '')
            desc_lower = desc.lower()

            if 'user' in desc_lower:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif 'item' in desc_lower:
                # Build item_list once: one dict per candidate item
                if not item_list:
                    for item_id in self.task['candidate_list']:
                        item = self.interaction_tool.get_item(item_id=item_id)
                        keys_to_extract = [
                            'item_id', 'name', 'stars', 'review_count', 'attributes',
                            'title', 'average_rating', 'rating_number', 'description',
                            'ratings_count', 'title_without_series'
                        ]
                        filtered_item = {
                            key: item[key]
                            for key in keys_to_extract
                            if key in item
                        }
                        item_list.append(filtered_item)

            elif 'review' in desc_lower:
                history_review = str(
                    self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                )
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(
                        encoding.encode(history_review)[:12000]
                    )
            else:
                pass

        # Retrieve relevant past trajectories from memory (improved query)
        memory_context = ""
        user_top_choices = []
        if getattr(self, "memory", None) is not None:
            try:
                # Query 1: Look for this specific user's past successful choices
                user_query = f"user_id={self.task['user_id']} successful top recommendation"
                user_memory = self.memory(user_query)
                
                # Query 2: Look for similar recommendation scenarios
                scenario_query = (
                    f"recommendation task; user_id={self.task['user_id']}; "
                    f"candidates={self.task['candidate_list'][:5]}..."  # First 5 for similarity
                )
                scenario_memory = self.memory(scenario_query)
                
                # Combine memories
                if user_memory:
                    memory_context += f"Past successful choices for this user:\n{user_memory}\n\n"
                if scenario_memory:
                    memory_context += f"Similar recommendation scenarios:\n{scenario_memory}\n\n"
                    
                # Extract top choices from memory if available
                if user_memory:
                    # Try to extract item IDs from memory text
                    import re
                    item_pattern = r'B[A-Z0-9]{9}'  # Amazon ASIN pattern
                    found_items = re.findall(item_pattern, user_memory)
                    user_top_choices = found_items[:5]  # Top 5 past choices
            except Exception as e:
                logging.warning(f"Memory retrieval failed: {e}")
                memory_context = ""

        memory_block = ""
        if memory_context:
            memory_block = (
                "IMPORTANT: Past successful recommendations for this user or similar scenarios:\n"
                f"{memory_context}"
                "Use these as strong signals - if an item appeared as a top choice before, "
                "it's likely a good match for this user.\n\n"
            )
        elif user_top_choices:
            memory_block = (
                f"Past top choices for this user: {', '.join(user_top_choices)}\n"
                "If any of these appear in your candidate list, strongly consider ranking them highly.\n\n"
            )

        candidate_ids = self.task['candidate_list']

        # Get model scores and ranking (BPR preferred, MF fallback)
        mf_result = get_mf_scores_and_ranking(self.task['user_id'], candidate_ids)
        mf_scores = None
        mf_ranking = None
        model_type = "none"
        if mf_result:
            mf_scores, mf_ranking = mf_result
            model_obj = load_recommender_model()
            model_type = model_obj.get("model_type", "mf") if model_obj else "mf"
            print(f"{model_type.upper()} ranking:", mf_ranking[:10], "...")
            print(f"{model_type.upper()} scores (0-10 scale, top 5):", {k: f"{v:.2f}" for k, v in list(mf_scores.items())[:5]})

        # Non-LLM baseline ranking (popularity-based) over the same candidates (secondary)
        heuristic_ranking = []
        if item_list:
            heuristic_ranking = rank_candidates_by_popularity(candidate_ids, item_list)
            print("Heuristic (popularity) ranking:", heuristic_ranking)

        # Build MF guidance block for LLM prompt
        mf_guidance = ""
        if mf_scores and mf_ranking:
            # Get top 5 and bottom 3 by MF score
            sorted_by_score = sorted(mf_scores.items(), key=lambda x: x[1], reverse=True)
            top_items = [f"{cid} (score: {score:.2f})" for cid, score in sorted_by_score[:5]]
            bottom_items = [f"{cid} (score: {score:.2f})" for cid, score in sorted_by_score[-3:]]
            
            mf_guidance = f"""
IMPORTANT: A collaborative filtering model (trained on millions of user-item interactions)
has analyzed your preferences and provided match scores for each candidate item.

MF Model Predictions (0-10 scale, higher = better match for you):
- Top recommended items: {', '.join(top_items)}
- Lower match items: {', '.join(bottom_items)}

The MF model learned patterns from users similar to you and items similar to ones you've
interacted with. Use these scores as a strong signal, but also consider:
- Your specific review text and explicit preferences
- Item attributes, categories, and descriptions
- How well items match your past 4-5 star ratings vs 1-2 star ratings

You should generally favor items with higher MF scores, but you can override if your
review history suggests a different preference pattern.
"""

        # Reasoning prompt: ask directly for a ranked list of item_ids (like baseline),
        # but keep all the stronger Amazon-specific guidance and include a small example.
        task_description = f"""
You are a real user on an Amazon-style online shopping platform.

Your historical item review text and star ratings are as follows:
{history_review}

You are given a fixed list of candidate items (Amazon product IDs):
{candidate_ids}

For each candidate, you can see detailed metadata such as:
- title and category,
- overall star rating and review_count,
- attributes and description,
- ratings_count and other similar signals.

The information for these candidate items is as follows:
{item_list}

{mf_guidance}

{memory_block}CRITICAL: If you see past successful top choices for this user in the memory above,
and any of those items appear in your candidate list, you should STRONGLY prioritize ranking them
in the top 3 positions. The memory shows what actually worked for this user before.

Combine all signals:
1. MF model scores (strong collaborative signal)
2. Past successful choices from memory (proven to work for this user)
3. Your review history and explicit preferences
4. Item attributes and similarity to liked/disliked items

EXAMPLE (for format and behavior ONLY — do NOT reuse these IDs):

Suppose your history shows that you loved high-quality wireless headphones and
you disliked low-battery cheap earbuds. You are given candidate items:
['X1', 'X2', 'X3'] where:
- X1: wireless over-ear headphones, many positive reviews about sound quality;
- X2: cheap wired earbuds, many complaints about durability;
- X3: wireless earbuds with mixed reviews.

You should rank them:
['X1', 'X3', 'X2']

because X1 best matches the headphones you rated highly, X3 is somewhat similar,
and X2 is similar to products you disliked.

YOUR JOB:

1. For EACH candidate item in {candidate_ids}, infer how much YOU would like it,
   based on:
   - MF model match scores (if provided above) — these are learned from millions of
     interactions and should be given HIGH WEIGHT in your ranking
   - your past reviews and star ratings
   - how similar the item is (in category, attributes, description, etc.)
     to items you rated 4–5 stars versus 1–2 stars

2. Use that reasoning to RANK ALL candidate items from most preferred to least
   preferred.

When ranking:
- STRONGLY PRIORITIZE items with high MF scores (8-10) — the collaborative filtering
  model has strong evidence these match your preferences
- Also consider items with medium MF scores (5-7) if they match your review text
  or explicit preferences
- Generally DEPRIORITIZE items with low MF scores (0-4) unless your review history
  shows a clear pattern contradicting the MF prediction
- Strongly INCREASE the rank for items whose attributes/description closely match
  products you rated 4–5 stars in your history
- Strongly DECREASE the rank for items similar to products you rated 1–2 stars
- Pay attention to specific aspects mentioned in your reviews (e.g., durability,
  comfort, writing style, genre, features) and prefer items that match those aspects
- If an item is in a category you never interacted with and it doesn't resemble your
  liked items, put it lower in the ranking

Your goal is to combine MF model predictions (which capture collaborative patterns)
with your personal review history to make the best ranking.

OUTPUT FORMAT (very important):

Return ONLY a Python-style list of strings, with NO extra text, in this exact structure:

['item_id_1', 'item_id_2', 'item_id_3', ...]

RULES:
- There must be exactly {len(candidate_ids)} entries.
- Each element must be one of: {candidate_ids}
- Do NOT invent new IDs.
- Do NOT omit any candidate.
"""

        result = self.reasoning(task_description)

        # Parse a Python-style list of item_ids, like the baseline agents.
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                list_str = match.group()
            else:
                print("No list found in reasoning output.")
                return ['']

            try:
                parsed = eval(list_str)
            except Exception as e:
                print(f"eval failed on reasoning output: {e}")
                return ['']

            # Basic sanity checks: list of strings and same candidates
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

            # Append any missing candidates at the end to keep length consistent
            for cid in candidate_ids:
                if cid not in cleaned:
                    cleaned.append(cid)

            # Combine model (BPR/MF), LLM, and heuristic rankings (rank fusion).
            # Heavily weight the trained model since it's optimized for this task.
            final_ranking = cleaned
            try:
                rank_l = {cid: i for i, cid in enumerate(cleaned)}
                rank_h = {cid: i for i, cid in enumerate(heuristic_ranking)} if heuristic_ranking else {}
                rank_m = {cid: i for i, cid in enumerate(mf_ranking)} if mf_ranking else {}
                max_rank = len(candidate_ids)
                combined = []
                for cid in candidate_ids:
                    rl = rank_l.get(cid, max_rank)
                    rh = rank_h.get(cid, max_rank)
                    rm = rank_m.get(cid, max_rank)
                    # Very heavy weight on trained model (BPR/MF), light weight on LLM/heuristic
                    # Model is trained on millions of interactions and optimized for ranking
                    if mf_ranking:
                        # 85% model, 10% LLM, 5% heuristic
                        score = -(0.85 * rm + 0.10 * rl + 0.05 * rh)
                    else:
                        # No model available, fall back to LLM + heuristic
                        score = -(0.7 * rl + 0.3 * rh) if heuristic_ranking else -rl
                    combined.append((cid, score))
                combined.sort(key=lambda x: x[1], reverse=True)
                final_ranking = [cid for cid, _ in combined]
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
                    trajectory = {
                        "user_id": user_id,
                        "candidate_list": candidate_ids,
                        "top_choice": top_choice,
                        "mf_top_choice": mf_ranking[0] if mf_ranking else None,
                        "ranking": final_ranking[:5],  # Store top 5
                    }
                    trajectory_str = (
                        f"User {user_id} recommendation: top choice was {top_choice}. "
                        f"MF model predicted {trajectory['mf_top_choice']}. "
                        f"Top 5 ranking: {final_ranking[:5]}"
                    )
                    self.memory(f"review:{trajectory_str}")
                except Exception as e:
                    logging.warning(f"Memory storage failed: {e}")

            return final_ranking

        except Exception as e:
            print(f'format error: {e}')
            return ['']


if __name__ == "__main__":
    task_set = "amazon"  # or "goodreads" / "yelp" if you change paths
    # Initialize Simulator
    simulator = Simulator(
        data_dir="/srv/output/data1/output",
        device="auto",
        cache=False  # Disabled to avoid permission errors with cache directory
    )

    # Load scenarios
    simulator.set_task_and_groundtruth(
        task_dir=f"/srv/CS_245_Project/example/track2/{task_set}/tasks",
        groundtruth_dir=f"/srv/CS_245_Project/example/track2/{task_set}/groundtruth"
    )

    # Set your custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    simulator.set_llm(GeminiLLM(api_key=gemini_api_key))

    # Run evaluation (all tasks)
    # Reduced workers to avoid OOM - WARP model is large (~3.5GB+)
    agent_outputs = simulator.run_simulation(
        number_of_tasks=None,
        enable_threading=True,
        max_workers=10
    )

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(
        '/srv/CS_245_Project/example/gemini_base_agent_evaluation_results.json',
        'w'
    ) as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
