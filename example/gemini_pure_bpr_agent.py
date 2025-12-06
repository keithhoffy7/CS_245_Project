import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase, GeminiLLM
import logging

logging.basicConfig(level=logging.INFO)

BPR_MODEL_PATH = "/srv/CS_245_Project/example/bpr_model.pkl"

_bpr_model_cache: Optional[Dict] = None


def load_bpr_model() -> Optional[Dict]:
    global _bpr_model_cache
    if _bpr_model_cache is not None:
        return _bpr_model_cache

    if os.path.exists(BPR_MODEL_PATH):
        try:
            with open(BPR_MODEL_PATH, "rb") as f:
                _bpr_model_cache = pickle.load(f)
            logging.info("Loaded BPR model.")
            return _bpr_model_cache
        except Exception as e:
            logging.warning("Failed to load BPR model: %s", e)

    logging.warning(
        "BPR model not found. Pure BPR agent will fall back to input order.")
    _bpr_model_cache = None
    return None


def get_bpr_ranking(user_id: str, candidate_ids: List[str], return_scores: bool = False):
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

    # collect indices for candidates
    existing = []
    for cid in candidate_ids:
        idx = item_id_to_idx.get(cid)
        if idx is not None and 0 <= idx < item_factors.shape[0]:
            existing.append((cid, idx))

    if not existing:
        return None

    # Compute dot product scores between user and candidate
    u_vec = user_factors[u_idx]
    cand_indices = np.array([idx for _, idx in existing], dtype=np.int64)
    cand_vecs = item_factors[cand_indices]
    scores = cand_vecs @ u_vec

    # Build ranking
    scored = list(zip([cid for cid, _ in existing], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = [cid for cid, _ in scored]

    # scores dictionary
    scores_dict = {cid: float(score) for cid, score in scored}

    # add missing candidates at the end
    for cid in candidate_ids:
        if cid not in ranked:
            ranked.append(cid)
            scores_dict[cid] = None

    if return_scores:
        return ranked, scores_dict
    return ranked


class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """

        candidate_ids = self.task['candidate_list']
        user_id = self.task['user_id']

        result = get_bpr_ranking(user_id, candidate_ids, return_scores=True)
        if result:
            bpr_ranking, scores_dict = result

            # print(f"\n{'='*80}")
            # print(f"User: {user_id}")
            # print(f"BPR Scores (sorted by score, descending):")
            # print(f"{'='*80}")
            # sorted_by_score = sorted(scores_dict.items(), key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
            # for rank, (item_id, score) in enumerate(sorted_by_score[:20], 1):
            #     if score is not None:
            #         print(f"  {rank:2d}. {item_id}: {score:8.4f}")
            #     else:
            #         print(f"  {rank:2d}. {item_id}: (not in BPR model)")
            # print(f"{'='*80}")
            # print(f"BPR ranking (top 10): {bpr_ranking[:10]}\n")

            return bpr_ranking

        # if no BPR ranking available, just return unchanged candidates list
        logging.warning(
            f"No BPR ranking available for user {user_id}; returning input order."
        )
        return candidate_ids


if __name__ == "__main__":
    task_set = "amazon"  # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(
        data_dir="/srv/output/data1/output", device="auto", cache=False)

    # Load scenarios
    simulator.set_task_and_groundtruth(
        task_dir=f"/srv/CS_245_Project/example/track2/{task_set}/tasks", groundtruth_dir=f"/srv/CS_245_Project/example/track2/{task_set}/groundtruth")

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
    with open('/srv/CS_245_Project/example/gemini_pure_bpr_agent_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
