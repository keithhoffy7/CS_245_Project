import json
import os
import pickle
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, GeminiLLM
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryVoyager
import re
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except Exception:
        print(encoding.encode(string))
        return 0
    return a


# ---- BPR model loading and ranking -----------------------------------------

BPR_MODEL_PATH = "/srv/CS_245_Project/example/bpr_model.pkl"
_bpr_model_cache = None


def load_bpr_model():
    """Lazy-load BPR model if available."""
    global _bpr_model_cache
    if _bpr_model_cache is not None:
        return _bpr_model_cache

    if os.path.exists(BPR_MODEL_PATH):
        try:
            with open(BPR_MODEL_PATH, "rb") as f:
                _bpr_model_cache = pickle.load(f)
            logging.info("Loaded BPR model for ranking (BPR).")
            return _bpr_model_cache
        except Exception as e:
            logging.warning("Failed to load BPR model: %s", e)

    logging.warning("BPR model not found. Using LLM + memory only.")
    _bpr_model_cache = None
    return None


def get_bpr_ranking(user_id, candidate_ids):
    """
    Get BPR ranking for candidates.
    Returns list of candidate_ids in BPR order, or None if model/user not available.
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

    # Append missing candidates at the end to preserve full candidate list
    for cid in candidate_ids:
        if cid not in ranked:
            ranked.append(cid)

    return ranked


class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}
'''
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=4000
        )

        return reasoning_result


class MyRecommendationAgent(RecommendationAgent):
    """
    Recommendation agent with Voyager memory on top of Keith's baseline.
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)
        # Voyager memory for cross-task few-shot recall
        self.memory = MemoryVoyager(llm=self.llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """

        candidate_ids = self.task['candidate_list']

        # Step 1: gather candidate item metadata
        item_list = []
        for n_bus in range(len(candidate_ids)):
            item = self.interaction_tool.get_item(
                item_id=candidate_ids[n_bus])
            keys_to_extract = ['item_id', 'title',
                               'average_rating', 'rating_number', 'description']
            filtered_item = {key: item[key]
                             for key in keys_to_extract if key in item}
            item_list.append(filtered_item)

        # Step 2: user review history
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

        # Step 3: item info for reviewed items
        reviewed_items = []
        for review in history_review_dict:
            item_id = review.get('item_id')
            if not item_id:
                continue
            item = self.interaction_tool.get_item(item_id=item_id)
            keys_to_extract = ['item_id', 'title',
                               'average_rating', 'rating_number', 'description']
            filtered_item = {key: item[key]
                             for key in keys_to_extract if key in item}
            reviewed_items.append(filtered_item)

        # Step 4: merge reviews with item info
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

        merged_reviews = self.reasoning(merged_reviews_prompt)

        # Step 5: make candidate list readable
        readable_item_list_prompt = f'''
        Below is a list of some information about certain candidate products. Please make this information into text that is readable and easy to understand.
        The information shown for each item should include the title, item id, average rating, rating number, and description in text format.
        
        {item_list}

        Your final output should only be this list of item information, DO NOT introduce any other text! Please number the items as well.
        '''

        readable_item_list = self.reasoning(readable_item_list_prompt)

        # Step 6: retrieve similar past trajectories from memory (if any)
        memory_context = ""
        if getattr(self, "memory", None) is not None:
            try:
                # Simple scenario descriptor: user_id + first few candidates
                memory_query = (
                    f"user_id={self.task['user_id']} "
                    f"candidates={candidate_ids[:5]}"
                )
                memory_context = self.memory(memory_query) or ""
                if memory_context:
                    print("Retrieved memory context:\n", memory_context)
            except Exception as e:
                logging.warning(f"Memory retrieval failed: {e}")
                memory_context = ""

        memory_section = ""
        if memory_context:
            memory_section = f"""
IMPORTANT CONTEXT FROM PAST SUCCESSFUL RECOMMENDATIONS:
{memory_context}
"""

        # Optional: BPR collaborative filtering hint
        bpr_section = ""
        bpr_ranking = get_bpr_ranking(self.task['user_id'], candidate_ids)
        if bpr_ranking:
            bpr_top5 = bpr_ranking[:5]
            print("BPR ranking (top 10):", bpr_ranking[:10])
            bpr_section = f"""
NOTE: A collaborative filtering model (BPR) trained on many user-item interactions
suggests these items might be highly relevant for this user: {', '.join(bpr_top5)}.
You should consider these as strong candidates, especially when they match the user's review history.
"""

        # Step 7: final ranking prompt with optional memory section
        task_description = f'''
        You are a real user on amazon's online shopping platform. 
        You have reviewed some products in the past on amazon's site. 
        Below is a list of some of the products you have reviewed in the past.
        Each review contains the number of stars, the title of the review, the review text, and relevant information about the product:
        
        {merged_reviews}

        {memory_section}
        {bpr_section}
        Now you need to rank the following 20 items: {candidate_ids} according to how likely you are to buy the product. 
        Make sure to use the information you have about yourself and the products you have reviewed in the past to make your ranking.
        Please rank the items that you are most interested in higher in the rank list.
        The information of the above 20 candidate items is as follows: 
        
        {readable_item_list}

        Your final output should be ONLY a ranked item list of {candidate_ids} with the following format, DO NOT introduce any other item ids!
        DO NOT output your analysis process!

        The correct output format:

        ['item id1', 'item id2', 'item id3', ...]

        '''

        result = self.reasoning(task_description)

        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
                return ['']

            final_ranking = eval(result)
            print('Processed Output:', final_ranking)

            # Step 8: store a concise memory snippet for future tasks
            if getattr(self, "memory", None) is not None:
                try:
                    memory_entry = (
                        f"user_id={self.task['user_id']} "
                        f"top_choice={final_ranking[0] if final_ranking else 'N/A'} "
                        f"candidates={self.task['candidate_list'][:5]} "
                        f"ranking={final_ranking[:5]}"
                    )
                    print("Storing memory entry:\n", memory_entry)
                    self.memory(f"review:{memory_entry}")
                except Exception as e:
                    logging.warning(f"Memory storage failed: {e}")

            return final_ranking
        except Exception as e:
            print('format error')
            logging.warning(f"Ranking parse failed: {e}")
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
        number_of_tasks=None, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'/srv/CS_245_Project/example/gemini_planning_context_bpr_memory_agent_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
