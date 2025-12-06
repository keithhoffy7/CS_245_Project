import json
import os
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


class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm, memory=None):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)

    def create_prompt(self, task_description: str, merged_reviews: str,
                      readable_item_list: str, candidate_ids: list[str]) -> str:
        prompt = '''You are a reasoning agent on an Amazon-style online shopping platform.
Your task is to rank products for a user based on their historical preferences and product information.

You have access to two key types of information:
1. USER + REVIEW: The user's historical product reviews and star ratings
2. ITEM: Detailed metadata for candidate products (titles, categories, ratings, descriptions, etc.)

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

STEP 4: Synthesize and rank
- Combine all signals: review history, item attributes, similarity to liked/disliked items
- Rank candidates from most preferred to least preferred
- Items matching highly-rated past purchases should rank higher
- Items similar to poorly-rated past purchases should rank lower

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

CANDIDATE LIST (you must rank all of these):
{candidate_ids}

Remember: Your output must be ONLY the ranked list, nothing else.
'''
        prompt = prompt.format(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids,
            num_candidates=len(candidate_ids)
        )
        return prompt

    def __call__(self, task_description: str, merged_reviews: str,
                 readable_item_list: str, candidate_ids: list[str]):
        """Override the parent class's __call__ method"""
        prompt = self.create_prompt(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids
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
    Participant's implementation of SimulationAgent
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(
            profile_type_prompt='', llm=self.llm, memory=None)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # Get candidate item information
        item_list = []
        for n_bus in range(len(self.task['candidate_list'])):
            item = self.interaction_tool.get_item(
                item_id=self.task['candidate_list'][n_bus])
            keys_to_extract = ['item_id', 'title',
                               'average_rating', 'rating_number', 'description']
            filtered_item = {key: item[key]
                             for key in keys_to_extract if key in item}
            item_list.append(filtered_item)

        # Get user's review history
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

        # Get item information for reviewed items
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

        # Merge reviews with item information
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

        messages_merge = [{"role": "user", "content": merged_reviews_prompt}]
        merged_reviews = self.llm(
            messages=messages_merge, temperature=0.1, max_tokens=4000)

        # Make candidate item list readable
        readable_item_list_prompt = f'''
        Below is a list of some information about certain candidate products. Please make this information into text that is readable and easy to understand.
        The information shown for each item should include the title, item id, average rating, rating number, and description in text format.
        
        {item_list}

        Your final output should only be this list of item information, DO NOT introduce any other text! Please number the items as well.
        '''

        messages_readable = [
            {"role": "user", "content": readable_item_list_prompt}]
        readable_item_list = self.llm(
            messages=messages_readable, temperature=0.1, max_tokens=4000)

        # Final ranking task
        candidate_ids = self.task['candidate_list']
        task_description = (
            "Rank the candidate products for this user based on their review history "
            "and item attributes."
        )

        result = self.reasoning(
            task_description=task_description,
            merged_reviews=merged_reviews,
            readable_item_list=readable_item_list,
            candidate_ids=candidate_ids
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

            # Filter for valid candidate_ids and preserve order
            candidate_set = set(candidate_ids)
            cleaned = []
            for cid in parsed:
                if cid in candidate_set and cid not in cleaned:
                    cleaned.append(cid)

            # Append any missing candidates at the end
            for cid in candidate_ids:
                if cid not in cleaned:
                    cleaned.append(cid)

            print('Processed Output:', cleaned)
            return cleaned

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
        number_of_tasks=None, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'/srv/CS_245_Project/example/gemini_planning_context_reasoning_agent_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
