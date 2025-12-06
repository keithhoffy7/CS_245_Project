import json
import os
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, GeminiLLM
from collections import Counter
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryVoyager
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
    Participant's implementation of SimulationAgent
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.memory = MemoryVoyager(llm=self.llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """

        def extract_candidate_keywords(items, max_keywords=8):
            stopwords = {
                'with', 'from', 'this', 'that', 'have', 'your', 'about', 'their',
                'which', 'will', 'into', 'these', 'those', 'because', 'there',
                'while', 'after', 'before', 'when', 'where', 'been', 'being',
                'also', 'only', 'such', 'most', 'more', 'make', 'made', 'into',
                'using', 'through', 'each', 'other', 'some', 'many', 'very',
                'than', 'over', 'under', 'between', 'among', 'good', 'great'
            }
            text_parts = []
            for item in items:
                title = item.get('title') or ''
                description = item.get('description') or ''

                # Some fields may be lists or other non-string types (always cast to string)
                if not isinstance(title, str):
                    title = str(title)
                if not isinstance(description, str):
                    description = str(description)

                text_parts.extend([title.lower(), description.lower()])
            combined_text = ' '.join(text_parts)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text)
            filtered = [w for w in words if w not in stopwords]
            word_counts = Counter(filtered)
            return [word for word, _ in word_counts.most_common(max_keywords)]

        def truncate_text(text: str, max_chars: int = 220) -> str:
            if not text:
                return ''
            return (text[:max_chars] + '...') if len(text) > max_chars else text

        def build_memory_summary(ranking, candidate_details, candidate_keywords, candidate_titles):
            top_items_details = []
            for idx, item_id in enumerate(ranking[:5], start=1):
                item_info = candidate_details.get(item_id)
                if not item_info:
                    continue
                title = item_info.get('title', 'Unknown item')
                avg_rating = item_info.get('average_rating', 'N/A')
                rating_number = item_info.get('rating_number', 'N/A')
                description = truncate_text(item_info.get('description', ''))
                top_items_details.append(
                    f"{idx}. {title} (rating {avg_rating}, {rating_number} reviews) – {description}"
                )

            keyword_sentence = ""
            if candidate_keywords:
                keyword_sentence = f"Key traits emphasized in this success: {', '.join(candidate_keywords[:8])}."

            candidate_titles_sentence = ""
            if candidate_titles:
                candidate_titles_sentence = (
                    f"Candidate pool included items such as {', '.join(candidate_titles[:5])}."
                )

            summary_sections = [
                "Successful recommendation scenario summary:",
                keyword_sentence,
                candidate_titles_sentence,
                "Top ranked items and notes:",
                '\n'.join(
                    top_items_details) if top_items_details else "Details unavailable."
            ]
            return '\n'.join(section for section in summary_sections if section)

        item_list = []
        for n_bus in range(len(self.task['candidate_list'])):
            item = self.interaction_tool.get_item(
                item_id=self.task['candidate_list'][n_bus])
            keys_to_extract = ['item_id', 'title',
                               'average_rating', 'rating_number', 'description']
            filtered_item = {key: item[key]
                             for key in keys_to_extract if key in item}
            item_list.append(filtered_item)

        candidate_details = {
            item['item_id']: item for item in item_list if item.get('item_id')}
        candidate_titles = [item.get('title')
                            for item in item_list if item.get('title')]
        candidate_keywords = extract_candidate_keywords(item_list)

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

        reviewed_items = []
        for review in history_review_dict:
            item_id = review.get('item_id')
            item = self.interaction_tool.get_item(item_id=item_id)
            keys_to_extract = ['item_id', 'title',
                               'average_rating', 'rating_number', 'description']
            filtered_item = {key: item[key]
                             for key in keys_to_extract if key in item}
            reviewed_items.append(filtered_item)

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

        readable_item_list_prompt = f'''
        Below is a list of some information about certain candidate products. Please make this information into text that is readable and easy to understand.
        The information shown for each item should include the title, item id, average rating, rating number, and description in text format.
        
        {item_list}

        Your final output should only be this list of item information, DO NOT introduce any other text! Please number the items as well.
        '''

        readable_item_list = self.reasoning(readable_item_list_prompt)

        # Retrieve similar past successful trajectories from memory using product-level traits
        memory_context = ""
        if getattr(self, "memory", None) is not None:
            try:
                memory_snippets = []
                keyword_phrase = ', '.join(
                    candidate_keywords[:6]) if candidate_keywords else ''
                if keyword_phrase:
                    trait_query = (
                        f"successful recommendation scenario focusing on {keyword_phrase} products"
                    )
                    trait_memory = self.memory(trait_query)
                    if trait_memory:
                        memory_snippets.append(trait_memory)

                illustrative_titles = [
                    title for title in candidate_titles[:5] if title]
                if illustrative_titles:
                    title_query = (
                        "recommendation summary involving products like "
                        f"{', '.join(illustrative_titles)}"
                    )
                    title_memory = self.memory(title_query)
                    if title_memory and title_memory not in memory_snippets:
                        memory_snippets.append(title_memory)

                # Fallback: if no specific trait/title memories found, retrieve a general successful scenario
                if not memory_snippets:
                    generic_query = "successful recommendation scenario for amazon products"
                    generic_memory = self.memory(generic_query)
                    if generic_memory:
                        memory_snippets.append(generic_memory)

                if memory_snippets:
                    memory_context = '\n'.join(memory_snippets)
            except Exception as e:
                logging.warning(f"Memory retrieval failed: {e}")
                memory_context = ""

        # Build memory section with proper introduction if memory exists
        memory_section = ""
        if memory_context:
            memory_section = f'''
        IMPORTANT CONTEXT FROM PAST SUCCESSFUL RECOMMENDATIONS:
        Below are examples of past successful recommendation rankings that you have made in similar situations. 
        These examples show patterns of how you ranked items based on your preferences and review history.
        Use these examples as reference to understand your ranking behavior and preferences, but remember that 
        each situation is unique. Consider the patterns shown in these examples when making your current ranking.
        
        {memory_context}
        
        '''

        task_description = f'''
        You are a real user on amazon's online shopping platform. 
        You have reviewed some products in the past on amazon's site. 
        Below is a list of some of the products you have reviewed in the past.
        Each review contains the number of stars, the title of the review, the review text, and relevant information about the product:
        
        {merged_reviews}
        
        {memory_section}
        Now you need to rank the following 20 items: {self.task['candidate_list']} according to how likely you are to buy the product. 
        Make sure to use the information you have about yourself and the products you have reviewed in the past to make your ranking.
        If you have past successful recommendation examples above, use them as guidance to understand your preferences and ranking patterns.
        Please rank the items that you are most interested in higher in the rank list.
        The information of the above 20 candidate items is as follows: 
        
        {readable_item_list}

        Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        DO NOT output your analysis process!

        The correct output format:

        ['item id1', 'item id2', 'item id3', ...]

        '''

        result = self.reasoning(task_description)

        try:
            # print('Meta Output:',result)
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
            print('Processed Output:', eval(result))
            final_ranking = eval(result)

            # Store successful trajectory in memory for future tasks
            if getattr(self, "memory", None) is not None:
                try:
                    summary_entry = build_memory_summary(
                        final_ranking,
                        candidate_details,
                        candidate_keywords,
                        candidate_titles
                    )
                    self.memory(f"review:{summary_entry}")
                except Exception as e:
                    logging.warning(f"Memory storage failed: {e}")

            return final_ranking
        except:
            print('format error')
            return ['']


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
    with open(f'/srv/CS_245_Project/example/gemini_planning_context_memory_agent_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
