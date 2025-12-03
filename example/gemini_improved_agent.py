import json
import os
import argparse
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import (
    PlanningBase, PlanningIO, PlanningDEPS, PlanningTD, 
    PlanningVoyager, PlanningOPENAGI, PlanningHUGGINGGPT
)
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

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

Task: {task_description}
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

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
            max_tokens=1000
        )
        
        return reasoning_result

class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase, planning_strategy='custom'):
        """
        Initialize the recommendation agent
        
        Args:
            llm: LLM instance
            planning_strategy: Planning strategy to use. Options:
                - 'custom': Use RecPlanning (default)
                - 'io': PlanningIO
                - 'deps': PlanningDEPS
                - 'td': PlanningTD (temporal dependencies)
                - 'voyager': PlanningVoyager
                - 'openagi': PlanningOPENAGI
                - 'hugginggpt': PlanningHUGGINGGPT
        """
        super().__init__(llm=llm)
        
        # Choose planning strategy
        if planning_strategy == 'custom':
            self.planning = RecPlanning(llm=self.llm)
        elif planning_strategy == 'io':
            self.planning = PlanningIO(llm=self.llm)
        elif planning_strategy == 'deps':
            self.planning = PlanningDEPS(llm=self.llm)
        elif planning_strategy == 'td':
            self.planning = PlanningTD(llm=self.llm)
        elif planning_strategy == 'voyager':
            self.planning = PlanningVoyager(llm=self.llm)
        elif planning_strategy == 'openagi':
            self.planning = PlanningOPENAGI(llm=self.llm)
        elif planning_strategy == 'hugginggpt':
            self.planning = PlanningHUGGINGGPT(llm=self.llm)
        else:
            raise ValueError(f"Unknown planning strategy: {planning_strategy}")
        
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)
        self.planning_strategy = planning_strategy

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # Generate plan using the selected planning strategy
        # Make the task description and example explicitly aligned with the Amazon-style recommendation setting.
        task_description = (
            "You are planning how to recommend products on an Amazon-style platform. "
            "You can use three kinds of information: (1) the target user's historical "
            "reviews and star ratings, (2) detailed metadata for a fixed list of "
            "candidate items (titles, categories, star ratings, review counts, "
            "attributes, descriptions, etc.), and (3) any additional user profile "
            "information if available. "
            "Decompose this into subgoals that clearly indicate when to query USER, "
            "ITEM, or REVIEW information via the available tools."
        )
        few_shot_example = (
            "I need to recommend 20 Amazon products to a user based on their past "
            "product reviews and ratings, and a given candidate item list."
        )
        
        try:
            plan = self.planning(
                task_type='Recommendation Task',
                task_description=task_description,
                feedback='',
                few_shot=few_shot_example
            )
            if not plan:
                print("Planner returned empty plan. Raw output likely not JSON-formatted.")
                raise ValueError("Empty plan")
            print(f"The generated plan is: {plan}")
            
            # Extract descriptions if plan is in dict format
            if plan and isinstance(plan[0], dict):
                plan_descriptions = [task.get('description', '') for task in plan]
            else:
                plan_descriptions = [str(task) for task in plan]
        except Exception as e:
            print(f"Planning failed: {e}. Using fallback plan.")
            # Attempt to log raw planner output if available
            if hasattr(self.planning, 'last_output'):
                print(f"Raw planner output: {self.planning.last_output}")
            # Fallback to hardcoded plan if planning fails
            plan = [
                {'description': 'First I need to find user information'},
                {'description': 'Next, I need to find item information'},
                {'description': 'Next, I need to find review information'}
            ]
            plan_descriptions = [task['description'] for task in plan]
            # Also log the fallback plan so every task shows a generated plan
            print(f"The fallback generated plan is: {plan}")

        user = ''
        item_list = []
        history_review = ''
        
        # Handle both dict format and string format plans
        if plan and isinstance(plan[0], dict):
            tasks_to_process = plan
        else:
            # Convert to dict format if needed
            tasks_to_process = [{'description': str(task)} for task in plan]
        
        for sub_task in tasks_to_process:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
                    keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                item_list.append(filtered_item)
                # print(item)
            elif 'review' in sub_task['description']:
                history_review = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        task_description = f'''
        You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        Please rank the more interested items more front in your rank list.
        The information of the above 20 candidate items is as follows: {item_list}.

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
            print('Processed Output:',eval(result))
            # time.sleep(4)
            return eval(result)
        except:
            print('format error')
            return ['']


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Gemini Recommendation Agent with different planning strategies')
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='custom',
        choices=['custom', 'io', 'deps', 'td', 'voyager', 'openagi', 'hugginggpt'],
        help='Planning strategy to use (default: custom)'
    )
    parser.add_argument(
        '--task-set',
        type=str,
        default='amazon',
        choices=['amazon', 'goodreads', 'yelp'],
        help='Task set to use (default: amazon)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Maximum number of worker threads (default: 10)'
    )
    parser.add_argument(
        '--num-tasks',
        type=int,
        default=None,
        help='Number of tasks to run (default: None, runs all tasks)'
    )
    
    args = parser.parse_args()
    
    task_set = args.task_set
    planning_strategy = args.strategy or os.getenv("PLANNING_STRATEGY", "custom")
    print(f"Using planning strategy: {planning_strategy}")
    print(f"Task set: {task_set}")
    
    # Initialize Simulator
    simulator = Simulator(data_dir="/srv/output/data1/output", device="auto", cache=False)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"/srv/CS_245_Project/example/track2/{task_set}/tasks", groundtruth_dir=f"/srv/CS_245_Project/example/track2/{task_set}/groundtruth")

    # Create a wrapper class that captures planning_strategy
    class AgentWrapper(MyRecommendationAgent):
        def __init__(self, llm):
            # Call parent with planning_strategy captured from outer scope
            super().__init__(llm=llm, planning_strategy=planning_strategy)
    
    # Set your custom agent 
    simulator.set_agent(AgentWrapper)

    # Set LLM client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    simulator.set_llm(GeminiLLM(api_key=gemini_api_key))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(
        number_of_tasks=args.num_tasks, 
        enable_threading=True, 
        max_workers=args.max_workers
    )

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'/srv/CS_245_Project/example/gemini_improved_agent_evaluation_{planning_strategy}_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
