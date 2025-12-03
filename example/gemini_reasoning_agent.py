import json
import os
import re
import logging
import argparse

from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningBase,
    ReasoningIO,
    ReasoningCOT,
    ReasoningCOTSC,
    ReasoningTOT,
    ReasoningDILU,
    ReasoningSelfRefine,
    ReasoningStepBack,
)

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except Exception:
        print(encoding.encode(string))
        a = 0
    return a


class RecPlanning(PlanningBase):
    """Inherits from PlanningBase (same as in gemini_base_agent)."""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == "":
            prompt = """You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

Task: {task_description}
"""
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = """You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
"""
            prompt = prompt.format(
                example=few_shot,
                task_description=task_description,
                task_type=task_type,
                feedback=feedback,
            )
        return prompt


class RecReasoningBase(ReasoningBase):
    """
    Simple reasoning module that just forwards the task description.
    This provides a baseline similar to the original gemini_base_agent RecReasoning.
    """

    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        prompt = """
{task_description}
"""
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )
        return reasoning_result


def build_reasoning_module(llm, strategy: str) -> ReasoningBase:
    """
    Factory to build a reasoning module based on a strategy name.

    Supported strategy names (case-insensitive):
        - "base"          -> RecReasoningBase (simple, no special prompting)
        - "io"            -> ReasoningIO
        - "cot"           -> ReasoningCOT
        - "cotsc"         -> ReasoningCOTSC
        - "cot_sc"        -> ReasoningCOTSC
        - "tot"           -> ReasoningTOT
        - "dilu"          -> ReasoningDILU
        - "self_refine"   -> ReasoningSelfRefine
        - "step_back"     -> ReasoningStepBack

    If an unknown strategy is provided, falls back to "base".
    """
    strategy_normalized = (strategy or "").strip().lower()

    mapping = {
        "base": RecReasoningBase,
        "io": ReasoningIO,
        "cot": ReasoningCOT,
        "cotsc": ReasoningCOTSC,
        "cot_sc": ReasoningCOTSC,
        "tot": ReasoningTOT,
        "dilu": ReasoningDILU,
        "self_refine": ReasoningSelfRefine,
        "step_back": ReasoningStepBack,
    }

    cls = mapping.get(strategy_normalized, RecReasoningBase)
    # All Reasoning* classes share the same init signature: (profile_type_prompt, memory, llm)
    return cls(profile_type_prompt="", memory=None, llm=llm)


class MyRecommendationAgent(RecommendationAgent):
    """
    Recommendation agent identical to the one in gemini_base_agent,
    except the reasoning strategy is configurable via constructor argument
    `reasoning_strategy` or a class-level default set at runtime.
    """
    # Class-level default, can be overridden before simulation starts
    reasoning_strategy_default: str = "base"

    def __init__(self, llm: LLMBase, reasoning_strategy: str | None = None):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)

        if reasoning_strategy is None:
            reasoning_strategy = self.reasoning_strategy_default

        logging.info(f"Using reasoning strategy: {reasoning_strategy}")
        self.reasoning = build_reasoning_module(self.llm, reasoning_strategy)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        # plan = self.planning(task_type='Recommendation Task',
        #                      task_description="Please make a plan to query user information, you can choose to query user, item, and review information",
        #                      feedback='',
        #                      few_shot='')
        # print(f"The plan is :{plan}")
        plan = [
            {"description": "First I need to find user information"},
            {"description": "Next, I need to find item information"},
            {"description": "Next, I need to find review information"},
        ]

        user = ""
        item_list = []
        history_review = ""
        for sub_task in plan:
            if "user" in sub_task["description"]:
                user = str(self.interaction_tool.get_user(user_id=self.task["user_id"]))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif "item" in sub_task["description"]:
                for n_bus in range(len(self.task["candidate_list"])):
                    item = self.interaction_tool.get_item(
                        item_id=self.task["candidate_list"][n_bus]
                    )
                    keys_to_extract = [
                        "item_id",
                        "name",
                        "stars",
                        "review_count",
                        "attributes",
                        "title",
                        "average_rating",
                        "rating_number",
                        "description",
                        "ratings_count",
                        "title_without_series",
                    ]
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                item_list.append(filtered_item)
            elif "review" in sub_task["description"]:
                history_review = str(
                    self.interaction_tool.get_reviews(user_id=self.task["user_id"])
                )
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(
                        encoding.encode(history_review)[:12000]
                    )
            else:
                pass
        task_description = f"""
        You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        Please rank the more interested items more front in your rank list.
        The information of the above 20 candidate items is as follows: {item_list}.

        Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        DO NOT output your analysis process!

        The correct output format:

        ['item id1', 'item id2', 'item id3', ...]

        """
        result = self.reasoning(task_description)

        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
            print("Processed Output:", eval(result))
            return eval(result)
        except Exception:
            print("format error")
            return [""]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gemini reasoning agent with a chosen reasoning strategy."
    )
    parser.add_argument(
        "--reasoning-strategy",
        type=str,
        default="base",
        help=(
            "Reasoning strategy to use. "
            "Options: base, io, cot, cotsc, cot_sc, tot, dilu, self_refine, step_back"
        ),
    )
    parser.add_argument(
        "--task-set",
        type=str,
        default="amazon",
        help='Task set to use (e.g., "amazon", "goodreads", "yelp").',
    )
    args = parser.parse_args()

    task_set = args.task_set  # "goodreads" or "yelp"

    # Initialize Simulator
    simulator = Simulator(data_dir="/srv/output/data1/output", device="auto", cache=False)

    # Load scenarios
    simulator.set_task_and_groundtruth(
        task_dir=f"/srv/CS_245_Project/example/track2/{task_set}/tasks",
        groundtruth_dir=f"/srv/CS_245_Project/example/track2/{task_set}/groundtruth",
    )

    # Set the default reasoning strategy on the agent class,
    # then register the class itself with the simulator.
    MyRecommendationAgent.reasoning_strategy_default = args.reasoning_strategy
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    simulator.set_llm(GeminiLLM(api_key=gemini_api_key))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(
        number_of_tasks=None, enable_threading=True, max_workers=10
    )

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(
        "/srv/CS_245_Project/example/gemini_reasoning_agent_evaluation_results.json", "w"
    ) as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
