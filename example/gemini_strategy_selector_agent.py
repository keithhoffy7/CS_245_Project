import argparse
import json
import logging
import os
import re

from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import (
    PlanningBase,
    PlanningIO,
    PlanningDEPS,
    PlanningTD,
    PlanningVoyager,
    PlanningOPENAGI,
    PlanningHUGGINGGPT,
)
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
from websocietysimulator.agent.modules.memory_modules import (
    MemoryBase,
    MemoryDILU,
    MemoryGenerative,
    MemoryTP,
    MemoryVoyager,
)
import tiktoken


logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except Exception:
        return 0


# ---- Factory helpers ------------------------------------------------------


PLANNING_STRATEGIES = {
    "none": None,
    "io": PlanningIO,
    "deps": PlanningDEPS,
    "td": PlanningTD,
    "voyager": PlanningVoyager,
    "openagi": PlanningOPENAGI,
    "hugginggpt": PlanningHUGGINGGPT,
}

REASONING_STRATEGIES = {
    "base": None,  # handled by RecReasoningBase
    "io": ReasoningIO,
    "cot": ReasoningCOT,
    "cotsc": ReasoningCOTSC,
    "cot_sc": ReasoningCOTSC,
    "tot": ReasoningTOT,
    "dilu": ReasoningDILU,
    "self_refine": ReasoningSelfRefine,
    "step_back": ReasoningStepBack,
}

MEMORY_STRATEGIES = {
    "none": None,
    "dilu": MemoryDILU,
    "generative": MemoryGenerative,
    "tp": MemoryTP,
    "voyager": MemoryVoyager,
}


class RecReasoningBase(ReasoningBase):
    """
    Simple reasoning module that just forwards the task description.
    Matches the baseline behavior in `gemini_reasoning_agent.RecReasoningBase`.
    """

    def __init__(self, profile_type_prompt: str, llm: LLMBase):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str, feedback: str = ""):
        prompt = f"""
{task_description}
"""
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )
        return reasoning_result


def build_planning_module(llm: LLMBase, strategy: str) -> PlanningBase | None:
    key = (strategy or "none").strip().lower()
    cls = PLANNING_STRATEGIES.get(key)
    if cls is None:
        return None
    return cls(llm=llm)


def build_memory_module(llm: LLMBase, strategy: str) -> MemoryBase | None:
    key = (strategy or "none").strip().lower()
    cls = MEMORY_STRATEGIES.get(key)
    if cls is None:
        return None
    return cls(llm=llm)


def build_reasoning_module(
    llm: LLMBase, strategy: str, memory: MemoryBase | None
) -> ReasoningBase:
    key = (strategy or "base").strip().lower()
    if key in (None, "", "base"):
        return RecReasoningBase(profile_type_prompt="", llm=llm)

    cls = REASONING_STRATEGIES.get(key)
    if cls is None:
        logging.warning(
            "Unknown reasoning strategy '%s', falling back to base reasoning.", strategy
        )
        return RecReasoningBase(profile_type_prompt="", llm=llm)

    return cls(profile_type_prompt="", memory=memory, llm=llm)


class StrategySelectorRecommendationAgent(RecommendationAgent):
    """
    Recommendation agent that wires together configurable planning, reasoning,
    and memory strategies.
    """

    # Class-level defaults that can be set before simulation starts
    planning_strategy_default: str = "none"
    reasoning_strategy_default: str = "base"
    memory_strategy_default: str = "none"

    def __init__(
        self,
        llm: LLMBase,
        planning_strategy: str | None = None,
        reasoning_strategy: str | None = None,
        memory_strategy: str | None = None,
    ):
        super().__init__(llm=llm)

        if planning_strategy is None:
            planning_strategy = self.planning_strategy_default
        if reasoning_strategy is None:
            reasoning_strategy = self.reasoning_strategy_default
        if memory_strategy is None:
            memory_strategy = self.memory_strategy_default

        self.memory = build_memory_module(llm=self.llm, strategy=memory_strategy)
        self.planning = build_planning_module(llm=self.llm, strategy=planning_strategy)
        self.reasoning = build_reasoning_module(
            llm=self.llm, strategy=reasoning_strategy, memory=self.memory
        )

        logging.info(
            "Using strategies - planning: %s, reasoning: %s, memory: %s",
            planning_strategy,
            reasoning_strategy,
            memory_strategy,
        )

    def workflow(self):
        """
        Simulate user behavior.
        Returns:
            list: Sorted list of item IDs
        """
        # ---- High-level planning -------------------------------------------------
        #
        # Default static plan if no planner is provided, or if the planner fails.
        plan = [
            {"description": "First I need to find user information"},
            {"description": "Next, I need to find item information"},
            {"description": "Next, I need to find review information"},
        ]

        # If a planning module exists, build a dynamic plan for this recommendation
        # task. We keep a conservative fallback to the static plan to remain
        # compatible with baselines and avoid hard failures.
        dynamic_plan = None
        if getattr(self, "planning", None) is not None:
            try:
                planning_task_description = (
                    "You need to make a personalized item recommendation.\n"
                    f"User id: {self.task.get('user_id')}\n"
                    f"Candidate items: {self.task.get('candidate_list')}\n"
                    "You can query: user profile, candidate item metadata, and "
                    "historical user reviews."
                )
                dynamic_plan = self.planning(
                    task_type="recommendation",
                    task_description=planning_task_description,
                    feedback="",
                    few_shot="",
                )
            except Exception:
                dynamic_plan = None

        if isinstance(dynamic_plan, list) and dynamic_plan:
            plan = dynamic_plan

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
                    filtered_item = {
                        key: item[key] for key in keys_to_extract if key in item
                    }
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

        # Optional: retrieve relevant past trajectories from memory, if enabled
        memory_context = ""
        if getattr(self, "memory", None) is not None:
            try:
                query = f"recommendation task; user_id={self.task['user_id']}; candidates={self.task['candidate_list']}"
                memory_context = self.memory(query)
            except Exception:
                memory_context = ""

        memory_block = ""
        if memory_context:
            memory_block = (
                "Here are some past successful recommendation trajectories that may help you:\n"
                f"{memory_context}\n\n"
            )

        # Make the (possibly dynamic) plan explicit in the task description so that
        # the reasoning module can condition on it.
        plan_descriptions = [step.get("description", "") for step in plan]
        plan_block = ""
        if any(plan_descriptions):
            serialized_plan = "\n".join(
                f"- {desc}" for desc in plan_descriptions if desc
            )
            plan_block = (
                "Here is the high-level plan you should conceptually follow:\n"
                f"{serialized_plan}\n\n"
            )

        task_description = f"""
        You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        Please rank the more interested items more front in your rank list.
        The information of the above 20 candidate items is as follows: {item_list}.

        {plan_block}{memory_block}Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
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
            parsed_result = eval(result)
            print("Processed Output:", parsed_result)

            # Optional: store current trajectory into memory, if enabled
            if getattr(self, "memory", None) is not None:
                try:
                    trajectory = {
                        "user_id": self.task.get("user_id"),
                        "candidate_list": self.task.get("candidate_list"),
                        "history_review": history_review,
                        "ranking": parsed_result,
                    }
                    self.memory(f"review:{trajectory}")
                except Exception:
                    pass

            return parsed_result
        except Exception:
            print("format error")
            return [""]


def build_output_filename(task_set: str, planning: str, reasoning: str, memory: str) -> str:
    safe = lambda s: (s or "none").replace("/", "_")
    return (
        f"/srv/CS_245_Project/example/"
        f"gemini_strategy_selector_"
        f"task-{safe(task_set)}_"
        f"plan-{safe(planning)}_"
        f"reason-{safe(reasoning)}_"
        f"mem-{safe(memory)}.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gemini recommendation agent with configurable strategies."
    )
    parser.add_argument(
        "--task-set",
        type=str,
        default="amazon",
        help='Task set to use (e.g., "amazon", "goodreads", "yelp").',
    )
    parser.add_argument(
        "--planning-strategy",
        type=str,
        default="none",
        choices=sorted(PLANNING_STRATEGIES.keys()),
        help=f"Planning strategy. Options: {', '.join(sorted(PLANNING_STRATEGIES.keys()))}",
    )
    parser.add_argument(
        "--reasoning-strategy",
        type=str,
        default="base",
        choices=sorted(set(k for k in REASONING_STRATEGIES.keys()) | {"base"}),
        help=(
            "Reasoning strategy. "
            "Options: base, " + ", ".join(sorted(k for k in REASONING_STRATEGIES.keys() if k))
        ),
    )
    parser.add_argument(
        "--memory-strategy",
        type=str,
        default="none",
        choices=sorted(MEMORY_STRATEGIES.keys()),
        help=f"Memory strategy. Options: {', '.join(sorted(MEMORY_STRATEGIES.keys()))}",
    )
    args = parser.parse_args()

    task_set = args.task_set

    simulator = Simulator(
        data_dir="/srv/output/data1/output",
        device="auto",
        cache=False,
    )
    simulator.set_task_and_groundtruth(
        task_dir=f"/srv/CS_245_Project/example/track2/{task_set}/tasks",
        groundtruth_dir=f"/srv/CS_245_Project/example/track2/{task_set}/groundtruth",
    )

    # Configure class-level defaults and register the agent class itself
    StrategySelectorRecommendationAgent.planning_strategy_default = args.planning_strategy
    StrategySelectorRecommendationAgent.reasoning_strategy_default = args.reasoning_strategy
    StrategySelectorRecommendationAgent.memory_strategy_default = args.memory_strategy
    simulator.set_agent(StrategySelectorRecommendationAgent)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    simulator.set_llm(GeminiLLM(api_key=gemini_api_key))

    agent_outputs = simulator.run_simulation(
        number_of_tasks=None,
        enable_threading=True,
        max_workers=10,
    )

    evaluation_results = simulator.evaluate()
    output_path = build_output_filename(
        task_set=task_set,
        planning=args.planning_strategy,
        reasoning=args.reasoning_strategy,
        memory=args.memory_strategy,
    )
    with open(output_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"Evaluation results saved to: {output_path}")
    print(f"The evaluation_results is :{evaluation_results}")
