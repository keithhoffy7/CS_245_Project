import re
import ast
import json

class PlanningBase():
    def __init__(self, llm):
        """
        Initialize the planning base class
        
        Args:
            llm: LLM instance used to generate planning
        """
        self.plan = []
        self.llm = llm
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        raise NotImplementedError("Subclasses should implement this method")
    
    def __call__(self, task_type, task_description, feedback, few_shot='few_shot'):
        prompt = self.create_prompt(task_type, task_description, feedback, few_shot)
        
        # Use the new LLM call method
        messages = [{"role": "user", "content": prompt}]
        string = self.llm(
            messages=messages,
            temperature=0.1
        )
        # Store raw LLM output for debugging
        self.last_output = string
        
        dicts = []
        dict_strings = re.findall(r"\{[^{}]*\}", string)
        if dict_strings:
            dicts = [ast.literal_eval(ds) for ds in dict_strings]
        else:
            json_match = re.search(r"\[[^\]]+\]", string, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, list):
                        dicts = parsed
                except json.JSONDecodeError:
                    pass

        if not dicts:
            bullet_lines = []
            for line in string.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if re.match(r"^(\d+\.|[-*•])\s+", stripped):
                    bullet_lines.append(re.sub(r"^(\d+\.|[-*•])\s*", "", stripped))
            if bullet_lines:
                dicts = [{"description": item} for item in bullet_lines]

        self.plan = dicts
        return self.plan
    
class PlanningIO(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class PlanningDEPS(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class PlanningTD(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
Your output format must follow the example below, specifying the order and dependencies.
The following are some examples:
Task: {example}

Task: {task_description}
'''
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
Your output format should follow the example below, specifying the order and dependencies.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)

class PlanningVoyager(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        base_prompt = '''You are a helpful assistant that generates subgoals to complete an {task_type} on an online shopping platform (similar to Amazon).
The goal is to recommend products to a user based on their historical reviews and a list of candidate items.
You must decompose the final recommendation task into a list of subgoals that can be executed in order.

Each subgoal should focus on using the available data sources:
- user profile and historical reviews: via interaction_tool.get_user(user_id=<user_id>) and interaction_tool.get_reviews(user_id=<user_id>)
- candidate item metadata (titles, categories, star ratings, review counts, attributes, descriptions, etc.): via interaction_tool.get_item(item_id=<item_id>)

Return ONLY a valid JSON array (no prose) where each element is an object containing:
    "description": short actionable text that clearly mentions whether it uses USER, ITEM, or REVIEW information,
    "reasoning instruction": why this step matters for improving product recommendations,
    "tool instruction": an explicit call pattern for the tool you would use, or "None" if no tool is needed.

Example output:
[
  {{"description": "Retrieve the target user's historical Amazon-style reviews and ratings",
    "reasoning instruction": "Understand the user's real purchase and review history to infer preferences",
    "tool instruction": "interaction_tool.get_reviews(user_id=<user_id>)"}},
  {{"description": "Fetch metadata (title, category, star rating, review_count, attributes, description) for each candidate item",
    "reasoning instruction": "Compare each product's attributes against the user's past liked items",
    "tool instruction": "interaction_tool.get_item(item_id=<item_id>)"}}
]

Do not wrap the JSON in markdown or explanations. The JSON array should be directly parseable.'''

        if feedback == '':
            prompt = f"""{base_prompt}
The following are some examples:
Task: {{example}}

Task: {{task_description}}
"""
        else:
            prompt = f"""{base_prompt}
The following are some examples:
Task: {{example}}

end
--------------------
reflexion:{{feedback}}
task:{{task_description}}
"""
        return prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)

class PlanningOPENAGI(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.  
Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
        else:
            prompt = '''You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.
Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)

class PlanningHUGGINGGPT(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

Task: {task_description}
'''
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
The following are some examples:
Task: {example}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
