INVOKE_SYS_PROMPT = "- **fine_filter_agent**: Scores sample quality via LLM on multiple dimensions, then keeps the top-ranked samples by weighted average."

PLANNER_SYS_PROMPT = "- **fine_filter_agent**: Uses LLM to score samples (1-5) on configurable quality dimensions, ranks by weighted total score, and retains the top percent."

EXECUTOR_SYS_PROMPT = """- **fine_filter_agent**: (dataset_dir: str, fine_filter_config: dict)
    - dataset_dir: Absolute path to the dataset directory. The agent processes all .json files and saves results to a 'fine/' subdirectory.
    - fine_filter_config: Configuration dict with one key:
        - dimensions (dict): Maps dimension names to {{"description": str, "weight": float}}. Weights MUST sum to 1.0.
          Example:
          {{
            "instruction_following": {{"description": "Does the output follow the instruction?", "weight": 0.4}},
            "logical_rigor": {{"description": "Is the reasoning logically consistent?", "weight": 0.4}},
            "information_density": {{"description": "Is the output comprehensive?", "weight": 0.2}}
          }}
          Choose dimensions and weights appropriate for the dataset and filtering goal.
"""
