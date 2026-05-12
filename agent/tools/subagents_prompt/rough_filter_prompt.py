INVOKE_SYS_PROMPT = "- **rough_filter_agent**: Filters samples using rule-based heuristics (length, format, regex patterns)."

PLANNER_SYS_PROMPT = "- **rough_filter_agent**: Applies rule-based filtering (length limits, refusal detection, noise ratio, language ratio) to remove obviously low-quality samples."

EXECUTOR_SYS_PROMPT = """- **rough_filter_agent**: (dataset_dir: str, policy: dict)
    - dataset_dir: Absolute path to the dataset directory. The agent processes all .json files and saves results to a 'rough/' subdirectory.
    - policy: A non-empty dict defining filter rules. You MUST include ALL keys below \u2014 do not add, remove, or rename any key:
        - min_total_len (int): Minimum total sample length. Default: 40
        - max_total_len (int): Maximum total sample length. Default: 16384
        - min_inst_len (int): Minimum instruction length. Default: 5
        - min_out_len (int): Minimum output length. Default: 15
        - out_inst_ratio_min (float): Minimum output/instruction length ratio. Default: 0.2
        - refusal_phrases (list[str]): Phrases indicating refusals. Default: ["sorry", "I cannot answer", "As an AI"]
        - noise_regex (str): Regex for noise characters. Default: r"[^\u4e00-\u9fa5\w\s,.;!?()/\uff0c\u3002\uff01\uff1f]"
          IMPORTANT: Adjust for domain-specific datasets:
          * Math: allow \u2211, \u222b, \u221a, \u221e, \u03c0, +, -, *, ^, _, etc.
          * Code: allow {}, [], <>, |, &, #, etc.
        - max_noise_ratio (float): Maximum noise-to-total ratio. Default: 0.15
        - require_en (bool): Whether to require English content. Default: true
        - min_en_ratio (float): Minimum English character ratio. Default: 0.3
"""
