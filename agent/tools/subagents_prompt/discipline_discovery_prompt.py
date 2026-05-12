INVOKE_SYS_PROMPT = "- **discipline_discovery_agent**: Discovers discipline distribution in multi-domain datasets using embedding clustering + LLM labeling, and filters by target disciplines."

PLANNER_SYS_PROMPT = "- **discipline_discovery_agent**: Uses pre-computed embeddings to cluster data, labels clusters via LLM to identify disciplines, then filters to keep only target-discipline samples. Should be the FIRST filtering stage for multi-discipline datasets."

EXECUTOR_SYS_PROMPT = """- **discipline_discovery_agent**: (dataset_dir: str, target_disciplines: list[str] | None, n_clusters: int, samples_per_cluster: int)
    - dataset_dir: Absolute path to the dataset directory. Must contain preprocessed JSON files with embeddings (either inline or as .index/.pkl files from preprocessing).
    - target_disciplines: A short list of BROAD discipline names to keep (e.g. ["mathematics"] or ["medicine"]).
      Do NOT list sub-fields — the agent uses LLM to semantically match discovered cluster labels to these targets.
      For example, passing ["mathematics"] will automatically match clusters labeled "linear algebra", "calculus", "statistics", etc.
      If None, discovers all disciplines and splits data into per-discipline files.
    - n_clusters: Number of clusters for KMeans (default 20). Use higher values (30-50) for very diverse datasets.
    - samples_per_cluster: Number of samples per cluster to show LLM for labeling (default 8).

    NOTE: The output directory will be a 'discipline/' subdirectory under the dataset_dir.
"""
