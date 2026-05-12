import glob
import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from agent.dispatch import global_dispatcher
from agent.tools.filters.discipline_filter import run_discipline_discovery
from agent.utils import get_agent_tgt_dataset_path, time_logger


@tool("discipline_discovery_agent", return_direct=False)
@time_logger
def discipline_discovery_agent(
    dataset_dir: str,
    target_disciplines: Optional[List[str]] = None,
    n_clusters: int = 20,
    samples_per_cluster: int = 8,
) -> str:
    """Discipline Discovery Agent: Discovers discipline distribution in a multi-domain dataset using embedding clustering and LLM labeling, then filters by target disciplines.

    This agent should be used as the FIRST stage when working with multi-discipline datasets.
    It requires the dataset to have pre-computed embeddings (from the preprocessing stage).

    Args:
        dataset_dir: Absolute path to the dataset directory (must contain preprocessed JSON files with embeddings).
        target_disciplines: List of discipline names to keep. If None, discovers all disciplines and splits data into per-discipline files.
        n_clusters: Number of clusters for KMeans fallback (default 20). Higher values detect finer-grained disciplines.
        samples_per_cluster: Number of samples per cluster to show the LLM for labeling (default 8).
    """
    global_dispatcher.emit_tool_call(
        name="discipline_discovery_agent_start",
        args={
            "dataset_dir": dataset_dir,
            "target_disciplines": target_disciplines,
            "n_clusters": n_clusters,
        },
        agent="discipline_discovery",
    )

    try:
        result_json = run_discipline_discovery(
            dataset_dir=dataset_dir,
            target_disciplines=target_disciplines,
            n_clusters=n_clusters,
            samples_per_cluster=samples_per_cluster,
        )
        result = json.loads(result_json)

        output_dir = result.get("output_dir", "")

        global_dispatcher.emit_tool_call(
            name="discipline_discovery_agent_done",
            args={
                "status": "success",
                "disciplines_found": result.get("disciplines_found", []),
                "filtered_count": result.get("filtered_count", 0),
                "total_items": result.get("total_items", 0),
                "output_dir": output_dir,
            },
            agent="discipline_discovery",
        )

        return json.dumps({
            "status": "success",
            "output_dir": output_dir,
            "original_count": result.get("total_items", 0),
            "filtered_count": result.get("filtered_count", 0),
            "disciplines_found": result.get("disciplines_found", []),
            "discipline_counts": result.get("discipline_counts", {}),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        error_msg = f"Discipline discovery failed: {str(e)}"
        print(f"[DisciplineDiscovery] {error_msg}")
        global_dispatcher.emit_tool_call(
            name="discipline_discovery_agent_error",
            args={"error": str(e)[:200]},
            agent="discipline_discovery",
        )
        return json.dumps({"status": "failure", "error": error_msg})
