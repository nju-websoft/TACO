import glob
import os
import json
from typing import Dict, Any
from langchain_core.tools import tool
from agent.dispatch import global_dispatcher
from agent.tools.filters.rough_filter import run_rough_filter
from agent.utils import get_agent_tgt_dataset_path, time_logger


@tool("rough_filter_agent", return_direct=False)
@time_logger
def rough_filter_agent(dataset_dir: str, policy: Dict[str, Any]) -> str:
    """Rough Filter Agent: Rapidly filters a dataset based on a configuration policy dict without LLM.
    
    Args:
        dataset_dir: Absolute path to the dataset directory.
        policy: A dictionary containing filter rules. Must follow this structure:
             { 
                "min_total_len": int, minimum allowed total length of an example (inclusive)
                "max_total_len": int, maximum allowed total length of an example (inclusive)
                
                "min_inst_len": int, minimum allowed length of the instruction part (inclusive)
                "min_out_len": int, minimum allowed length of the output part (inclusive)
                "out_inst_ratio_min": float, minimum required ratio of output length to total length
                                
                "refusal_phrases": List[str], filter phrases that indicate refusal or inability to answer
                "noise_regex": str, regex pattern to match and filter out noise characters
                "max_noise_ratio": float, maximum allowed ratio of noise characters to total characters
                "require_en": bool, whether to require English characters in the dataset
                "min_en_ratio": float, minimum required ratio of English characters to total characters
             }
    """
    # Resolve targets
    if os.path.isdir(dataset_dir):
        targets = glob.glob(os.path.join(dataset_dir, "*.json"))
    elif "*" in dataset_dir or "?" in dataset_dir:
        targets = glob.glob(dataset_dir)
    else:
        targets = [dataset_dir]
        
    targets = [t for t in targets if os.path.exists(t) and "rough" not in t] # Avoid re-processing output files
    
    if not targets:
        return f"Error: No valid dataset files found matching {dataset_dir}"

    global_dispatcher.emit_tool_call(
        name="rough_filter_start",
        args={"file_count": len(targets), "policy_keys": list(policy.keys())},
        agent="rough_filter"
    )

    summary = []
    try:
        for i, tgt in enumerate(targets):
            global_dispatcher.emit_progress(
                name="rough_filter_progress",
                current=i + 1, total=len(targets),
                agent="rough_filter",
                file=os.path.basename(tgt)
            )
            result_json = run_rough_filter(tgt, policy)
            summary.append(json.loads(result_json))

        total_original = sum(d.get("original_count", 0) for d in summary)
        total_refined = sum(d.get("refined_count", 0) for d in summary)
        global_dispatcher.emit_tool_call(
            name="rough_filter_done",
            args={"file_count": len(targets), "original": total_original, "retained": total_refined,
                  "retention_rate": f"{round(total_refined / total_original * 100)}%" if total_original else "N/A"},
            agent="rough_filter"
        )

        return json.dumps({
            "status": "success",
            "processed_files": len(targets),
            "details": summary,
            "output_dir": get_agent_tgt_dataset_path(dataset_dir, "rough")
        }, ensure_ascii=False)
    except Exception as e:
        return f"Error executing rough filter: {str(e)}"
