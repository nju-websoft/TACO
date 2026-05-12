import glob
import os
import json
from typing import Dict, Any
from langchain_core.tools import tool
from agent.dispatch import global_dispatcher
from agent.tools.filters.fine_filter import run_fine_filter_single_file
from agent.utils import get_agent_tgt_dataset_path, time_logger, _load_json_or_jsonl
from agent.config import load_config
import math


@tool("fine_filter_agent", return_direct=False)
@time_logger
def fine_filter_agent(dataset_dir: str, fine_filter_config: Dict[str, Any]) -> str:
    """Fine Filter Agent: Evaluates dataset samples on multiple quality dimensions using LLM batch processing.
    
    Args:
        dataset_dir: Absolute path to the rough-filtered dataset directory.
        fine_filter_config: Configuration dictionary. For example:
            {
                "dimensions": { 
                        "instruction_following":  # Value of dimensions are ONLY for example, they should be set according to state of dataset
                        {
                            "description": "Does the output strictly follow the given instruction and constraints?",
                            "weight": "xxx, set appropriate value"
                        },
                        "Logical Rigor":  # Value of dimensions are ONLY for example, they should be set according to state of dataset
                        {
                            "description": "Is the output logically consistent and free of contradictions?",
                            "weight": "xxx, set appropriate value"
                        },
                        "Information Density":  # Value of dimensions are ONLY for example, they should be set according to state of dataset
                        {
                            "description": "Is the output comprehensive and covers all aspects of the request?",
                            "weight": "xxx, set appropriate value"
                        }
                }
            }
    """
    # Set default percent if not provided
    cfg = load_config()
    target_num = cfg.get("target_num", 1000)
    # Count total samples across all target files to compute ratio
    import glob as _glob
    if os.path.isdir(dataset_dir):
        _targets = _glob.glob(os.path.join(dataset_dir, "*.json"))
    else:
        _targets = [dataset_dir]
    _total = sum(len(_load_json_or_jsonl(t)) for t in _targets if os.path.exists(t) and "fine" not in t)
    target_ratio = min(target_num / _total, 1.0) if _total > 0 else 0.1
    if "basemodel" in dataset_dir.split('/')[-1]:
        fine_filter_config["percent"] = target_ratio
    else:
        fine_filter_config["percent"] = round(math.sqrt(target_ratio), 2)
    
    # Resolve targets
    if os.path.isdir(dataset_dir):
        targets = glob.glob(os.path.join(dataset_dir, "*.json"))
    elif "*" in dataset_dir or "?" in dataset_dir:
        targets = glob.glob(dataset_dir)
    else:
        targets = [dataset_dir]
        
    targets = [t for t in targets if os.path.exists(t) and "fine" not in t] # Avoid re-processing
    
    if not targets:
        return f"Error: No valid dataset files found matching {dataset_dir}"

    global_dispatcher.emit_tool_call(
        name="fine_filter_start",
        args={"file_count": len(targets),
              "dimensions": list(fine_filter_config.get("dimensions", {}).keys())},
        agent="fine_filter"
    )

    all_file_data = [] # Global buffer for all files
    summary_details = []

    try:
        # 1. Evaluate all files and collect data
        for file_idx, tgt in enumerate(targets):
            global_dispatcher.emit_progress(
                name="fine_filter_file_progress",
                current=file_idx + 1, total=len(targets),
                agent="fine_filter",
                file=os.path.basename(tgt)
            )
            result = run_fine_filter_single_file(tgt, fine_filter_config)
            if result.get("status") == "success":
                # Add source file info to each item for later grouping
                for item in result.get("data", []):
                    item["_source_file"] = result.get("source_file")
                all_file_data.extend(result.get("data", []))
            
            summary_details.append({
                "file": tgt,
                "status": result.get("status"),
                "processed_count": result.get("processed_count", 0)
            })

        # 2. Global Sorting
        all_file_data.sort(key=lambda x: x.get("fine_evaluation", {}).get("total_score", 0), reverse=True)

        # 3. Global Filtering (Percent)
        percent = fine_filter_config.get("percent")
        if percent is not None:
            try:
                percent_val = float(percent)
                if 0 < percent_val < 1:
                    keep_count = max(1, int(len(all_file_data) * percent_val))
                    all_file_data = all_file_data[:keep_count]
            except (ValueError, TypeError):
                pass
        
        # 4. Save back to files (maintaining structure)
        # Group by source file
        from collections import defaultdict
        file_groups = defaultdict(list)
        for item in all_file_data:
            src = item.pop("_source_file", None) # Remove temp key
            if src:
                file_groups[src].append(item)
        
        saved_files = []
        for tgt in targets:
            output_dir = get_agent_tgt_dataset_path(tgt, "fine")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(tgt))
            # Get data for this file (might be empty if all filtered out)
            data_to_save = file_groups.get(tgt, [])
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            saved_files.append({
                "file": output_path,
                "count": len(data_to_save)
            })

        total_processed = sum(d["processed_count"] for d in summary_details)
        total_retained = len(all_file_data)
        global_dispatcher.emit_tool_call(
            name="fine_filter_done",
            args={"file_count": len(targets), "processed": total_processed, "retained": total_retained,
                  "retention_rate": f"{round(total_retained / total_processed * 100)}%" if total_processed else "N/A"},
            agent="fine_filter"
        )

        return json.dumps({
            "status": "success",
            "processed_files": len(targets),
            "total_processed_samples": total_processed,
            "total_retained_samples": total_retained,
            "details": saved_files,
            "output_dir": output_dir
        }, ensure_ascii=False)
    except Exception as e:
        return f"Error executing fine evaluation: {str(e)}"
