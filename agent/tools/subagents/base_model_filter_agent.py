from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
import random
import math
from langchain_core.messages import SystemMessage, HumanMessage
from agent.model import subagent_llm as agent_llm
from agent.utils import _paced_invoke, get_clean_content
from agent.dispatch import global_dispatcher
from agent.config import load_config
import json
import os
from agent.tools.subagents_prompt.base_model_filter_prompt import SYS_PROMPT
from agent.tools.filters.base_model_filter import BaseModelTools
from agent.utils import get_agent_tgt_dataset_path, time_logger, _load_json_or_jsonl
import glob

# Mapping from tool_dict key -> weight dict key
TOOL_METRIC_MAP = {
    "get_loss_metrics": "loss",
    "get_entropy_metrics": "entropy",
    "get_drift_metrics": "drift",
    "get_mean_diff_metrics": "mean_diff",
}


# Helper functions
def safe_get(lst, idx, default=0.0):
    if idx < len(lst): return lst[idx]
    print('[BASE MODEL FILTER] Warning: Safe get return default value')
    return default


@tool("base_model_filter_agent", return_direct=False)
@time_logger
def base_model_filter_agent(dataset_dir: str, goal: str, weight: Dict[str, float]) -> str:
    """Base Model Filter Agent: Analyzes and filters dataset using base model metrics (NLL, Entropy, etc.).
    
    Args:
        dataset_dir: Absolute path to the dataset directory.
        goal: The specific analysis or filtering goal.
        weight: A dictionary containing weights for each indicator.
    """
    cfg = load_config()
    target_num = cfg.get("target_num", 1000)
    # Count total samples across all target files to compute ratio
    if os.path.isdir(dataset_dir):
        _targets = glob.glob(os.path.join(dataset_dir, "*.json"))
    else:
        _targets = [dataset_dir]
    _total = sum(len(_load_json_or_jsonl(t)) for t in _targets if os.path.exists(t))
    target_ratio = min(target_num / _total, 1.0) if _total > 0 else 0.1
    if "fine" in dataset_dir.split('/')[-1]:
        percent = target_ratio
    else:
        percent = round(math.sqrt(target_ratio), 2)
    
    # Resolve targets
    if os.path.isdir(dataset_dir):
        targets = glob.glob(os.path.join(dataset_dir, "*.json"))
    elif "*" in dataset_dir or "?" in dataset_dir:
        targets = glob.glob(dataset_dir)
    else:
        targets = [dataset_dir]
        
    targets = [t for t in targets if os.path.exists(t)]# and "rough" in t]
    
    if not targets:
        return f"Error: No valid dataset files found matching {dataset_dir}"

    global_dispatcher.emit_tool_call(
        name="base_model_filter_start",
        args={"file_count": len(targets),
              "active_weights": {k: v for k, v in weight.items() if v > 0},
              "keep_percent": percent},
        agent="base_model_filter"
    )

    overall_results = []
    all_samples_buffer = []

    for idx, tgt_file in enumerate(targets):
        try:
            bm_tools = BaseModelTools()
        except Exception as e:
            return f"Error initializing Base Model Tools: {e}"
        
        try:
            bm_tools.load_dataset(tgt_file)
            dataset = bm_tools.dataset
        except Exception as e:
            overall_results.append(f"Error loading {tgt_file}: {e}")
            continue
            
        if not dataset:
            overall_results.append(f"Dataset at {tgt_file} is empty.")
            continue
        
        global_dispatcher.emit_progress(
            name="base_model_filter_file",
            current=idx + 1, total=len(targets),
            agent="base_model_filter",
            file=os.path.basename(tgt_file), dataset_size=len(dataset) if dataset else 0
        )
        # 2. ReAct Loop (Per File)
        max_steps = 50
        history: List[str] = []
        last_ts = {"t": 0.0}
        
        # Reset tool state for new file
        tool_dict = {
            "get_loss_metrics": {"func": bm_tools.get_loss_metrics, "tgt": [], "called": False},
            "get_entropy_metrics": {"func": bm_tools.get_entropy_metrics, "tgt": [], "called": False},
            "get_drift_metrics": {"func": bm_tools.get_drift_metrics, "tgt": [], "called": False},
            "get_mean_diff_metrics": {"func": bm_tools.get_mean_diff_metrics, "tgt": [], "called": False},
        }
        # Remove tools whose weight is 0 — avoids wasting ReAct steps
        tool_dict = {k: v for k, v in tool_dict.items()
                     if weight.get(TOOL_METRIC_MAP[k], 0.0) > 0}

        read_sample_count = 0

        for step in range(max_steps):
            # Prepare context
            history_text = "\n".join(history) if history else "None"
            hum_msg = HumanMessage(content=f"Dataset Path: {tgt_file}\nDataset Size: {len(dataset)}\nGoal: {goal}\n\nPrevious Steps:\n{history_text}")
            
            # Invoke LLM
            resp = _paced_invoke(agent_llm, [SYS_PROMPT, hum_msg], last_ts)
            if not resp:
                print(f"Error: Agent LLM returned no response. Step: {step+1}")
                continue   
            content = getattr(resp, "content", "") or ""
            
            # Parse JSON
            try:
                action_obj = get_clean_content(content)
            except Exception:
                history.append(f"Step {step+1}: Output parsing failed. Content: {content[:100]}...")
                continue
                
            thought = action_obj.get("thought", "")
            action = action_obj.get("action", "")

            # Dispatch logs
            global_dispatcher.emit_tool_call(
                name="base_model_filter_step",
                args={"step": step + 1, "max_steps": max_steps,
                      "action": action, "thought": thought[:80]},
                agent="base_model_filter"
            )
            
            # Execute Action
            result = ""
            
            if action == "finish":
                missing_metrics = []
                missing_tools = []
                for tool_key, metric_name in TOOL_METRIC_MAP.items():
                    if weight.get(metric_name, 0.0) > 0 and not tool_dict[tool_key]["called"]:
                        missing_metrics.append(metric_name)
                        missing_tools.append(tool_key)
                
                if missing_metrics:
                    result = (
                        f"REJECTED: Cannot finish — the following metrics have positive weight "
                        f"but have NOT been computed yet: {', '.join(missing_metrics)}. "
                        f"You MUST call these tools first: {', '.join(missing_tools)}. "
                        f"Then call finish again."
                    )
                else:
                    # Store to buffer — only include metrics present in tool_dict
                    for i in range(len(dataset)):
                        metrics = {TOOL_METRIC_MAP[k]: safe_get(v["tgt"], i)
                                   for k, v in tool_dict.items()}
                        all_samples_buffer.append({
                            "sample": dataset[i],
                            "metrics": metrics,
                            "source_file": tgt_file
                        })
                    break
            elif action == "read_dataset_sample":
                read_sample_count += 1
                if read_sample_count > 5:
                    result = "Error: Max sample reads exceeded."
                else:
                    try:
                        sample = random.choice(dataset)
                        result = f"Random Sample: {sample}\nUsed {read_sample_count}/5 reads."
                    except Exception as e:
                        result = f"Error: {e}"
            elif action in tool_dict.keys():
                if tool_dict[action]["called"]:
                    result = f"Error: '{action}' already called."
                else:
                    tool_dict[action]["called"] = True
                    global_dispatcher.emit_tool_call(name="base_model_filter_metric", args={"metric": action, "dataset_len": len(dataset)}, agent="base_model_filter")
                    
                    metric_name = TOOL_METRIC_MAP[action]
                    if weight.get(metric_name, 0.0) == 0:
                        result = f"Skipped: '{action}' (metric '{metric_name}') has weight 0. No computation needed."
                    else:
                        bmf_cfg = cfg.get("base_model_filter", {})
                        batch_size = bmf_cfg.get("metrics_batch_size", 32)
                            
                        for i in range(0, len(dataset), batch_size):
                            tool_dict[action]["func"](i, min(i + batch_size, len(dataset)))
                            
                        bm_tools.normalize_metrics(metric_name)

                        # Read normalized values from cache
                        tool_dict[action]["tgt"] = [bm_tools.metrics_cache.get(i, {}).get(metric_name, 0.0) for i in range(len(dataset))]
                        result = f"Metrics for {metric_name} calculated and normalized to [0, 1]."
            elif action in TOOL_METRIC_MAP:
                result = f"Skipped: '{action}' has weight 0 — do not call it again."
            else:
                result = f"Error: Unknown action '{action}'"
                
            history.append(f"Step {step+1}:\nThought: {thought}\nAction: {action})\nResult: {result}\n")

    # 3. Global Filter & Save
    if all_samples_buffer:
        # Calculate weighted scores (only over metrics that were computed)
        active_metrics = list(all_samples_buffer[0]["metrics"].keys()) if all_samples_buffer else []

        for i, sample in enumerate(all_samples_buffer):
            score = sum(sample["metrics"].get(m, 0.0) * weight.get(m, 0.0) for m in active_metrics)
            sample["average"] = score
            sample["final_data"] = {
                "sample": sample["sample"],
                "average": score,
                **sample["metrics"]
            }
            
        # Sort & Filter
        all_samples_buffer.sort(key=lambda x: x["average"], reverse=True)
        keep_count = int(len(all_samples_buffer) * percent)
        filtered_samples = all_samples_buffer[:keep_count]
        
        # Group by file
        from collections import defaultdict
        file_groups = defaultdict(list)
        for s in filtered_samples:
            file_groups[s["source_file"]].append(s["final_data"]["sample"])
            
        # Save files
        output_dir = get_agent_tgt_dataset_path(dataset_dir, "basemodel")
        os.makedirs(output_dir, exist_ok=True)
        for tgt_file in targets:
            output_path = os.path.join(output_dir, os.path.basename(tgt_file))
            
            data_to_save = file_groups.get(tgt_file, [])
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                
            overall_results.append(f"Saved {len(data_to_save)} samples to {output_path}")
    else:
        print("[Warning] No valid samples to save.")
        
    total_retained = len(filtered_samples) if 'filtered_samples' in dir() else 0
    global_dispatcher.emit_tool_call(
        name="base_model_filter_done",
        args={"file_count": len(targets), "total_samples": len(all_samples_buffer),
              "retained": total_retained,
              "retention_rate": f"{round(total_retained / len(all_samples_buffer) * 100)}%" if all_samples_buffer else "N/A"},
        agent="base_model_filter"
    )

    return json.dumps({
        "status": "success",
        "processed_files": len(targets),
        "details": overall_results,
        "output_dir": output_dir
    }, ensure_ascii=False)


if __name__ == "__main__":
    # Test configuration
    test_dataset = "dataset/trial_set"
    test_goal = "Filter out samples with high loss and entropy"
    test_weight = {"loss": 0.2, "entropy": 0.15, "drift": 0.25, 'mean_diff': 0.4}
    
    print(f"Testing base_model_filter_agent with:")
    print(f"Dataset: {test_dataset}")
    print(f"Goal: {test_goal}")
    print(f"Weights: {test_weight}")
    
    result = base_model_filter_agent.invoke({
        "dataset_dir": test_dataset,
        "goal": test_goal,
        "weight": test_weight,
    })
    print("\nFinal Result:")
    print(result)
