import enum
from typing import List, Optional, Dict, Any
import time
from langchain_core.messages import SystemMessage, HumanMessage
from agent.dispatch import global_dispatcher
from agent.config import limits
import os
import json
from pathlib import Path
from functools import wraps
from datetime import datetime


TYPING_SPEED = limits.get("typing_speed", 1e4)
BASE_DIR = os.getenv("BASE_DIR", os.getcwd())

# Global variable to store the session timestamp
_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# Global variable to store aggregated execution times: {func_name: {"total_time": float, "count": int}}
_EXECUTION_STATS = {}

SUBAGENT_CONTEXT_LEN = int(limits.get("max_context_chars", 30000))

def time_logger(func):
    """Decorator to log function execution time to a file.
    Aggregates stats in memory and appends new entry to log file.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        # Update aggregated stats
        func_name = func.__name__
        if func_name not in _EXECUTION_STATS:
            _EXECUTION_STATS[func_name] = {"total_time": 0.0, "count": 0}
        
        _EXECUTION_STATS[func_name]["total_time"] += duration
        _EXECUTION_STATS[func_name]["count"] += 1

        log_dir = os.path.join(get_root_dir(), "log", _SESSION_TIMESTAMP)
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "execution_time.log")
        
        stats = _EXECUTION_STATS[func_name]
        avg_time = stats["total_time"] / stats["count"]
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Function '{func_name}' executed in {duration:.4f} seconds. "
                    f"(Total calls: {stats['count']}, Total time: {stats['total_time']:.4f}s, Avg time: {avg_time:.4f}s)\n")
            
        return result
    return wrapper

def get_root_dir() -> str:
    import os
    p = os.getenv("ROOT_DIR")
    if isinstance(p, str) and p.strip():
        return p.strip()
    return BASE_DIR

RATE_MIN_MS = limits.get("paced_invoke_rate_minms", 800)
MAX_RETRIES = limits.get("paced_invoke_max_retries", 3)
MAX_SLEEP_TIME = limits.get("paced_invoke_sleep_ms", 5)

def _paced_invoke(llm, msgs, last_ts: dict):
    now = time.time() * 1000.0
    wait = RATE_MIN_MS - (now - last_ts["t"])
    if wait > 0:
        time.sleep(wait / 1000.0)
    tries = 0
    delay = max(0.5, RATE_MIN_MS / 1000.0)
    while True:
        try:
            resp = llm.invoke(msgs)
            last_ts["t"] = time.time() * 1000.0
            return resp
        except Exception:
            if tries >= MAX_RETRIES:
                return None
            time.sleep(delay)
            delay = min(MAX_SLEEP_TIME, delay * 2.0)
            tries += 1

def _summarize_history(llm, hist: List[str], current_goal: str, sum_sys: SystemMessage, last_ts: dict) -> str:
    raw_text = "\n\n".join(hist)
    if len(raw_text) < SUBAGENT_CONTEXT_LEN:
        return raw_text

    sum_hum = HumanMessage(content=f"Goal: {current_goal}\n\nHistory:\n{raw_text}")
    global_dispatcher.emit_tool_call(name="context_summarize", args={"original_chars": len(raw_text), "limit_chars": SUBAGENT_CONTEXT_LEN}, agent="context")
    try:
        resp = _paced_invoke(llm, [sum_sys, sum_hum], last_ts)
        return getattr(resp, "content", "") or raw_text[:SUBAGENT_CONTEXT_LEN]
    except Exception:
        return raw_text[:SUBAGENT_CONTEXT_LEN]


TODO_MEM: List[dict] = []

def _todo_read() -> List[dict]:
    return list(TODO_MEM)

def _todo_write(items: List[dict]) -> str:
    global TODO_MEM
    TODO_MEM = list(items)
    return "ok"


def _get_base_model_prefix() -> str:
    """Extract a short model name from config for directory prefixing."""
    try:
        from agent.config import load_config
        cfg = load_config()
        model_path = (cfg.get("bd") or {}).get("base_model_path", "")
        if model_path:
            # "/path/to/Qwen/Qwen2.5-0.5B" → "Qwen2.5-0.5B"
            name = os.path.basename(model_path.rstrip("/"))
            if name:
                return name
    except Exception:
        pass
    return ""


def get_agent_tgt_dataset_path(dataset_dir: str, agent_type: str) -> str:
    dataset_real_dir = dataset_dir
    if not os.path.isdir(dataset_dir):
        print(f"Warning: {dataset_dir} is not a directory. Assuming it as a file.")
        dataset_real_dir = os.path.dirname(dataset_dir)

    prefix = _get_base_model_prefix()
    prefixed_type = f"{prefix}_{agent_type}" if prefix else agent_type

    base_dir_name = os.path.basename(dataset_real_dir)
    if "rough" in base_dir_name or "basemodel" in base_dir_name or "fine" in base_dir_name or "discipline" in base_dir_name:
        return f"{dataset_real_dir}_{agent_type}"
    else:
        return os.path.join(dataset_real_dir, prefixed_type)


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    
    try:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                else:
                    data = [content]
    except Exception:
        pass

    print(f"[Load] Loaded {len(data)} samples from {path}")
    check_field = ["input", "instruction", "output", "text", "id"]
    for item in data:
        for field in check_field:
            if field not in item:
                print(f"[Load] Warning: Sample {item} is missing field {field}.")
                assert 0, f"Sample {item} is missing field {field}."
    return data


def safe_truncate(tokenizer, text, max_tokens=8000):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return text


def get_clean_content(content):
    json_content = content.strip()
    if "```json" in json_content:
        json_content = json_content.split("```json")[1].split("```")[0]
    elif "```" in json_content:
        json_content = json_content.split("```")[1].split("```")[0]
    return json.loads(json_content)
