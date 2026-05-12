import re
import json
import os
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import html
from agent.utils import get_agent_tgt_dataset_path, _load_json_or_jsonl


def _get_noise_ratio(text: str, noise_regex: str) -> float:
    if not text:
        return 0.0
    if not noise_regex:
        return 0.0
    
    try:
        matches = re.findall(noise_regex, text)
        return len(matches) / len(text)
    except Exception:
        return 0.0


def _get_en_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    # Match English characters
    en_chars = re.findall(r'[a-zA-Z]', text)
    return len(en_chars) / len(text)


def _has_refusal_patterns(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        if p in text:
            return True
    return False


def apply_filters(data: List[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    filtered_data = []
    
    # 1. Basic Policy
    min_total = policy.get("min_total_len", 40)
    max_total = policy.get("max_total_len", 4096)
    # 2. Instruction Specific
    min_inst_len = policy.get("min_inst_len", 5)
    min_out_len = policy.get("min_out_len", 15)
    ratio_min = policy.get("out_inst_ratio_min", 0.2)
    # 3. Patterns
    refusal_list = policy.get("refusal_phrases", [])
    noise_regex = policy.get("noise_regex", "")
    max_noise_ratio = policy.get("max_noise_ratio", 0.15)
    # 4. Lang
    require_en = policy.get("require_en", True)
    min_en_ratio = policy.get("min_en_ratio", 0.3)

    check_fields = ["instruction", "input", "output", "text"]
    for item in data:
        # Check Fields Existence
        if any(f not in item for f in check_fields):
            continue
            
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        full_text = str(item.get("text", "")).strip()

        # 1. Basic: Total Length
        if len(full_text) < min_total or len(full_text) > max_total:
            continue
            
        # 2. Instruction Specific
        if len(instruction) < min_inst_len:
            continue
        if len(output) < min_out_len:
            continue
        if len(instruction) > 0 and (len(output) < ratio_min * len(instruction)):
            continue
            
        # 3. Patterns
        # Refusal
        if _has_refusal_patterns(output, refusal_list):
            continue
        # Noise
        if noise_regex and _get_noise_ratio(full_text, noise_regex) > max_noise_ratio:
            continue
            
        # 4. Lang
        if require_en:
            if _get_en_char_ratio(full_text) < min_en_ratio:
                continue
        
        filtered_data.append(item)
            
    return filtered_data


def run_rough_filter(dataset_path: str, policy: Dict[str, Any]) -> str:
    raw_data = _load_json_or_jsonl(dataset_path)
    if not raw_data:
        return f"Error: No data found at {dataset_path}"
        
    # Filter
    refined_data = apply_filters(raw_data, policy)
    
    output_name = os.path.basename(dataset_path)
    output_dir = get_agent_tgt_dataset_path(dataset_path, "rough")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_name)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=2)
    
    return json.dumps({
        "status": "success",
        "original_count": len(raw_data),
        "refined_count": len(refined_data),
        "output_path": output_file_path
    }, ensure_ascii=False)
