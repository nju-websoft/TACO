from agent.config import load_config
import json
import os
from typing import List, Dict, Any
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from agent.model import subagent_llm as agent_llm
from agent.utils import _paced_invoke, _load_json_or_jsonl, get_clean_content, time_logger
from agent.dispatch import global_dispatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

# Default Evaluation Dimensions
DEFAULT_DIMENSIONS = {
    "instruction_following":
    {
        "description": "Does the output strictly follow the given instruction and constraints?",
        "weight": 0.4
    },
    "Logical Rigor":
    {
        "description": "Is the output logically consistent and free of contradictions?",
        "weight": 0.4
    },
    "Information Density":
    {
        "description": "Is the output comprehensive and covers all aspects of the request?",
        "weight": 0.2
    }
}

EVALUATION_SYSTEM_PROMPT = """**Role**: Professional Instruction-Data Quality Auditor.
**Task**: Evaluate the quality of a batch of dataset samples **comparatively** based on specific quality dimensions.

You will receive {batch_size} samples at once. By seeing them side by side, you can calibrate your ratings — use the full 1-5 scale, differentiating strong samples from weak ones within this batch.

**Evaluation Dimensions:**
{dimensions_desc}

**Rating Scale (1-5):**
1: Terrible (Completely failed)
2: Poor (Major issues)
3: Fair (Acceptable but needs improvement)
4: Good (Minor issues)
5: Excellent (Perfect)

**Output Format:**
Return a JSON **array** with exactly {batch_size} objects, one per sample, in the same order as the input:
[
  {{{{
    "sample_id": <id_from_input>,
    "scores": {{{{
      "<dimension_name>": <1-5>,
      ...
    }}}},
    "reasoning": "Brief explanation of the scores (1-2 sentences)",
    "flagged": <true/false>
  }}}},
  ...
]

**Important:**
- Compare samples against each other to maintain consistent, objective scoring across the batch.
- Use the full rating range: if there are clearly better and worse samples, reflect that in the scores.
- If harmful/illegal/NSFW content, set "flagged" to true and give low scores.
- Return ONLY the JSON array with no additional text.
"""


def _format_sample_text(sample: Dict[str, Any], sample_id: int) -> str:
    """Format a single sample for inclusion in the batch prompt."""
    instruction = str(sample.get("instruction", "")).strip()
    inp = str(sample.get("input", "")).strip()
    output = str(sample.get("output", "")).strip()

    text = f"--- Sample {sample_id} ---\nInstruction: {instruction}\n"
    if inp:
        text += f"Input: {inp}\n"
    text += f"Output: {output}\n"
    return text


@time_logger
def evaluate_batch(batch: List[Dict[str, Any]], dimensions: Dict[str, Dict[str, Any]],
                   last_ts: dict, start_index: int = 0) -> List[Dict[str, Any]]:
    """Send the entire batch to the LLM in one call for comparative evaluation."""

    dim_desc = "\n".join([
        f"- {k}: {v.get('description', '')} (Weight: {v.get('weight', 0)})"
        for k, v in dimensions.items()
    ])
    sys_msg = SystemMessage(content=EVALUATION_SYSTEM_PROMPT.format(
        batch_size=len(batch),
        dimensions_desc=dim_desc,
    ))

    # Build the multi-sample prompt
    samples_text = "\n".join(
        _format_sample_text(sample, start_index + idx)
        for idx, sample in enumerate(batch)
    )
    hum_msg = HumanMessage(content=samples_text)

    # Single LLM call for the whole batch
    try:
        resp = _paced_invoke(agent_llm, [sys_msg, hum_msg], last_ts)
        if not resp:
            eval_results = []
        else:
            content = getattr(resp, "content", "") or "[]"
            eval_results = get_clean_content(content)
            if not isinstance(eval_results, list):
                eval_results = [eval_results]
    except Exception as e:
        print(f"[FineFilter] Batch eval failed: {e}")
        eval_results = []

    # Attach results to samples
    enriched = []
    for idx, sample in enumerate(batch):
        new_item = sample.copy()
        if idx < len(eval_results) and isinstance(eval_results[idx], dict):
            eval_res = eval_results[idx]
            scores = eval_res.get("scores", {})
            total_score = sum(
                scores.get(dim, 0) * dimensions.get(dim, {}).get("weight", 0)
                for dim in dimensions
            )
            eval_res["total_score"] = round(total_score, 2)
        else:
            eval_res = {"error": "Missing evaluation", "scores": {}, "total_score": 0, "flagged": False}
        new_item["fine_evaluation"] = eval_res
        enriched.append(new_item)

    return enriched


def run_fine_filter_single_file(dataset_path: str, fine_filter_config: Dict[str, Any]) -> Dict[str, Any]:
    ff_cfg = load_config().get("fine_filter", {})
    samples_per_round = ff_cfg.get("samples_per_round", 50)
    num_batches = ff_cfg.get("num_batches", 5)
    batch_size = max(1, samples_per_round // num_batches)

    dimensions = fine_filter_config.get("dimensions", DEFAULT_DIMENSIONS)

    raw_data = _load_json_or_jsonl(dataset_path)
    if not raw_data:
        return {"status": "error", "message": f"Error: No data found at {dataset_path}", "data": []}

    all_evaluated_data = []
    total = len(raw_data)

    # Process in rounds of samples_per_round
    for round_start in range(0, total, samples_per_round):
        round_data = raw_data[round_start : round_start + samples_per_round]

        # Split round into batches
        batches = [round_data[i : i + batch_size] for i in range(0, len(round_data), batch_size)]

        # Each batch gets its own last_ts to avoid thread contention
        def _eval_one_batch(args):
            b, b_start = args
            ts = {"t": 0.0}
            return evaluate_batch(b, dimensions, ts, start_index=b_start)

        batch_args = [
            (b, round_start + i * batch_size)
            for i, b in enumerate(batches)
        ]

        # Process batches concurrently
        with ThreadPoolExecutor(max_workers=len(batches)) as executor:
            futures = {executor.submit(_eval_one_batch, arg): i for i, arg in enumerate(batch_args)}
            round_results = [None] * len(batches)
            for future in as_completed(futures):
                idx = futures[future]
                round_results[idx] = future.result()

        for batch_result in round_results:
            if batch_result:
                all_evaluated_data.extend(batch_result)

        processed_so_far = min(round_start + samples_per_round, total)
        global_dispatcher.emit_progress(
            name="fine_filter_eval_progress",
            current=processed_so_far, total=total,
            agent="fine_filter",
            file=os.path.basename(dataset_path)
        )

    return {
        "status": "success",
        "processed_count": total,
        "data": all_evaluated_data,
        "source_file": dataset_path
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test fine filter on a single file')
    parser.add_argument('--file', type=str, default='/data/fqzhou/dataset_agent/dataset/trial_set/data_0.json', help='Path to dataset file')
    parser.add_argument('--output', type=str, default='/data/fqzhou/dataset_agent/dataset/trial_set/fine_debug/data_0.json', help='Output file path')
    args = parser.parse_args()

    config = {"dimensions": DEFAULT_DIMENSIONS}

    print(f"Processing file: {args.file}")
    result = run_fine_filter_single_file(args.file, config)

    if result["status"] == "success":
        print(f"\nSuccess! Processed {result['processed_count']} samples")

        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result['data'], f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output}")
        else:
            print("\nPreview (first 3 samples):")
            for i, sample in enumerate(result['data'][:3]):
                eval_res = sample.get('fine_evaluation', {})
                print(f"\nSample {i}:")
                print(f"  Total Score: {eval_res.get('total_score', 'N/A')}")
                print(f"  Scores: {eval_res.get('scores', {})}")
                print(f"  Flagged: {eval_res.get('flagged', False)}")
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")
