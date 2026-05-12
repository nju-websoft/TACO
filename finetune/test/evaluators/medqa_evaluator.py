"""MedQA-USMLE evaluator for finetune/test framework.

Loads MedQA-USMLE-4-options from local JSON, generates answers with DeepSpeed,
and evaluates by matching the predicted choice letter against ground truth.
"""
import json
import os
import re
from typing import Dict, List

import torch
import deepspeed


# ============================================================
# Data loading
# ============================================================

def load_medqa_local(test_data_dir: str) -> List[Dict]:
    """Load MedQA problems from local JSON file."""
    json_files = sorted(f for f in os.listdir(test_data_dir) if f.endswith(".json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {test_data_dir}")
    problems = []
    for jf in json_files:
        with open(os.path.join(test_data_dir, jf), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                problems.extend(data)
    return problems


# ============================================================
# Prompt generation and answer extraction
# ============================================================

def generate_prompt(problem: Dict, tokenizer=None) -> str:
    """Generate prompt for MedQA multiple choice question."""
    question = problem["question"]
    options_text = problem.get("options_text", "")
    if not options_text and "options" in problem:
        options_text = "\n".join(
            f"{k}. {v}" for k, v in sorted(problem["options"].items())
        )

    prompt_text = f"{question}\n\n{options_text}"

    instruction = (
        f"{prompt_text}\n\n"
        "Choose the correct answer. "
        "Output your answer as a single letter wrapped in double brackets, "
        "e.g. [[A]], [[B]], [[C]], or [[D]]."
    )

    if tokenizer and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )


def extract_choice(response: str) -> str:
    """Extract letter choice (A-D) from model response.

    Priority:
      1. [[A]] / [[B]] / [[C]] / [[D]]  — prompted format
      2. (A) / (B) / (C) / (D)           — common alternative
      3. "answer is X" / "answer: X"     — natural language pattern
      4. Last standalone A-D letter       — likely the conclusion
      5. First character fallback
    """
    text = response.strip()
    upper = text.upper()

    # 1. Prompted format  [[X]]
    m = re.search(r'\[\[([A-D])\]\]', upper)
    if m:
        return m.group(1)

    # 2. Parenthesised  (X)
    m = re.search(r'\(([A-D])\)', upper)
    if m:
        return m.group(1)

    # 3. "answer is X" / "answer: X"
    m = re.search(r'(?:ANSWER|CHOICE)\s*(?:IS|:)\s*([A-D])\b', upper)
    if m:
        return m.group(1)

    # 4. Last standalone letter (model often reasons first, concludes last)
    matches = re.findall(r'\b([A-D])\b', upper)
    if matches:
        return matches[-1]

    # 5. First character fallback
    if text and text[0].upper() in "ABCD":
        return text[0].upper()

    return ""


# ============================================================
# Code generation with DeepSpeed
# ============================================================

def batch_generate(model, tokenizer, problems, max_length=512, batch_size=16, rank=0):
    """Generate answers for MedQA problems."""
    all_responses = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params_to_gather = [p for p in model.parameters()]
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
        total_batches = (len(problems) + batch_size - 1) // batch_size
        for i in range(0, len(problems), batch_size):
            if rank == 0:
                print(f"[MedQA] batch {i // batch_size + 1}/{total_batches}")

            batch = problems[i : i + batch_size]
            prompts = [generate_prompt(p, tokenizer) for p in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            for output in outputs:
                response = tokenizer.decode(output, skip_special_tokens=True)
                if "assistant\n" in response:
                    response = response.split("assistant\n")[-1]
                elif "### Response:\n" in response:
                    response = response.split("### Response:\n")[-1]
                all_responses.append(response)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    tokenizer.padding_side = original_padding_side
    return all_responses


# ============================================================
# Main evaluation entry point
# ============================================================

def evaluate_medqa(
    model,
    tokenizer,
    test_data_dir,
    rank=0,
    world_size=1,
    max_samples=None,
    max_gen_length=512,
    batch_size=16,
):
    """
    Evaluate on MedQA-USMLE-4-options (accuracy).

    Args:
        model: DeepSpeed model engine or HF model
        tokenizer: tokenizer
        test_data_dir: path to directory containing test.json
        rank, world_size: for distributed evaluation
        max_samples: limit number of samples per rank
        max_gen_length: max tokens to generate
        batch_size: generation batch size

    Returns:
        (accuracy, correct, total, details) - compatible with evaluate_test_set
    """
    if rank == 0:
        print(f"[MedQA] Loading problems from {test_data_dir}")

    problems = load_medqa_local(test_data_dir)

    if rank == 0:
        print(f"[MedQA] Total problems: {len(problems)}")

    # Shard across ranks
    problems = [p for i, p in enumerate(problems) if i % world_size == rank]
    if max_samples:
        problems = problems[:max_samples]

    if rank == 0:
        print(f"[MedQA] Rank {rank}: {len(problems)} problems")

    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero problems after sharding.
    raw_outputs = batch_generate(
        model, tokenizer, problems, max_gen_length, batch_size, rank
    )

    local_correct = 0
    local_total = len(problems)

    for idx, (problem, pred_text) in enumerate(zip(problems, raw_outputs)):
        gt_answer = problem["answer_idx"].strip().upper()
        pred_answer = extract_choice(pred_text)
        is_correct = pred_answer == gt_answer

        if is_correct:
            local_correct += 1

        if rank == 0:
            details.append(
                {
                    "question": problem["question"],
                    "options": problem.get("options", {}),
                    "response": pred_text,
                    "pred_answer": pred_answer,
                    "gt_answer": gt_answer,
                    "gt_answer_text": problem.get("answer", ""),
                    "correct": is_correct,
                }
            )
            if idx < 3:
                print(
                    f"[MedQA] Sample {idx}: Pred={pred_answer}, GT={gt_answer} ({'✓' if is_correct else '✗'})"
                )

    # Aggregate across ranks
    correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=model.device)
    total_tensor = torch.tensor([local_total], dtype=torch.long, device=model.device)
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

    correct = correct_tensor.item()
    total = total_tensor.item()
    accuracy = correct / total if total > 0 else 0.0

    if rank == 0:
        print(f"[MedQA] Accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy, correct, total, details
