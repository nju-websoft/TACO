"""CRUXEval Output Prediction evaluator for finetune/test framework.

Loads CRUXEval from local JSONL, generates output predictions with DeepSpeed,
and evaluates by executing assert statements in a sandbox (pass@1).

CRUXEval-O: Given a Python function and input, predict the output.

Test execution runs in isolated subprocesses via _cruxeval_sandbox.py
to avoid inheriting model memory from the training process.
"""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import torch
import deepspeed


# ============================================================
# Constants
# ============================================================

TIMEOUT_LIMIT = 5.0

_SANDBOX_SCRIPT = os.path.join(os.path.dirname(__file__), "_cruxeval_sandbox.py")

# Re-export for test_verify.py compatibility
from ._cruxeval_sandbox import check_correctness


# ============================================================
# Data loading
# ============================================================

def load_cruxeval_local(test_data_dir: str) -> List[Dict]:
    """Load CRUXEval problems from local JSONL file."""
    jsonl_files = [f for f in os.listdir(test_data_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {test_data_dir}")
    jsonl_path = os.path.join(test_data_dir, jsonl_files[0])
    problems = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    problems.sort(key=lambda x: x["id"])
    return problems


# ============================================================
# Prompt generation and answer extraction
# ============================================================

_OUTPUT_PROMPT_TEMPLATE = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
"""


def generate_prompt(problem: Dict, tokenizer=None) -> str:
    """Generate prompt for CRUXEval output prediction."""
    prompt_text = _OUTPUT_PROMPT_TEMPLATE.format(
        code=problem["code"], input=problem["input"]
    )
    if tokenizer and getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": "You are an expert Python programmer. You can execute Python code in your head and predict the output."},
            {"role": "user", "content": prompt_text},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt_text


def extract_answer(raw_output: str) -> str:
    """Extract predicted output from model generation."""
    text = raw_output.strip()
    if "[/ANSWER]" in text:
        text = text.split("[/ANSWER]")[0].strip()
    if "[ANSWER]" in text:
        text = text.split("[ANSWER]")[-1].strip()
    lines = text.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("assert ") and "==" in line:
            text = line
            break
    if "==" in text:
        text = text.split("==")[-1].strip()
    return text.strip()


# ============================================================
# Code generation with DeepSpeed
# ============================================================

def batch_generate_code(model, tokenizer, problems, max_length=512, batch_size=1, rank=0):
    """Generate output predictions for CRUXEval problems."""
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
                print(f"[CRUXEval-O] batch {i // batch_size + 1}/{total_batches}")

            batch = problems[i:i + batch_size]
            prompts = [generate_prompt(p, tokenizer) for p in batch]
            inputs = tokenizer(prompts, return_tensors="pt", truncation=True,
                               max_length=2048, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_length, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, use_cache=True)

            for output in outputs:
                response = tokenizer.decode(output, skip_special_tokens=True)
                if 'assistant\n' in response:
                    response = response.split('assistant\n')[-1]
                elif '### Response:\n' in response:
                    response = response.split('### Response:\n')[-1]
                elif '[ANSWER]\n' in response:
                    response = response.split('[ANSWER]\n')[-1]
                all_responses.append(response)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    tokenizer.padding_side = original_padding_side
    return all_responses


# ============================================================
# Parallel test execution (subprocess-based)
# ============================================================

def _eval_single_subprocess(idx, code, input_val, output_val, predicted):
    """Evaluate a single output prediction in an isolated subprocess."""
    if f"f({input_val})" in predicted:
        return (idx, False, "repeated function call")

    check_program = f"{code}\nassert {output_val} == {predicted}"
    input_fd, input_path = tempfile.mkstemp(suffix=".json", prefix="crux_o_")
    result_path = input_path + ".result"
    try:
        with os.fdopen(input_fd, "w") as f:
            json.dump({"check_program": check_program}, f)

        proc = subprocess.run(
            [sys.executable, _SANDBOX_SCRIPT, input_path, result_path],
            capture_output=True,
            timeout=TIMEOUT_LIMIT + 30,
        )

        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            return (idx, r["passed"], "" if r["passed"] else check_program)
        else:
            return (idx, False, f"no result file. stderr: {proc.stderr.decode(errors='replace')[:500]}")
    except subprocess.TimeoutExpired:
        return (idx, False, "subprocess timed out")
    except Exception as e:
        return (idx, False, str(e))
    finally:
        for p in [input_path, result_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def run_tests_parallel(problems, predictions, num_workers=8):
    """Run tests in isolated subprocesses using ThreadPoolExecutor."""
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _eval_single_subprocess, idx,
                problem["code"], problem["input"],
                problem["output"], predicted
            ): idx
            for idx, (problem, predicted) in enumerate(zip(problems, predictions))
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append((idx, False, str(e)))

    results.sort(key=lambda x: x[0])
    return results


# ============================================================
# Main evaluation entry point
# ============================================================

def evaluate_cruxeval_output(model, tokenizer, test_data_dir,
                             rank=0, world_size=1, max_samples=None,
                             max_gen_length=512, batch_size=1):
    """
    Evaluate on CRUXEval-O (Output Prediction, pass@1).

    Returns:
        (accuracy, correct, total, details) - compatible with evaluate_test_set
    """
    if rank == 0:
        print(f"[CRUXEval-O] Loading problems from {test_data_dir}")

    problems = load_cruxeval_local(test_data_dir)

    if rank == 0:
        print(f"[CRUXEval-O] Total problems: {len(problems)}")

    # Shard across ranks
    problems = [p for i, p in enumerate(problems) if i % world_size == rank]
    if max_samples:
        problems = problems[:max_samples]

    if rank == 0:
        print(f"[CRUXEval-O] Rank {rank}: {len(problems)} problems")

    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero problems after sharding.
    # Generate predictions
    raw_outputs = batch_generate_code(model, tokenizer, problems,
                                      max_gen_length, batch_size, rank)

    # Extract answers
    predictions = [extract_answer(raw) for raw in raw_outputs]

    if rank == 0:
        print(f"[CRUXEval-O] Running sandbox tests...")

    # Run tests in isolated subprocesses
    test_results = run_tests_parallel(problems, predictions,
                                      num_workers=min(12, len(problems)))

    local_correct = sum(1 for _, passed, _ in test_results if passed)
    local_total = len(problems)

    if rank == 0:
        for idx, passed, info in test_results:
            details.append({
                "id": problems[idx]["id"],
                "code": problems[idx]["code"],
                "input": problems[idx]["input"],
                "expected_output": problems[idx]["output"],
                "raw_output": raw_outputs[idx],
                "predicted": predictions[idx],
                "passed": passed,
                "info": info,
            })

    # Aggregate across ranks
    correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=model.device)
    total_tensor = torch.tensor([local_total], dtype=torch.long, device=model.device)
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

    correct = correct_tensor.item()
    total = total_tensor.item()
    accuracy = correct / total if total > 0 else 0.0

    if rank == 0:
        print(f"[CRUXEval-O] pass@1: {accuracy:.2%} ({correct}/{total})")

    return accuracy, correct, total, details
