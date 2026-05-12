"""HumanEval evaluator for finetune/test framework.

Loads HumanEval from local JSONL, generates code with DeepSpeed,
and evaluates using sandboxed execution (pass@1).

Test execution runs in isolated subprocesses via _humaneval_sandbox.py
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

TIMEOUT_LIMIT = 10.0

_SANDBOX_SCRIPT = os.path.join(os.path.dirname(__file__), "_humaneval_sandbox.py")

# Re-export for test_verify.py compatibility
from ._humaneval_sandbox import check_correctness


# ============================================================
# Data loading
# ============================================================

def load_humaneval_local(test_data_dir: str) -> List[Dict]:
    """Load HumanEval problems from local JSONL file."""
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
    problems.sort(key=lambda x: x["task_id"])
    return problems


# ============================================================
# Prompt generation and code extraction
# ============================================================

def generate_prompt(problem: Dict, tokenizer=None) -> str:
    """Generate prompt for HumanEval function completion.

    Chat models: instruction + function signature/docstring.
    Base models: raw prompt (function signature + docstring) for continuation.
    """
    prompt = problem["prompt"]
    if tokenizer and getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": "You are an expert Python programmer. Complete the given function."},
            {"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"""### Instruction:\nComplete the following Python function:\n\n{prompt}\n### Response:\n"""


def extract_code(raw_output: str, prompt: str) -> str:
    """Extract generated function body from model output.

    The model should generate the function body to complete the prompt.
    We prepend the original prompt to get a complete function.
    """
    code = raw_output.strip()

    # If model repeated the prompt, strip it
    if code.startswith(prompt.strip()):
        code = code[len(prompt.strip()):]

    # If output is wrapped in markdown code block, extract it
    if "```python" in code:
        parts = code.split("```python")
        if len(parts) > 1:
            code = parts[-1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0]

    return code


# ============================================================
# Code generation with DeepSpeed
# ============================================================

def batch_generate_code(model, tokenizer, problems, max_length=512, batch_size=1, rank=0):
    """Generate completions for HumanEval problems."""
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
                print(f"[HumanEval] batch {i // batch_size + 1}/{total_batches}")

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

def _eval_single_subprocess(idx, problem, completion):
    """Evaluate a single problem in an isolated subprocess via _humaneval_sandbox.py."""
    input_fd, input_path = tempfile.mkstemp(suffix=".json", prefix="he_in_")
    result_path = input_path + ".result"
    try:
        with os.fdopen(input_fd, "w") as f:
            json.dump({
                "problem": problem,
                "completion": completion,
            }, f)

        proc = subprocess.run(
            [sys.executable, _SANDBOX_SCRIPT, input_path, result_path],
            capture_output=True,
            timeout=TIMEOUT_LIMIT + 30,
        )

        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            return (idx, r)
        else:
            stderr_msg = proc.stderr.decode(errors="replace")[:500]
            return (idx, {
                "task_id": problem["task_id"],
                "passed": False,
                "result": f"failed: no result file. stderr: {stderr_msg}",
            })
    except subprocess.TimeoutExpired:
        return (idx, {
            "task_id": problem["task_id"],
            "passed": False,
            "result": "timed out",
        })
    except Exception as e:
        return (idx, {
            "task_id": problem["task_id"],
            "passed": False,
            "result": f"failed: {e}",
        })
    finally:
        for p in [input_path, result_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def run_tests_parallel(problems, completions, num_workers=8):
    """Run tests in isolated subprocesses using ThreadPoolExecutor.

    Each test is dispatched to _humaneval_sandbox.py via subprocess.run,
    so child processes do not inherit the training process memory.
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_eval_single_subprocess, idx, problem, completion): idx
            for idx, (problem, completion) in enumerate(zip(problems, completions))
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append((idx, {
                    "task_id": problems[idx]["task_id"],
                    "passed": False,
                    "result": f"failed: {e}",
                }))

    results.sort(key=lambda x: x[0])
    return results


# ============================================================
# Main evaluation entry point
# ============================================================

def evaluate_humaneval(model, tokenizer, test_data_dir,
                       rank=0, world_size=1, max_samples=None,
                       max_gen_length=512, batch_size=1):
    """
    Evaluate on HumanEval (pass@1).

    Args:
        model: DeepSpeed model engine or HF model
        tokenizer: tokenizer
        test_data_dir: path to directory containing JSONL file
        rank, world_size: for distributed evaluation
        max_samples: limit number of samples per rank
        max_gen_length: max tokens to generate
        batch_size: generation batch size

    Returns:
        (accuracy, correct, total, details) - compatible with evaluate_test_set
    """
    if rank == 0:
        print(f"[HumanEval] Loading problems from {test_data_dir}")

    problems = load_humaneval_local(test_data_dir)

    if rank == 0:
        print(f"[HumanEval] Total problems: {len(problems)}")

    # Shard across ranks
    problems = [p for i, p in enumerate(problems) if i % world_size == rank]
    if max_samples:
        problems = problems[:max_samples]

    if rank == 0:
        print(f"[HumanEval] Rank {rank}: {len(problems)} problems")

    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero problems after sharding.
    # Generate completions
    raw_outputs = batch_generate_code(model, tokenizer, problems,
                                      max_gen_length, batch_size, rank)

    # Extract code from outputs
    completions = [extract_code(raw, p["prompt"]) for raw, p in zip(raw_outputs, problems)]

    if rank == 0:
        print(f"[HumanEval] Running functional tests...")

    # Run tests in isolated subprocesses
    test_results = run_tests_parallel(problems, completions, num_workers=min(12, len(problems)))

    local_correct = sum(1 for _, r in test_results if r["passed"])
    local_total = len(problems)

    if rank == 0:
        for idx, r in test_results:
            details.append({
                "task_id": problems[idx]["task_id"],
                "entry_point": problems[idx]["entry_point"],
                "raw_output": raw_outputs[idx],
                "extracted_code": completions[idx],
                "passed": r["passed"],
                "result": r["result"],
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
        print(f"[HumanEval] pass@1: {accuracy:.2%} ({correct}/{total})")

    return accuracy, correct, total, details
