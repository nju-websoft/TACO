"""BigCodeBench evaluator for finetune/test framework.

Loads BigCodeBench v0.1.4 from local parquet, generates code with DeepSpeed,
and evaluates using sandboxed unittest execution (pass@1).

Test execution runs in isolated subprocesses via _bigcodebench_sandbox.py
to avoid inheriting model memory from the training process.
"""
import itertools
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import deepspeed


# ============================================================
# Constants
# ============================================================

TIMEOUT_LIMIT = 240.0
PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SANDBOX_SCRIPT = os.path.join(os.path.dirname(__file__), "_bigcodebench_sandbox.py")

# Re-export for test_verify.py compatibility
from ._bigcodebench_sandbox import untrusted_check


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


# ============================================================
# Code generation and evaluation
# ============================================================

def load_bigcodebench_local(test_data_dir: str) -> List[Dict]:
    """Load BigCodeBench tasks from local parquet file."""
    parquet_files = [f for f in os.listdir(test_data_dir) if f.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {test_data_dir}")
    parquet_path = os.path.join(test_data_dir, parquet_files[0])
    df = pd.read_parquet(parquet_path)
    tasks = df.to_dict("records")
    tasks.sort(key=lambda x: x["task_id"])
    return tasks


def generate_prompt(task, tokenizer=None):
    """Generate prompt for BigCodeBench-Complete task."""
    prompt = task["complete_prompt"]
    if tokenizer and getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": "You are an expert Python programmer. Complete the given function."},
            {"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"""### Instruction:\nComplete the following Python function:\n\n{prompt}\n### Response:\n"""


def extract_code(raw_output: str, prompt: str) -> str:
    """Extract generated code from model output.

    For BigCodeBench-Complete, the model should continue the function body.
    We prepend the original prompt (function signature + docstring) to get a
    complete function.
    """
    code = raw_output.strip()
    # If model repeated the prompt, strip it
    if code.startswith(prompt.strip()):
        code = code[len(prompt.strip()):]

    # If output is wrapped in markdown code block, extract it
    if "```python" in code:
        parts = code.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0]

    # Combine prompt (function signature) with generated body
    full_code = prompt + code
    return full_code


def batch_generate_code(model, tokenizer, tasks, max_length=2048, batch_size=1, rank=0):
    """Generate code for BigCodeBench tasks using DeepSpeed distributed inference."""
    all_responses = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params_to_gather = [p for p in model.parameters()]
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        for i in range(0, len(tasks), batch_size):
            if rank == 0:
                print(f"[BigCodeBench] batch {i // batch_size + 1}/{total_batches}")

            batch = tasks[i:i + batch_size]
            prompts = [generate_prompt(t, tokenizer) for t in batch]
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


def _eval_single_subprocess(idx, task, code):
    """Evaluate a single task in an isolated subprocess via _bigcodebench_sandbox.py."""
    input_fd, input_path = tempfile.mkstemp(suffix=".json", prefix="bcb_in_")
    result_path = input_path + ".result"
    try:
        with os.fdopen(input_fd, "w") as f:
            json.dump({
                "code": code,
                "test_code": task["test"],
                "entry_point": task["entry_point"],
            }, f)

        proc = subprocess.run(
            [sys.executable, _SANDBOX_SCRIPT, input_path, result_path],
            capture_output=True,
            timeout=TIMEOUT_LIMIT + 70,
        )

        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            return (idx, result["stat"], result["details"])
        else:
            stderr_msg = proc.stderr.decode(errors="replace")[:500]
            return (idx, FAIL, {"ALL": f"No result file. stderr: {stderr_msg}"})
    except subprocess.TimeoutExpired:
        return (idx, TIMEOUT, {"ALL": "Subprocess timed out"})
    except Exception as e:
        return (idx, FAIL, {"ALL": str(e)})
    finally:
        for p in [input_path, result_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def run_tests_parallel(tasks, code_list, num_workers=8):
    """Run sandbox tests in isolated subprocesses using ThreadPoolExecutor.

    Each test is dispatched to _bigcodebench_sandbox.py via subprocess.run,
    so child processes do not inherit the training process memory (model weights).
    ThreadPoolExecutor manages concurrency without forking.
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_eval_single_subprocess, idx, task, code): idx
            for idx, (task, code) in enumerate(zip(tasks, code_list))
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append((idx, FAIL, {"ALL": str(e)}))

    results.sort(key=lambda x: x[0])
    return results


def evaluate_bigcodebench(model, tokenizer, test_data_dir,
                          rank=0, world_size=1, max_samples=None,
                          max_gen_length=2048, batch_size=1):
    """
    Evaluate on BigCodeBench (Complete, v0.1.4).

    Args:
        model: DeepSpeed model engine or HF model
        tokenizer: tokenizer
        test_data_dir: path to directory containing parquet file
        rank, world_size: for distributed evaluation
        max_samples: limit number of samples per rank
        max_gen_length: max tokens to generate
        batch_size: generation batch size

    Returns:
        (accuracy, correct, total, details) - compatible with evaluate_test_set interface
    """
    if rank == 0:
        print(f"[BigCodeBench] Loading tasks from {test_data_dir}")

    tasks = load_bigcodebench_local(test_data_dir)

    if rank == 0:
        print(f"[BigCodeBench] Total tasks: {len(tasks)}")

    # Shard across ranks
    tasks = [t for i, t in enumerate(tasks) if i % world_size == rank]
    if max_samples:
        tasks = tasks[:max_samples]

    if rank == 0:
        print(f"[BigCodeBench] Rank {rank}: {len(tasks)} tasks")

    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero tasks after sharding.
    # Generate code
    raw_outputs = batch_generate_code(model, tokenizer, tasks,
                                      max_gen_length, batch_size, rank)

    # Extract complete code (prompt + generated body)
    code_list = [extract_code(raw, task["complete_prompt"])
                 for raw, task in zip(raw_outputs, tasks)]

    if rank == 0:
        print(f"[BigCodeBench] Running sandbox tests...")

    # Run tests in isolated subprocesses
    test_results = run_tests_parallel(tasks, code_list, num_workers=min(12, len(tasks)))

    local_correct = sum(1 for _, stat, _ in test_results if stat == PASS)
    local_total = len(tasks)

    if rank == 0:
        for idx, stat, test_details in test_results:
            task = tasks[idx]
            details.append({
                "task_id": task["task_id"],
                "entry_point": task["entry_point"],
                "raw_output": raw_outputs[idx],
                "extracted_code": code_list[idx],
                "status": stat,
                "test_details": test_details if stat != PASS else {},
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
        print(f"[BigCodeBench] pass@1: {accuracy:.2%} ({correct}/{total})")

    return accuracy, correct, total, details
