"""Code generation evaluator using LiveCodeBench.

Core evaluation logic is copied from LiveCodeBench without modification.
Only the generation part is adapted for local model inference with DeepSpeed.
"""
import torch
import deepspeed
from ..lcb.benchmarks.code_generation import CodeGenerationProblem, load_code_generation_dataset
from ..lcb.prompts.code_generation import format_prompt_generation
from ..lcb.lm_styles import LMStyle
from ..lcb.evaluation.compute_code_generation_metrics import codegen_metrics


def extract_code(model_output: str) -> str:
    """Extract code from model output, handling both raw code and markdown blocks.

    Handles these cases:
    1. Output wrapped in ```python ... ``` -> extract last code block content
    2. Output wrapped in ``` ... ``` -> extract last code block content
    3. Raw code (no markdown) -> return as-is (stripped)
    """
    stripped = model_output.strip()
    if not stripped:
        return ""

    lines = stripped.split("\n")
    # Find all ``` delimiter lines
    fence_indices = [i for i, line in enumerate(lines) if line.strip().startswith("```")]

    if len(fence_indices) >= 2:
        # Has code block(s) - take the last complete block
        start = fence_indices[-2]
        end = fence_indices[-1]
        # Skip the opening ``` line (may contain "```python")
        code_lines = lines[start + 1 : end]
        return "\n".join(code_lines)

    # No code block found - return raw output (base model behavior)
    return stripped


def generate_prompt(question, tokenizer=None):
    """Generate prompt for a LiveCodeBench problem.

    Chat models: direct instruction (no few-shot) with clear output format.
    Base models: original LCB few-shot format (GenericBase).
    """
    if tokenizer and getattr(tokenizer, "chat_template", None):
        problem_text = question.question_content

        if question.starter_code:
            user_content = (
                f"Solve the following programming problem.\n\n"
                f"## Problem\n{problem_text}\n\n"
                f"## Starter Code\n```python\n{question.starter_code}\n```\n\n"
                f"Complete the starter code. Output your solution in a ```python code block."
            )
        else:
            user_content = (
                f"Solve the following programming problem.\n\n"
                f"## Problem\n{problem_text}\n\n"
                f"Write a complete Python program that reads input from stdin and "
                f"writes output to stdout. Output your solution in a ```python code block."
            )

        msgs = [
            {"role": "system", "content": "You are an expert Python programmer. You solve competitive programming problems by writing correct and efficient code."},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Base model: wrap LCB format with response marker for reliable prompt stripping
    base_prompt = format_prompt_generation(question, LMStyle.GenericBase)
    return base_prompt.rstrip() + '\n### Response:\n'


def batch_generate_code(model, tokenizer, problems, max_length=2048, batch_size=1, rank=0):
    """Generate code solutions for problems using local model."""
    all_responses = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params_to_gather = [p for p in model.parameters()]
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
        total_batches = (len(problems) + batch_size - 1) // batch_size
        for i in range(0, len(problems), batch_size):
            if rank == 0:
                print(f"[Code Evaluator] batch {i // batch_size + 1}/{total_batches}")

            batch = problems[i:i + batch_size]
            prompts = [generate_prompt(p, tokenizer) for p in batch]
            inputs = tokenizer(prompts, return_tensors='pt', truncation=True,
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


def evaluate_code_generation(model, tokenizer, release_version='release_latest',
                             rank=0, world_size=1, max_samples=None,
                             max_gen_length=2048, batch_size=1):
    """
    Evaluate code generation on LiveCodeBench.

    Returns:
        (pass_at_1, correct, total) - same interface as math/mmlu evaluators
    """
    if rank == 0:
        print(f"[Code Evaluator] Loading LiveCodeBench: {release_version}")

    benchmark = load_code_generation_dataset(release_version)
    benchmark = sorted(benchmark, key=lambda x: x.question_id)

    if rank == 0:
        print(f"[Code Evaluator] Total problems: {len(benchmark)}")

    # Shard for distributed
    benchmark = [b for i, b in enumerate(benchmark) if i % world_size == rank]
    if max_samples:
        benchmark = benchmark[:max_samples]

    if rank == 0:
        print(f"[Code Evaluator] Rank {rank}: {len(benchmark)} problems")

    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero benchmark after sharding.
    # Generate
    raw_outputs = batch_generate_code(model, tokenizer, benchmark,
                                      max_gen_length, batch_size, rank)

    # Extract code from outputs
    code_outputs = [extract_code(o) for o in raw_outputs]

    # Build evaluation samples
    eval_samples = [p.get_evaluation_sample() for p in benchmark]
    generations = [[code] for code in code_outputs]

    if rank == 0:
        print(f"[Code Evaluator] Running code execution tests...")

    metrics, results, _ = codegen_metrics(
        eval_samples, generations, num_process_evaluate=12, timeout=6)

    pass_at_1 = metrics["pass@1"]
    local_correct = int(pass_at_1 * len(benchmark))
    local_total = len(benchmark)

    if rank == 0:
        for i, (prob, raw_out, code_out) in enumerate(zip(benchmark, raw_outputs, code_outputs)):
            passed = results[i][0] if results[i] else False
            details.append({
                'question_id': prob.question_id,
                'question_title': prob.question_title,
                'raw_output': raw_out,
                'extracted_code': code_out,
                'passed': bool(passed),
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
        print(f"[Code Evaluator] pass@1: {accuracy:.2%} ({correct}/{total})")

    return accuracy, correct, total, details
