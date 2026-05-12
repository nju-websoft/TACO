"""Physics evaluator for PHYSICS dataset."""
import torch
import deepspeed
import json
import os
import glob
from ..utils import load_test_data
from ..utils.extract_boxed import extract_final_answer_allform
from ..utils.equation_equivalency import is_equiv


def generate_prompt(question, tokenizer=None):
    """Generate prompt for physics questions."""
    system_prompt = "You are an AI expert specializing in answering advanced physics questions. Please provide detailed reasoning and put your final answer in \\boxed{}."

    if tokenizer and getattr(tokenizer, "chat_template", None):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"""{system_prompt}

### Question:
{question}

### Answer:
"""


def batch_generate_answers(model, tokenizer, questions, max_length=2048, batch_size=1, rank=0):
    """Generate answers for a batch of physics questions."""
    all_responses = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params_to_gather = [p for p in model.parameters()]

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
        for i in range(0, len(questions), batch_size):
            if rank == 0:
                print(f"[Physics Evaluator] Rank {rank}: batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")

            batch_questions = questions[i:i+batch_size]
            prompts = [generate_prompt(q, tokenizer) for q in batch_questions]
            inputs = tokenizer(prompts, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)

            for output in outputs:
                response = tokenizer.decode(output, skip_special_tokens=True)
                if 'assistant\n' in response:
                    response = response.split('assistant\n')[-1].strip()
                elif '### Answer:' in response:
                    response = response.split('### Answer:')[-1].strip()
                all_responses.append(response)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    tokenizer.padding_side = original_padding_side
    return all_responses


def load_physics_data(test_data_dir, rank, world_size):
    """Load all physics jsonl files from directory."""
    all_samples = []
    jsonl_files = glob.glob(os.path.join(test_data_dir, '*.jsonl'))

    for fpath in sorted(jsonl_files):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                all_samples.append(sample)

    # Shard by rank
    shard_samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]
    return shard_samples


def evaluate_physics(model, tokenizer, test_data_dir, rank=0, world_size=1, max_samples=None, max_gen_length=2048, batch_size=1):
    """Evaluate physics reasoning with SymPy + LLM fallback."""
    if rank == 0:
        print(f"[Physics Evaluator] Distributed evaluation on {world_size} GPUs, batch_size={batch_size}")

    test_samples = load_physics_data(test_data_dir, rank, world_size)
    if max_samples:
        test_samples = test_samples[:max_samples]

    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero test_samples after sharding.
    questions = [s['questions'] for s in test_samples]
    responses = batch_generate_answers(model, tokenizer, questions, max_gen_length, batch_size, rank)

    local_correct = 0
    local_total = len(test_samples)

    for idx, (sample, pred_text) in enumerate(zip(test_samples, responses)):
        gt_answers = sample['final_answers']
        pred_answers = extract_final_answer_allform(pred_text, answer_type='list')

        is_correct = False
        equiv_detail = None

        if pred_answers:
            pred_answer = pred_answers[0]
            for gt_answer in gt_answers:
                try:
                    result = is_equiv(pred_answer, gt_answer)
                    if result.get('final_result'):
                        is_correct = True
                        equiv_detail = result
                        break
                except Exception as e:
                    continue

        if is_correct:
            local_correct += 1

        if rank == 0:
            details.append({
                'question': sample['questions'],
                'response': pred_text,
                'pred_answer': pred_answers[0] if pred_answers else None,
                'gt_answer': gt_answers,
                'correct': is_correct,
                'equiv_detail': equiv_detail,
            })
            if idx < 3:
                print(f"[Physics Evaluator] Sample {idx}: Pred={pred_answers}, GT={gt_answers}, Correct={is_correct}")

    correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=model.device)
    total_tensor = torch.tensor([local_total], dtype=torch.long, device=model.device)
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

    correct = correct_tensor.item()
    total = total_tensor.item()
    accuracy = correct / total if total > 0 else 0.0

    if rank == 0:
        print(f"[Physics Evaluator] Complete: {correct}/{total} = {accuracy:.2%}")

    return accuracy, correct, total, details
