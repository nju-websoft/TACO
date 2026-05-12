"""OpenFinData evaluator for financial domain tasks."""
import torch
import deepspeed
import json
import os
import glob


def last_capital_postprocess(text: str) -> str:
    """Extract last capital letter from text."""
    for t in text[::-1]:
        if t.isupper():
            return t
    return ''


def build_prompt(item: dict, task_type: str, tokenizer=None) -> str:
    """Build prompt based on task type."""
    question = item['question']

    if tokenizer and getattr(tokenizer, "chat_template", None):
        messages = [{'role': 'user', 'content': question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"""Below is a question. Please answer it.

### Question:
{question}

### Answer:
"""


def batch_generate_answers(model, tokenizer, items, task_type, max_length=2048, batch_size=1, rank=0):
    """Generate answers for a batch of questions."""
    all_responses = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params_to_gather = [p for p in model.parameters()]

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
        for i in range(0, len(items), batch_size):
            if rank == 0:
                print(f"[OpenFinData Evaluator] Rank {rank}: batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")

            batch_items = items[i:i+batch_size]
            prompts = [build_prompt(item, task_type, tokenizer) for item in batch_items]
            inputs = tokenizer(prompts, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False,
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


def load_openfindata(test_data_dir, rank, world_size):
    """Load all openfindata json files and aggregate by task type."""
    task_configs = {
        'emotion_identification': '3choices',
        'entity_disambiguation': '3choices',
        'financial_facts': '3choices',
        'data_inspection': '4choices',
        'financial_terminology': '4choices',
        'metric_calculation': '4choices',
        'value_extraction': '4choices',
        'intent_understanding': '5choices',
        'entity_recognition': 'keyword',
    }

    all_tasks = []
    for task_name, task_type in task_configs.items():
        fpath = os.path.join(test_data_dir, f'{task_name}.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    item['task_name'] = task_name
                    item['task_type'] = task_type
                    all_tasks.append(item)

    shard_tasks = [t for i, t in enumerate(all_tasks) if i % world_size == rank]
    return shard_tasks


def evaluate_openfindata(model, tokenizer, test_data_dir, rank=0, world_size=1, max_samples=None, max_gen_length=2048, batch_size=1):
    """Evaluate OpenFinData financial domain tasks."""
    if rank == 0:
        print(f"[OpenFinData Evaluator] Distributed evaluation on {world_size} GPUs, batch_size={batch_size}")

    test_samples = load_openfindata(test_data_dir, rank, world_size)
    if max_samples:
        test_samples = test_samples[:max_samples]

    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero test_samples after sharding.
    responses = batch_generate_answers(model, tokenizer, test_samples, None, max_gen_length, batch_size, rank)

    local_correct = 0
    local_total = len(test_samples)

    for idx, (sample, pred_text) in enumerate(zip(test_samples, responses)):
        task_type = sample['task_type']
        gt_answer = sample['answer']

        if task_type == 'keyword':
            pred_answer = pred_text
            judgement = gt_answer.split('、')
            is_correct = all(item in pred_answer for item in judgement)
        else:
            pred_answer = last_capital_postprocess(pred_text)
            is_correct = pred_answer == gt_answer

        if is_correct:
            local_correct += 1

        if rank == 0:
            details.append({
                'task_name': sample['task_name'],
                'question': sample['question'],
                'response': pred_text,
                'pred_answer': pred_answer,
                'gt_answer': gt_answer,
                'correct': is_correct,
            })
            if idx < 3:
                print(f"[OpenFinData Evaluator] Sample {idx}: Task={sample['task_name']}, Pred={pred_answer}, GT={gt_answer}, Correct={is_correct}")

    correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=model.device)
    total_tensor = torch.tensor([local_total], dtype=torch.long, device=model.device)
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

    correct = correct_tensor.item()
    total = total_tensor.item()
    accuracy = correct / total if total > 0 else 0.0

    if rank == 0:
        print(f"[OpenFinData Evaluator] Complete: {correct}/{total} = {accuracy:.2%}")

    return accuracy, correct, total, details
