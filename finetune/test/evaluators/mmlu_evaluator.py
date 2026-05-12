"""MMLU-Pro evaluator for multiple choice questions"""
import torch
import deepspeed
import re
from ..utils import load_test_data


def generate_prompt_mmlu(question, tokenizer=None):
    """Generate prompt for MMLU-Pro multiple choice questions"""
    if tokenizer and getattr(tokenizer, "chat_template", None):
        messages = [{'role': 'user', 'content': f'{question}\n\nChoose the correct answer and respond with ONLY the letter (A, B, C, D, etc.) of your choice.'}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"""Below is a multiple choice question. Choose the correct answer and respond with ONLY the letter (A, B, C, D, etc.) of your choice.
### Instruction:
{question}

### Response:
"""


def extract_choice_answer(response):
    """Extract letter choice from model response.

    Robust against models that emit reasoning before the final letter.
    Priority order:
      1. ``[[A]]`` / ``[[B]]`` / ...      — prompted-bracket format if model used it
      2. ``(A)`` / ``(B)`` / ...           — parenthesised letter
      3. ``answer is X`` / ``answer: X``  — natural language pattern
      4. **Last** standalone A-Z letter   — likely the conclusion line
      5. First character if it is a letter
    """
    text = (response or "").strip()
    upper = text.upper()

    m = re.search(r"\[\[\s*([A-Z])\s*\]\]", upper)
    if m:
        return m.group(1)
    m = re.search(r"\(\s*([A-Z])\s*\)", upper)
    if m:
        return m.group(1)
    m = re.search(r"ANSWER\s*(?:IS|:|=)\s*([A-Z])\b", upper)
    if m:
        return m.group(1)
    matches = re.findall(r"\b([A-Z])\b", upper)
    if matches:
        return matches[-1]
    if upper and upper[0] in "ABCDEFGHIJKLMN":
        return upper[0]
    return ""


def batch_generate_mmlu(model, tokenizer, questions, max_length=512, batch_size=16, rank=0):
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
                print(f"[MMLU Evaluator] Rank {rank}: batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")

            batch_questions = questions[i:i+batch_size]
            prompts = [generate_prompt_mmlu(q, tokenizer) for q in batch_questions]
            inputs = tokenizer(prompts, return_tensors='pt', truncation=True, max_length=2048, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            input_padded_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)

            # `outputs` from `model.generate()` has shape [B, input_padded_len + gen_len]
            # and prepends the (left-padded) input ids. To recover ONLY the generated
            # tokens we slice off the first `input_padded_len` positions; without
            # this slice the chat-template path leaks the prompt's "A. ..."/"B. ..."
            # option labels into `extract_choice_answer`, which then unconditionally
            # returns "A" and produces the spurious 23.60% accuracy floor that
            # exactly matches the share of A-answers in the test set.
            for output in outputs:
                gen_tokens = output[input_padded_len:]
                response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                # Defensive fallback for legacy "### Instruction:..." prompts that
                # may still echo the prompt back when chat_template is absent.
                if '### Response:' in response:
                    response = response.split('### Response:')[-1].strip()
                all_responses.append(response)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    tokenizer.padding_side = original_padding_side
    return all_responses


def evaluate_mmlu_choice(model, tokenizer, test_data_dir, rank=0, world_size=1, max_samples=None, max_gen_length=512, batch_size=16):
    if rank == 0:
        print(f"[MMLU Evaluator] Distributed evaluation on {world_size} GPUs, batch_size={batch_size}")

    test_samples = load_test_data(test_data_dir, rank, world_size)
    if max_samples:
        test_samples = test_samples[:max_samples]

    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero test_samples after sharding.
    questions = [s['question'] for s in test_samples]
    all_responses = batch_generate_mmlu(model, tokenizer, questions, max_gen_length, batch_size, rank)

    local_correct = 0
    local_total = len(test_samples)
    details = []

    for idx, (sample, pred_text) in enumerate(zip(test_samples, all_responses)):
        gt_answer = sample['answer'].strip().upper()
        pred_answer = extract_choice_answer(pred_text)

        is_correct = pred_answer == gt_answer
        if is_correct:
            local_correct += 1

        if rank == 0:
            details.append({
                'question': sample['question'],
                'response': pred_text,
                'pred_answer': pred_answer,
                'gt_answer': gt_answer,
                'correct': is_correct,
            })
            if idx < 3:
                print(f"[MMLU Evaluator] Sample {idx}: Pred={pred_answer}, GT={gt_answer}")

    correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=model.device)
    total_tensor = torch.tensor([local_total], dtype=torch.long, device=model.device)
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

    correct = correct_tensor.item()
    total = total_tensor.item()
    accuracy = correct / total if total > 0 else 0.0

    if rank == 0:
        print(f"[MMLU Evaluator] Complete: {correct}/{total} = {accuracy:.2%}")

    return accuracy, correct, total, details
