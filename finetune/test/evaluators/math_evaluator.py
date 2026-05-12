"""Math reasoning evaluator for datasets like GSM8K, MATH, Minerva, etc."""
import torch
import deepspeed
from ..utils import UnitTextManager, StringCleaner, AnswerExtractor, load_test_data, math_verify_compare


def generate_prompt(question, tokenizer=None):
    if tokenizer and getattr(tokenizer, "chat_template", None):
        messages = [{'role': 'user', 'content': f'{question}\n\nPut your final answer in \\boxed{{}}.'}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes this question. Put your final answer in \\boxed{{}}.
### Instruction: 
{question}
    
### Response:
"""


def batch_generate_answers(model, tokenizer, questions, max_length=1024, batch_size=16, rank=0):
    all_responses = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gather all model parameters so generate() doesn't need cross-rank allgather
    params_to_gather = [p for p in model.parameters()]

    # Sync all ranks before entering gathered context
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
        for i in range(0, len(questions), batch_size):
            if rank == 0:
                print(f"[Test Evaluator] Rank {rank}: batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")

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
                elif '### Response:' in response:
                    response = response.split('### Response:')[-1].strip()
                all_responses.append(response)

    # Sync after all generation is done
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    tokenizer.padding_side = original_padding_side
    return all_responses


def evaluate_exact_match(model, tokenizer, test_data_dir, rank=0, world_size=1, max_samples=None, max_gen_length=1024, batch_size=16):
    if rank == 0:
        print(f"[Test Evaluator] Distributed evaluation on {world_size} GPUs, batch_size={batch_size}")
    
    test_samples = load_test_data(test_data_dir, rank, world_size)
    if max_samples:
        test_samples = test_samples[:max_samples]
    
    details = []
    # Always call batch_generate so its collectives (barrier,
    # GatheredParameters) fire on every rank, even when this rank
    # received zero test_samples after sharding.
    unit_manager = UnitTextManager()
    string_cleaner = StringCleaner(unit_manager)
    answer_extractor = AnswerExtractor(string_cleaner)
        
    questions = [s['question'] for s in test_samples]
    responses = batch_generate_answers(model, tokenizer, questions, max_gen_length, batch_size, rank)
        
    local_correct = 0
    local_total = len(test_samples)
        
    for idx, (sample, pred_text) in enumerate(zip(test_samples, responses)):
        gt_answer = sample['answer']
        pred_answer = answer_extractor.extract_answer(pred_text)
        gt_answer_clean = answer_extractor.extract_answer(gt_answer)
            
        is_correct = math_verify_compare(pred_answer, gt_answer_clean)
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
                print(f"[Test Evaluator] Sample {idx}: Pred-text={pred_text}, Pred={pred_answer}, GT={gt_answer_clean}")
    
    correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=model.device)
    total_tensor = torch.tensor([local_total], dtype=torch.long, device=model.device)
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
    
    correct = correct_tensor.item()
    total = total_tensor.item()
    accuracy = correct / total if total > 0 else 0.0
    
    if rank == 0:
        print(f"[Test Evaluator] Complete: {correct}/{total} = {accuracy:.2%}")
    
    return accuracy, correct, total, details

