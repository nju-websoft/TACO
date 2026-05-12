"""Unified test evaluation interface"""
import os
from .evaluators import evaluate_exact_match, evaluate_mmlu_choice, evaluate_code_generation, evaluate_bigcodebench, evaluate_humaneval, evaluate_cruxeval_output, evaluate_cruxeval_input
from .evaluators.physics_evaluator import evaluate_physics
from .evaluators.openfindata_evaluator import evaluate_openfindata
from .evaluators.medqa_evaluator import evaluate_medqa


def evaluate_test_set(model, tokenizer, test_data_dir, rank=0, world_size=1, max_samples=None, max_gen_length=1024, batch_size=16):
    """
    Unified evaluation interface that automatically selects the correct evaluator based on test_data_dir.

    Supported datasets:
    - gsm8k, math, minerva, numina_cot: Math reasoning with boxed answers
    - mmlu_pro, mmlu: Multiple choice questions with letter answers
    - livecodebench: Code generation (pass@1)
    - physics: Physics reasoning with SymPy + LLM fallback
    - openfindata: Financial domain tasks
    - cruxeval_O: CRUXEval output prediction (pass@1)
    - cruxeval_I: CRUXEval input prediction (pass@1)
    """
    dataset_name = os.path.basename(test_data_dir.rstrip('/'))

    if rank == 0:
        print(f"[Unified Evaluator] Detected dataset: {dataset_name}")

    if dataset_name in ['mmlu_pro', 'mmlu', 'mmlu_medical']:
        return evaluate_mmlu_choice(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)
    elif dataset_name in ['humaneval']:
        return evaluate_humaneval(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)
    elif dataset_name in ['bigcodebench']:
        return evaluate_bigcodebench(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)
    elif dataset_name in ['livecodebench', 'lbc_trial']:
        return evaluate_code_generation(model, tokenizer, rank=rank, world_size=world_size, max_samples=max_samples, max_gen_length=max_gen_length, batch_size=batch_size)
    elif dataset_name == 'cruxeval_O':
        return evaluate_cruxeval_output(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)
    elif dataset_name == 'cruxeval_I':
        return evaluate_cruxeval_input(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)
    elif dataset_name == 'physics':
        return evaluate_physics(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)
    elif dataset_name == 'openfindata':
        return evaluate_openfindata(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)
    elif dataset_name == 'medqa':
        return evaluate_medqa(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)
    else:
        return evaluate_exact_match(model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size)


__all__ = ['evaluate_test_set']
