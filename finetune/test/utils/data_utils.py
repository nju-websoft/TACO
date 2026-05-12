"""Data loading utilities"""
import json
import glob
import os


def load_test_data(test_data_dir, rank=0, world_size=1):
    all_files = sorted(glob.glob(os.path.join(test_data_dir, '*.json')))
    test_samples = []
    if rank == 0:
        print(f"[Test Evaluator] Found {len(all_files)} JSON files")
    for f in all_files:
        with open(f, 'r', encoding='utf-8') as fd:
            data = json.load(fd)
            if isinstance(data, list):
                for item in data:
                    # Support both alpaca-style ('instruction'/'output') and
                    # MMLU-style ('question'/'answer') field naming. Without
                    # this fallback mmlu_medical (which uses question/answer)
                    # silently loads 0 samples and reports 0/0 = 0.00%.
                    question = item.get('instruction') or item.get('question') or ''
                    answer = item.get('output') or item.get('answer') or ''
                    if question and answer:
                        test_samples.append({'question': question, 'answer': answer})
    test_samples = [s for i, s in enumerate(test_samples) if i % world_size == rank]
    if rank == 0:
        print(f"[Test Evaluator] Rank {rank}: {len(test_samples)} samples")
    return test_samples
