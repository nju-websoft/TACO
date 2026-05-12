import os
import json
import time
import glob
import torch
import random
from torch.utils.data import IterableDataset


class DynamicAlpacaDataset(IterableDataset):
    def __init__(self, data_dir, tokenizer, max_length=8192, rank=0, world_size=1,
                 file_list=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rank = rank
        self.world_size = world_size
        # When set, overrides the per-iter glob() so every rank sees the
        # same file list in the same order. The training loop assigns this
        # after the Judger to broadcast rank 0's view to all ranks.
        self.file_list = file_list

    def _resolve_files(self):
        if self.file_list is not None:
            return list(self.file_list)
        return sorted(glob.glob(os.path.join(self.data_dir, "*.json")))

    def __len__(self):
        try:
            all_files = self._resolve_files()
            count = 0
            for f in all_files:
                with open(f, 'r', encoding='utf-8') as fd:
                    data = json.load(fd)
                    if isinstance(data, list):
                        count += len(data)
            return count
        except Exception:
            return 0


    def __iter__(self):
        rank = self.rank
        world_size = self.world_size

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        all_files = self._resolve_files()

        valid_files = []
        for f in all_files:
            try:
                with open(f, 'r', encoding='utf-8') as fd:
                    data = json.load(fd)
                    if isinstance(data, list) and len(data) > 0:
                        valid_files.append(f)
            except Exception:
                pass

        all_files = valid_files
        if not all_files:
            return

        all_records = []
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list) or len(data) == 0:
                    continue
                all_records.extend(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        random.Random(42).shuffle(all_records)  # Disabled to prevent re-shuffling

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id
        worker_records = [item for idx, item in enumerate(all_records) if idx % total_workers == global_worker_id]

        print(f"[Rank {rank} Worker {worker_id}] {len(worker_records)} records")

        for item in worker_records:
            text = item.get("text", "")

            if text.find("####") != -1 and "gsm8k" in self.data_dir:
                text = text[:text.find("####")]

            if not text:
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")
                if inst and out:
                    text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n###Input:\n{inp}\n\n### Response:\n{out}"

            # Skip empty text
            if not text or len(text.strip()) == 0:
                continue

            sep = "### Response:"
            sep_pos = text.find(sep)

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = enc.input_ids[0]

            # Skip if tokenization resulted in empty sequence
            if len(input_ids) == 0:
                continue

            labels = input_ids.clone()

            if sep_pos != -1:
                prefix_text = text[:sep_pos + len(sep)]
                prefix_enc = self.tokenizer(
                    prefix_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                prefix_len = len(prefix_enc.input_ids[0])
                labels[:prefix_len] = -100
            else:
                labels[:] = -100

            # Skip if all labels are masked
            if (labels != -100).sum() == 0:
                continue

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "raw_text": text,
                "id": item.get("id", "unknown")
            }

