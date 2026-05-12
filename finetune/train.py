
import os
import sys
# Add project root to path to allow importing finetune
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import re
import glob
import json
import math
import datetime
import torch
import argparse
import deepspeed
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from finetune.dataset import DynamicAlpacaDataset
from finetune.judger import Judger
from finetune.test import evaluate_test_set
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader


def get_original_vocab_size(model_path):
    """Get original vocab size from config before Zero-3 padding"""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    return config.vocab_size

def get_args():
    parser = argparse.ArgumentParser(description="Train base model on Alpaca")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed tra ning")

    parser.add_argument("--data_dir", type=str, default="dataset/math_trial", help="Path to dataset directory")
    parser.add_argument("--original_data_dir", type=str, default="dataset/math_trial", help="Path to original dataset directory")

    parser.add_argument("--model_path", type=str, default="/data/fqzhou/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B", help="Path to model directory")
    parser.add_argument("--steps_per_print", type=int, default=10, help="Logging frequency")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")

    parser.add_argument("--test_data_dirs", type=str, nargs="+", default=['benchmark/medqa'], help="Paths to test datasets (supports multiple)")
    parser.add_argument("--test_bs", type=int, default=4)
    
    parser.add_argument("--validate_interval", type=int, default=1, help="Validation interval (in epochs)")
    parser.add_argument("--judge_interval", type=int, default=1, help="Number of validations before triggering judger")
    parser.add_argument("--enable_judger", action="store_true", help="Enable judger analysis")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--max_gen_length", type=int, default=1024, help="Max generation length for test")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def collate_fn(batch, pad_token_id):
    # Batch is a list of dicts: [{"input_ids": ...}, ...]
    max_len = max(len(x["input_ids"]) for x in batch)
    
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    raw_texts = []
    sample_ids = []
    
    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]
        length = len(input_ids)
        
        # Collect raw text if available
        if "raw_text" in item:
            raw_texts.append(item["raw_text"])
            
        # Collect ID if available
        if "id" in item:
            sample_ids.append(item["id"])
        
        pad_len = max_len - length
        # Pad right
        padded_ids = torch.cat([input_ids, torch.tensor([pad_token_id] * pad_len, dtype=torch.long)])
        padded_lbl = torch.cat([labels, torch.tensor([-100] * pad_len, dtype=torch.long)]) # -100 ignore index
        
        mask = torch.tensor([1] * length + [0] * pad_len, dtype=torch.long)
        
        padded_input_ids.append(padded_ids)
        padded_labels.append(padded_lbl)
        attention_masks.append(mask)
        
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
        "raw_texts": raw_texts, # Pass raw texts to training loop
        "sample_ids": sample_ids # Pass IDs to training loop
    }

def main():
    args = get_args()

    # Initialize DeepSpeed Distributed
    deepspeed.init_distributed(timeout=datetime.timedelta(minutes=180))

    # Get distributed info
    rank = deepspeed.comm.get_rank()
    world_size = deepspeed.comm.get_world_size()

    if rank == 0:
        print(f"Initializing distributed training with {world_size} GPU(s)")
        print(f"Data directory: {args.data_dir}")
        if args.test_data_dirs:
            print(f"Test data directories: {args.test_data_dirs}")
        print(f"Model path: {args.model_path}")

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", use_cache=False)

    # Dataset
    dataset = DynamicAlpacaDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size
    )

    # Enable gradient checkpointing to reduce activation memory (critical for 3B+ models)
    model.gradient_checkpointing_enable()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    from deepspeed.runtime.lr_schedules import WarmupDecayLR
    _total_samples = len(dataset)
    _steps_per_epoch = max(1, _total_samples // (args.per_device_train_batch_size * world_size))
    _total_steps = _steps_per_epoch
    _peak_lr = model_engine.optimizer.param_groups[0]["lr"]
    _warmup_steps = int(_total_steps * 0.05)
    model_engine.lr_scheduler = WarmupDecayLR(
        model_engine.optimizer,
        total_num_steps=_total_steps,
        warmup_min_lr=0,
        warmup_max_lr=_peak_lr,
        warmup_num_steps=_warmup_steps
    )
    if rank == 0:
        print(f"Scheduler: total_steps={_total_steps}, warmup_steps=1, peak_lr={_peak_lr}")

    # Manually create DataLoader for IterableDataset
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_results = {}  # {dataset_name: [accuracies]}
    # Initialize logging - synchronize timestamp across all ranks
    log_root = "finetune/log"
    if rank == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = None
    # Broadcast timestamp from rank 0 to all ranks
    timestamp_list = [timestamp]
    torch.distributed.broadcast_object_list(timestamp_list, src=0)
    timestamp = timestamp_list[0]
    base_log_dir = os.path.join(log_root, timestamp)
    test_eval_epochs = []
    for test_dir in args.test_data_dirs:
        test_results[test_dir] = []
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Save training config
    if rank == 0:
        config_info = {
            'model_path': args.model_path,
            'data_dir': args.data_dir,
            'test_data_dirs': args.test_data_dirs,
            'epochs': args.epochs,
            'batch_size': args.per_device_train_batch_size,
            'learning_rate': _peak_lr,
            'timestamp': timestamp
        }
        with open(os.path.join(base_log_dir, 'config.json'), 'w') as f:
            json.dump(config_info, f, indent=2)
    epoch_losses = []
    epoch_loss_stds = []       # std of per-batch train losses each epoch
    pending_log_dirs = []

    def get_segment_dir(val_count):
        n = args.validate_interval
        train_count = n * val_count
        dirname = f"{train_count}_train_{val_count}_val"
        return os.path.join(base_log_dir, dirname)

    current_val_round = 1
    if rank == 0:
        current_log_dir = get_segment_dir(current_val_round)
        os.makedirs(current_log_dir, exist_ok=True)

        loss_log_path = os.path.join(base_log_dir, "loss_log.txt")
        plot_path = os.path.join(base_log_dir, f"loss_curve_{args.model_path.split('/')[-1]}.png")
        train_loss_id_map_path = os.path.join(current_log_dir, "train_loss_ids.jsonl")
        val_loss_id_map_path = os.path.join(current_log_dir, "val_loss_ids.jsonl")

        print(f"Logging training and validation info to {current_log_dir}")

    # TensorBoard writer
    if rank == 0:
        tb_log_dir = os.path.join(base_log_dir, 'tensorboard')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(tb_log_dir)
        print(f"TensorBoard logs: {tb_log_dir}")
    else:
        writer = None

    def update_loss_plot():
        """Redraw plots with train loss and test exact match."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        train_x = list(range(1, len(epoch_losses) + 1))
        ax1.plot(train_x, epoch_losses, marker="o", label="Train Loss", color="tab:blue")
        for x, y, s in zip(train_x, epoch_losses, epoch_loss_stds):
            ax1.annotate(f"σ={s:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7, color="tab:blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Average Loss")
        ax1.set_title("Train Loss Curve")
        ax1.legend()
        ax1.grid(True)
        if test_eval_epochs:
            colors = ["tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:red", "tab:blue", "tab:yellow"]
            for idx, (test_dir, accs) in enumerate(test_results.items()):
                if len(accs) == len(test_eval_epochs):
                    color = colors[idx % len(colors)]
                    label = test_dir.split("/")[-1]
                    ax2.plot(test_eval_epochs, accs, marker="s", label=label, color=color)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Exact Match Accuracy")
        ax2.set_title("Test Exact Match Curve")
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    # === Pre-training Test Evaluation (epoch 0 baseline) ===
    if args.test_data_dirs:
        if rank == 0:
            print("--- Pre-training Test Evaluation (Epoch 0 baseline) ---")
        
        model_engine.eval()
        for test_dir in args.test_data_dirs:
            accuracy, correct, total, details = evaluate_test_set(
                model_engine.module, tokenizer, test_dir, rank, world_size,
            max_samples=None, max_gen_length=args.max_gen_length, batch_size=args.test_bs
            )
            test_results[test_dir].append(accuracy)
            if rank == 0:
                print(f"  [{test_dir}] Exact Match: {accuracy:.2%} ({correct}/{total})")
                if writer:
                    writer.add_scalar(f"Test/{test_dir.split('/')[-1]}_accuracy", accuracy, 0)
                # Save response details
                if details:
                    results_dir = os.path.join(base_log_dir, "test_results", "epoch_0")
                    os.makedirs(results_dir, exist_ok=True)
                    dataset_name = os.path.basename(test_dir.rstrip('/'))
                    with open(os.path.join(results_dir, f"{dataset_name}.jsonl"), 'w', encoding='utf-8') as f:
                        for d in details:
                            f.write(json.dumps(d, ensure_ascii=False) + '\n')
                    print(f"  Saved {len(details)} responses to test_results/epoch_0/{dataset_name}.jsonl")
        test_eval_epochs.append(0)
        if rank == 0: update_loss_plot()

    torch.distributed.barrier()

    # Training Loop
    global_step = 0  # Track total steps across all epochs
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"Starting Epoch {epoch+1}/{args.epochs}")

        epoch_loss_sum = 0.0
        num_batches = 0
        epoch_batch_losses = []    # per-batch losses for std calculation
        epoch_loss_ids = [] # Store (id, loss) for the whole epoch

        # 1. Training Phase
        model_engine.train()
        train_iter = iter(train_dataloader)
        step = 0
        while True:
            # Try to get next batch; signal 1 if available, 0 if exhausted.
            try:
                batch = next(train_iter)
                has_data = torch.tensor([1], dtype=torch.long, device=model_engine.device)
            except StopIteration:
                has_data = torch.tensor([0], dtype=torch.long, device=model_engine.device)

            # All ranks vote: if ANY rank ran out of data, everyone stops.
            # This prevents deadlock when shards have unequal batch counts.
            torch.distributed.all_reduce(has_data, op=torch.distributed.ReduceOp.MIN)
            if has_data.item() == 0:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)

            # Forward
            outputs = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

            # Symmetric NaN detection: every rank checks its own loss and
            # all_reduces a flag, so when ANY rank sees NaN they all abort
            # together (avoiding rank 0 dying alone and others timing out
            # on the next collective).
            nan_flag = torch.tensor(
                [int(torch.isnan(loss).any() or torch.isinf(loss).any())],
                dtype=torch.long, device=model_engine.device,
            )
            torch.distributed.all_reduce(nan_flag, op=torch.distributed.ReduceOp.MAX)
            if nan_flag.item() > 0:
                if rank == 0:
                    print(f"ABNORMAL LOSS (NaN/Inf) at epoch {epoch+1} step {step}, aborting all ranks")
                raise RuntimeError("NaN/Inf loss detected; aborting distributed training")

            # Backward & Step
            model_engine.backward(loss)
            model_engine.step()

            # Accumulate loss for logging
            dist_loss = loss.detach().clone()
            torch.distributed.all_reduce(dist_loss, op=torch.distributed.ReduceOp.AVG)
            avg_loss = dist_loss.item()

            epoch_loss_sum += avg_loss
            num_batches += 1
            epoch_batch_losses.append(avg_loss)

            # Collect (loss, id) pairs (Only on Rank 0)
            if rank == 0:
                sample_ids = batch.get("sample_ids", [])
                epoch_loss_ids.append({
                    "loss": avg_loss,
                    "ids": sample_ids
                })

            if rank == 0:
                if step % args.steps_per_print == 0:
                    print(f"Epoch {epoch+1} [Train], Step {step}, Loss: {avg_loss}")
                global_step += 1
                if writer:
                    writer.add_scalar("Train/loss", avg_loss, global_step)
                    writer.add_scalar("Train/lr", model_engine.optimizer.param_groups[0]["lr"], global_step)

            step += 1

        # End of Epoch Training Logging
        if rank == 0:
            avg_epoch_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
            epoch_std = float(np.std(epoch_batch_losses)) if len(epoch_batch_losses) > 1 else 0.0
            epoch_losses.append(avg_epoch_loss)
            if writer:
                writer.add_scalar("Train/epoch_loss", avg_epoch_loss, epoch + 1)
            epoch_loss_stds.append(epoch_std)

            # Write loss-id map for this epoch
            with open(train_loss_id_map_path, "a", encoding="utf-8") as f:
                record = {
                    "epoch": epoch + 1,
                    "avg_loss": avg_epoch_loss,
                    "data": epoch_loss_ids
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Write to txt
            with open(loss_log_path, "a") as f:
                f.write(f"Epoch {epoch+1}: {avg_epoch_loss}\n")

            # Update Plot
            update_loss_plot()

            print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_epoch_loss}")


        # 2. Validation Phase (if triggered)
        if (epoch + 1) % args.validate_interval == 0:
            if rank == 0:
                print(f"--- Starting Validation Round (Epoch {epoch+1}) ---")

            model_engine.eval()
            val_epoch_loss_ids = []
            avg_val_losses = []

            # Create fresh validation dataloader for this epoch
            val_dataset = DynamicAlpacaDataset(
                data_dir=args.data_dir,
                tokenizer=tokenizer,
                rank=rank,
                world_size=world_size
            )
            # Reuse the file list broadcast from rank 0 by the Judger block so
            # validation sees the SAME files in the SAME order as training did
            # — without this, post-Judger filesystem propagation delay or
            # non-deterministic glob ordering can desync the val loss.
            val_dataset.file_list = dataset.file_list
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.per_device_train_batch_size,
                collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                num_workers=args.num_workers,
                pin_memory=True
            )

            # Calculate max steps on rank 0 and broadcast to ensure all ranks
            # use the same value even if files are being added concurrently.
            if rank == 0:
                total_samples = len(val_dataset)
                max_val_steps = (total_samples + args.per_device_train_batch_size * world_size - 1) // (args.per_device_train_batch_size * world_size)
            else:
                max_val_steps = 0

            max_val_steps_tensor = torch.tensor([max_val_steps], dtype=torch.long).to(model_engine.device)
            torch.distributed.broadcast(max_val_steps_tensor, src=0)
            max_val_steps = max_val_steps_tensor.item()

            if rank == 0:
                print(f"Validation: {total_samples} samples, {max_val_steps} steps")

            # Validate with fixed number of steps to ensure all ranks iterate same times
            with torch.no_grad():
                val_iter = iter(val_dataloader)
                val_step = 0
                while val_step < max_val_steps:
                    try:
                        batch = next(val_iter)
                        has_data = torch.tensor([1], dtype=torch.long, device=model_engine.device)
                    except StopIteration:
                        has_data = torch.tensor([0], dtype=torch.long, device=model_engine.device)

                    torch.distributed.all_reduce(has_data, op=torch.distributed.ReduceOp.MIN)
                    if has_data.item() == 0:
                        break

                    input_ids = batch["input_ids"].to(model_engine.device)
                    labels = batch["labels"].to(model_engine.device)
                    attention_mask = batch["attention_mask"].to(model_engine.device)

                    outputs = model_engine(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                    loss = outputs.loss

                    dist_loss = loss.detach().clone()
                    torch.distributed.all_reduce(dist_loss, op=torch.distributed.ReduceOp.AVG)
                    avg_val_loss = dist_loss.item()
                    avg_val_losses.append(avg_val_loss)

                    # Collect for Top-K (Only on Rank 0)
                    if rank == 0:
                        sample_ids = batch.get("sample_ids", [])
                        val_epoch_loss_ids.append({
                            "loss": avg_val_loss,
                            "ids": sample_ids
                        })

                    val_step += 1

            # Synchronize all ranks after validation loop
            torch.distributed.barrier()

            # 2b. Test Set Evaluation (exact match)
            if args.test_data_dirs:
                if rank == 0:
                    print(f"--- Starting Test Evaluation (Epoch {epoch+1}) ---")
                
                for test_dir in args.test_data_dirs:
                    accuracy, correct, total, details = evaluate_test_set(
                        model_engine.module, tokenizer, test_dir, rank, world_size,
                        max_samples=None, max_gen_length=args.max_gen_length, batch_size=args.test_bs
                    )
                    if rank == 0:
                        test_results[test_dir].append(accuracy)
                    if rank == 0:
                        print(f"  [{test_dir}] Exact Match: {accuracy:.2%} ({correct}/{total})")
                        if writer:
                            writer.add_scalar(f"Test/{test_dir.split('/')[-1]}_accuracy", accuracy, epoch + 1)
                        # Save response details
                        if details:
                            results_dir = os.path.join(base_log_dir, "test_results", f"epoch_{epoch + 1}")
                            os.makedirs(results_dir, exist_ok=True)
                            dataset_name = os.path.basename(test_dir.rstrip('/'))
                            with open(os.path.join(results_dir, f"{dataset_name}.jsonl"), 'w', encoding='utf-8') as f:
                                for d in details:
                                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
                            print(f"  Saved {len(details)} responses to test_results/epoch_{epoch + 1}/{dataset_name}.jsonl")
                if rank == 0:
                    test_eval_epochs.append(epoch + 1)
                
                if rank == 0: update_loss_plot()

            torch.distributed.barrier()

            if rank == 0:
                avg_val_loss_total = sum(avg_val_losses) / len(avg_val_losses) if len(avg_val_losses) > 0 else 0
                # Write val loss-id map for this epoch
                with open(val_loss_id_map_path, "a", encoding="utf-8") as f:
                    record = {
                        "epoch": epoch + 1,
                        "avg_loss": avg_val_loss_total,
                        "data": val_epoch_loss_ids
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"Validation finished.")

                # Add current log dir to pending list
                pending_log_dirs.append(current_log_dir)

                # Prepare for next segment
                current_val_round += 1
                current_log_dir = get_segment_dir(current_val_round)
                os.makedirs(current_log_dir, exist_ok=True)

                # Update paths
                train_loss_id_map_path = os.path.join(current_log_dir, "train_loss_ids.jsonl")
                val_loss_id_map_path = os.path.join(current_log_dir, "val_loss_ids.jsonl")

                print(f"Logging switched to new segment: {current_log_dir}")

            torch.distributed.barrier()

            # Trigger condition is `epoch + 1` based, NOT pending_log_dirs based,
            # so every rank evaluates it identically and the barrier below is
            # symmetric. Earlier the condition keyed off pending_log_dirs which
            # is only populated on rank 0, causing a NCCL collective mismatch
            # at the next epoch boundary.
            if (epoch + 1) % args.judge_interval == 0:
                if rank == 0:
                    print(f"Triggering Judger on {len(pending_log_dirs)} segments...")
                    try:
                        print(f"[Judger] Analyzing {pending_log_dirs}")
                        if args.enable_judger:
                            judger = Judger(pending_log_dirs, args.data_dir, args.original_data_dir)
                            judger.analyze()
                    except Exception as e:
                        print(f"Judger failed: {e}")
                    pending_log_dirs = []
                # Wait for rank 0 Judger to finish before all ranks proceed
                torch.distributed.barrier()
                # Broadcast the post-Judger data-dir file list from rank 0 so
                # all ranks load the SAME files in the SAME order in the next
                # epoch's iter(). Without this, filesystem propagation delay
                # or non-deterministic glob ordering can desync the stride-
                # based sharding in DynamicAlpacaDataset.
                file_list_holder = [None]
                if rank == 0:
                    file_list_holder[0] = sorted(
                        glob.glob(os.path.join(args.data_dir, "*.json"))
                    )
                torch.distributed.broadcast_object_list(file_list_holder, src=0)
                dataset.file_list = file_list_holder[0]
        
        # 3. Save Checkpoint (final epoch only)
        if epoch == args.epochs - 1:
            if rank == 0:
                print(f"Saving final checkpoint at Epoch {epoch+1}...")
            ckpt_id = f"epoch_{epoch+1}"
            model_engine.save_checkpoint(base_log_dir, ckpt_id)

    if rank == 0 and writer:
        writer.close()
    if rank == 0:
        print("Training finished.")

    # Clean up distributed process group
    if torch.distributed.is_initialized():
        if rank == 0 and writer:
            writer.close()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()