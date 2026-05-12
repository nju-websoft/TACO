import argparse
import os
import sys
import subprocess
from preprocess.preprocess import run_preprocess
from agent_interface import chat
from pathlib import Path
import glob
import json


def execute_preprocess(args):
    print("\n" + "="*50)
    print("STAGE 1: PREPROCESS")
    print("="*50)
    try:
        run_preprocess(args.origin_data_dir, batch_size=args.preprocess_batch_size, workers=args.preprocess_workers)
        print("Preprocessing completed successfully.")
        breakpoint()
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)

def execute_agent(instruction, dataset_dir):
    print("\n" + "="*50)
    print("STAGE 2: AGENT EXECUTION")
    print("="*50)
    try:
        chat(instruction, dataset_dir=dataset_dir)
        print("Agent execution completed.")
    except Exception as e:
        print(f"Error during agent execution: {e}")
        sys.exit(1)

def execute_train(args):
    print("\n" + "="*50)
    print("STAGE 3: TRAINING")
    print("="*50)

    # Extract model short name to match model-prefixed directories
    model_name = os.path.basename(args.model_path.rstrip('/'))

    target_folders = [
        folder.resolve()
        for folder in Path(args.origin_data_dir).iterdir()
        if folder.is_dir()
        and "basemodel" in folder.name
        and "fine" in folder.name
        and "rough" in folder.name
        and folder.name.startswith(model_name)
    ]
    if len(target_folders) == 0:
        raise ValueError(
            f"No filtered dataset folder found for model '{model_name}' in {args.origin_data_dir}. "
            f"Expected a folder starting with '{model_name}' containing 'rough', 'basemodel', and 'fine'."
        )
    if len(target_folders) > 1:
        raise ValueError(
            f"Multiple filtered dataset folders found for model '{model_name}': {[f.name for f in target_folders]}. Expected exactly one."
        )

    # Parse GPU specification
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)

    # Build DeepSpeed command with GPU specification
    cmd = [
        "deepspeed",
        "--include", f"localhost:{','.join(map(str, gpu_ids))}", # Specify which GPUs to use
        "finetune/train.py",
        "--data_dir", str(target_folders[0]),
        "--original_data_dir", args.origin_data_dir,
        "--epochs", str(args.epochs),
        "--validate_interval", str(args.validate_interval),
        "--judge_interval", str(args.judge_interval),
        "--test_data_dirs", *args.test_data_dirs,
        "--model_path", args.model_path,
        "--enable_judger",
        "--test_bs", args.test_bs,
        "--deepspeed",
        "--deepspeed_config", args.deepspeed_config
    ]

    print(f"Training on {num_gpus} GPU(s): {gpu_ids}")
    print(f"Executing command: {' '.join(cmd)}")

    breakpoint()
    
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Pipeline: Preprocess -> Agent -> Train")

    parser.add_argument("--instruction", type=str, 
                        default="Extract high-quality math reasoning from this multi-discipline dataset.", 
                        help="Natural language instruction for the agent")
    
    parser.add_argument("--model_path", type=str, default="/data/fqzhou/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B", help="Path to model directory")

    parser.add_argument("--origin_data_dir", type=str, default="dataset/tulu_alpaca", help="Directory containing processed data for training")
    parser.add_argument("--test_data_dirs", type=str, nargs="+", default=["benchmark/aime24"], help="Paths to test datasets (supports multiple)")
    parser.add_argument("--test_bs", type=int, default=32)
    
    parser.add_argument("--preprocess_batch_size", type=int, default=64, help="Batch size for preprocessing")
    parser.add_argument("--preprocess_workers", type=int, default=8, help="Number of workers for preprocessing")
    
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--validate_interval", type=int, default=1, help="Validation interval (in epochs)")
    parser.add_argument("--judge_interval", type=int, default=1, help="Number of validations before triggering judger")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    
    parser.add_argument("--deepspeed_config", type=str, default="ds_config/ds_config_3b.json", help="Path to DeepSpeed config")
    parser.add_argument("--gpus", type=str, default="2,3", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    
    args = parser.parse_args()

    print(f"Preprocessing files in {args.origin_data_dir}...")
    execute_preprocess(args)

    execute_agent(args.instruction, args.origin_data_dir)

    execute_train(args)
