import pandas as pd
import argparse
import glob
import os
import json
import pyarrow.parquet as pq
from tqdm import tqdm

def process_batch_and_save(batch_df, file_counter, output_dir, tgt_file_prefix):
    """
    Process a DataFrame batch and save it as a separate JSON file.
    """
    json_list = batch_df.to_dict(orient="records")
    fixed_json_list = []
    
    for json_unit in json_list:
        # Handle cases where keys might be different or missing
        # Adjust these keys based on your specific parquet schema  
        problem = json_unit.get('prompt').replace("### Response", "### Response:")
        solution = json_unit.get('response')
        
        fixed_json_list.append(
            {
                "instruction": problem,
                "input": "",
                "output": solution,
                # "text": f"Below is an instruction that describes a task. Write response that appropriately solve this task.\n\n### Instruction:\n{problem}\n\n### Response:\n{solution}",
                "text": f"Below is an instruction that describes a task. Write response that appropriately solve this task.\n\n### Instruction:\n{problem}\n\n{solution}",
            }
        )
    
    if not fixed_json_list:
        return 0

    output_filename = f"{tgt_file_prefix}_{file_counter}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed_json_list, f, ensure_ascii=False, indent=2)
    
    return len(fixed_json_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parquet file to JSON list in chunks.")
    parser.add_argument("--parquet_file_dir", default='/data/fqzhou/dataset_agent/dataset/oss_instruct_50k', help="Path to the parquet directory.")
    parser.add_argument("--tgt_data_dir", default="/data/fqzhou/dataset_agent/dataset/oss_instruct_50k")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of records per output JSON file.")
    args = parser.parse_args()
    
    tgt_file_prefix = os.path.basename(args.parquet_file_dir)

    all_data_files = glob.glob(os.path.join(args.parquet_file_dir, "*.parquet"))
    
    if not all_data_files:
        print(f"No .parquet files found in {args.parquet_file_dir}")
        exit(0)
        
    print(f"Found {len(all_data_files)} parquet files.")
    
    current_buffer = []
    file_counter = 0
    total_processed = 0
    
    # Ensure output directory exists (same as input dir based on original logic)
    output_dir = args.tgt_data_dir
    
    for parquet_path in tqdm(all_data_files, desc="Processing files"):
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            # Iterate over row groups to avoid loading entire file
            for batch in parquet_file.iter_batches(batch_size=args.chunk_size):
                batch_df = batch.to_pandas()
                current_buffer.extend(batch_df.to_dict(orient="records"))
                
                # If buffer exceeds chunk size, write to file
                while len(current_buffer) >= args.chunk_size:
                    chunk_to_write = current_buffer[:args.chunk_size]
                    current_buffer = current_buffer[args.chunk_size:]
                    
                    # Convert to dataframe just for the helper function (or refactor helper)
                    # Here we can just process the list directly to save overhead
                    temp_df = pd.DataFrame(chunk_to_write) 
                    saved_count = process_batch_and_save(temp_df, file_counter, output_dir, tgt_file_prefix)
                    file_counter += 1
                    total_processed += saved_count
                    
        except Exception as e:
            print(f"Error processing {parquet_path}: {e}")

    # Write remaining data
    if current_buffer:
        temp_df = pd.DataFrame(current_buffer)
        saved_count = process_batch_and_save(temp_df, file_counter, output_dir, tgt_file_prefix)
        total_processed += saved_count

    print(f"Processing complete.")
    print(f"Total records processed: {total_processed}")
    print(f"Total files created: {file_counter + 1 if current_buffer else file_counter}")
