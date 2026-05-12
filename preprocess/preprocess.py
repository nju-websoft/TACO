import sys
import os
import argparse

# Add project root to sys.path to allow imports from agent and sibling modules if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the functions from the moved files
# Since we added project_root to sys.path, we can import from preprocess package
# or directly if we are in the same directory. 
# Using fully qualified imports based on project root is safer if we treat it as a package.
try:
    from preprocess.add_ids import add_ids_to_json
    from preprocess.add_embeddings import add_embeddings_to_json
except ImportError:
    # Fallback for when running directly inside the folder without package structure recognition
    from add_ids import add_ids_to_json
    from add_embeddings import add_embeddings_to_json

import glob
from preprocess.add_embeddings import merge_and_save_index

def run_preprocess(file_dir, batch_size=32, workers=16):
    targets = glob.glob(os.path.join(file_dir, "*.json"))
    output_dir = file_dir
        
    targets = [os.path.abspath(p) for p in targets if os.path.exists(p)]
    
    if not targets:
        print(f"Error: No files found matching '{file_dir}'.")
        return

    # Check for existing index and pkl files
    base_name = os.path.basename(file_dir)
    index_file = os.path.join(output_dir, f"{base_name}.index")
    pkl_file = os.path.join(output_dir, f"{base_name}.pkl")
    print(f"Index File path: {index_file}")
    print(f"Pkl File path: {pkl_file}")
    
    skip_embedding = False
    if os.path.exists(index_file) and os.path.exists(pkl_file):
        print(f"\nFound existing index ({index_file}) and pkl ({pkl_file}).")
        skip_embedding = True

    print(f"Starting preprocessing pipeline for {len(targets)} files.")
    
    all_embeddings = []
    all_ids = []

    for i, abs_path in enumerate(targets):
        print(f"\nProcessing File {i+1}/{len(targets)}: {abs_path}")
        
        # Step 1: Add IDs
        print("[Step 1/2] Adding IDs...")
        try:
            add_ids_to_json(abs_path)
        except Exception as e:
            print(f"Failed during ID addition for {abs_path}: {e}")
            continue

        if skip_embedding:
            print("Skipping embedding generation and index merging steps.")
            continue

        # Step 2: Add Embeddings (Skip individual index generation)
        print("[Step 2/2] Adding Embeddings...")
        try:
            # Pass skip_index=True to avoid creating .index per file
            result = add_embeddings_to_json(abs_path, batch_size=batch_size, max_workers=workers, skip_index=True)
            if result:
                embs, ids = result
                all_embeddings.extend(embs)
                all_ids.extend(ids)
        except Exception as e:
            print(f"Failed during Embedding generation for {abs_path}: {e}")
            continue
            
    # Step 3: Merge Index
    if all_embeddings and not skip_embedding:
        print("\n" + "="*50)
        print("[Step 3/3] Merging Indices...")
        print("="*50)
        # Use a common name for the merged index, e.g., based on directory name or 'total'
        # If processing "dataset/math", save as "dataset/math/total.index"
        merge_and_save_index(all_embeddings, all_ids, output_dir, index_name=os.path.basename(output_dir))
    
    print("\n" + "="*50)
    print("Pipeline completed.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline (add IDs -> add embeddings).")
    parser.add_argument("--file_dir", type=str, default="dataset/alpaca", help="Path to the JSON dataset directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embeddings.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for embeddings.")
    
    args = parser.parse_args()
    
    run_preprocess(args.file_dir, batch_size=args.batch_size, workers=args.workers)
