import json
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from agent.config import load_config
from tqdm import tqdm
import pickle
import time
from transformers import AutoTokenizer
from agent.utils import safe_truncate


def process_batch_with_retry(tokenizer, client, model, batch_items, retries=3):
    """
    带重试机制的批处理，确保尽量不丢件
    """
    for attempt in range(retries):
        try:
            texts = [safe_truncate(tokenizer, item['text']) for item in batch_items]
            response = client.embeddings.create(input=texts, model=model)
            results = sorted(response.data, key=lambda x: x.index)
            
            for i, res in enumerate(results):
                batch_items[i]['embedding'] = res.embedding
            return True
        except Exception as e:
            print(f"\n尝试 {attempt+1}/{retries} 失败: {e}")
            if attempt < retries - 1:
                time.sleep(1) # 简单退避
    return False

def add_embeddings_to_json(file_path, batch_size=32, max_workers=16, skip_index=False):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    # Load configuration
    cfg = load_config()
    bd_cfg = cfg.get("bd", {})
    
    embed_url = bd_cfg.get("embed_url")
    embed_api_model = bd_cfg.get("embed_api_model_name")
    api_key = bd_cfg.get("api_key", "EMPTY") 

    tokenizer = AutoTokenizer.from_pretrained(bd_cfg.get("embed_model_path", "weights/Qwen3-Embedding-0.6B"))
    
    if not embed_url or not embed_api_model:
        print("Error: 'embed_url' or 'embed_api_model' not found in config.yaml under 'bd' section.")
        return None

    print(f"Using Embedding Model: {embed_api_model} at {embed_url}")

    embed_client = OpenAI(
        base_url=embed_url,
        api_key=api_key,
    )

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read '{file_path}'.\n{e}")
        return None

    if not isinstance(data, list):
        print(f"Error: The JSON content in '{file_path}' is not a list.")
        return None

    items_to_process = [item for item in data if isinstance(item, dict) and 'text' in item and 'embedding' not in item]
    
    if items_to_process:
        print(f"Generating embeddings for {len(items_to_process)} items in {os.path.basename(file_path)}...")
        batches = [items_to_process[i:i + batch_size] for i in range(0, len(items_to_process), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch_with_retry, tokenizer, embed_client, embed_api_model, batch) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Embeddings"):
                future.result()

    # Return collected embeddings for external merging if needed
    final_embeddings, final_ids = [],[]
    for item in data:
        emb = item.get('embedding')
        item_id = item.get('id')
        if item_id is not None and isinstance(emb, list) and len(emb) > 0:
            final_embeddings.append(emb)
            final_ids.append(item_id)
    
    if skip_index:
        return final_embeddings, final_ids

    # Legacy Single File Logic
    if not final_embeddings:
        print("[Warning] No valid embeddings found to build FAISS index.")
        return None

    print(f"Building FAISS index with {len(final_embeddings)} vectors...")

    try:
        # Convert to float32 numpy array
        embeddings_matrix = np.array(final_embeddings, dtype='float32')
        dimension = embeddings_matrix.shape[1]
        
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_matrix)
        index.add(embeddings_matrix)
        
        base_name = os.path.splitext(file_path)[0]
        faiss_path = base_name + ".index"
        
        faiss.write_index(index, faiss_path)
        print(f"Successfully saved FAISS index to '{faiss_path}'.")
        
        id_mapping = final_ids
        with open(f"{base_name}.pkl", 'wb') as f:
            pickle.dump(id_mapping, f)

    except Exception as e:
        print(f"Error building or saving FAISS index: {e}")
        
    return final_embeddings, final_ids


def merge_and_save_index(all_embeddings, all_ids, output_dir, index_name="merged"):
    if not all_embeddings:
        print("No embeddings to merge.")
        return

    print(f"Merging {len(all_embeddings)} vectors into unified index...")
    try:
        embeddings_matrix = np.array(all_embeddings, dtype='float32')
        dimension = embeddings_matrix.shape[1]
        
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_matrix)
        index.add(embeddings_matrix)
        
        faiss_path = os.path.join(output_dir, f"{index_name}.index")
        faiss.write_index(index, faiss_path)
        
        pkl_path = os.path.join(output_dir, f"{index_name}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(all_ids, f)
            
        print(f"Merged index saved to {faiss_path}")
        print(f"Merged ID mapping saved to {pkl_path}")
        
    except Exception as e:
        print(f"Error merging index: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add embeddings to a JSON dataset and save FAISS index.")
    parser.add_argument("file_path", help="Path to the JSON file.")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of items per batch (default: 32).")
    parser.add_argument("--workers", type=int, default=16, help="Number of concurrent workers (default: 16).")
    
    if len(sys.argv) < 2:
        parser.print_help()
    else:
        args = parser.parse_args()
        add_embeddings_to_json(args.file_path, batch_size=args.batch_size, max_workers=args.workers)
