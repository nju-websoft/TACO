"""seed_similarity_filter.py — Embedding-similarity baseline for §4.5.

For each (pool, target-domain) pair we:
  1. sample N seed instructions from the oracle subset of that domain;
  2. encode the seeds with BAAI/bge-m3 (L2-normalized);
  3. stream the full mixed pool, encoding each sample's instruction;
  4. score each pool sample = max cosine similarity to any of the N seeds;
  5. write the top-K samples (default 100k) to out_dir/data.json in
     Alpaca schema with the "### Response:\\n" anchor used by ADCF.

A pool-embedding cache (.npz) is saved on first run; subsequent runs
on the same pool reuse it.

CLI:
  python seed_similarity_filter.py \\
    --pool openhermes \\
    --seed_dir openhermes_math_oracle \\
    --out_dir openhermes_math_sim
"""
import argparse
import json
import os
import random

import numpy as np
import pyarrow.parquet as pq
import ijson
import torch
from sentence_transformers import SentenceTransformer

OH_PATH = "/data/fqzhou/dataset_agent/dataset/openhermes_labeled/openhermes2_5.json"
TULU_DIR = "/data/fqzhou/.cache/modelscope/hub/datasets/allenai/tulu-3-sft-mixture/data"
OUT_ROOT = "/data/fqzhou/dataset_agent/dataset"

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n### Response:\n{output}"
)


def to_alpaca(instr, out, inp=""):
    text = ALPACA_PROMPT.format(instruction=instr, input=inp, output=out)
    return {"instruction": instr, "input": inp, "output": out, "text": text}


def first_human_assistant(conversations):
    if not conversations:
        return None
    user, asst = None, None
    for turn in conversations:
        role = turn.get("from")
        if role in ("human", "user") and user is None:
            user = turn.get("value", "")
        elif role in ("gpt", "assistant", "model") and user is not None and asst is None:
            asst = turn.get("value", "")
            break
    if user is None or asst is None:
        return None
    return user, asst


def first_user_assistant_msgs(messages):
    if messages is None:
        return None
    try:
        n = len(messages)
    except TypeError:
        return None
    if n == 0:
        return None
    user, asst = None, None
    for m in messages:
        role = m.get("role")
        if role == "user" and user is None:
            user = m.get("content", "")
        elif role == "assistant" and user is not None and asst is None:
            asst = m.get("content", "")
            break
    if user is None or asst is None:
        return None
    return user, asst


def iter_pool_openhermes():
    with open(OH_PATH, "rb") as f:
        for item in ijson.items(f, "item"):
            pair = first_human_assistant(item.get("conversations", []))
            if pair is None:
                continue
            yield {"instruction": pair[0], "output": pair[1]}


def iter_pool_tulu():
    parquet_files = sorted(
        os.path.join(TULU_DIR, f) for f in os.listdir(TULU_DIR) if f.endswith(".parquet")
    )
    for path in parquet_files:
        table = pq.read_table(path, columns=["messages"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            pair = first_user_assistant_msgs(row.get("messages"))
            if pair is None:
                continue
            yield {"instruction": pair[0], "output": pair[1]}


def truncate(s, max_chars=2000):
    """bge-m3 truncates internally at 8192 tokens; we cap chars too as a guard."""
    return s[:max_chars] if isinstance(s, str) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=["openhermes", "tulu"])
    ap.add_argument("--seed_dir", required=True,
                    help="Oracle dirname (relative to dataset/), e.g., openhermes_math_oracle")
    ap.add_argument("--out_dir", required=True,
                    help="Output dirname (relative to dataset/), e.g., openhermes_math_sim")
    ap.add_argument("--n_seeds", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=100_000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="/data/fqzhou/dataset_agent/dataset/_emb_cache")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 1. Load seed samples ──
    seed_path = os.path.join(OUT_ROOT, args.seed_dir, "data.json")
    print(f"[seed] loading from {seed_path}")
    with open(seed_path, "r", encoding="utf-8") as f:
        seed_records = json.load(f)
    seed_records = random.sample(seed_records, k=min(args.n_seeds, len(seed_records)))
    seed_texts = [truncate(r["instruction"]) for r in seed_records]
    print(f"[seed] sampled {len(seed_texts)} seed instructions")

    # ── 2. Load embedding model ──
    print(f"[model] loading BAAI/bge-m3 on {args.device}")
    model = SentenceTransformer("/data/fqzhou/.cache/modelscope/hub/models/BAAI/bge-m3", device=args.device)

    # ── 3. Encode seeds ──
    print(f"[encode] seeds...")
    seed_emb = model.encode(seed_texts, convert_to_numpy=True, batch_size=args.batch_size,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    print(f"[encode] seed_emb shape: {seed_emb.shape}")

    # ── 4. Pool: stream + encode (or load cache) ──
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"{args.pool}_pool_emb.npy")
    records_cache = os.path.join(args.cache_dir, f"{args.pool}_pool_records.json")

    iter_fn = iter_pool_openhermes if args.pool == "openhermes" else iter_pool_tulu

    if os.path.exists(cache_path) and os.path.exists(records_cache):
        print(f"[cache] loading pool_emb from {cache_path}")
        pool_emb = np.load(cache_path)
        print(f"[cache] loading pool_records from {records_cache}")
        with open(records_cache, "r", encoding="utf-8") as f:
            pool_records = json.load(f)
        assert len(pool_records) == pool_emb.shape[0], \
            f"cache mismatch: {len(pool_records)} records vs {pool_emb.shape[0]} embeddings"
        print(f"[cache] pool_emb={pool_emb.shape}, pool_records={len(pool_records)}")
    else:
        print(f"[encode] streaming pool ({args.pool}) and encoding (this can take ~15-30 min)...")
        pool_records = []
        emb_chunks = []
        buf_texts = []
        for i, rec in enumerate(iter_fn(), start=1):
            pool_records.append(rec)
            buf_texts.append(truncate(rec["instruction"]))
            if len(buf_texts) >= args.batch_size:
                emb = model.encode(buf_texts, convert_to_numpy=True,
                                   batch_size=args.batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
                emb_chunks.append(emb.astype(np.float32))
                buf_texts = []
            if i % 20000 == 0:
                print(f"[encode] processed {i} pool samples")
        if buf_texts:
            emb = model.encode(buf_texts, convert_to_numpy=True,
                               batch_size=args.batch_size,
                               normalize_embeddings=True, show_progress_bar=False)
            emb_chunks.append(emb.astype(np.float32))
        pool_emb = np.concatenate(emb_chunks, axis=0)
        print(f"[encode] pool_emb shape: {pool_emb.shape}; saving cache")
        np.save(cache_path, pool_emb)
        with open(records_cache, "w", encoding="utf-8") as f:
            json.dump(pool_records, f, ensure_ascii=False)
        print(f"[cache] saved {cache_path} and {records_cache}")

    # ── 5. Max cosine similarity ──
    print(f"[sim] computing max cosine sim ({pool_emb.shape[0]} × {seed_emb.shape[0]})")
    # Both are L2-normalized → dot product == cosine.
    # 1M × 1024 @ 100 × 1024 → 1M × 100 = 400 MB float32, fits in RAM.
    sim = pool_emb @ seed_emb.T
    max_sim = sim.max(axis=1)
    print(f"[sim] stats min={max_sim.min():.3f} max={max_sim.max():.3f} mean={max_sim.mean():.3f}")

    # ── 6. Top-K ──
    k = min(args.top_k, len(pool_records))
    print(f"[topk] selecting top {k}")
    top_idx = np.argpartition(-max_sim, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-max_sim[top_idx])]  # sort within top-k descending
    print(f"[topk] threshold (min selected sim) = {max_sim[top_idx[-1]]:.4f}")

    # ── 7. Write output ──
    out_dir = os.path.join(OUT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.json")
    selected = [to_alpaca(pool_records[i]["instruction"], pool_records[i]["output"])
                for i in top_idx]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(selected)} records -> {out_path}")


if __name__ == "__main__":
    main()
