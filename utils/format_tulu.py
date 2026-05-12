"""format_tulu.py — Convert Tulu 3 SFT mixture (parquet) to
OpenHermes-style chunked JSON files.

Source layout (read-only):
  /data/fqzhou/.cache/modelscope/hub/datasets/allenai/tulu-3-sft-mixture/data/*.parquet

Output layout:
  /data/fqzhou/dataset_agent/dataset/tulu/tulu_000.json
  /data/fqzhou/dataset_agent/dataset/tulu/tulu_001.json
  ...
  100,000 samples per file (last file may be shorter).

Per-sample schema (matches OpenHermes-2.5 conversations style;
source/id/category dropped):
  {
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt",   "value": "..."}
    ]
  }

Multi-turn dialogues are preserved. Role mapping:
  user      -> human
  assistant -> gpt
  system    -> system (kept as-is)
"""
import json
import os
import sys

import pyarrow.parquet as pq

TULU_DIR = "/data/fqzhou/.cache/modelscope/hub/datasets/allenai/tulu-3-sft-mixture/data"
OUT_DIR = "/data/fqzhou/dataset_agent/dataset/tulu"
CHUNK_SIZE = 100_000

ROLE_MAP = {"user": "human", "assistant": "gpt", "system": "system"}

os.makedirs(OUT_DIR, exist_ok=True)

parquet_files = sorted(
    os.path.join(TULU_DIR, f) for f in os.listdir(TULU_DIR) if f.endswith(".parquet")
)
if not parquet_files:
    print(f"ERROR: no parquet files in {TULU_DIR}", file=sys.stderr)
    sys.exit(1)
print(f"found {len(parquet_files)} parquet files")

buffer = []
chunk_idx = 0
total_in = 0
total_out = 0


def flush():
    global buffer, chunk_idx
    if not buffer:
        return
    out_path = os.path.join(OUT_DIR, f"tulu_{chunk_idx:03d}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(buffer, f, ensure_ascii=False, indent=2)
    sz = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  wrote {len(buffer):,} samples -> {out_path}  ({sz:.1f} MiB)")
    buffer.clear()
    chunk_idx += 1


for path in parquet_files:
    print(f"reading {os.path.basename(path)}...")
    table = pq.read_table(path, columns=["messages"])
    df = table.to_pandas()
    for _, row in df.iterrows():
        total_in += 1
        msgs = row.get("messages")
        if msgs is None:
            continue
        try:
            n = len(msgs)
        except TypeError:
            continue
        if n == 0:
            continue
        conv = []
        for m in msgs:
            role = m.get("role")
            if role is None:
                continue
            content = m.get("content", "")
            mapped = ROLE_MAP.get(role, role)
            conv.append({"from": mapped, "value": content})
        if len(conv) >= 2:  # require at least one human/gpt exchange
            buffer.append({"conversations": conv})
            total_out += 1
            if len(buffer) >= CHUNK_SIZE:
                flush()

flush()
print(f"\nDone. read {total_in:,} rows; wrote {total_out:,} samples to {chunk_idx} files in {OUT_DIR}")
