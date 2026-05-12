"""Build math/code oracle subsets from OpenHermes-Labeled and Tulu-3-SFT.

Outputs:
  /data/fqzhou/dataset_agent/dataset/openhermes_math_oracle/data.json
  /data/fqzhou/dataset_agent/dataset/openhermes_code_oracle/data.json
  /data/fqzhou/dataset_agent/dataset/tulu_math_oracle/data.json
  /data/fqzhou/dataset_agent/dataset/tulu_code_oracle/data.json

Each sample is converted to the Alpaca schema expected by the ADCF
pipeline: {instruction, input, output, text} where `text` follows the
"### Response:\n" anchor convention used by base_model_filter.py.

Run with --skip-openhermes if OpenHermes outputs are already written.
"""
import argparse
import json
import os
import re

import ijson
import pyarrow.parquet as pq

OH_PATH = "/data/fqzhou/dataset_agent/dataset/openhermes_labeled/openhermes2_5.json"
TULU_DIR = "/data/fqzhou/.cache/modelscope/hub/datasets/allenai/tulu-3-sft-mixture/data"
OUT_ROOT = "/data/fqzhou/dataset_agent/dataset"

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n### Response:\n{output}"
)


def to_alpaca(instr: str, out: str, inp: str = "") -> dict:
    text = ALPACA_PROMPT.format(instruction=instr, input=inp, output=out)
    return {"instruction": instr, "input": inp, "output": out, "text": text}


def first_human_assistant(conversations):
    if conversations is None or len(conversations) == 0:
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
    # messages from parquet is a numpy array of dicts; bool/`or` are unsafe.
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


def _save(records, dirname):
    out_dir = os.path.join(OUT_ROOT, dirname)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  wrote {len(records)} -> {out_path}")


def _strip(records):
    return [{k: v for k, v in r.items() if not k.startswith("__")} for r in records]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--skip-openhermes", action="store_true")
ap.add_argument("--skip-tulu", action="store_true")
args = ap.parse_args()


# ---------------------------------------------------------------------------
# OpenHermes — strict-match: source contains 'math' (= metamath) for math;
# source contains 'code' OR category in {coding,code} for code.
# ---------------------------------------------------------------------------
if not args.skip_openhermes:
    print("=" * 70)
    print("OpenHermes — filtering math + code (strict source/category)")
    print("=" * 70)

    OH_MATH_SRC = re.compile(r"math|metamath|gsm|mathlib|mathinst", re.I)
    OH_CODE_SRC = re.compile(r"code|magicoder|codealpaca|leetcode|glaive-code", re.I)

    oh_math, oh_code = [], []
    with open(OH_PATH, "rb") as f:
        for item in ijson.items(f, "item"):
            s = str(item.get("source", "") or "")
            c = str(item.get("category", "") or "")
            pair = first_human_assistant(item.get("conversations", []))
            if pair is None:
                continue
            user, asst = pair
            rec = to_alpaca(user, asst)

            is_math = bool(OH_MATH_SRC.search(s)) or c.lower() == "math"
            is_code = bool(OH_CODE_SRC.search(s)) or c.lower() in {"coding", "code"}

            if is_math and not is_code:
                oh_math.append(rec)
            elif is_code and not is_math:
                oh_code.append(rec)

    print(f"  oh_math: {len(oh_math)}")
    print(f"  oh_code: {len(oh_code)}")

    print("\nWriting OpenHermes outputs...")
    _save(oh_math, "openhermes_math_oracle")
    _save(oh_code, "openhermes_code_oracle")
else:
    print("[skip] OpenHermes filter (per --skip-openhermes)")


# ---------------------------------------------------------------------------
# Tulu 3 — exact source-name match
# ---------------------------------------------------------------------------
if not args.skip_tulu:
    print("\n" + "=" * 70)
    print("Tulu-3 — filtering math + code (exact source match)")
    print("=" * 70)

    TULU_MATH_SOURCES = {
        "ai2-adapt-dev/personahub_math_v5_regen_149960",
        "ai2-adapt-dev/numinamath_tir_math_decontaminated",
        "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
        "allenai/tulu-3-sft-personas-math-grade",
        "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
    }
    TULU_CODE_SOURCES = {
        "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
        "ai2-adapt-dev/personahub_code_v2_34999",
    }

    tulu_math, tulu_code = [], []
    parquet_files = sorted(
        os.path.join(TULU_DIR, f) for f in os.listdir(TULU_DIR) if f.endswith(".parquet")
    )
    for path in parquet_files:
        table = pq.read_table(path, columns=["id", "messages", "source"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            s = str(row.get("source", "") or "")
            if s not in TULU_MATH_SOURCES and s not in TULU_CODE_SOURCES:
                continue
            pair = first_user_assistant_msgs(row.get("messages"))
            if pair is None:
                continue
            user, asst = pair
            rec = to_alpaca(user, asst)
            if s in TULU_MATH_SOURCES:
                tulu_math.append(rec)
            else:
                tulu_code.append(rec)
        print(f"  scanned {os.path.basename(path)}: math={len(tulu_math)} code={len(tulu_code)}")

    print(f"\n  tulu_math: {len(tulu_math)}")
    print(f"  tulu_code: {len(tulu_code)}")

    print("\nWriting Tulu outputs...")
    _save(tulu_math, "tulu_math_oracle")
    _save(tulu_code, "tulu_code_oracle")
else:
    print("[skip] Tulu filter (per --skip-tulu)")


print("\nDone.")
