"""Connectivity test for all API endpoints defined in config.yaml.

Usage:
    python test_connection.py            # test all endpoints
    python test_connection.py llm        # test agent LLM only
    python test_connection.py base       # test base model LLM only
    python test_connection.py embed      # test embedding model only
"""

import sys
import os
import time
import yaml
import requests


def load_config():
    config_path = os.path.join("/data/fqzhou/dataset_agent/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_llm(base_url, api_key, model, label="LLM"):
    """Test an OpenAI-compatible chat completion endpoint."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }

    print(f"\n[{label}] POST {url}")
    print(f"  model: {model}")
    try:
        start = time.time()
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  ✓ OK ({elapsed:.2f}s) — response: {content[:80]}")
            return True
        else:
            print(f"  ✗ FAIL status={resp.status_code} — {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ ERROR — {e}")
        return False


def test_embedding(base_url, api_key, model, label="Embedding"):
    """Test an OpenAI-compatible embedding endpoint."""
    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "input": "hello world",
    }

    print(f"\n[{label}] POST {url}")
    print(f"  model: {model}")
    try:
        start = time.time()
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            vec = data["data"][0]["embedding"]
            print(f"  ✓ OK ({elapsed:.2f}s) — dim={len(vec)}, first 3: {vec[:3]}")
            return True
        else:
            print(f"  ✗ FAIL status={resp.status_code} — {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ ERROR — {e}")
        return False


def main():
    cfg = load_config()
    provider = cfg.get("provider", "bd")
    provider_cfg = cfg.get(provider, {})

    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}

    # 1. Agent LLM
    if target in ("all", "llm"):
        results["Agent LLM"] = test_llm(
            base_url=provider_cfg.get("base_url") or provider_cfg.get("azure_endpoint", ""),
            api_key=provider_cfg.get("api_key", ""),
            model=provider_cfg.get("model", ""),
            label="Agent LLM",
        )

    # 2. Base Model LLM (local vLLM)
    if target in ("all", "base"):
        base_url = provider_cfg.get("base_model_url")
        if base_url:
            results["Base Model LLM"] = test_llm(
                base_url=base_url,
                api_key="",
                model=provider_cfg.get("base_api_model_name", ""),
                label="Base Model LLM",
            )
        else:
            print("\n[Base Model LLM] skipped — no base_model_url in config")

    # 3. Embedding Model
    if target in ("all", "embed"):
        embed_url = provider_cfg.get("embed_url")
        if embed_url:
            results["Embedding"] = test_embedding(
                base_url=embed_url,
                api_key="",
                model=provider_cfg.get("embed_api_model_name", ""),
                label="Embedding",
            )
        else:
            print("\n[Embedding] skipped — no embed_url in config")

    # Summary
    print("\n" + "=" * 40)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name}: {status}")
    print("=" * 40)


if __name__ == "__main__":
    main()
