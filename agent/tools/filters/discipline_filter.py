"""Discipline Discovery Filter.

Uses pre-computed embeddings + KMeans clustering + LLM labeling to:
1. Discover discipline distribution in a multi-domain dataset
2. Filter/split data by target disciplines

Designed to run BEFORE rough_filter in the multi-discipline workflow,
reducing irrelevant data volume early so downstream filters are cheaper.
"""

import json
import os
import re
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from agent.utils import _load_json_or_jsonl, _paced_invoke, get_agent_tgt_dataset_path
from agent.model import subagent_llm as agent_llm
from agent.dispatch import global_dispatcher
from langchain_core.messages import SystemMessage, HumanMessage


# ── Embedding helpers ────────────────────────────────────────────

def _load_items_with_embeddings(dataset_dir: str) -> Tuple[List[Dict], np.ndarray]:
    """Load all JSON items and their embeddings.

    Supports two storage layouts:
      1. Inline: each JSON item has an 'embedding' field (list of floats).
      2. External: a FAISS .index file + .pkl file (list of ids) sit alongside
         the JSON files. Embeddings are reconstructed from the index and matched
         to JSON items by id.

    Returns:
        items: list of dicts (full records, without embedding field)
        embeddings: (N, D) float32 numpy array, aligned with items
    """
    import glob as _glob

    if os.path.isdir(dataset_dir):
        targets = sorted(_glob.glob(os.path.join(dataset_dir, "*.json")))
    else:
        targets = [dataset_dir]
        dataset_dir = os.path.dirname(dataset_dir)

    # ── Try inline embeddings first ──
    items: List[Dict] = []
    emb_list: List[List[float]] = []

    for tgt in targets:
        if not os.path.exists(tgt):
            continue
        data = _load_json_or_jsonl(tgt)
        for item in data:
            if "embedding" in item:
                items.append(item)
                emb_list.append(item["embedding"])

    if emb_list:
        print(f"[DisciplineDiscovery] Loaded {len(emb_list)} inline embeddings")
        return items, np.array(emb_list, dtype=np.float32)

    # ── Fallback: load from FAISS index + pkl ──
    index_files = _glob.glob(os.path.join(dataset_dir, "*.index"))
    pkl_files = _glob.glob(os.path.join(dataset_dir, "*.pkl"))

    if not index_files or not pkl_files:
        raise ValueError(
            f"No embeddings found in {dataset_dir}. "
            "Neither inline 'embedding' fields in JSON nor .index/.pkl files exist. "
            "Run preprocessing (add_embeddings) first."
        )

    import faiss
    import pickle

    index_path = index_files[0]
    pkl_path = pkl_files[0]

    print(f"[DisciplineDiscovery] Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)

    print(f"[DisciplineDiscovery] Loading id list from {pkl_path}")
    with open(pkl_path, "rb") as f:
        id_list = pickle.load(f)  # List[str], length = index.ntotal

    print(f"[DisciplineDiscovery] Index: {index.ntotal} vectors, dim={index.d}; pkl: {len(id_list)} ids")

    # Load all JSON items and build id → item mapping
    all_items: List[Dict] = []
    for tgt in targets:
        if not os.path.exists(tgt):
            continue
        data = _load_json_or_jsonl(tgt)
        all_items.extend(data)

    id_to_item = {}
    for item in all_items:
        item_id = item.get("id")
        if item_id is not None:
            id_to_item[str(item_id)] = item

    print(f"[DisciplineDiscovery] JSON items: {len(all_items)}, with id: {len(id_to_item)}")

    # Batch-reconstruct all vectors at once (much faster than per-item loop)
    all_embeddings = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for i in range(index.ntotal):
        index.reconstruct(i, all_embeddings[i])

    # Match embeddings to items by id
    matched_items: List[Dict] = []
    matched_indices: List[int] = []

    for idx, item_id in enumerate(id_list):
        item = id_to_item.get(str(item_id))
        if item is None:
            continue
        matched_items.append(item)
        matched_indices.append(idx)

    matched_embeddings = all_embeddings[matched_indices]

    if not matched_items:
        raise ValueError(
            f"Could not match any embeddings to JSON items. "
            f"pkl ids sample: {id_list[:3]}, JSON id sample: {list(id_to_item.keys())[:3]}"
        )

    print(f"[DisciplineDiscovery] Matched {len(matched_items)}/{len(id_list)} items with embeddings")
    return matched_items, np.array(matched_embeddings, dtype=np.float32)


# ── Clustering ───────────────────────────────────────────────────

def _cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 20) -> np.ndarray:
    """Over-cluster with KMeans. Returns cluster labels (N,)."""
    from sklearn.cluster import KMeans

    # Clamp n_clusters to data size
    n_clusters = min(n_clusters, len(embeddings))
    km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=300, random_state=42)
    labels = km.fit_predict(embeddings)
    return labels


def _try_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 50) -> Optional[np.ndarray]:
    """Try HDBSCAN (auto K, noise-aware). Returns None if < 2 clusters found."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    n_found = len(set(labels) - {-1})
    if n_found < 2:
        return None  # single cluster or all noise — not useful
    return labels


_HDBSCAN_MAX_SAMPLES = 50_000  # HDBSCAN is O(N²); skip above this threshold


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 20,
) -> np.ndarray:
    """Cluster embeddings. Prefers HDBSCAN for small datasets; uses KMeans for large ones.

    HDBSCAN is O(N²) in time/space and will OOM on large datasets (>50k in high dim).
    KMeans is O(N·K·D·I) and scales linearly with N.
    """
    n = len(embeddings)

    if n <= _HDBSCAN_MAX_SAMPLES:
        try:
            labels = _try_hdbscan(embeddings)
            if labels is not None:
                n_found = len(set(labels) - {-1})
                n_noise = int((labels == -1).sum())
                print(f"[DisciplineDiscovery] HDBSCAN found {n_found} clusters "
                      f"({n_noise} noise points)")
                return labels
            print("[DisciplineDiscovery] HDBSCAN returned < 2 clusters, falling back to KMeans")
        except Exception as e:
            print(f"[DisciplineDiscovery] HDBSCAN failed ({e}), falling back to KMeans")
    else:
        print(f"[DisciplineDiscovery] N={n} > {_HDBSCAN_MAX_SAMPLES}, skipping HDBSCAN (O(N²))")

    print(f"[DisciplineDiscovery] Using KMeans with K={n_clusters}")
    return _cluster_kmeans(embeddings, n_clusters)


# ── LLM labeling ────────────────────────────────────────────────

LABEL_SYSTEM_PROMPT = """You are a dataset discipline classifier. You will be shown several data samples from the same cluster.

Your task: determine the **primary academic discipline or subject area** these samples belong to.

Rules:
- Return a single short label in English (e.g. "cardiology", "organic chemistry", "linear algebra", "Python programming", "classical mechanics").
- Be specific but not overly narrow. Use the most natural discipline name.
- If samples span multiple related areas, use the broader discipline (e.g. "mathematics" instead of "calculus" if samples mix calculus and algebra).
- If samples are too diverse to fit one discipline, return "mixed".

Respond with ONLY the discipline label, nothing else."""


def _format_samples_for_labeling(samples: List[Dict], max_chars: int = 3000) -> str:
    """Format a few samples for the LLM to label."""
    parts = []
    total = 0
    for i, s in enumerate(samples):
        inst = str(s.get("instruction", ""))[:400]
        out = str(s.get("output", ""))[:200]
        block = f"--- Sample {i+1} ---\nInstruction: {inst}\nOutput (truncated): {out}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def label_clusters_with_llm(
    items: List[Dict],
    cluster_labels: np.ndarray,
    samples_per_cluster: int = 8,
) -> Dict[int, str]:
    """Label each cluster's discipline via LLM.

    Returns: {cluster_id: discipline_label}
    """
    unique_clusters = sorted(set(cluster_labels) - {-1})  # exclude noise
    cluster_disciplines: Dict[int, str] = {}
    last_ts = {"t": 0}

    global_dispatcher.emit_tool_call(
        name="discipline_labeling_start",
        args={"n_clusters": len(unique_clusters)},
        agent="discipline_discovery",
    )

    for idx, cid in enumerate(unique_clusters):
        # Gather items in this cluster
        members = [items[i] for i, l in enumerate(cluster_labels) if l == cid]
        sampled = random.sample(members, min(samples_per_cluster, len(members)))

        prompt_text = _format_samples_for_labeling(sampled)
        msgs = [
            SystemMessage(content=LABEL_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]

        resp = _paced_invoke(agent_llm, msgs, last_ts)
        if resp is None:
            label = "unknown"
        else:
            label = (getattr(resp, "content", "") or "unknown").strip().lower()
            # Clean up quotes or extra text
            label = label.strip('"\'').strip()
            if len(label) > 60:
                label = label[:60]

        cluster_disciplines[cid] = label

        global_dispatcher.emit_progress(
            name="discipline_labeling_progress",
            current=idx + 1,
            total=len(unique_clusters),
            agent="discipline_discovery",
            cluster=cid,
            label=label,
        )

    return cluster_disciplines


# ── Merge similar clusters ───────────────────────────────────────

MERGE_SYSTEM_PROMPT = """You are given a list of discipline labels discovered from dataset clusters.
Some labels may refer to the same or very similar discipline (e.g. "calculus" and "integral calculus", or "cardiology" and "heart disease").

Your task: merge labels that refer to the same discipline into a canonical name.

Output a JSON object mapping each original label to its canonical form:
{{"original_label_1": "canonical_name", "original_label_2": "canonical_name", ...}}

Rules:
- Keep labels that are genuinely different disciplines separate.
- Use the most common or natural discipline name as canonical.
- "mixed" or "unknown" labels should map to themselves.
- Return ONLY the JSON object."""


def merge_similar_labels(
    cluster_disciplines: Dict[int, str],
) -> Dict[int, str]:
    """Use LLM to merge cluster labels that refer to the same discipline."""
    unique_labels = list(set(cluster_disciplines.values()))

    if len(unique_labels) <= 2:
        return cluster_disciplines  # nothing to merge

    msgs = [
        SystemMessage(content=MERGE_SYSTEM_PROMPT),
        HumanMessage(content=f"Labels: {json.dumps(unique_labels)}"),
    ]

    last_ts = {"t": 0}
    resp = _paced_invoke(agent_llm, msgs, last_ts)
    if resp is None:
        return cluster_disciplines

    try:
        content = (getattr(resp, "content", "") or "").strip()
        # Extract JSON from response
        match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if match:
            mapping = json.loads(match.group())
        else:
            mapping = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return cluster_disciplines  # merge failed, keep originals

    # Apply mapping
    merged = {}
    for cid, label in cluster_disciplines.items():
        merged[cid] = mapping.get(label, label)

    return merged


# ── Filtering / splitting ────────────────────────────────────────

_MATCH_SYSTEM_PROMPT = """You are given a list of discipline labels discovered from a dataset, and a list of target disciplines the user wants to keep.

Determine which discovered labels belong to (or are sub-fields of) any target discipline.

For example, if the target is ["mathematics"]:
- "linear algebra" → YES (sub-field of mathematics)
- "calculus" → YES
- "python programming" → NO
- "statistics" → YES (mathematical discipline)
- "organic chemistry" → NO

Return a JSON object mapping each discovered label to true (keep) or false (discard):
{"linear algebra": true, "python programming": false, ...}

Return ONLY the JSON object."""


def _match_disciplines_with_llm(
    discovered_labels: List[str],
    target_disciplines: List[str],
) -> set:
    """Use LLM to determine which discovered labels match target disciplines.

    Returns: set of discovered labels that should be kept.
    """
    msgs = [
        SystemMessage(content=_MATCH_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "discovered_labels": discovered_labels,
            "target_disciplines": target_disciplines,
        })),
    ]

    last_ts = {"t": 0}
    resp = _paced_invoke(agent_llm, msgs, last_ts)
    if resp is None:
        # Fallback: substring matching
        print("[DisciplineDiscovery] LLM matching failed, falling back to substring match")
        return _match_disciplines_substring(discovered_labels, target_disciplines)

    try:
        content = (getattr(resp, "content", "") or "").strip()
        match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if match:
            mapping = json.loads(match.group())
        else:
            mapping = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        print("[DisciplineDiscovery] LLM match parse failed, falling back to substring match")
        return _match_disciplines_substring(discovered_labels, target_disciplines)

    kept = set()
    for label, should_keep in mapping.items():
        if should_keep is True or (isinstance(should_keep, str) and should_keep.lower() == "true"):
            kept.add(label.lower().strip())

    print(f"[DisciplineDiscovery] LLM matched {len(kept)}/{len(discovered_labels)} labels to targets")
    return kept


def _match_disciplines_substring(
    discovered_labels: List[str],
    target_disciplines: List[str],
) -> set:
    """Fallback: substring matching between discovered labels and targets."""
    target_lower = [t.lower().strip() for t in target_disciplines]
    kept = set()
    for label in discovered_labels:
        ll = label.lower().strip()
        for t in target_lower:
            if t in ll or ll in t:
                kept.add(ll)
                break
    return kept


def filter_by_disciplines(
    items: List[Dict],
    cluster_labels: np.ndarray,
    cluster_disciplines: Dict[int, str],
    target_disciplines: Optional[List[str]] = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Filter items to keep only those in target disciplines.

    If target_disciplines is None, keep ALL items but add 'discipline' field.
    Uses LLM to semantically match discovered labels against target disciplines.

    Returns:
        filtered_items: items that match target disciplines
        discipline_counts: {discipline: count}
    """
    discipline_counts: Dict[str, int] = defaultdict(int)

    # Build a per-item discipline label
    item_disciplines = []
    for i, item in enumerate(items):
        cid = cluster_labels[i]
        if cid == -1:
            disc = "unclassified"
        else:
            disc = cluster_disciplines.get(cid, "unknown")
        item_disciplines.append(disc)
        discipline_counts[disc] += 1

    if target_disciplines is None:
        # No filtering, just annotate
        for item, disc in zip(items, item_disciplines):
            item["_discipline"] = disc
        return items, dict(discipline_counts)

    # Use LLM to match discovered labels to target disciplines
    all_labels = sorted(set(cluster_disciplines.values()))
    kept_labels = _match_disciplines_with_llm(all_labels, target_disciplines)

    filtered = []
    for item, disc in zip(items, item_disciplines):
        if disc.lower().strip() in kept_labels:
            item["_discipline"] = disc
            filtered.append(item)

    return filtered, dict(discipline_counts)


# ── Main entry point ─────────────────────────────────────────────

def run_discipline_discovery(
    dataset_dir: str,
    target_disciplines: Optional[List[str]] = None,
    n_clusters: int = 20,
    samples_per_cluster: int = 8,
) -> str:
    """Discover disciplines in dataset and filter by targets.

    Args:
        dataset_dir: path to dataset directory (must have embeddings)
        target_disciplines: list of discipline names to keep; None = discover only
        n_clusters: number of clusters for KMeans fallback
        samples_per_cluster: samples to show LLM per cluster for labeling

    Returns:
        JSON string with result summary
    """
    print(f"[DisciplineDiscovery] Loading embeddings from {dataset_dir}")
    items, embeddings = _load_items_with_embeddings(dataset_dir)
    print(f"[DisciplineDiscovery] Loaded {len(items)} items with {embeddings.shape[1]}-dim embeddings")

    global_dispatcher.emit_tool_call(
        name="discipline_discovery_start",
        args={"n_items": len(items), "n_clusters_hint": n_clusters},
        agent="discipline_discovery",
    )

    # 1. Cluster
    cluster_labels = cluster_embeddings(embeddings, n_clusters=n_clusters)
    n_clusters_found = len(set(cluster_labels) - {-1})
    print(f"[DisciplineDiscovery] Found {n_clusters_found} clusters")

    # 2. Label clusters via LLM
    cluster_disciplines = label_clusters_with_llm(items, cluster_labels, samples_per_cluster)
    print(f"[DisciplineDiscovery] Raw labels: {cluster_disciplines}")

    # 3. Merge similar labels
    cluster_disciplines = merge_similar_labels(cluster_disciplines)
    final_disciplines = sorted(set(cluster_disciplines.values()))
    print(f"[DisciplineDiscovery] Merged disciplines: {final_disciplines}")

    # 4. Filter by target disciplines
    filtered_items, discipline_counts = filter_by_disciplines(
        items, cluster_labels, cluster_disciplines, target_disciplines
    )

    # 5. Save results (chunked, 10000 items per file)
    output_dir = get_agent_tgt_dataset_path(dataset_dir, "discipline")
    os.makedirs(output_dir, exist_ok=True)

    CHUNK_SIZE = 10_000

    if target_disciplines:
        # Clean items: remove embedding and internal fields
        out_items = []
        for item in filtered_items:
            clean = {k: v for k, v in item.items()
                     if k not in ("embedding", "_discipline", "_source_file")}
            out_items.append(clean)
    else:
        # Discovery-only: save per-discipline, each discipline chunked
        disc_items = defaultdict(list)
        for item in filtered_items:
            disc = item.pop("_discipline", "unknown")
            clean = {k: v for k, v in item.items() if k != "embedding"}
            disc_items[disc].append(clean)

        out_items = []
        for disc_name, disc_data in disc_items.items():
            safe_name = re.sub(r'[^\w\-]', '_', disc_name)
            for ci in range(0, len(disc_data), CHUNK_SIZE):
                chunk = disc_data[ci:ci + CHUNK_SIZE]
                chunk_idx = ci // CHUNK_SIZE
                out_path = os.path.join(output_dir, f"{safe_name}_{chunk_idx}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
            out_items.extend(disc_data)

    # Write chunked output for target-filtered mode
    if target_disciplines:
        for ci in range(0, len(out_items), CHUNK_SIZE):
            chunk = out_items[ci:ci + CHUNK_SIZE]
            chunk_idx = ci // CHUNK_SIZE
            out_path = os.path.join(output_dir, f"filtered_{chunk_idx}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)

    # 6. Print summary to terminal (no file output)
    summary = {
        "total_items": len(items),
        "n_clusters": n_clusters_found,
        "disciplines_found": final_disciplines,
        "discipline_counts": discipline_counts,
        "target_disciplines": target_disciplines,
        "filtered_count": len(filtered_items),
        "output_dir": output_dir,
    }

    print(f"\n{'='*60}")
    print(f"[DisciplineDiscovery] Summary")
    print(f"{'='*60}")
    print(f"  Total items:        {len(items)}")
    print(f"  Clusters found:     {n_clusters_found}")
    print(f"  Disciplines found:  {final_disciplines}")
    print(f"  Discipline counts:")
    for disc, cnt in sorted(discipline_counts.items(), key=lambda x: -x[1]):
        print(f"    {disc}: {cnt}")
    print(f"  Target disciplines: {target_disciplines}")
    print(f"  Filtered count:     {len(filtered_items)}")
    print(f"  Output dir:         {output_dir}")
    n_chunks = (len(out_items) + CHUNK_SIZE - 1) // CHUNK_SIZE if out_items else 0
    print(f"  Output files:       {n_chunks} chunks (max {CHUNK_SIZE} each)")
    print(f"{'='*60}\n")

    global_dispatcher.emit_tool_call(
        name="discipline_discovery_done",
        args={
            "disciplines": final_disciplines,
            "counts": discipline_counts,
            "filtered": len(filtered_items),
            "output_dir": output_dir,
        },
        agent="discipline_discovery",
    )

    return json.dumps(summary, ensure_ascii=False, indent=2)
