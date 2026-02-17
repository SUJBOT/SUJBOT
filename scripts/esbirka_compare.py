"""
e-Sbírka Benchmark Comparison — GT vs Pipeline extraction metrics.

Compares ground-truth entities/relationships from structured e-Sbírka data
against the VL+KG pipeline extraction results. Reports P/R/F1 per entity and
relationship type, plus cross-document dedup evaluation.

Three-phase matching strategy (language-agnostic):
  Phase 1: Exact normalized string match (fast, free)
  Phase 2: Semantic embedding similarity via multilingual-e5-small (language-agnostic)
  Phase 3: LLM judge for borderline candidates (optional, --llm-judge flag)

Usage:
    uv run python scripts/esbirka_compare.py [--threshold 0.7] [--llm-judge]
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEP = "=" * 70
BENCHMARK_DIR = Path("data/esbirka_benchmark")

# Types that only the pipeline produces (GT has no instances)
# Computed dynamically — no hardcoded lists


# =========================================================================
# Name normalization (structural, language-agnostic)
# =========================================================================


def normalize_for_matching(name: str) -> str:
    """Normalize entity name for exact matching (structural only)."""
    s = name.lower().strip()
    # Normalize dashes
    s = re.sub(r"[–—]", "-", s)
    # Remove quotes
    s = re.sub(r'["\'""\u201e\u201c\u201d]', "", s)
    # Remove parens
    s = re.sub(r"[()]", "", s)
    # Normalize spaces around dashes/slashes
    s = re.sub(r"\s*[-/]\s*", "-", s)
    # Comma → space
    s = re.sub(r",\s*", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_citation_number(name: str) -> Optional[str]:
    """Extract numeric citation (e.g., '263/2016') from any entity name."""
    m = re.search(r"(\d+/\d{4})", name)
    return m.group(1) if m else None


def trigram_similarity(a: str, b: str) -> float:
    """Compute trigram (Jaccard) similarity."""
    if not a or not b:
        return 0.0

    def trigrams(s: str) -> Set[str]:
        s = f"  {s.lower()} "
        return {s[i : i + 3] for i in range(len(s) - 2)}

    ta = trigrams(a)
    tb = trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# =========================================================================
# Semantic embedding (language-agnostic via multilingual-e5-small)
# =========================================================================


def build_embedding_index(
    entities: List[Dict],
) -> Tuple[np.ndarray, List[str]]:
    """Embed entity names+descriptions and return matrix + text list."""
    from src.graph.embedder import GraphEmbedder

    embedder = GraphEmbedder()
    texts = []
    for e in entities:
        desc = e.get("description", "")
        text = f"{e['name']}: {desc}" if desc else e["name"]
        texts.append(text)

    if not texts:
        return np.array([]), texts

    embeddings = embedder.encode_passages(texts)
    return embeddings, texts


def semantic_similarity_matrix(
    gt_embeddings: np.ndarray,
    pipeline_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity matrix between GT and pipeline embeddings."""
    if gt_embeddings.size == 0 or pipeline_embeddings.size == 0:
        return np.array([])
    # Both are already L2-normalized by encode_passages
    return gt_embeddings @ pipeline_embeddings.T


# =========================================================================
# LLM Judge (optional, for borderline cases)
# =========================================================================

LLM_JUDGE_PROMPT = """You are comparing two entity extractions from the same document.
Determine if they refer to the SAME real-world entity/concept, despite possible naming differences.

Ground truth entity:
  Name: {gt_name}
  Type: {gt_type}
  Description: {gt_desc}

Pipeline entity:
  Name: {pipe_name}
  Type: {pipe_type}
  Description: {pipe_desc}

Answer ONLY "MATCH" or "NO_MATCH". A match means they refer to the same entity
even if named differently (e.g., abbreviation vs full name, different grammatical form,
with or without a qualifying citation suffix)."""


def llm_judge_match(
    gt_ent: Dict,
    pipe_ent: Dict,
    provider: Any,
) -> bool:
    """Use LLM to judge whether two entities match."""
    prompt = LLM_JUDGE_PROMPT.format(
        gt_name=gt_ent["name"],
        gt_type=gt_ent["type"],
        gt_desc=gt_ent.get("description", ""),
        pipe_name=pipe_ent["name"],
        pipe_type=pipe_ent["type"],
        pipe_desc=pipe_ent.get("description", ""),
    )
    try:
        response = provider.create_message(
            system="You are a precise entity matching judge.",
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            max_tokens=10,
            temperature=0.0,
        )
        # ProviderResponse.content is List[Dict] (Anthropic content blocks)
        answer = response.content[0].get("text", "").strip().upper()
        return "MATCH" in answer
    except Exception as e:
        logger.warning(f"LLM judge error: {e}")
        return False


# =========================================================================
# Three-phase Entity Matching
# =========================================================================


def match_entities(
    gt_entities: List[Dict],
    pipeline_entities: List[Dict],
    threshold: float = 0.7,
    semantic_threshold: float = 0.75,
    cross_type_threshold: float = 0.85,
    llm_provider: Any = None,
    llm_judge_range: Tuple[float, float] = (0.55, 0.75),
) -> Dict[str, Any]:
    """
    Match GT entities against pipeline entities using four phases:
    1. Exact normalized string match (same type)
    2. Semantic embedding similarity (same type, above semantic_threshold)
    3. Cross-type semantic matching (different types, above cross_type_threshold or LLM judge)
    4. LLM judge for all borderline semantic matches (optional)
    """
    matched = []
    gt_matched_indices: Set[int] = set()
    pipeline_matched_indices: Set[int] = set()

    n_gt = len(gt_entities)
    n_pipe = len(pipeline_entities)

    # Pre-compute normalized names
    gt_normalized = [normalize_for_matching(e["name"]) for e in gt_entities]
    pipe_normalized = [normalize_for_matching(e["name"]) for e in pipeline_entities]

    # Also extract citation numbers for entities with numeric IDs (X/YYYY)
    gt_citations = [extract_citation_number(e["name"]) for e in gt_entities]
    pipe_citations = [extract_citation_number(e["name"]) for e in pipeline_entities]

    # --- Phase 1: Exact normalized match (same type) ---
    logger.info("  Phase 1: Exact normalized matching...")
    pipe_by_type_norm: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for pi in range(n_pipe):
        pipe_by_type_norm[pipeline_entities[pi]["type"]].append((pi, pipe_normalized[pi]))

    for gi in range(n_gt):
        if gi in gt_matched_indices:
            continue
        gt_type = gt_entities[gi]["type"]
        gt_norm = gt_normalized[gi]

        for pi, p_norm in pipe_by_type_norm.get(gt_type, []):
            if pi in pipeline_matched_indices:
                continue
            if gt_norm == p_norm:
                matched.append({
                    "gt": gt_entities[gi],
                    "pipeline": pipeline_entities[pi],
                    "match_type": "exact",
                    "similarity": 1.0,
                })
                gt_matched_indices.add(gi)
                pipeline_matched_indices.add(pi)
                break

    # Phase 1b: Citation number match (structural, format-based)
    logger.info("  Phase 1b: Citation number matching...")
    for gi in range(n_gt):
        if gi in gt_matched_indices:
            continue
        gt_cit = gt_citations[gi]
        if not gt_cit:
            continue
        gt_type = gt_entities[gi]["type"]

        for pi, p_norm in pipe_by_type_norm.get(gt_type, []):
            if pi in pipeline_matched_indices:
                continue
            p_cit = pipe_citations[pi]
            if p_cit and gt_cit == p_cit:
                matched.append({
                    "gt": gt_entities[gi],
                    "pipeline": pipeline_entities[pi],
                    "match_type": "citation",
                    "similarity": 0.95,
                })
                gt_matched_indices.add(gi)
                pipeline_matched_indices.add(pi)
                break

    phase1_count = len(matched)
    logger.info(f"    Phase 1 matches: {phase1_count}")

    # --- Phase 2+3: Semantic embedding match ---
    gt_unmatched = [i for i in range(n_gt) if i not in gt_matched_indices]
    pipe_unmatched = [i for i in range(n_pipe) if i not in pipeline_matched_indices]

    if gt_unmatched and pipe_unmatched:
        logger.info(
            f"  Phase 2: Semantic matching ({len(gt_unmatched)} GT × {len(pipe_unmatched)} pipeline)..."
        )
        t0 = time.time()

        gt_sub = [gt_entities[i] for i in gt_unmatched]
        pipe_sub = [pipeline_entities[i] for i in pipe_unmatched]

        gt_emb, _ = build_embedding_index(gt_sub)
        pipe_emb, _ = build_embedding_index(pipe_sub)

        if gt_emb.size > 0 and pipe_emb.size > 0:
            sim_matrix = semantic_similarity_matrix(gt_emb, pipe_emb)

            # Collect ALL candidate pairs (same-type and cross-type)
            same_type_candidates = []
            cross_type_candidates = []

            for li, gi in enumerate(gt_unmatched):
                gt_type = gt_entities[gi]["type"]
                for lj, pi in enumerate(pipe_unmatched):
                    sim = float(sim_matrix[li, lj])
                    if sim < llm_judge_range[0]:
                        continue
                    if pipeline_entities[pi]["type"] == gt_type:
                        same_type_candidates.append((sim, gi, pi))
                    else:
                        cross_type_candidates.append((sim, gi, pi))

            # Phase 2a: Same-type semantic (lower threshold)
            same_type_candidates.sort(reverse=True, key=lambda x: x[0])
            llm_judge_candidates = []

            for sim, gi, pi in same_type_candidates:
                if gi in gt_matched_indices or pi in pipeline_matched_indices:
                    continue
                if sim >= semantic_threshold:
                    matched.append({
                        "gt": gt_entities[gi],
                        "pipeline": pipeline_entities[pi],
                        "match_type": "semantic",
                        "similarity": sim,
                    })
                    gt_matched_indices.add(gi)
                    pipeline_matched_indices.add(pi)
                elif llm_provider and sim >= llm_judge_range[0]:
                    llm_judge_candidates.append((sim, gi, pi))

            elapsed = time.time() - t0
            phase2_count = len(matched) - phase1_count
            logger.info(f"    Phase 2a (same-type) matches: {phase2_count} ({elapsed:.1f}s)")

            # Phase 2b: Cross-type semantic (higher threshold)
            cross_type_candidates.sort(reverse=True, key=lambda x: x[0])
            phase2b_start = len(matched)

            for sim, gi, pi in cross_type_candidates:
                if gi in gt_matched_indices or pi in pipeline_matched_indices:
                    continue
                if sim >= cross_type_threshold:
                    matched.append({
                        "gt": gt_entities[gi],
                        "pipeline": pipeline_entities[pi],
                        "match_type": "semantic_cross_type",
                        "similarity": sim,
                    })
                    gt_matched_indices.add(gi)
                    pipeline_matched_indices.add(pi)
                elif llm_provider and sim >= llm_judge_range[0]:
                    llm_judge_candidates.append((sim, gi, pi))

            phase2b_count = len(matched) - phase2b_start
            logger.info(f"    Phase 2b (cross-type) matches: {phase2b_count}")

            # --- Phase 3: LLM judge for all borderline cases ---
            if llm_provider and llm_judge_candidates:
                logger.info(
                    f"  Phase 3: LLM judge for {len(llm_judge_candidates)} borderline cases..."
                )
                t0 = time.time()
                llm_matches = 0

                for sim, gi, pi in sorted(llm_judge_candidates, reverse=True, key=lambda x: x[0]):
                    if gi in gt_matched_indices or pi in pipeline_matched_indices:
                        continue

                    is_match = llm_judge_match(
                        gt_entities[gi], pipeline_entities[pi], llm_provider
                    )
                    if is_match:
                        same = gt_entities[gi]["type"] == pipeline_entities[pi]["type"]
                        matched.append({
                            "gt": gt_entities[gi],
                            "pipeline": pipeline_entities[pi],
                            "match_type": "llm_judge" if same else "llm_judge_cross_type",
                            "similarity": sim,
                        })
                        gt_matched_indices.add(gi)
                        pipeline_matched_indices.add(pi)
                        llm_matches += 1

                elapsed = time.time() - t0
                logger.info(f"    Phase 3 matches: {llm_matches} ({elapsed:.1f}s)")

    # Unmatched
    gt_only = [gt_entities[i] for i in range(n_gt) if i not in gt_matched_indices]
    pipeline_only = [pipeline_entities[i] for i in range(n_pipe) if i not in pipeline_matched_indices]

    return {
        "matched": matched,
        "gt_only": gt_only,
        "pipeline_only": pipeline_only,
    }


# =========================================================================
# Relationship Matching (uses entity match map)
# =========================================================================


LLM_REL_JUDGE_PROMPT = """You are comparing two relationship extractions from the same document.
Determine if they represent the SAME relationship between the SAME entities.

Ground truth relationship:
  Source: {gt_src}
  Type: {gt_type}
  Target: {gt_tgt}
  Description: {gt_desc}

Pipeline relationship:
  Source: {pipe_src}
  Type: {pipe_type}
  Target: {pipe_tgt}
  Description: {pipe_desc}

Answer ONLY "MATCH" or "NO_MATCH". A match means:
- The source entities refer to the same real-world entity (possibly named differently)
- The target entities refer to the same real-world entity
- The relationship type captures the same semantic connection (even if labeled differently)"""


def _rel_to_text(rel: Dict) -> str:
    """Convert a relationship to text for embedding."""
    desc = rel.get("description", "")
    return f"{rel['source']} --[{rel['type']}]--> {rel['target']}: {desc}" if desc else \
        f"{rel['source']} --[{rel['type']}]--> {rel['target']}"


def match_relationships(
    gt_rels: List[Dict],
    pipeline_rels: List[Dict],
    entity_match_map: Dict[str, str],
    threshold: float = 0.7,
    semantic_threshold: float = 0.75,
    llm_provider: Any = None,
) -> Dict[str, Any]:
    """
    Match GT relationships against pipeline relationships.

    Three-phase matching:
    1. Exact: entity match map resolution + normalized string comparison (same type)
    2. Semantic: embed relationship triples, match by cosine similarity (any type)
    3. LLM judge: for borderline semantic matches (optional)
    """
    matched = []
    gt_matched: Set[int] = set()
    pipeline_matched: Set[int] = set()

    # --- Phase 1: Exact endpoint match (same relationship type) ---
    logger.info("  Rel Phase 1: Exact endpoint matching...")
    pipe_rels_normalized = []
    for p_rel in pipeline_rels:
        p_src = normalize_for_matching(p_rel["source"])
        p_tgt = normalize_for_matching(p_rel["target"])
        pipe_rels_normalized.append((p_src, p_tgt))

    for gi, gt_rel in enumerate(gt_rels):
        if gi in gt_matched:
            continue

        gt_src_resolved = entity_match_map.get(gt_rel["source"], gt_rel["source"])
        gt_tgt_resolved = entity_match_map.get(gt_rel["target"], gt_rel["target"])
        gt_src = normalize_for_matching(gt_src_resolved)
        gt_tgt = normalize_for_matching(gt_tgt_resolved)

        for pi, (p_src, p_tgt) in enumerate(pipe_rels_normalized):
            if pi in pipeline_matched:
                continue
            if gt_rel["type"] != pipeline_rels[pi]["type"]:
                continue

            src_match = gt_src == p_src or trigram_similarity(gt_src, p_src) >= threshold
            if not src_match:
                continue
            tgt_match = gt_tgt == p_tgt or trigram_similarity(gt_tgt, p_tgt) >= threshold
            if not tgt_match:
                continue

            matched.append({
                "gt": gt_rel,
                "pipeline": pipeline_rels[pi],
                "match_type": "exact",
            })
            gt_matched.add(gi)
            pipeline_matched.add(pi)
            break

    phase1_count = len(matched)
    logger.info(f"    Rel Phase 1 matches: {phase1_count}")

    # --- Phase 2: Semantic relationship matching ---
    gt_unmatched = [i for i in range(len(gt_rels)) if i not in gt_matched]
    pipe_unmatched = [i for i in range(len(pipeline_rels)) if i not in pipeline_matched]

    if gt_unmatched and pipe_unmatched:
        logger.info(
            f"  Rel Phase 2: Semantic matching ({len(gt_unmatched)} GT × {len(pipe_unmatched)} pipeline)..."
        )
        t0 = time.time()

        gt_texts = [_rel_to_text(gt_rels[i]) for i in gt_unmatched]
        pipe_texts = [_rel_to_text(pipeline_rels[i]) for i in pipe_unmatched]

        from src.graph.embedder import GraphEmbedder
        embedder = GraphEmbedder()
        gt_emb = embedder.encode_passages(gt_texts)
        pipe_emb = embedder.encode_passages(pipe_texts)

        if gt_emb.size > 0 and pipe_emb.size > 0:
            sim_matrix = gt_emb @ pipe_emb.T

            # Collect candidates (same-type gets lower threshold, cross-type gets higher)
            candidates = []
            for li, gi in enumerate(gt_unmatched):
                for lj, pi in enumerate(pipe_unmatched):
                    sim = float(sim_matrix[li, lj])
                    same_type = gt_rels[gi]["type"] == pipeline_rels[pi]["type"]
                    min_sim = 0.55 if same_type else 0.70
                    if sim >= min_sim:
                        candidates.append((sim, gi, pi, same_type))

            candidates.sort(reverse=True, key=lambda x: x[0])

            llm_candidates = []
            for sim, gi, pi, same_type in candidates:
                if gi in gt_matched or pi in pipeline_matched:
                    continue
                accept_threshold = semantic_threshold if same_type else 0.85
                if sim >= accept_threshold:
                    matched.append({
                        "gt": gt_rels[gi],
                        "pipeline": pipeline_rels[pi],
                        "match_type": "semantic" if same_type else "semantic_cross_type",
                    })
                    gt_matched.add(gi)
                    pipeline_matched.add(pi)
                elif llm_provider:
                    llm_candidates.append((sim, gi, pi, same_type))

            elapsed = time.time() - t0
            phase2_count = len(matched) - phase1_count
            logger.info(f"    Rel Phase 2 matches: {phase2_count} ({elapsed:.1f}s)")

            # --- Phase 3: LLM judge ---
            if llm_provider and llm_candidates:
                logger.info(f"  Rel Phase 3: LLM judge for {len(llm_candidates)} candidates...")
                t0 = time.time()
                llm_matches = 0

                for sim, gi, pi, same_type in llm_candidates[:100]:  # Cap at 100 LLM calls
                    if gi in gt_matched or pi in pipeline_matched:
                        continue
                    try:
                        prompt = LLM_REL_JUDGE_PROMPT.format(
                            gt_src=gt_rels[gi]["source"],
                            gt_type=gt_rels[gi]["type"],
                            gt_tgt=gt_rels[gi]["target"],
                            gt_desc=gt_rels[gi].get("description", ""),
                            pipe_src=pipeline_rels[pi]["source"],
                            pipe_type=pipeline_rels[pi]["type"],
                            pipe_tgt=pipeline_rels[pi]["target"],
                            pipe_desc=pipeline_rels[pi].get("description", ""),
                        )
                        response = llm_provider.create_message(
                            system="You are a precise relationship matching judge.",
                            messages=[{"role": "user", "content": prompt}],
                            tools=[],
                            max_tokens=10,
                            temperature=0.0,
                        )
                        answer = response.content[0].get("text", "").strip().upper()
                        if "MATCH" in answer:
                            matched.append({
                                "gt": gt_rels[gi],
                                "pipeline": pipeline_rels[pi],
                                "match_type": "llm_judge" if same_type else "llm_judge_cross_type",
                            })
                            gt_matched.add(gi)
                            pipeline_matched.add(pi)
                            llm_matches += 1
                    except Exception as e:
                        logger.warning(f"LLM rel judge error: {e}")

                elapsed = time.time() - t0
                logger.info(f"    Rel Phase 3 matches: {llm_matches} ({elapsed:.1f}s)")

    gt_only = [gt_rels[i] for i in range(len(gt_rels)) if i not in gt_matched]
    pipeline_only = [pipeline_rels[i] for i in range(len(pipeline_rels)) if i not in pipeline_matched]

    return {
        "matched": matched,
        "gt_only": gt_only,
        "pipeline_only": pipeline_only,
    }


def build_entity_match_map(match_result: Dict) -> Dict[str, str]:
    """Build a mapping from GT entity name → pipeline entity name from match results."""
    mapping = {}
    for m in match_result["matched"]:
        gt_name = m["gt"]["name"]
        pipe_name = m["pipeline"]["name"]
        mapping[gt_name] = pipe_name
    return mapping


# =========================================================================
# Metrics
# =========================================================================


def compute_prf(matched: int, gt_total: int, pipeline_total: int) -> Dict[str, float]:
    """Compute precision, recall, F1."""
    precision = matched / pipeline_total if pipeline_total > 0 else 0.0
    recall = matched / gt_total if gt_total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": matched,
        "gt_total": gt_total,
        "pipeline_total": pipeline_total,
    }


def compute_per_type_metrics(
    match_result: Dict,
    type_key: str = "type",
) -> Dict[str, Dict[str, float]]:
    """Compute P/R/F1 per entity or relationship type.

    For cross-type matches, recall is counted by GT type (how many GT entities
    of this type were found?) and precision by pipeline type (how many pipeline
    entities of this type were correct?).
    """
    # For recall: count matches by GT type
    matched_by_gt_type: Dict[str, int] = defaultdict(int)
    for m in match_result["matched"]:
        matched_by_gt_type[m["gt"][type_key]] += 1

    # For precision: count matches by pipeline type
    matched_by_pipe_type: Dict[str, int] = defaultdict(int)
    for m in match_result["matched"]:
        matched_by_pipe_type[m["pipeline"][type_key]] += 1

    # GT totals (by GT type)
    gt_by_type: Dict[str, int] = defaultdict(int)
    for m in match_result["matched"]:
        gt_by_type[m["gt"][type_key]] += 1
    for e in match_result["gt_only"]:
        gt_by_type[e[type_key]] += 1

    # Pipeline totals (by pipeline type)
    pipeline_by_type: Dict[str, int] = defaultdict(int)
    for m in match_result["matched"]:
        pipeline_by_type[m["pipeline"][type_key]] += 1
    for e in match_result["pipeline_only"]:
        pipeline_by_type[e[type_key]] += 1

    all_types = set(gt_by_type.keys()) | set(pipeline_by_type.keys())

    metrics = {}
    for t in sorted(all_types):
        gt_total = gt_by_type.get(t, 0)
        pipe_total = pipeline_by_type.get(t, 0)
        recall_matched = matched_by_gt_type.get(t, 0)
        precision_matched = matched_by_pipe_type.get(t, 0)

        precision = precision_matched / pipe_total if pipe_total > 0 else 0.0
        recall = recall_matched / gt_total if gt_total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[t] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "matched": recall_matched,  # Display GT-side matches (recall-oriented)
            "gt_total": gt_total,
            "pipeline_total": pipe_total,
        }

    return metrics


# =========================================================================
# Dedup Evaluation
# =========================================================================


def evaluate_dedup(
    gt_combined: Dict,
    pipeline_combined: Dict,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """Evaluate cross-document deduplication quality."""
    expected_groups = gt_combined.get("expected_dedup_groups", [])
    pipeline_entities = pipeline_combined.get("cross_doc_merged_entities", [])

    results = {
        "expected_groups": len(expected_groups),
        "correctly_merged": 0,
        "incorrectly_split": 0,
        "details": [],
    }

    if not expected_groups:
        return results

    # Build pipeline entity lookup by normalized name
    pipeline_names = {normalize_for_matching(e["name"]): e for e in pipeline_entities}

    for group in expected_groups:
        gt_name = group["name"]
        gt_type = group["type"]
        gt_docs = group["documents"]

        gt_norm = normalize_for_matching(gt_name)

        found = gt_norm in pipeline_names
        if not found:
            for p_norm, p_ent in pipeline_names.items():
                if p_ent.get("type") == gt_type and trigram_similarity(gt_norm, p_norm) >= threshold:
                    found = True
                    break

        if found:
            results["correctly_merged"] += 1
            status = "MERGED"
        else:
            results["incorrectly_split"] += 1
            status = "SPLIT"

        results["details"].append({
            "name": gt_name,
            "type": gt_type,
            "documents": gt_docs,
            "status": status,
        })

    results["merge_rate"] = round(
        results["correctly_merged"] / len(expected_groups) if expected_groups else 0.0,
        4,
    )

    return results


# =========================================================================
# Report Generation
# =========================================================================


def print_entity_metrics(
    metrics: Dict[str, Dict],
    title: str = "Entity Metrics",
    gt_type_set: Optional[Set[str]] = None,
) -> None:
    """Print entity metrics table, optionally marking pipeline-only types."""
    print(f"\n{title}")
    print("-" * 88)
    print(
        f"{'Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'Match':>6} {'GT':>6} {'Pipe':>6}  Note"
    )
    print("-" * 88)

    total_matched = 0
    total_gt = 0
    total_pipeline = 0
    shared_matched = 0
    shared_gt = 0
    shared_pipeline = 0

    for etype, m in sorted(metrics.items()):
        is_shared = gt_type_set is None or etype in gt_type_set
        note = "" if is_shared else "(pipe-only)"
        print(
            f"{etype:<20} {m['precision']:>10.2%} {m['recall']:>10.2%} {m['f1']:>10.2%} "
            f"{m['matched']:>6} {m['gt_total']:>6} {m['pipeline_total']:>6}  {note}"
        )
        total_matched += m["matched"]
        total_gt += m["gt_total"]
        total_pipeline += m["pipeline_total"]
        if is_shared:
            shared_matched += m["matched"]
            shared_gt += m["gt_total"]
            shared_pipeline += m["pipeline_total"]

    print("-" * 88)
    overall = compute_prf(total_matched, total_gt, total_pipeline)
    print(
        f"{'OVERALL':<20} {overall['precision']:>10.2%} {overall['recall']:>10.2%} "
        f"{overall['f1']:>10.2%} {total_matched:>6} {total_gt:>6} {total_pipeline:>6}"
    )
    if gt_type_set and shared_gt != total_gt:
        shared = compute_prf(shared_matched, shared_gt, shared_pipeline)
        print(
            f"{'SHARED TYPES ONLY':<20} {shared['precision']:>10.2%} {shared['recall']:>10.2%} "
            f"{shared['f1']:>10.2%} {shared_matched:>6} {shared_gt:>6} {shared_pipeline:>6}"
        )


def print_match_type_breakdown(matched: List[Dict]) -> None:
    """Print breakdown of how entities were matched."""
    by_type = defaultdict(int)
    for m in matched:
        by_type[m["match_type"]] += 1
    if not by_type:
        return
    print("\n  Match method breakdown:")
    for mt, count in sorted(by_type.items()):
        print(f"    {mt}: {count}")


def print_missed_entities(gt_only: List[Dict], limit: int = 20) -> None:
    """Print entities in GT but not found by pipeline."""
    if not gt_only:
        return
    print(f"\nMissed by pipeline ({len(gt_only)} entities):")
    by_type = defaultdict(int)
    for ent in gt_only:
        by_type[ent["type"]] += 1
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    print(f"\n  Sample missed:")
    for ent in gt_only[:limit]:
        print(f"    [{ent['type']:<15}] {ent['name'][:70]}")
    if len(gt_only) > limit:
        print(f"    ... and {len(gt_only) - limit} more")


def print_hallucinated_entities(pipeline_only: List[Dict], limit: int = 15) -> None:
    """Print entities found by pipeline but not in GT."""
    if not pipeline_only:
        return
    print(f"\nExtra from pipeline ({len(pipeline_only)} entities):")
    by_type = defaultdict(int)
    for ent in pipeline_only:
        by_type[ent["type"]] += 1
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    print(f"\n  Sample extra:")
    for ent in pipeline_only[:limit]:
        desc = ent.get("description", "")[:50]
        print(f"    [{ent['type']:<15}] {ent['name'][:60]}  {desc}")
    if len(pipeline_only) > limit:
        print(f"    ... and {len(pipeline_only) - limit} more")


def print_dedup_report(dedup_result: Dict) -> None:
    """Print deduplication evaluation report."""
    print(f"\nCross-Document Dedup Evaluation:")
    print(f"  Expected groups:    {dedup_result['expected_groups']}")
    print(f"  Correctly merged:   {dedup_result['correctly_merged']}")
    print(f"  Incorrectly split:  {dedup_result['incorrectly_split']}")
    print(f"  Merge rate:         {dedup_result.get('merge_rate', 0):.1%}")

    for detail in dedup_result.get("details", []):
        status_icon = "+" if detail["status"] == "MERGED" else "-"
        print(
            f"  [{status_icon}] {detail['name'][:40]:<40} [{detail['type']:<12}] "
            f"docs={detail['documents']}"
        )


# =========================================================================
# Main comparison
# =========================================================================


def compare_single_document(
    gt_path: Path,
    pipeline_path: Path,
    threshold: float,
    semantic_threshold: float,
    cross_type_threshold: float = 0.85,
    llm_provider: Any = None,
) -> Dict[str, Any]:
    """Compare GT vs pipeline for a single document."""
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))

    doc_id = gt["document_id"]
    print(f"\n{'=' * 80}")
    print(f"Document: {doc_id}")
    print(f"  GT: {len(gt['entities'])} entities, {len(gt['relationships'])} relationships")

    pipeline_entities = pipeline.get("merged_entities", [])
    pipeline_rels = pipeline.get("merged_relationships", [])
    print(f"  Pipeline: {len(pipeline_entities)} entities, {len(pipeline_rels)} relationships")
    print(f"  Pages processed: {pipeline.get('pages_processed', '?')}")

    # Determine which entity types GT has
    gt_types = {e["type"] for e in gt["entities"]}

    # Entity comparison
    entity_result = match_entities(
        gt["entities"],
        pipeline_entities,
        threshold=threshold,
        semantic_threshold=semantic_threshold,
        cross_type_threshold=cross_type_threshold,
        llm_provider=llm_provider,
    )
    entity_metrics = compute_per_type_metrics(entity_result)
    print_entity_metrics(entity_metrics, f"Entity Metrics — {doc_id}", gt_types)
    print_match_type_breakdown(entity_result["matched"])
    print_missed_entities(entity_result["gt_only"])
    print_hallucinated_entities(entity_result["pipeline_only"])

    # Build entity match map for relationship matching
    entity_match_map = build_entity_match_map(entity_result)

    # Relationship comparison
    rel_result = match_relationships(
        gt["relationships"], pipeline_rels, entity_match_map,
        threshold=threshold,
        semantic_threshold=semantic_threshold,
        llm_provider=llm_provider,
    )
    rel_metrics = compute_per_type_metrics(rel_result)
    gt_rel_types = {r["type"] for r in gt["relationships"]}
    print_entity_metrics(rel_metrics, f"Relationship Metrics — {doc_id}", gt_rel_types)
    if rel_result["matched"]:
        print_match_type_breakdown(rel_result["matched"])

    if rel_result["gt_only"]:
        print(f"\nMissed relationships ({len(rel_result['gt_only'])}):")
        for rel in rel_result["gt_only"][:10]:
            print(f"  {rel['source'][:35]} --[{rel['type']}]--> {rel['target'][:35]}")

    return {
        "document_id": doc_id,
        "entity_metrics": entity_metrics,
        "entity_overall": compute_prf(
            len(entity_result["matched"]),
            len(gt["entities"]),
            len(pipeline_entities),
        ),
        "relationship_metrics": rel_metrics,
        "relationship_overall": compute_prf(
            len(rel_result["matched"]),
            len(gt["relationships"]),
            len(pipeline_rels),
        ),
        "match_type_breakdown": {
            m["match_type"]: sum(1 for x in entity_result["matched"] if x["match_type"] == m["match_type"])
            for m in entity_result["matched"]
        },
        "missed_entities": entity_result["gt_only"],
        "hallucinated_entities": entity_result["pipeline_only"],
        "missed_relationships": rel_result["gt_only"],
        "hallucinated_relationships": rel_result["pipeline_only"],
    }


def compare_combined(
    threshold: float,
    semantic_threshold: float,
    cross_type_threshold: float = 0.85,
    llm_provider: Any = None,
) -> Optional[Dict]:
    """Compare combined (cross-document) results."""
    gt_path = BENCHMARK_DIR / "gt_combined.json"
    pipeline_path = BENCHMARK_DIR / "pipeline_combined.json"

    if not gt_path.exists() or not pipeline_path.exists():
        logger.warning("Combined GT or pipeline results not found, skipping cross-doc comparison")
        return None

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))

    print(f"\n{'=' * 80}")
    print("CROSS-DOCUMENT COMPARISON")
    print(f"  GT: {len(gt['entities'])} entities, {len(gt['relationships'])} relationships")

    pipeline_entities = pipeline.get("cross_doc_merged_entities", [])
    pipeline_rels = pipeline.get("cross_doc_merged_relationships", [])
    print(f"  Pipeline: {len(pipeline_entities)} entities, {len(pipeline_rels)} relationships")

    gt_types = {e["type"] for e in gt["entities"]}

    # Entity comparison
    entity_result = match_entities(
        gt["entities"],
        pipeline_entities,
        threshold=threshold,
        semantic_threshold=semantic_threshold,
        cross_type_threshold=cross_type_threshold,
        llm_provider=llm_provider,
    )
    entity_metrics = compute_per_type_metrics(entity_result)
    print_entity_metrics(entity_metrics, "Entity Metrics — Combined", gt_types)
    print_match_type_breakdown(entity_result["matched"])

    # Relationship comparison
    entity_match_map = build_entity_match_map(entity_result)
    rel_result = match_relationships(
        gt["relationships"], pipeline_rels, entity_match_map,
        threshold=threshold,
        semantic_threshold=semantic_threshold,
        llm_provider=llm_provider,
    )
    rel_metrics = compute_per_type_metrics(rel_result)
    gt_rel_types = {r["type"] for r in gt["relationships"]}
    print_entity_metrics(rel_metrics, "Relationship Metrics — Combined", gt_rel_types)
    if rel_result["matched"]:
        print_match_type_breakdown(rel_result["matched"])

    # Dedup evaluation
    dedup_result = evaluate_dedup(gt, pipeline, threshold)
    print_dedup_report(dedup_result)

    return {
        "entity_metrics": entity_metrics,
        "entity_overall": compute_prf(
            len(entity_result["matched"]),
            len(gt["entities"]),
            len(pipeline_entities),
        ),
        "relationship_metrics": rel_metrics,
        "relationship_overall": compute_prf(
            len(rel_result["matched"]),
            len(gt["relationships"]),
            len(pipeline_rels),
        ),
        "dedup": dedup_result,
    }


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare e-Sbírka GT vs pipeline extraction results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Trigram similarity threshold for fuzzy matching (default: 0.7)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.75,
        help="Semantic embedding similarity threshold (default: 0.75)",
    )
    parser.add_argument(
        "--cross-type-threshold",
        type=float,
        default=0.85,
        help="Semantic threshold for cross-type entity matching (default: 0.85)",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use LLM judge for borderline semantic matches (costs API calls)",
    )
    parser.add_argument(
        "--llm-model",
        default="claude-haiku-4-5",
        help="Model for LLM judge (default: claude-haiku-4-5)",
    )
    args = parser.parse_args()

    # Initialize LLM provider if requested
    llm_provider = None
    if args.llm_judge:
        from dotenv import load_dotenv

        load_dotenv()
        from src.agent.providers.factory import create_provider

        llm_provider = create_provider(args.llm_model)
        logger.info(f"LLM judge enabled: {args.llm_model}")

    print(SEP)
    print("e-Sbírka Benchmark: GT vs Pipeline Comparison")
    print(f"Matching: exact + semantic (threshold={args.semantic_threshold})"
          + (f" + LLM judge ({args.llm_model})" if llm_provider else ""))
    print(SEP)

    # Find GT and pipeline files
    gt_files = sorted(BENCHMARK_DIR.glob("gt_sb_*.json"))
    if not gt_files:
        logger.error(f"No GT files found in {BENCHMARK_DIR}. Run esbirka_gt_dataset.py first.")
        sys.exit(1)

    results = {
        "documents": [],
        "threshold": args.threshold,
        "semantic_threshold": args.semantic_threshold,
        "llm_judge": args.llm_judge,
    }

    for gt_path in gt_files:
        doc_id = gt_path.stem.replace("gt_", "")
        pipeline_path = BENCHMARK_DIR / f"pipeline_{doc_id}.json"

        if not pipeline_path.exists():
            logger.warning(f"Pipeline result not found for {doc_id}, skipping")
            continue

        doc_result = compare_single_document(
            gt_path, pipeline_path, args.threshold, args.semantic_threshold,
            args.cross_type_threshold, llm_provider,
        )
        results["documents"].append(doc_result)

    # Combined comparison
    combined_result = compare_combined(
        args.threshold, args.semantic_threshold, args.cross_type_threshold, llm_provider
    )
    if combined_result:
        results["combined"] = combined_result

    # Save results
    out_path = BENCHMARK_DIR / "comparison_results.json"
    serializable = _make_serializable(results)
    out_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{SEP}")
    print(f"Results saved to: {out_path}")

    # Print overall summary
    print(f"\n{SEP}")
    print("OVERALL SUMMARY")
    print(SEP)
    for doc_result in results["documents"]:
        eo = doc_result["entity_overall"]
        ro = doc_result["relationship_overall"]
        print(
            f"  {doc_result['document_id']}:"
            f"  entities P={eo['precision']:.2%} R={eo['recall']:.2%} F1={eo['f1']:.2%}"
            f"  |  rels P={ro['precision']:.2%} R={ro['recall']:.2%} F1={ro['f1']:.2%}"
        )
    if combined_result:
        eo = combined_result["entity_overall"]
        ro = combined_result["relationship_overall"]
        dr = combined_result.get("dedup", {})
        print(
            f"  Combined:"
            f"  entities P={eo['precision']:.2%} R={eo['recall']:.2%} F1={eo['f1']:.2%}"
            f"  |  rels P={ro['precision']:.2%} R={ro['recall']:.2%} F1={ro['f1']:.2%}"
            f"  |  dedup rate={dr.get('merge_rate', 0):.0%}"
        )


def _make_serializable(obj: Any) -> Any:
    """Ensure all values are JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, float):
        if obj != obj:  # NaN
            return 0.0
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


if __name__ == "__main__":
    main()
