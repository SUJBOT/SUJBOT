"""
e-Sbírka Pipeline Extraction — run VL+KG pipeline on benchmark PDFs.

Processes downloaded e-Sbírka PDFs through the existing VL rendering + KG entity
extraction pipeline, saving results as JSON for comparison with the GT dataset.

Usage:
    uv run python scripts/esbirka_pipeline_extract.py [--model claude-haiku-4-5] [--max-pages 0]

Requires:
    - PDFs in data/esbirka_benchmark/ (run esbirka_gt_dataset.py first)
    - LLM API key (ANTHROPIC_API_KEY or DEEPINFRA_API_KEY)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEP = "=" * 70

BENCHMARK_DIR = Path("data/esbirka_benchmark")
PAGE_STORE_DIR = "data/esbirka_benchmark/vl_pages"


# =========================================================================
# Normalization helpers (adapted from graph_normalize_dedup.py for in-memory use)
# =========================================================================


def normalize_name(name: str) -> str:
    """Normalize entity name for dedup: dashes, quotes, parens, spaces."""
    s = name.lower()
    s = re.sub(r"[–—]", "-", s)  # em/en-dash → hyphen
    s = re.sub(r'["\'""\u201e\u201c\u201d]', "", s)  # remove quotes
    s = re.sub(r"[()]", "", s)  # remove parens
    s = re.sub(r"\s*[-/]\s*", "-", s)  # normalize spaces around dashes/slashes
    s = re.sub(r",\s*", " ", s)  # comma → space
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
    return s


def trigram_similarity(a: str, b: str) -> float:
    """Compute trigram (Jaccard) similarity between two strings."""
    if not a or not b:
        return 0.0
    a_lower = a.lower()
    b_lower = b.lower()

    def trigrams(s: str) -> Set[str]:
        s = f"  {s} "  # Pad for edge trigrams
        return {s[i : i + 3] for i in range(len(s) - 2)}

    ta = trigrams(a_lower)
    tb = trigrams(b_lower)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def deduplicate_entities_in_memory(
    raw_entities: List[Dict],
    raw_relationships: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Deduplicate entities and remap relationships (in-memory, no DB needed).

    Phase 1: Exact normalization match (safe)
    Phase 2: Trigram similarity > 0.7 (conservative)

    Returns:
        Tuple of (merged_entities, merged_relationships)
    """

    # Phase 1: Group by (normalized_name, type)
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for ent in raw_entities:
        key = (normalize_name(ent["name"]), ent["type"])
        groups[key].append(ent)

    # Build canonical name map
    canonical_map: Dict[str, str] = {}  # original_name → canonical_name
    merged_entities = []

    for (norm_name, etype), group in groups.items():
        # Pick canonical: longest description, then longest name
        group.sort(key=lambda e: (-len(e.get("description", "")), -len(e["name"])))
        canonical = group[0]
        merged_entities.append(canonical)

        # Map all variants to canonical
        for ent in group:
            canonical_map[ent["name"]] = canonical["name"]

    # Phase 2: Trigram similarity on merged entities
    # Only merge if trigram > 0.7 and types match
    further_merges: Dict[str, str] = {}  # name → canonical_name
    entity_by_name = {e["name"]: e for e in merged_entities}
    checked = set()

    for i, e1 in enumerate(merged_entities):
        for j, e2 in enumerate(merged_entities):
            if j <= i:
                continue
            if e1["type"] != e2["type"]:
                continue
            pair_key = (min(e1["name"], e2["name"]), max(e1["name"], e2["name"]))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            sim = trigram_similarity(e1["name"], e2["name"])
            if sim > 0.7:
                # Merge shorter name into longer (more specific)
                if len(e1["name"]) >= len(e2["name"]):
                    further_merges[e2["name"]] = e1["name"]
                else:
                    further_merges[e1["name"]] = e2["name"]

    # Apply phase 2 merges
    if further_merges:
        # Resolve transitive chains
        def resolve(name: str) -> str:
            visited = set()
            while name in further_merges and name not in visited:
                visited.add(name)
                name = further_merges[name]
            return name

        # Update canonical map
        for orig, canon in list(canonical_map.items()):
            resolved = resolve(canon)
            canonical_map[orig] = resolved

        # Filter merged entities
        keep_names = set(resolve(e["name"]) for e in merged_entities)
        merged_entities = [e for e in merged_entities if e["name"] in keep_names]

    # Remap relationships
    merged_relationships = []
    seen_rels = set()
    for rel in raw_relationships:
        source = canonical_map.get(rel["source"], rel["source"])
        target = canonical_map.get(rel["target"], rel["target"])

        # Skip self-references created by merging
        if source == target:
            continue

        rel_key = (source.lower(), target.lower(), rel["type"])
        if rel_key in seen_rels:
            continue
        seen_rels.add(rel_key)

        merged_relationships.append({
            "source": source,
            "target": target,
            "type": rel["type"],
            "description": rel.get("description", ""),
        })

    return merged_entities, merged_relationships


# =========================================================================
# Pipeline Execution
# =========================================================================


def run_pipeline(model: str, max_pages: int = 0) -> None:
    """Run VL+KG extraction pipeline on benchmark PDFs."""

    # Import pipeline components
    from src.vl.page_store import PageStore
    from src.graph.entity_extractor import EntityExtractor
    from src.agent.providers.factory import create_provider

    # Find PDFs
    pdf_files = sorted(BENCHMARK_DIR.glob("sb_*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {BENCHMARK_DIR}. Run esbirka_gt_dataset.py first.")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDFs to process")
    logger.info(f"Model: {model}")
    if max_pages > 0:
        logger.info(f"Max pages per document: {max_pages}")

    # Initialize components
    page_store = PageStore(
        store_dir=PAGE_STORE_DIR,
        source_pdf_dir=str(BENCHMARK_DIR),
    )
    provider = create_provider(model)
    extractor = EntityExtractor(provider)

    all_results = []

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem  # e.g., "sb_2016_263"

        logger.info(f"\n{SEP}")
        logger.info(f"Processing: {pdf_path.name} (doc_id: {doc_id})")
        logger.info(SEP)

        # 1. Render PDF to page images
        logger.info("  Rendering PDF pages...")
        page_ids = page_store.render_pdf_pages(str(pdf_path), doc_id)
        logger.info(f"  Rendered {len(page_ids)} pages")

        if max_pages > 0:
            page_ids = page_ids[:max_pages]
            logger.info(f"  Limited to {len(page_ids)} pages")

        # 2. Extract entities from each page
        raw_entities: List[Dict] = []
        raw_relationships: List[Dict] = []
        page_extractions: List[Dict] = []

        for idx, page_id in enumerate(page_ids):
            logger.info(f"  Extracting page {idx + 1}/{len(page_ids)}: {page_id}")
            t0 = time.time()

            result = extractor.extract_from_page(page_id, page_store)

            elapsed = time.time() - t0
            n_ent = len(result["entities"])
            n_rel = len(result["relationships"])
            logger.info(f"    → {n_ent} entities, {n_rel} relationships ({elapsed:.1f}s)")

            # Collect raw extractions
            page_extractions.append({
                "page_id": page_id,
                "entities": result["entities"],
                "relationships": result["relationships"],
            })
            raw_entities.extend(result["entities"])
            raw_relationships.extend(result["relationships"])

        logger.info(f"\n  Raw totals: {len(raw_entities)} entities, {len(raw_relationships)} relationships")

        # 3. Cross-page deduplication
        logger.info("  Running cross-page deduplication...")
        merged_entities, merged_relationships = deduplicate_entities_in_memory(
            raw_entities, raw_relationships
        )
        logger.info(
            f"  After dedup: {len(merged_entities)} entities, "
            f"{len(merged_relationships)} relationships"
        )

        # 4. Build result
        result = {
            "document_id": doc_id,
            "model": model,
            "pages_processed": len(page_ids),
            "raw_extractions": page_extractions,
            "raw_entity_count": len(raw_entities),
            "raw_relationship_count": len(raw_relationships),
            "merged_entities": merged_entities,
            "merged_relationships": merged_relationships,
            "entity_type_counts": _count_by_type(merged_entities, "type"),
            "relationship_type_counts": _count_by_type(merged_relationships, "type"),
        }
        all_results.append(result)

        # Save per-document result
        out_path = BENCHMARK_DIR / f"pipeline_{doc_id}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"  Saved: {out_path}")

    # 5. Cross-document dedup
    logger.info(f"\n{SEP}")
    logger.info("Cross-document deduplication")
    logger.info(SEP)

    all_merged_entities = []
    all_merged_rels = []
    for r in all_results:
        for ent in r["merged_entities"]:
            ent_copy = {**ent, "document_id": r["document_id"]}
            all_merged_entities.append(ent_copy)
        for rel in r["merged_relationships"]:
            rel_copy = {**rel, "document_id": r["document_id"]}
            all_merged_rels.append(rel_copy)

    cross_doc_entities, cross_doc_rels = deduplicate_entities_in_memory(
        all_merged_entities, all_merged_rels
    )

    cross_doc_result = {
        "documents": [r["document_id"] for r in all_results],
        "model": model,
        "per_document_entity_counts": {r["document_id"]: len(r["merged_entities"]) for r in all_results},
        "cross_doc_merged_entities": cross_doc_entities,
        "cross_doc_merged_relationships": cross_doc_rels,
        "entity_type_counts": _count_by_type(cross_doc_entities, "type"),
        "relationship_type_counts": _count_by_type(cross_doc_rels, "type"),
    }

    out_path = BENCHMARK_DIR / "pipeline_combined.json"
    out_path.write_text(json.dumps(cross_doc_result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved combined: {out_path}")
    logger.info(
        f"  Cross-doc: {len(cross_doc_entities)} entities, {len(cross_doc_rels)} relationships"
    )

    # Summary
    logger.info(f"\n{SEP}")
    logger.info("Pipeline extraction complete!")
    logger.info(SEP)
    for r in all_results:
        logger.info(
            f"  {r['document_id']}: {r['pages_processed']} pages → "
            f"{len(r['merged_entities'])} entities, {len(r['merged_relationships'])} rels"
        )
    logger.info(f"  Combined (cross-doc dedup): {len(cross_doc_entities)} entities, {len(cross_doc_rels)} rels")


def _count_by_type(items: List[Dict], key: str) -> Dict[str, int]:
    """Count items by a key."""
    counts: Dict[str, int] = {}
    for item in items:
        t = item.get(key, "UNKNOWN")
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


# =========================================================================
# CLI
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run VL+KG extraction pipeline on e-Sbírka benchmark PDFs"
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="LLM model for entity extraction (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages per document (0 = all pages, default: 0)",
    )
    args = parser.parse_args()

    run_pipeline(model=args.model, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
