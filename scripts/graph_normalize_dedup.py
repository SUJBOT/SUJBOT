"""
Targeted normalization-based entity deduplication.

Phase 1:  Safe merges — entities differing only in formatting (dashes, quotes, parens, spaces)
Phase 1b: Substring containment — shorter name fully contained in longer name (same type+doc)
Phase 2:  Trigram similarity + word overlap + LLM arbitration — near-duplicates, abbreviations

Usage:
    DATABASE_URL="postgresql://postgres:sujbot_secure_password@localhost:5432/sujbot" \
    uv run python scripts/graph_normalize_dedup.py
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import nest_asyncio

nest_asyncio.apply()

import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEP = "=" * 70


def normalize_name(name: str) -> str:
    """Normalize entity name for safe dedup: dashes, quotes, parens, spaces."""
    s = name.lower()
    s = re.sub(r"[–—]", "-", s)  # em/en-dash → hyphen
    s = re.sub(r'["\'""\u201e\u201c\u201d]', "", s)  # remove quotes
    s = re.sub(r"[()]", "", s)  # remove parens
    s = re.sub(r"\s*[-/]\s*", "-", s)  # normalize spaces around dashes/slashes
    s = re.sub(r",\s*", " ", s)  # comma → space
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
    return s


async def merge_entity_group(
    conn,
    canonical_id: int,
    duplicate_ids: List[int],
    merged_description: Optional[str] = None,
    canonical_name: Optional[str] = None,
) -> int:
    """Merge duplicate entities into canonical (same logic as GraphStorageAdapter)."""
    # Remap relationships
    for dup_id in duplicate_ids:
        await conn.execute(
            "UPDATE graph.relationships SET source_entity_id = $1 WHERE source_entity_id = $2",
            canonical_id, dup_id,
        )
        await conn.execute(
            "UPDATE graph.relationships SET target_entity_id = $1 WHERE target_entity_id = $2",
            canonical_id, dup_id,
        )

    # Remove self-references
    await conn.execute(
        "DELETE FROM graph.relationships "
        "WHERE source_entity_id = target_entity_id AND source_entity_id = $1",
        canonical_id,
    )

    # Deduplicate edges
    await conn.execute(
        """
        DELETE FROM graph.relationships r1
        USING graph.relationships r2
        WHERE r1.source_entity_id = r2.source_entity_id
          AND r1.target_entity_id = r2.target_entity_id
          AND r1.relationship_type = r2.relationship_type
          AND r1.relationship_id > r2.relationship_id
          AND (r1.source_entity_id = $1 OR r1.target_entity_id = $1)
        """,
        canonical_id,
    )

    # Merge source_page_ids
    all_ids = [canonical_id] + duplicate_ids
    ph = ", ".join(f"${i + 1}" for i in range(len(all_ids)))
    merged_pages = await conn.fetchval(
        f"SELECT array_agg(DISTINCT pid) FROM ("
        f"  SELECT unnest(source_page_ids) AS pid FROM graph.entities "
        f"  WHERE entity_id IN ({ph})"
        f") sub WHERE pid IS NOT NULL",
        *all_ids,
    )

    # Collect duplicate names for aliases before deleting
    dup_names_for_alias = []
    for dup_id in duplicate_ids:
        dup_name = await conn.fetchval(
            "SELECT name FROM graph.entities WHERE entity_id = $1", dup_id
        )
        if dup_name:
            dup_names_for_alias.append(dup_name)

    # Move aliases from duplicates to canonical BEFORE deleting (FK cascade would lose them)
    for dup_id in duplicate_ids:
        await conn.execute(
            "UPDATE graph.entity_aliases SET entity_id = $1 WHERE entity_id = $2",
            canonical_id, dup_id,
        )

    # Delete duplicates FIRST to avoid unique constraint violation on (name, type, doc)
    dup_ph = ", ".join(f"${i + 1}" for i in range(len(duplicate_ids)))
    result = await conn.execute(
        f"DELETE FROM graph.entities WHERE entity_id IN ({dup_ph})", *duplicate_ids
    )
    count = int(result.split()[-1]) if result else 0

    # Now update canonical entity (no conflict since duplicates are gone)
    update_parts = ["search_embedding = NULL"]
    params: List[Any] = []
    if canonical_name is not None:
        params.append(canonical_name)
        update_parts.append(f"name = ${len(params)}")
    if merged_description is not None:
        params.append(merged_description)
        update_parts.append(f"description = ${len(params)}")
    params.append(merged_pages or [])
    update_parts.append(f"source_page_ids = ${len(params)}")
    params.append(canonical_id)
    await conn.execute(
        f"UPDATE graph.entities SET {', '.join(update_parts)} WHERE entity_id = ${len(params)}",
        *params,
    )

    # Add duplicate names as aliases of canonical
    for dup_name in dup_names_for_alias:
        if canonical_name and dup_name.lower() != canonical_name.lower():
            await conn.execute(
                "INSERT INTO graph.entity_aliases (entity_id, alias) VALUES ($1, $2) "
                "ON CONFLICT DO NOTHING",
                canonical_id, dup_name,
            )

    return count


async def llm_should_merge(provider, name1: str, name2: str, entity_type: str) -> bool:
    """Ask LLM whether two entities refer to the same real-world thing."""
    prompt = (
        f"Do these two entity names refer to the SAME real-world thing?\n\n"
        f"Entity type: {entity_type}\n"
        f"Name 1: {name1}\n"
        f"Name 2: {name2}\n\n"
        f"Consider: word order differences, abbreviations, minor spelling variants, "
        f"singular/plural forms (systém vs systémy, reaktor vs reaktory), "
        f"and formatting differences are the SAME entity. "
        f"Different numerical values (A₁ vs A₂, 10² vs 10³, kategorie II vs III) "
        f"are DIFFERENT entities.\n\n"
        f"Answer ONLY 'YES' or 'NO'."
    )
    try:
        response = await asyncio.to_thread(
            provider.create_message,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system="You are a deduplication judge. Answer only YES or NO.",
            max_tokens=10,
            temperature=0.0,
        )
        answer = (response.text or "").strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        logger.warning(f"LLM arbitration failed for '{name1}' vs '{name2}': {e}")
        return False


async def phase1_normalization_dedup(pool: asyncpg.Pool) -> Dict:
    """Phase 1: Merge entities with identical normalized names (safe, no LLM needed)."""
    logger.info(SEP)
    logger.info("PHASE 1: Normalization-based dedup (dashes, quotes, parens, spaces)")
    logger.info(SEP)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT entity_id, name, entity_type, coalesce(description, '') AS description "
            "FROM graph.entities ORDER BY entity_id"
        )

    # Group by (normalized_name, entity_type)
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        key = (normalize_name(r["name"]), r["entity_type"])
        groups.setdefault(key, []).append(dict(r))

    # Filter to groups with duplicates
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}

    if not dup_groups:
        logger.info("No normalization duplicates found")
        return {"merged": 0, "removed": 0}

    logger.info(f"Found {len(dup_groups)} normalization duplicate groups\n")

    total_removed = 0
    total_merged = 0

    for (norm_name, entity_type), entities in sorted(dup_groups.items()):
        # Pick canonical: longest name (most descriptive), then lowest ID
        entities.sort(key=lambda e: (-len(e["name"]), e["entity_id"]))
        canonical = entities[0]
        duplicates = entities[1:]

        # Merge descriptions
        descs = [e["description"] for e in entities if e["description"]]
        distinct_descs = list(dict.fromkeys(descs))
        merged_desc = " | ".join(distinct_descs)[:2000] if distinct_descs else None

        dup_names = [e["name"] for e in duplicates]
        logger.info(
            f"  MERGE [{entity_type}]: '{canonical['name']}' ← {dup_names}"
        )

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    removed = await merge_entity_group(
                        conn,
                        canonical["entity_id"],
                        [e["entity_id"] for e in duplicates],
                        merged_description=merged_desc,
                        canonical_name=canonical["name"],
                    )
                    total_removed += removed
                    total_merged += 1
        except Exception as e:
            logger.error(f"  FAILED: {e}")

    stats = {"merged": total_merged, "removed": total_removed}
    logger.info(f"\nPhase 1 complete: {stats}")
    return stats


# Types where numeric/identifier differences mean different entities — skip substring dedup
_SKIP_SUBSTRING_TYPES = frozenset({"SANCTION", "DEADLINE", "SECTION", "REQUIREMENT"})


async def phase1b_substring_dedup(pool: asyncpg.Pool, provider) -> Dict:
    """Phase 1b: Merge entities where shorter name is contained in longer name (same type+doc).

    Catches cases like:
    - "fyzikální spouštění" ⊂ "fyzikální spouštění podle vnitřních předpisů"
    - "nakládání se zdrojem" ⊂ "bezpečné nakládání se zdrojem"
    - "údržba" ⊂ "údržba a opravy"

    Uses LLM confirmation for safety. Keeps the shorter (more canonical) name.
    """
    logger.info(SEP)
    logger.info("PHASE 1b: Substring containment dedup (short name ⊂ long name)")
    logger.info(SEP)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT entity_id, name, entity_type, document_id, "
            "coalesce(description, '') AS description "
            "FROM graph.entities ORDER BY entity_id"
        )

    # Group by (entity_type, document_id)
    groups: Dict[Tuple[str, Optional[str]], List[Dict]] = {}
    for r in rows:
        key = (r["entity_type"], r["document_id"])
        groups.setdefault(key, []).append(dict(r))

    candidates = []
    for (etype, doc_id), entities in groups.items():
        if etype in _SKIP_SUBSTRING_TYPES:
            continue
        # Compare all pairs within the group
        for i, short in enumerate(entities):
            short_norm = normalize_name(short["name"])
            short_words = len(short_norm.split())
            if short_words < 2:
                continue  # Avoid single-word matches like "záření" matching everything
            for long in entities[i + 1 :]:
                long_norm = normalize_name(long["name"])
                if len(short_norm) >= len(long_norm):
                    # Ensure short is actually shorter
                    if len(short_norm) == len(long_norm):
                        continue
                    short, long = long, short
                    short_norm, long_norm = long_norm, short_norm
                    short_words = len(short_norm.split())
                    if short_words < 2:
                        continue
                # Check substring containment
                if short_norm not in long_norm:
                    continue
                # Short name must be at least 40% of long name's length
                if len(short_norm) < 0.4 * len(long_norm):
                    continue
                candidates.append(
                    {
                        "short_id": short["entity_id"],
                        "short_name": short["name"],
                        "long_id": long["entity_id"],
                        "long_name": long["name"],
                        "entity_type": etype,
                        "short_desc": short["description"],
                        "long_desc": long["description"],
                    }
                )

    if not candidates:
        logger.info("No substring containment candidates found")
        return {"candidates": 0, "llm_confirmed": 0, "merged": 0, "removed": 0}

    logger.info(f"Found {len(candidates)} substring candidates for LLM review\n")

    confirmed = []
    rejected = []
    for c in candidates:
        should_merge = await llm_should_merge(
            provider, c["short_name"], c["long_name"], c["entity_type"]
        )
        if should_merge:
            confirmed.append(c)
            logger.info(
                f"  ✓ LLM CONFIRMED [{c['entity_type']}]: "
                f"'{c['short_name']}' ⊂ '{c['long_name']}'"
            )
        else:
            rejected.append(c)
            logger.info(
                f"  ✗ LLM REJECTED  [{c['entity_type']}]: "
                f"'{c['short_name']}' ⊄ '{c['long_name']}'"
            )

    # Merge confirmed pairs — keep the shorter name as canonical
    total_removed = 0
    total_merged = 0
    merged_ids: set = set()  # Track already-merged entity IDs

    for c in confirmed:
        if c["long_id"] in merged_ids or c["short_id"] in merged_ids:
            continue  # Already merged in a previous iteration

        # Merge descriptions
        descs = [d for d in [c["short_desc"], c["long_desc"]] if d]
        distinct_descs = list(dict.fromkeys(descs))
        merged_desc = " | ".join(distinct_descs)[:2000] if distinct_descs else None

        logger.info(
            f"  MERGE [{c['entity_type']}]: '{c['short_name']}' ← ['{c['long_name']}']"
        )

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    removed = await merge_entity_group(
                        conn,
                        c["short_id"],
                        [c["long_id"]],
                        merged_description=merged_desc,
                        canonical_name=c["short_name"],
                    )
                    total_removed += removed
                    total_merged += 1
                    merged_ids.add(c["long_id"])
        except Exception as e:
            logger.error(f"  FAILED: {e}")

    stats = {
        "candidates": len(candidates),
        "llm_confirmed": len(confirmed),
        "llm_rejected": len(rejected),
        "merged": total_merged,
        "removed": total_removed,
    }
    logger.info(f"\nPhase 1b complete: {stats}")
    return stats


def word_overlap(name1: str, name2: str) -> float:
    """Compute word-level Jaccard overlap between two entity names."""
    words1 = set(normalize_name(name1).split())
    words2 = set(normalize_name(name2).split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


async def phase2_trigram_llm_dedup(pool: asyncpg.Pool, provider) -> Dict:
    """Phase 2: Trigram similarity + word overlap + LLM arbitration for near-duplicates."""
    logger.info(SEP)
    logger.info("PHASE 2: Trigram similarity + word overlap + LLM arbitration")
    logger.info(SEP)

    # --- Source A: Trigram similarity > 0.75 (existing logic) ---
    async with pool.acquire() as conn:
        trigram_candidates = await conn.fetch(
            """
            SELECT e1.entity_id AS id1, e1.name AS name1,
                   e2.entity_id AS id2, e2.name AS name2,
                   e1.entity_type,
                   similarity(lower(e1.name), lower(e2.name)) AS sim
            FROM graph.entities e1
            JOIN graph.entities e2
                ON e1.entity_type = e2.entity_type
                AND e1.entity_id < e2.entity_id
                AND similarity(lower(e1.name), lower(e2.name)) > 0.75
            WHERE e1.entity_type NOT IN ('REQUIREMENT')
            ORDER BY similarity(lower(e1.name), lower(e2.name)) DESC
            LIMIT 200
            """,
        )

    # --- Source B: Word overlap > 0.6 for pairs with lower trigram sim ---
    # This catches cases where trigram fails due to length difference
    # (e.g., "údržba" vs "údržba a opravy" — low trigram sim, high word overlap)
    async with pool.acquire() as conn:
        overlap_pool_rows = await conn.fetch(
            """
            SELECT e1.entity_id AS id1, e1.name AS name1,
                   e2.entity_id AS id2, e2.name AS name2,
                   e1.entity_type,
                   similarity(lower(e1.name), lower(e2.name)) AS sim
            FROM graph.entities e1
            JOIN graph.entities e2
                ON e1.entity_type = e2.entity_type
                AND e1.document_id = e2.document_id
                AND e1.entity_id < e2.entity_id
                AND similarity(lower(e1.name), lower(e2.name)) BETWEEN 0.3 AND 0.75
            WHERE e1.entity_type NOT IN ('REQUIREMENT', 'SANCTION', 'DEADLINE', 'SECTION')
            ORDER BY similarity(lower(e1.name), lower(e2.name)) DESC
            LIMIT 500
            """,
        )

    # Filter word-overlap candidates by actual word overlap
    word_overlap_candidates = []
    for r in overlap_pool_rows:
        wo = word_overlap(r["name1"], r["name2"])
        if wo > 0.6:
            word_overlap_candidates.append(dict(r))

    # Combine both sources, dedup by (id1, id2)
    seen_pairs: set = set()
    candidates = []
    for c in trigram_candidates:
        pair = (c["id1"], c["id2"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            candidates.append(dict(c))
    for c in word_overlap_candidates:
        pair = (c["id1"], c["id2"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            candidates.append(c)

    logger.info(
        f"Found {len(trigram_candidates)} trigram + "
        f"{len(word_overlap_candidates)} word-overlap candidates "
        f"({len(candidates)} unique pairs)\n"
    )

    if not candidates:
        logger.info("No candidates found")
        return {"candidates": 0, "llm_confirmed": 0, "merged": 0, "removed": 0}

    # Filter out pairs already handled by normalization or substring phases
    filtered = []
    for c in candidates:
        if normalize_name(c["name1"]) == normalize_name(c["name2"]):
            continue  # Already merged in phase 1
        # Skip if one is substring of the other (handled in phase 1b)
        n1, n2 = normalize_name(c["name1"]), normalize_name(c["name2"])
        if n1 in n2 or n2 in n1:
            continue
        filtered.append(c)

    if not filtered:
        logger.info("All candidates already handled by earlier phases")
        return {"candidates": 0, "llm_confirmed": 0, "merged": 0, "removed": 0}

    logger.info(f"  {len(filtered)} candidates remain after filtering\n")

    confirmed = []
    rejected = []
    for c in filtered:
        should_merge = await llm_should_merge(
            provider, c["name1"], c["name2"], c["entity_type"]
        )
        if should_merge:
            confirmed.append(c)
            logger.info(
                f"  ✓ LLM CONFIRMED [{c['entity_type']}]: "
                f"'{c['name1']}' ≈ '{c['name2']}' (sim={c['sim']:.3f})"
            )
        else:
            rejected.append(c)
            logger.info(
                f"  ✗ LLM REJECTED  [{c['entity_type']}]: "
                f"'{c['name1']}' ≠ '{c['name2']}' (sim={c['sim']:.3f})"
            )

    # Merge confirmed pairs
    total_removed = 0
    total_merged = 0

    # Build Union-Find for transitive closure
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for c in confirmed:
        union(c["id1"], c["id2"])

    # Group by canonical
    merge_groups: Dict[int, List[int]] = {}
    all_ids = set()
    for c in confirmed:
        all_ids.add(c["id1"])
        all_ids.add(c["id2"])
    for eid in all_ids:
        root = find(eid)
        merge_groups.setdefault(root, []).append(eid)
    for root in merge_groups:
        merge_groups[root] = sorted(set(merge_groups[root]))

    # Fetch entity info for name selection
    if all_ids:
        ph = ", ".join(f"${i + 1}" for i in range(len(all_ids)))
        async with pool.acquire() as conn:
            info_rows = await conn.fetch(
                f"SELECT entity_id, name, coalesce(description, '') AS description "
                f"FROM graph.entities WHERE entity_id IN ({ph})",
                *sorted(all_ids),
            )
        entity_info = {r["entity_id"]: dict(r) for r in info_rows}

        for canonical_id, group_ids in merge_groups.items():
            duplicate_ids = [eid for eid in group_ids if eid != canonical_id]
            if not duplicate_ids:
                continue

            # Pick longest name as canonical
            best_name = max(
                (entity_info[eid]["name"] for eid in group_ids if eid in entity_info),
                key=len,
                default=None,
            )

            descs = [entity_info[eid]["description"] for eid in group_ids if eid in entity_info and entity_info[eid]["description"]]
            distinct_descs = list(dict.fromkeys(descs))
            merged_desc = " | ".join(distinct_descs)[:2000] if distinct_descs else None

            dup_names = [entity_info[eid]["name"] for eid in duplicate_ids if eid in entity_info]
            # Get entity_type from any member of the group
            etype = next(
                (c["entity_type"] for c in confirmed if c["id1"] in group_ids or c["id2"] in group_ids),
                "?",
            )
            logger.info(
                f"  MERGE [{etype}]: '{best_name}' ← {dup_names}"
            )

            try:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        removed = await merge_entity_group(
                            conn,
                            canonical_id,
                            duplicate_ids,
                            merged_description=merged_desc,
                            canonical_name=best_name,
                        )
                        total_removed += removed
                        total_merged += 1
            except Exception as e:
                logger.error(f"  FAILED: {e}")

    stats = {
        "candidates": len(filtered),
        "llm_confirmed": len(confirmed),
        "llm_rejected": len(rejected),
        "merged": total_merged,
        "removed": total_removed,
    }
    logger.info(f"\nPhase 2 complete: {stats}")
    return stats


async def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    pool = await asyncpg.create_pool(dsn=db_url, min_size=2, max_size=10)

    # Phase 1: Safe normalization dedup
    stats1 = await phase1_normalization_dedup(pool)

    # LLM provider for phases 1b and 2
    logger.info(SEP)
    logger.info("Setting up LLM provider for Phases 1b and 2...")
    from src.agent.providers.factory import create_provider
    provider = create_provider("claude-haiku-4-5-20251001")

    # Phase 1b: Substring containment + LLM
    stats1b = await phase1b_substring_dedup(pool, provider)

    # Phase 2: Trigram + word overlap + LLM
    stats2 = await phase2_trigram_llm_dedup(pool, provider)

    # Summary
    logger.info(SEP)
    logger.info("FINAL SUMMARY")
    logger.info(SEP)
    logger.info(f"Phase 1  (normalization):  {stats1}")
    logger.info(f"Phase 1b (substring+LLM):  {stats1b}")
    logger.info(f"Phase 2  (trigram+overlap): {stats2}")

    total_removed = stats1["removed"] + stats1b["removed"] + stats2["removed"]
    logger.info(f"Total entities removed: {total_removed}")

    # Print final stats
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT count(*) as entities FROM graph.entities"
        )
        row2 = await conn.fetchrow(
            "SELECT count(*) as rels FROM graph.relationships"
        )
        logger.info(f"Graph now: {row['entities']} entities, {row2['rels']} relationships")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
