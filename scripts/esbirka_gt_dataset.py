"""
e-Sbírka Ground Truth Dataset Builder.

Downloads structured legal data from the Czech e-Sbírka REST API and extracts
a ground-truth entity/relationship dataset for benchmarking the KG extraction pipeline.

Entity extraction uses LLM on fragment text for comprehensive coverage,
combined with structural data from API endpoints (metadata, souvislosti, fragments).

Documents:
  - Zákon č. 263/2016 Sb. (atomový zákon)
  - Vyhláška č. 422/2016 Sb. (radiační ochrana)

Usage:
    uv run python scripts/esbirka_gt_dataset.py [--model claude-haiku-4-5] [--max-fragments 0]
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nest_asyncio

nest_asyncio.apply()

from src.agent.providers.factory import create_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEP = "=" * 70

# =========================================================================
# Configuration
# =========================================================================

API_BASE = "https://www.e-sbirka.cz/sbr-cache"
API_KEY = os.getenv("ESBIRKA_API_KEY", "")

BENCHMARK_DIR = Path("data/esbirka_benchmark")

DOCUMENTS = [
    {
        "id": "sb_2016_263",
        "staleUrl": "/sb/2016/263",
        "citation": "263/2016 Sb.",
        "name": "atomový zákon",
    },
    {
        "id": "sb_2016_422",
        "staleUrl": "/sb/2016/422",
        "citation": "422/2016 Sb.",
        "name": "radiační ochrana",
    },
]

# LLM prompt for text-based entity extraction
GT_EXTRACTION_PROMPT = Path("prompts/graph_gt_text_extraction.txt").read_text()


# =========================================================================
# API Client
# =========================================================================


def _headers() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if API_KEY:
        h["esel-api-access-key"] = API_KEY
    return h


def _encode_stable_url(stable_url: str) -> str:
    """URL-encode staleUrl for use in API path: /sb/2016/263 → %2Fsb%2F2016%2F263."""
    from urllib.parse import quote
    return quote(stable_url, safe="")


def api_get(client: httpx.Client, path: str, params: Optional[Dict] = None) -> Any:
    """Make GET request to e-Sbírka sbr-cache API with retry."""
    url = f"{API_BASE}{path}"
    for attempt in range(3):
        try:
            resp = client.get(url, headers=_headers(), params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            logger.error(f"HTTP {e.response.status_code} for {url}: {e.response.text[:200]}")
            raise
        except httpx.RequestError as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise
    raise RuntimeError(f"Failed after 3 attempts: {url}")


def download_binary(client: httpx.Client, url: str, dest: Path) -> None:
    """Download binary file (PDF) from URL."""
    for attempt in range(3):
        try:
            resp = client.get(url, headers=_headers(), timeout=60.0, follow_redirects=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info(f"  Downloaded {dest.name} ({len(resp.content)} bytes)")
            return
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt < 2:
                time.sleep(2)
                continue
            logger.error(f"Failed to download {url}: {e}")
            raise


# =========================================================================
# Data Download
# =========================================================================


def fetch_document_metadata(client: httpx.Client, stable_url: str) -> Dict:
    """Fetch document metadata from e-Sbírka."""
    encoded = _encode_stable_url(stable_url)
    logger.info(f"  Fetching metadata for {stable_url}")
    return api_get(client, f"/dokumenty-sbirky/{encoded}")


def fetch_fragments(client: httpx.Client, stable_url: str) -> List[Dict]:
    """Fetch all fragments (paginated, 0-indexed) for a document."""
    encoded = _encode_stable_url(stable_url)
    all_fragments = []
    page = 0  # e-Sbírka uses 0-indexed pages
    while True:
        logger.info(f"  Fetching fragments page {page}...")
        data = api_get(
            client,
            f"/dokumenty-sbirky/{encoded}/fragmenty",
            params={"cisloStranky": page},
        )

        # Response structure: {"seznam": [...], "pocetStranek": N}
        fragments = data.get("seznam", data.get("fragmenty", []))
        if not fragments:
            if isinstance(data, list):
                fragments = data
            else:
                break

        all_fragments.extend(fragments)
        logger.info(f"    Got {len(fragments)} fragments (total: {len(all_fragments)})")

        total_pages = data.get("pocetStranek", data.get("celkovyPocetStranek", 1))
        if page + 1 >= total_pages:
            break
        page += 1

    return all_fragments


def fetch_souvislosti(client: httpx.Client, stable_url: str) -> List[Dict]:
    """Fetch inter-law relationships (souvislosti) for a document."""
    encoded = _encode_stable_url(stable_url)
    logger.info(f"  Fetching souvislosti for {stable_url}")
    try:
        data = api_get(client, f"/dokumenty-sbirky/{encoded}/souvislosti")
        if isinstance(data, list):
            return data
        return data.get("souvislosti", [])
    except httpx.HTTPStatusError:
        logger.warning(f"  Souvislosti not available")
        return []


def fetch_download_links(client: httpx.Client, stable_url: str) -> Dict:
    """Fetch PDF download links for a document. Returns the full links dict."""
    encoded = _encode_stable_url(stable_url)
    logger.info(f"  Fetching download links for {stable_url}")
    try:
        return api_get(client, f"/dokumenty-sbirky/{encoded}/odkazy-ke-stazeni")
    except httpx.HTTPStatusError:
        logger.warning(f"  Download links not available")
        return {}


def download_pdf(client: httpx.Client, doc_config: Dict, metadata: Dict, links: Dict) -> Optional[Path]:
    """Download or copy PDF for a document. Prefers local copies over API download."""
    import shutil

    pdf_path = BENCHMARK_DIR / f"{doc_config['id']}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 5000:
        logger.info(f"  PDF already exists: {pdf_path}")
        return pdf_path

    # Try local PDFs first (more reliable than API download)
    clean_citation = doc_config["citation"].replace(" Sb.", "")
    parts = clean_citation.split("/")
    if len(parts) == 2:
        number, year = parts
        local_patterns = [
            f"Sb_{year}_{number}*",
            f"*{number}_{year}*",
            f"*{clean_citation.replace('/', '_')}*",
        ]
        for pattern in local_patterns:
            candidates = [p for p in Path("data").glob(pattern) if p.suffix == ".pdf"]
            if candidates:
                shutil.copy2(candidates[0], pdf_path)
                logger.info(f"  Copied local PDF: {candidates[0].name} → {pdf_path.name}")
                return pdf_path

    # Fallback: try API download
    doc_id = None
    for section in ["informativniZneni", "overeneZneni"]:
        section_data = links.get(section, {})
        odkaz_pdf = section_data.get("odkazPdf", {})
        if isinstance(odkaz_pdf, dict) and "dokumentId" in odkaz_pdf:
            doc_id = odkaz_pdf["dokumentId"]
            break

    if doc_id:
        try:
            url = f"https://www.e-sbirka.cz/stahni/overena-zneni/{doc_id}"
            download_binary(client, url, pdf_path)
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    header = f.read(4)
                if header != b"%PDF":
                    logger.warning(f"  Downloaded file is not a PDF (header: {header!r}), removing")
                    pdf_path.unlink()
                else:
                    return pdf_path
        except Exception as e:
            logger.warning(f"  Failed to download PDF via API: {e}")

    logger.warning(f"  Could not find PDF for {doc_config['citation']}")
    return None


# =========================================================================
# LLM-based Entity Extraction from Text
# =========================================================================


def extract_entities_from_text_llm(
    fragment_texts: List[Tuple[str, str]],
    provider,
    batch_char_limit: int = 15000,
) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relationships from fragment text using LLM.

    Args:
        fragment_texts: List of (frag_id, text) tuples.
        provider: LLM provider for extraction.
        batch_char_limit: Max characters per LLM batch.

    Returns:
        Tuple of (entities, relationships) extracted by LLM.
    """
    if not fragment_texts:
        return [], []

    # Filter out trivial fragments (< 30 chars of actual content)
    meaningful = [(fid, t) for fid, t in fragment_texts if len(t) > 30]
    if not meaningful:
        return [], []

    # Batch by character count
    batches: List[List[Tuple[str, str]]] = []
    current_batch: List[Tuple[str, str]] = []
    current_chars = 0
    for fid, text in meaningful:
        if current_chars + len(text) > batch_char_limit and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append((fid, text))
        current_chars += len(text)
    if current_batch:
        batches.append(current_batch)

    logger.info(f"    LLM text extraction: {len(meaningful)} fragments → {len(batches)} batches")

    all_entities: List[Dict] = []
    all_relationships: List[Dict] = []
    t_start = time.time()

    for i, batch in enumerate(batches):
        combined_text = "\n\n---\n\n".join(t for _, t in batch)

        user_msg = f"{GT_EXTRACTION_PROMPT}\n\n## Text to analyze\n\n{combined_text}"

        try:
            response = provider.create_message(
                system="You are a precise entity and relationship extraction system for legal documents.",
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                max_tokens=4096,
                temperature=0.0,
            )

            # Parse response
            resp_text = response.content[0].get("text", "") if response.content else ""

            # Extract JSON from possible markdown code block
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", resp_text, re.DOTALL)
            if json_match:
                resp_text = json_match.group(1)

            data = json.loads(resp_text.strip())
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            # Tag source
            for ent in entities:
                ent["source"] = "llm_extraction"
            for rel in relationships:
                rel["source_info"] = "llm_extraction"

            all_entities.extend(entities)
            all_relationships.extend(relationships)

        except json.JSONDecodeError as e:
            logger.warning(f"    Batch {i+1}/{len(batches)}: JSON parse error: {e}")
            continue
        except Exception as e:
            logger.warning(f"    Batch {i+1}/{len(batches)}: LLM call failed: {e}")
            continue

        # Progress logging every 5 batches
        if (i + 1) % 5 == 0 or i == len(batches) - 1:
            elapsed = time.time() - t_start
            avg = elapsed / (i + 1)
            remaining = avg * (len(batches) - i - 1)
            logger.info(
                f"    Batch {i+1}/{len(batches)}: "
                f"{len(all_entities)} entities, {len(all_relationships)} rels "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    return all_entities, all_relationships


# =========================================================================
# Structural Entity Extraction from API Data
# =========================================================================


def extract_entities_from_metadata(metadata: Dict, doc_config: Dict) -> List[Dict]:
    """Extract entities from document metadata."""
    entities = []

    # The document itself is a REGULATION entity
    citation = metadata.get("uplnaCitace", "")
    if not citation:
        citation = (
            f"zákon č. {doc_config['citation']}"
            if "zákon" in doc_config["name"].lower()
            else f"vyhláška č. {doc_config['citation']}"
        )

    entities.append({
        "name": citation.lower() if citation[0].isupper() and "zákon" in citation.lower() else citation,
        "type": "REGULATION",
        "description": metadata.get("nazev", doc_config["name"]),
        "source": "metadata",
    })

    # Extract amendments from metadata
    novely = metadata.get("novely", [])
    for novela in novely:
        kod = novela.get("kodDokumentuSbirky", "")
        nazev = novela.get("nazev", "")
        if kod:
            entities.append({
                "name": f"zákon č. {kod}" if "zákon" in nazev.lower() else kod,
                "type": "AMENDMENT",
                "description": nazev,
                "source": "metadata.novely",
            })

    return entities


def extract_entities_from_souvislosti(souvislosti: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relationships from inter-law relationships."""
    entities = []
    relationships = []
    seen_entities = set()

    rel_type_map = {
        "JE_MENEN": "AMENDS",
        "MENI": "AMENDS",
        "RUSI": "SUPERSEDES",
        "JE_PROVADEN": "DERIVED_FROM",
        "PROVADI": "DERIVED_FROM",
        "ODKAZUJE": "REFERENCES",
        "JE_ODKAZOVAN": "REFERENCES",
        "OSTATNI_OBSAHUJE": "REFERENCES",
    }

    for item in souvislosti:
        typ = item.get("typ", "")
        rel_type = rel_type_map.get(typ)

        docs = item.get("dokumentySbirky", [])
        for doc in docs:
            target_kod = doc.get("kodDokumentuSbirky", "")
            target_nazev = doc.get("nazev", "")

            if not target_kod:
                continue

            nazev_lower = target_nazev.lower()
            if "zákon" in nazev_lower or "zák" in nazev_lower:
                entity_name = f"zákon č. {target_kod}"
            elif "vyhláška" in nazev_lower or "vyhl" in nazev_lower:
                entity_name = f"vyhláška č. {target_kod}"
            elif "nařízení" in nazev_lower:
                entity_name = f"nařízení č. {target_kod}"
            else:
                entity_name = target_kod

            if entity_name not in seen_entities:
                entities.append({
                    "name": entity_name,
                    "type": "REGULATION",
                    "description": target_nazev,
                    "source": "souvislosti",
                })
                seen_entities.add(entity_name)

            if rel_type:
                relationships.append({
                    "source": entity_name,
                    "target": "__SELF__",
                    "type": rel_type,
                    "description": f"{typ}: {target_nazev}",
                    "source_info": "souvislosti",
                })

    return entities, relationships


def extract_entities_from_fragments(
    fragments: List[Dict],
    provider=None,
    max_fragments: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """Extract entities and relationships from fragment structure and text.

    Structural extraction (SECTION, PART_OF, DEFINITION, cross-references)
    always runs from API data. If provider is given, LLM extracts additional
    entities/relationships from fragment text content.
    """
    entities = []
    relationships = []
    seen_entities = set()
    seen_relationships = set()

    # Build fragment hierarchy for PART_OF relationships
    fragment_map: Dict[str, Dict] = {}
    for frag in fragments:
        frag_id = str(frag.get("fragmentId", frag.get("id", "")))
        if frag_id:
            fragment_map[frag_id] = frag

    # Collect fragment texts for LLM extraction
    fragment_texts: List[Tuple[str, str]] = []

    for frag in fragments:
        frag_type = frag.get("kodTypuFragmentu", "")
        citation = frag.get("zkracenaCitace", "")
        xhtml = frag.get("xhtml", "")
        nazev = frag.get("nazev", "")
        hloubka = frag.get("hloubka", 0)
        frag_id = str(frag.get("id", frag.get("fragmentId", "")))
        parent_id = str(frag.get("nadrazenyFragmentId", ""))

        # Strip HTML tags for text
        text = re.sub(r"<[^>]+>", " ", xhtml) if xhtml else ""
        text = re.sub(r"\s+", " ", text).strip()

        # --- Structural extraction: SECTION entities ---
        section_types = {
            "Paragraf", "Clanek", "Cast", "Hlava", "Dil", "Oddil",
            "Pismeno", "Odstavec", "Bod", "Priloha",
            "Odstavec_Dc", "Pismeno_Dc", "Bod_Dc",
        }
        if frag_type in section_types and citation:
            entity_key = (citation, "SECTION")
            if entity_key not in seen_entities:
                desc = nazev if nazev else ""
                entities.append({
                    "name": citation,
                    "type": "SECTION",
                    "description": desc,
                    "source": f"fragment:{frag_id}",
                })
                seen_entities.add(entity_key)

                # PART_OF relationship to parent
                if parent_id and parent_id in fragment_map:
                    parent_frag = fragment_map[parent_id]
                    parent_citation = parent_frag.get("zkracenaCitace", parent_frag.get("citace", ""))
                    if parent_citation:
                        rel_key = (citation, parent_citation, "PART_OF")
                        if rel_key not in seen_relationships:
                            relationships.append({
                                "source": citation,
                                "target": parent_citation,
                                "type": "PART_OF",
                                "description": f"structural hierarchy (hloubka {hloubka})",
                                "source_info": "fragment_hierarchy",
                            })
                            seen_relationships.add(rel_key)

        # --- Structural extraction: DEFINITION entities ---
        if nazev and ("vymezení pojmů" in nazev.lower() or "definice" in nazev.lower()):
            if citation:
                def_entity_key = (citation, "DEFINITION")
                if def_entity_key not in seen_entities:
                    entities.append({
                        "name": citation,
                        "type": "DEFINITION",
                        "description": f"Definice: {nazev}",
                        "source": f"fragment:{frag_id}",
                    })
                    seen_entities.add(def_entity_key)

        # --- Structural extraction: cross-references ---
        odkazy = frag.get("odkazyZFragmentu", [])
        if isinstance(odkazy, list):
            for odkaz in odkazy:
                cil = odkaz.get("cil", {})
                if not isinstance(cil, dict):
                    continue
                target_url = cil.get("staleUrl", "")
                if not target_url or not citation:
                    continue

                ext_match = re.match(r"^/sb/(\d{4})/(\d+)(?:/|$)", target_url)
                if ext_match:
                    year = ext_match.group(1)
                    number = ext_match.group(2)
                    target_citation = f"{number}/{year} Sb."

                    entity_key = (target_citation, "REGULATION")
                    if entity_key not in seen_entities:
                        entities.append({
                            "name": target_citation,
                            "type": "REGULATION",
                            "description": "Cross-reference target",
                            "source": f"reference:{frag_id}",
                        })
                        seen_entities.add(entity_key)

                    rel_key = (citation, target_citation, "REFERENCES")
                    if rel_key not in seen_relationships:
                        relationships.append({
                            "source": citation,
                            "target": target_citation,
                            "type": "REFERENCES",
                            "description": f"cross-reference from {citation}",
                            "source_info": "fragment_reference",
                        })
                        seen_relationships.add(rel_key)

        # Collect text for LLM extraction
        if text and len(text) > 30:
            fragment_texts.append((frag_id, text))

    # --- LLM-based extraction from fragment text ---
    if provider and fragment_texts:
        if max_fragments > 0:
            fragment_texts = fragment_texts[:max_fragments]
            logger.info(f"    Limited to {len(fragment_texts)} fragments for LLM extraction")

        llm_entities, llm_relationships = extract_entities_from_text_llm(
            fragment_texts, provider
        )

        # Merge LLM results with dedup
        for ent in llm_entities:
            key = (ent["name"].lower(), ent["type"])
            if key not in seen_entities:
                entities.append(ent)
                seen_entities.add(key)

        for rel in llm_relationships:
            key = (rel["source"].lower(), rel["target"].lower(), rel["type"])
            if key not in seen_relationships:
                relationships.append(rel)
                seen_relationships.add(key)

        logger.info(
            f"    After merge: {len(entities)} entities, {len(relationships)} relationships"
        )
    elif not provider:
        logger.info("    No LLM provider — skipping text entity extraction")

    return entities, relationships


# =========================================================================
# Dataset Assembly
# =========================================================================


def build_gt_dataset(
    doc_config: Dict,
    metadata: Dict,
    fragments: List[Dict],
    souvislosti: List[Dict],
    provider=None,
    max_fragments: int = 0,
) -> Dict:
    """Assemble ground-truth dataset for a single document."""

    # Extract entities from all sources
    meta_entities = extract_entities_from_metadata(metadata, doc_config)
    souv_entities, souv_relationships = extract_entities_from_souvislosti(souvislosti)
    frag_entities, frag_relationships = extract_entities_from_fragments(
        fragments, provider=provider, max_fragments=max_fragments
    )

    # Merge entities (deduplicate by name+type)
    all_entities = []
    seen = set()
    for ent in meta_entities + souv_entities + frag_entities:
        key = (ent["name"].lower(), ent["type"])
        if key not in seen:
            all_entities.append(ent)
            seen.add(key)

    # Fix relationship targets: replace __SELF__ with document's own regulation name
    doc_name = meta_entities[0]["name"] if meta_entities else doc_config["citation"]
    all_relationships = []
    for rel in souv_relationships + frag_relationships:
        if rel["target"] == "__SELF__":
            rel["target"] = doc_name
        # Flip AMENDS direction based on souvislosti semantics
        if rel.get("source_info") == "souvislosti":
            if rel["type"] == "AMENDS":
                rel["source"], rel["target"] = rel["target"], rel["source"]
        all_relationships.append(rel)

    # Deduplicate relationships
    deduped_rels = []
    seen_rels = set()
    for rel in all_relationships:
        key = (rel["source"].lower(), rel["target"].lower(), rel["type"])
        if key not in seen_rels:
            deduped_rels.append(rel)
            seen_rels.add(key)

    return {
        "document_id": doc_config["id"],
        "metadata": {
            "kodDokumentuSbirky": doc_config["citation"],
            "nazev": metadata.get("nazev", doc_config["name"]),
            "uplnaCitace": metadata.get("uplnaCitace", ""),
            "staleUrl": doc_config["staleUrl"],
        },
        "entities": all_entities,
        "relationships": deduped_rels,
        "fragments_count": len(fragments),
        "entity_type_counts": _count_by_type(all_entities, "type"),
        "relationship_type_counts": _count_by_type(deduped_rels, "type"),
    }


def _count_by_type(items: List[Dict], key: str) -> Dict[str, int]:
    """Count items by a key."""
    counts: Dict[str, int] = {}
    for item in items:
        t = item.get(key, "UNKNOWN")
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


def build_combined_gt(datasets: List[Dict]) -> Dict:
    """Merge GT datasets from multiple documents for cross-document dedup testing."""
    all_entities = []
    all_relationships = []
    seen_entities = set()
    seen_rels = set()

    for ds in datasets:
        doc_id = ds["document_id"]
        for ent in ds["entities"]:
            key = (ent["name"].lower(), ent["type"])
            if key not in seen_entities:
                ent_copy = {**ent, "document_id": doc_id}
                all_entities.append(ent_copy)
                seen_entities.add(key)
        for rel in ds["relationships"]:
            key = (rel["source"].lower(), rel["target"].lower(), rel["type"])
            if key not in seen_rels:
                rel_copy = {**rel, "document_id": doc_id}
                all_relationships.append(rel_copy)
                seen_rels.add(key)

    # Identify expected cross-document dedup groups
    entity_name_docs: Dict[Tuple[str, str], List[str]] = {}
    for ds in datasets:
        doc_id = ds["document_id"]
        for ent in ds["entities"]:
            key = (ent["name"].lower(), ent["type"])
            entity_name_docs.setdefault(key, [])
            if doc_id not in entity_name_docs[key]:
                entity_name_docs[key].append(doc_id)

    expected_dedup_groups = [
        {"name": key[0], "type": key[1], "documents": docs}
        for key, docs in entity_name_docs.items()
        if len(docs) > 1
    ]

    return {
        "documents": [ds["document_id"] for ds in datasets],
        "entities": all_entities,
        "relationships": all_relationships,
        "entity_type_counts": _count_by_type(all_entities, "type"),
        "relationship_type_counts": _count_by_type(all_relationships, "type"),
        "expected_dedup_groups": expected_dedup_groups,
    }


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="e-Sbírka Ground Truth Dataset Builder"
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="LLM model for text entity extraction (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--max-fragments",
        type=int,
        default=0,
        help="Max fragments per document for LLM extraction (0 = all, default: 0)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM extraction, only use structural API data",
    )
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(SEP)
    logger.info("e-Sbírka Ground Truth Dataset Builder")
    logger.info(SEP)

    # Initialize LLM provider
    provider = None
    if not args.no_llm:
        logger.info(f"LLM model: {args.model}")
        provider = create_provider(args.model)
    else:
        logger.info("LLM extraction disabled (--no-llm)")

    datasets = []

    with httpx.Client() as client:
        for doc_config in DOCUMENTS:
            logger.info(f"\n{SEP}")
            logger.info(f"Processing: {doc_config['citation']} ({doc_config['name']})")
            logger.info(SEP)

            # 1. Fetch metadata
            metadata = fetch_document_metadata(client, doc_config["staleUrl"])

            # 2. Fetch fragments
            fragments = fetch_fragments(client, doc_config["staleUrl"])
            logger.info(f"  Total fragments: {len(fragments)}")

            # 3. Fetch souvislosti
            souvislosti = fetch_souvislosti(client, doc_config["staleUrl"])
            logger.info(f"  Total souvislosti: {len(souvislosti)}")

            # 4. Download PDF
            links = fetch_download_links(client, doc_config["staleUrl"])
            pdf_path = download_pdf(client, doc_config, metadata, links)
            if pdf_path:
                logger.info(f"  PDF: {pdf_path}")

            # 5. Build GT dataset
            gt = build_gt_dataset(
                doc_config, metadata, fragments, souvislosti,
                provider=provider,
                max_fragments=args.max_fragments,
            )
            datasets.append(gt)

            # 6. Save per-document GT
            out_path = BENCHMARK_DIR / f"gt_{doc_config['id']}.json"
            out_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"  Saved GT: {out_path}")
            logger.info(f"  Entities: {len(gt['entities'])} ({gt['entity_type_counts']})")
            logger.info(f"  Relationships: {len(gt['relationships'])} ({gt['relationship_type_counts']})")

    # 7. Build combined GT
    combined = build_combined_gt(datasets)
    combined_path = BENCHMARK_DIR / "gt_combined.json"
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"\n{SEP}")
    logger.info(f"Combined GT: {combined_path}")
    logger.info(f"  Total entities: {len(combined['entities'])}")
    logger.info(f"  Total relationships: {len(combined['relationships'])}")
    logger.info(f"  Expected dedup groups: {len(combined['expected_dedup_groups'])}")
    for group in combined["expected_dedup_groups"][:10]:
        logger.info(f"    {group['name']} [{group['type']}] in {group['documents']}")

    logger.info(f"\n{SEP}")
    logger.info("Done! Files in data/esbirka_benchmark/:")
    for f in sorted(BENCHMARK_DIR.iterdir()):
        logger.info(f"  {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
