#!/usr/bin/env python3
"""
Generate synthetic eval dataset from current document indexation.

This script:
1. Queries PostgreSQL for all current chunks in vectors.layer3
2. Samples chunks strategically (stratified by document)
3. Uses gpt-4o to generate natural Czech questions for each chunk
4. Outputs synthetic_eval_dataset.json for conformal prediction calibration

Usage:
    uv run python rag_confidence/generate_synthetic_dataset.py
    uv run python rag_confidence/generate_synthetic_dataset.py --dry-run
    uv run python rag_confidence/generate_synthetic_dataset.py --target-queries 500
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import psycopg
from openai import OpenAI
from psycopg.rows import dict_row
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
TARGET_QUERIES = 1000
MIN_CHUNK_LENGTH = 100  # Minimum characters for a chunk to be eligible
MAX_CHUNK_LENGTH = 3000  # Truncate very long chunks in prompts
QUESTIONS_PER_CHUNK = 1  # Generate exactly 1 question per chunk for diversity
GENERATION_MODEL = "gpt-4o"
BATCH_SIZE = 10  # Process chunks in batches for progress tracking

# Cost tracking (gpt-4o pricing as of Jan 2026)
# https://openai.com/api/pricing/
GPT4O_INPUT_COST_PER_1M = 2.50  # $2.50 per 1M input tokens
GPT4O_OUTPUT_COST_PER_1M = 10.00  # $10.00 per 1M output tokens


class CostTracker:
    """Track OpenAI API usage and costs."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add usage from a single API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    def get_cost(self) -> float:
        """Calculate total cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * GPT4O_INPUT_COST_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * GPT4O_OUTPUT_COST_PER_1M
        return input_cost + output_cost

    def get_summary(self) -> dict:
        """Get usage summary."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.get_cost(), 4),
        }

QUESTION_PROMPT = """Vytvoř JEDNU přirozenou otázku v češtině, kterou by uživatel mohl položit
a kterou by tento konkrétní chunk dokumentu zodpověděl.

Chunk ID: {chunk_id}
Dokument: {document_id}
Sekce: {section_title}

Text chunku:
{content}

Pravidla:
- Ptej se na PODSTATU a OBSAH - požadavky, povinnosti, podmínky, postupy, zásady, definice, limity, lhůty
- NIKDY se neptej na textové změny typu "jaké číslo bylo změněno" nebo "jaká změna byla provedena"
- Používej přirozený jazyk jako právník nebo odborník v oboru
- Otázka musí být zodpověditelná tímto chunkem

DŮLEŽITÉ: Hoď si v hlavě kostkou 1-10 a podle výsledku vyber typ otázky:
1-2: "Jaké [požadavky/podmínky/limity]..." (co platí)
3: "Kdo [je odpovědný/musí/může]..." (odpovědnost)
4: "Kdy [je nutné/musí být/nastává]..." (časové podmínky)
5: "V jakých případech..." (situační podmínky)
6: "Jak se [stanovuje/provádí/určuje]..." (postupy)
7: "Kde [se nachází/probíhá/je uloženo]..." (místo)
8: "Kolik [činí/je/musí být]..." (hodnoty, limity)
9: "Proč [je důležité/se vyžaduje]..." (odůvodnění)
10: "Co [obsahuje/zahrnuje/stanovuje]..." (obsah)

Příklady DOBRÝCH otázek (každá jiný typ):
- "Jaké bezpečnostní limity platí pro kontaminaci povrchů alfa zářiči?"
- "Kdo je oprávněn provádět kontrolu jaderných materiálů?"
- "Kdy musí být podána žádost o prodloužení povolení?"
- "V jakých případech není vyžadováno povolení k přepravě?"
- "Jak se stanovuje výše pojistného krytí pro provozovatele?"
- "Kde se provádí měření příkonu dávkového ekvivalentu?"
- "Kolik činí minimální limit pojištění odpovědnosti?"
- "Proč je nutné dodržovat limity kontaminace při přepravě?"
- "Co musí obsahovat žádost o povolení k nakládání s RAO?"

Příklady ŠPATNÝCH otázek (NEPOUŽÍVEJ):
- "Jaká změna byla provedena v příloze č. 1?"
- "Na jaké číslo bylo změněno číslo 37?"
- "Který bod byl upraven v této novele?"

Vrať POUZE validní JSON (bez markdown):
{{
  "question": "Tvoje otázka zde?"
}}"""


def get_db_connection():
    """Get PostgreSQL connection from environment."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg.connect(db_url)

    return psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "sujbot"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def get_openai_client() -> OpenAI:
    """Initialize OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")
    return OpenAI(api_key=api_key)


def load_all_chunks(conn) -> list[dict]:
    """Load all chunks from PostgreSQL layer3."""
    logger.info("Loading chunks from PostgreSQL vectors.layer3...")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT
                chunk_id,
                document_id,
                content,
                section_title,
                section_path,
                hierarchical_path
            FROM vectors.layer3
            WHERE section_path IS NULL OR section_path NOT LIKE 'Literatura%'
            ORDER BY document_id, chunk_id
        """)
        chunks = cur.fetchall()

    logger.info(f"Loaded {len(chunks)} chunks from database")
    return chunks


def sample_chunks_stratified(
    chunks: list[dict], target_count: int, min_length: int = MIN_CHUNK_LENGTH
) -> list[dict]:
    """
    Sample chunks with stratified sampling across documents.

    Ensures representation from all documents while filtering short chunks.
    """
    # Filter by minimum length
    eligible_chunks = [
        c for c in chunks if c["content"] and len(c["content"]) >= min_length
    ]
    logger.info(
        f"Eligible chunks after length filter (>={min_length} chars): {len(eligible_chunks)}"
    )

    # Group by document
    chunks_by_doc = defaultdict(list)
    for chunk in eligible_chunks:
        chunks_by_doc[chunk["document_id"]].append(chunk)

    n_docs = len(chunks_by_doc)
    logger.info(f"Found {n_docs} unique documents")

    # Calculate chunks per document (proportional to doc size, min 1)
    total_eligible = len(eligible_chunks)
    sampled = []

    # First pass: proportional allocation
    for doc_id, doc_chunks in chunks_by_doc.items():
        # Proportion of total chunks this doc represents
        proportion = len(doc_chunks) / total_eligible
        # Allocate proportionally, at least 1 chunk per doc
        n_to_sample = max(1, int(target_count * proportion))
        n_to_sample = min(n_to_sample, len(doc_chunks))

        # Random sample from this document
        doc_sample = random.sample(doc_chunks, n_to_sample)
        sampled.extend(doc_sample)

    # Adjust if we have too few or too many
    if len(sampled) < target_count:
        # Add more from remaining pool
        remaining = [c for c in eligible_chunks if c not in sampled]
        extra_needed = target_count - len(sampled)
        if remaining:
            extra = random.sample(remaining, min(extra_needed, len(remaining)))
            sampled.extend(extra)
    elif len(sampled) > target_count:
        # Trim to target
        sampled = random.sample(sampled, target_count)

    logger.info(f"Sampled {len(sampled)} chunks across {n_docs} documents")

    # Log distribution
    final_by_doc = defaultdict(int)
    for c in sampled:
        final_by_doc[c["document_id"]] += 1
    logger.info(f"Document distribution: min={min(final_by_doc.values())}, max={max(final_by_doc.values())}, avg={sum(final_by_doc.values())/len(final_by_doc):.1f}")

    return sampled


def generate_question_for_chunk(
    client: OpenAI,
    chunk: dict,
    cost_tracker: CostTracker,
) -> str | None:
    """Generate a single synthetic question for a chunk using gpt-4o."""
    content = chunk["content"]
    if len(content) > MAX_CHUNK_LENGTH:
        content = content[:MAX_CHUNK_LENGTH] + "..."

    prompt = QUESTION_PROMPT.format(
        chunk_id=chunk["chunk_id"],
        document_id=chunk["document_id"],
        section_title=chunk.get("section_title") or "N/A",
        content=content,
    )

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,  # Higher temperature for more diverse questions
            max_tokens=150,
        )

        # Track token usage
        if response.usage:
            cost_tracker.add_usage(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

        response_text = response.choices[0].message.content or ""

        # Parse JSON response
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start >= 0 and end > start:
            json_str = response_text[start : end + 1]
            data = json.loads(json_str)
            # Handle both "question" (new) and "questions" (old) formats
            if "question" in data:
                return data["question"]
            elif "questions" in data and data["questions"]:
                return data["questions"][0]

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for {chunk['chunk_id']}: {e}")
    except Exception as e:
        logger.warning(f"API error for {chunk['chunk_id']}: {e}")

    return None


def build_dataset(
    sampled_chunks: list[dict],
    client: OpenAI,
    target_queries: int,
    dry_run: bool = False,
) -> tuple[dict, CostTracker]:
    """Build the synthetic eval dataset with 1 question per chunk."""
    queries = []
    query_id = 1
    cost_tracker = CostTracker()

    if dry_run:
        logger.info("DRY RUN - generating sample questions for first 10 chunks only")
        sampled_chunks = sampled_chunks[:10]

    progress = tqdm(sampled_chunks, desc="Generating questions")

    for chunk in progress:
        q_text = generate_question_for_chunk(client, chunk, cost_tracker)

        if q_text and len(q_text) >= 10:
            query = {
                "query_id": f"synthetic_{query_id:04d}",
                "query_text": q_text,
                "relevant_chunk_ids": [chunk["chunk_id"]],
                "source_document": chunk["document_id"],
            }
            queries.append(query)
            query_id += 1

        # Stop if we have enough
        if len(queries) >= target_queries:
            break

        # Update progress bar with cost
        cost = cost_tracker.get_cost()
        progress.set_postfix({"queries": len(queries), "cost": f"${cost:.2f}"})

    # Build final dataset
    dataset = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_queries": len(queries),
            "n_chunks_sampled": len(sampled_chunks),
            "embedding_model": "Qwen/Qwen3-Embedding-8B",
            "generation_model": GENERATION_MODEL,
            "target_queries": target_queries,
            "api_cost": cost_tracker.get_summary(),
        },
        "queries": queries,
    }

    return dataset, cost_tracker


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic eval dataset for RAG confidence calibration"
    )
    parser.add_argument(
        "--target-queries",
        type=int,
        default=TARGET_QUERIES,
        help=f"Target number of queries (default: {TARGET_QUERIES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: rag_confidence/synthetic_eval_dataset.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate only 5 sample questions for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Output path
    data_dir = Path(__file__).parent
    output_path = args.output or data_dir / "synthetic_eval_dataset.json"

    # Connect to database
    logger.info("Connecting to PostgreSQL...")
    conn = get_db_connection()

    try:
        # Load chunks
        chunks = load_all_chunks(conn)
    finally:
        conn.close()

    if not chunks:
        logger.error("No chunks found in database!")
        return

    # Sample chunks (1 chunk = 1 question, add 5% buffer for failed generations)
    chunks_needed = int(args.target_queries * 1.05)
    sampled = sample_chunks_stratified(chunks, chunks_needed)

    # Shuffle to ensure random order (stratified sampling groups by document)
    random.shuffle(sampled)

    # Initialize OpenAI client
    logger.info("Initializing OpenAI client...")
    client = get_openai_client()

    # Generate dataset
    logger.info(f"Generating {args.target_queries} synthetic queries using {GENERATION_MODEL}...")
    dataset, cost_tracker = build_dataset(
        sampled, client, args.target_queries, dry_run=args.dry_run
    )

    # Save
    logger.info(f"Saving dataset to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Summary
    cost_summary = cost_tracker.get_summary()
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Queries generated:    {dataset['metadata']['n_queries']}")
    print(f"Chunks sampled:       {dataset['metadata']['n_chunks_sampled']}")
    print(f"Generation model:     {dataset['metadata']['generation_model']}")
    print(f"Output file:          {output_path}")
    print("")
    print("API COST SUMMARY:")
    print(f"  Total requests:     {cost_summary['total_requests']}")
    print(f"  Input tokens:       {cost_summary['total_input_tokens']:,}")
    print(f"  Output tokens:      {cost_summary['total_output_tokens']:,}")
    print(f"  Total tokens:       {cost_summary['total_tokens']:,}")
    print(f"  Estimated cost:     ${cost_summary['estimated_cost_usd']:.4f}")
    print("=" * 60)

    if not args.dry_run:
        print("\nNext steps:")
        print("  1. uv run python rag_confidence/regenerate_matrices.py")
        print("  2. uv run python rag_confidence/calibrate.py --alpha 0.1 --evaluate")


if __name__ == "__main__":
    main()
