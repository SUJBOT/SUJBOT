#!/usr/bin/env python3
"""
Criteria Evaluation Script for SUJBOT Single-Agent (VL mode).

Runs 2191 Czech nuclear safety criteria questions through SingleAgentRunner,
uses an LLM judge to classify each response as yes/no/dont_know, and produces
a scored report with confusion matrix, precision, recall, F1.

Usage:
    # Quick smoke test (5 entries)
    uv run python benchmark_criteria/criteria_eval.py --limit 5

    # 50 random entries (representative sample)
    uv run python benchmark_criteria/criteria_eval.py --limit 50 --shuffle

    # Full run (2191 entries, ~9 hours with Sonnet 4.5)
    uv run python benchmark_criteria/criteria_eval.py

    # Budget run with Haiku 4.5
    uv run python benchmark_criteria/criteria_eval.py --model claude-haiku-4-5-20251001
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from litellm import acompletion

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nest_asyncio

nest_asyncio.apply()

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Classification Judge
# =============================================================================

CLASSIFICATION_PROMPT = """\
You are a classification judge. Given a QUESTION and an AGENT ANSWER, classify \
the agent's answer into exactly one of three categories:

- "yes": The agent's answer indicates that the document DOES contain / DOES satisfy \
the requirement described in the question.
- "no": The agent's answer indicates that the document does NOT contain / does NOT \
satisfy the requirement described in the question.
- "dont_know": The agent could not find the information, explicitly says it doesn't \
know, or the answer is too ambiguous to classify.

Rules:
- Focus on the SUBSTANCE of the answer, not exact wording.
- If the agent found relevant information and confirms the requirement is met → "yes"
- If the agent explicitly states the information is missing or requirement is not met → "no"
- If the agent hedges extensively, says it cannot find info, or answer is unrelated → "dont_know"
- The question is in Czech. The answer may be in Czech or English.

Return ONLY a JSON object with this exact schema:
{{"classification": "yes"|"no"|"dont_know", "reasoning": "<brief explanation>"}}

QUESTION:
{question}

AGENT ANSWER:
{agent_answer}
"""


async def classify_answer(
    question: str,
    agent_answer: str,
    judge_model: str = "gpt-4o-mini",
) -> dict[str, str]:
    """Use LLM judge to classify agent answer as yes/no/dont_know."""
    truncated = agent_answer[:2000] if len(agent_answer) > 2000 else agent_answer
    prompt = CLASSIFICATION_PROMPT.format(question=question, agent_answer=truncated)

    try:
        resp = await acompletion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)
        classification = result.get("classification", "dont_know")
        if classification not in ("yes", "no", "dont_know"):
            classification = "dont_know"
        return {
            "classification": classification,
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        logger.warning(f"Judge classification failed: {e}")
        return {"classification": "dont_know", "reasoning": f"Judge error: {e}"}


# =============================================================================
# Criteria Runner
# =============================================================================


class CriteriaRunner:
    """Runs criteria questions through SingleAgentRunner and judges results."""

    def __init__(
        self,
        config: dict[str, Any],
        model: str,
        judge_model: str,
        prime_document: str,
    ):
        self.config = config
        self.model = model
        self.judge_model = judge_model
        self.prime_document = prime_document
        self.runner = None
        self._initialized = False

    async def initialize(self):
        """Initialize the SingleAgentRunner."""
        if self._initialized:
            return

        from src.single_agent.runner import SingleAgentRunner

        self.runner = SingleAgentRunner(self.config)
        await self.runner.initialize()
        self._initialized = True
        logger.info(f"SingleAgentRunner initialized (model={self.model})")

    def _build_priming_history(self) -> list[dict[str, str]]:
        """Build conversation history that restricts agent to the target document."""
        return [
            {
                "role": "user",
                "content": (
                    f"Odpovídej výhradně na základě dokumentu {self.prime_document}. "
                    f"Hledej pouze v tomto dokumentu."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Rozumím, budu odpovídat výhradně na základě dokumentu "
                    f"{self.prime_document}."
                ),
            },
        ]

    async def run_single(self, entry: dict[str, Any], index: int) -> dict[str, Any]:
        """Run one question through agent → judge → scoring."""
        question = entry["question"]
        expected = entry["expected_answer"]
        start_time = time.time()

        # Run agent with fresh priming history
        agent_result = None
        async for event in self.runner.run_query(
            query=question,
            model=self.model,
            stream_progress=False,
            conversation_history=self._build_priming_history(),
        ):
            if event.get("type") == "final":
                agent_result = event

        elapsed = time.time() - start_time

        if agent_result is None:
            agent_result = {
                "success": False,
                "final_answer": "ERROR: No response from agent",
                "total_cost_cents": 0,
                "tools_used": [],
            }

        agent_answer = agent_result.get("final_answer", "")
        agent_success = agent_result.get("success", False)
        agent_cost = agent_result.get("total_cost_cents", 0)
        tools_used = agent_result.get("tools_used", [])

        # Judge classification
        judge_result = await classify_answer(question, agent_answer, self.judge_model)
        classification = judge_result["classification"]

        # Score
        if classification == "dont_know":
            outcome = "abstained"
        elif classification == expected:
            outcome = "correct"
        else:
            outcome = "incorrect"

        result = {
            "index": index,
            "criterion_id": entry.get("criterion_id"),
            "snippet_id": entry.get("snippet_id"),
            "binding": entry.get("binding", ""),
            "is_negated": entry.get("is_negated", False),
            "question": question,
            "expected_answer": expected,
            "agent_answer": agent_answer[:500],
            "agent_success": agent_success,
            "agent_cost_cents": agent_cost,
            "tools_used": tools_used,
            "classification": classification,
            "judge_reasoning": judge_result["reasoning"],
            "outcome": outcome,
            "elapsed_seconds": round(elapsed, 1),
        }

        status = {"correct": "+", "incorrect": "X", "abstained": "?"}[outcome]
        logger.info(
            f"[{status}] #{index} (crit={entry.get('criterion_id')}) "
            f"{outcome} | classified={classification} expected={expected} | "
            f"{elapsed:.1f}s ${agent_cost / 100:.3f}"
        )

        return result

    async def shutdown(self):
        """Clean shutdown."""
        if self.runner:
            try:
                await self.runner.shutdown_async()
            except Exception as e:
                logger.warning(f"Error during shutdown: {e}")


# =============================================================================
# Scoring
# =============================================================================


def compute_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute confusion matrix, precision, recall, F1 from results."""
    # Confusion matrix (positive class = "yes")
    tp = fp = fn = tn = 0
    abstained = 0

    for r in results:
        if r["outcome"] == "abstained":
            abstained += 1
            continue
        expected = r["expected_answer"]
        classified = r["classification"]
        if expected == "yes" and classified == "yes":
            tp += 1
        elif expected == "no" and classified == "yes":
            fp += 1
        elif expected == "yes" and classified == "no":
            fn += 1
        elif expected == "no" and classified == "no":
            tn += 1

    total = len(results)
    decided = tp + fp + fn + tn
    correct = tp + tn
    incorrect = fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = correct / decided if decided > 0 else 0.0
    abstention_rate = abstained / total if total > 0 else 0.0

    # Breakdown by positive vs negated
    positive = [r for r in results if not r.get("is_negated", False)]
    negated = [r for r in results if r.get("is_negated", False)]

    def subset_accuracy(subset: list[dict]) -> dict[str, Any]:
        c = sum(1 for r in subset if r["outcome"] == "correct")
        i = sum(1 for r in subset if r["outcome"] == "incorrect")
        a = sum(1 for r in subset if r["outcome"] == "abstained")
        d = c + i
        return {
            "total": len(subset),
            "correct": c,
            "incorrect": i,
            "abstained": a,
            "accuracy": c / d if d > 0 else 0.0,
        }

    # Breakdown by binding
    povinne = [r for r in results if r.get("binding") == "Povinné"]
    doporucene = [r for r in results if r.get("binding") == "Doporučené"]

    # Per-criterion aggregation
    criteria_stats: dict[int, dict] = {}
    for r in results:
        cid = r.get("criterion_id")
        if cid is None:
            continue
        if cid not in criteria_stats:
            criteria_stats[cid] = {"total": 0, "correct": 0, "incorrect": 0, "abstained": 0}
        criteria_stats[cid]["total"] += 1
        criteria_stats[cid][r["outcome"]] += 1

    # Sort by worst performing (most incorrect)
    worst_criteria = sorted(
        criteria_stats.items(),
        key=lambda x: (-x[1]["incorrect"], -x[1]["total"]),
    )[:10]

    return {
        "total": total,
        "decided": decided,
        "correct": correct,
        "incorrect": incorrect,
        "abstained": abstained,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "abstention_rate": abstention_rate,
        "positive_split": subset_accuracy(positive),
        "negated_split": subset_accuracy(negated),
        "binding_povinne": subset_accuracy(povinne),
        "binding_doporucene": subset_accuracy(doporucene),
        "worst_criteria": worst_criteria,
    }


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(
    results: list[dict[str, Any]],
    scores: dict[str, Any],
    args: argparse.Namespace,
    total_elapsed: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate markdown and JSON reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    n = len(results)
    suffix = f"_{n}examples" if args.limit else "_full"

    # Cost stats
    total_cost = sum(r.get("agent_cost_cents", 0) for r in results)
    avg_cost = total_cost / n if n > 0 else 0
    avg_time = total_elapsed / n if n > 0 else 0

    cm = scores["confusion_matrix"]

    # Markdown report
    md_lines = [
        f"# Criteria Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Agent model:** {args.model}",
        f"**Judge model:** {args.judge_model}",
        f"**Document:** {args.prime_document}",
        f"**Entries evaluated:** {n}",
        f"**Total time:** {total_elapsed / 60:.1f} min",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy (excl. abstentions) | **{scores['accuracy']:.1%}** |",
        f"| Precision (yes) | {scores['precision']:.3f} |",
        f"| Recall (yes) | {scores['recall']:.3f} |",
        f"| F1 Score | **{scores['f1']:.3f}** |",
        f"| Abstention rate | {scores['abstention_rate']:.1%} |",
        f"| Correct | {scores['correct']} |",
        f"| Incorrect | {scores['incorrect']} |",
        f"| Abstained | {scores['abstained']} |",
        "",
        "## Confusion Matrix",
        "",
        "```",
        "                 Predicted YES    Predicted NO",
        f"Expected YES     TP = {cm['tp']:<12}  FN = {cm['fn']}",
        f"Expected NO      FP = {cm['fp']:<12}  TN = {cm['tn']}",
        "```",
        "",
        "## Positive vs Negated Split",
        "",
        "| Split | Total | Correct | Incorrect | Abstained | Accuracy |",
        "|-------|-------|---------|-----------|-----------|----------|",
    ]

    for label, key in [("Positive", "positive_split"), ("Negated", "negated_split")]:
        s = scores[key]
        md_lines.append(
            f"| {label} | {s['total']} | {s['correct']} | "
            f"{s['incorrect']} | {s['abstained']} | {s['accuracy']:.1%} |"
        )

    md_lines += [
        "",
        "## Binding Split",
        "",
        "| Binding | Total | Correct | Incorrect | Abstained | Accuracy |",
        "|---------|-------|---------|-----------|-----------|----------|",
    ]

    for label, key in [("Povinné", "binding_povinne"), ("Doporučené", "binding_doporucene")]:
        s = scores[key]
        md_lines.append(
            f"| {label} | {s['total']} | {s['correct']} | "
            f"{s['incorrect']} | {s['abstained']} | {s['accuracy']:.1%} |"
        )

    # Worst criteria
    if scores["worst_criteria"]:
        md_lines += [
            "",
            "## Top 10 Worst-Performing Criteria",
            "",
            "| Criterion ID | Total | Correct | Incorrect | Abstained |",
            "|-------------|-------|---------|-----------|-----------|",
        ]
        for cid, stats in scores["worst_criteria"]:
            md_lines.append(
                f"| {cid} | {stats['total']} | {stats['correct']} | "
                f"{stats['incorrect']} | {stats['abstained']} |"
            )

    # Sample incorrect answers
    incorrect_samples = [r for r in results if r["outcome"] == "incorrect"][:10]
    if incorrect_samples:
        md_lines += [
            "",
            "## Sample Incorrect Answers (first 10)",
            "",
        ]
        for r in incorrect_samples:
            q_short = r["question"][:150] + "..." if len(r["question"]) > 150 else r["question"]
            md_lines += [
                f"### Criterion {r['criterion_id']} (expected={r['expected_answer']}, "
                f"classified={r['classification']})",
                "",
                f"**Q:** {q_short}",
                "",
                f"**Agent:** {r['agent_answer'][:300]}...",
                "",
                f"**Judge:** {r['judge_reasoning']}",
                "",
            ]

    # Cost and timing
    md_lines += [
        "## Cost & Timing",
        "",
        f"- **Total agent cost:** ${total_cost / 100:.2f}",
        f"- **Avg cost per query:** ${avg_cost / 100:.4f}",
        f"- **Avg time per query:** {avg_time:.1f}s",
        f"- **Total elapsed:** {total_elapsed / 60:.1f} min",
        "",
    ]

    md_content = "\n".join(md_lines) + "\n"
    md_path = output_dir / f"{timestamp}_criteria_eval{suffix}.md"
    md_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Markdown report: {md_path}")

    # JSON report
    json_data = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "agent_model": args.model,
            "judge_model": args.judge_model,
            "prime_document": args.prime_document,
            "total_entries": n,
            "total_elapsed_seconds": round(total_elapsed, 1),
            "limit": args.limit,
            "offset": args.offset,
            "shuffle": args.shuffle,
        },
        "scores": {
            "accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "abstention_rate": scores["abstention_rate"],
            "confusion_matrix": cm,
            "correct": scores["correct"],
            "incorrect": scores["incorrect"],
            "abstained": scores["abstained"],
            "positive_split": scores["positive_split"],
            "negated_split": scores["negated_split"],
            "binding_povinne": scores["binding_povinne"],
            "binding_doporucene": scores["binding_doporucene"],
        },
        "cost": {
            "total_agent_cents": total_cost,
            "avg_per_query_cents": avg_cost,
        },
        "results": results,
    }

    json_path = output_dir / f"{timestamp}_criteria_eval{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON report: {json_path}")

    return md_path, json_path


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run criteria evaluation on SUJBOT single-agent (VL mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=PROJECT_ROOT / "benchmark_criteria" / "criteria_dataset.json",
        help="Path to criteria dataset JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to evaluate",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N entries (for resuming)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Agent model (must support VL/vision). Default: claude-sonnet-4-5-20250929",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="LLM judge model for classification (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--prime-document",
        type=str,
        default="BZ_VR1",
        help="Document name to restrict agent to (default: BZ_VR1)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize entry order (seeded for reproducibility)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config_path = PROJECT_ROOT / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load dataset
    with open(args.dataset_path) as f:
        dataset = json.load(f)
    entries = dataset["entries"]
    logger.info(f"Loaded {len(entries)} entries from {args.dataset_path}")

    # Apply offset
    if args.offset > 0:
        entries = entries[args.offset :]
        logger.info(f"Skipped first {args.offset} entries (offset)")

    # Shuffle
    if args.shuffle:
        rng = random.Random(42)
        entries = list(entries)
        rng.shuffle(entries)
        logger.info("Shuffled entries (seed=42)")

    # Apply limit
    if args.limit:
        entries = entries[: args.limit]
        logger.info(f"Limited to {len(entries)} entries")

    logger.info(
        f"Running {len(entries)} entries | agent={args.model} | "
        f"judge={args.judge_model} | doc={args.prime_document}"
    )

    # Initialize runner
    runner = CriteriaRunner(
        config=config,
        model=args.model,
        judge_model=args.judge_model,
        prime_document=args.prime_document,
    )
    await runner.initialize()

    # Run evaluations sequentially
    results: list[dict[str, Any]] = []
    total_start = time.time()

    try:
        for i, entry in enumerate(entries):
            try:
                result = await runner.run_single(entry, index=i)
                results.append(result)
            except Exception as e:
                logger.error(f"Entry {i} failed: {e}", exc_info=True)
                results.append(
                    {
                        "index": i,
                        "criterion_id": entry.get("criterion_id"),
                        "snippet_id": entry.get("snippet_id"),
                        "binding": entry.get("binding", ""),
                        "is_negated": entry.get("is_negated", False),
                        "question": entry["question"],
                        "expected_answer": entry["expected_answer"],
                        "agent_answer": f"ERROR: {e}",
                        "agent_success": False,
                        "agent_cost_cents": 0,
                        "tools_used": [],
                        "classification": "dont_know",
                        "judge_reasoning": f"Agent error: {e}",
                        "outcome": "abstained",
                        "elapsed_seconds": 0,
                    }
                )

            # Progress summary every 10 entries
            if (i + 1) % 10 == 0:
                c = sum(1 for r in results if r["outcome"] == "correct")
                x = sum(1 for r in results if r["outcome"] == "incorrect")
                a = sum(1 for r in results if r["outcome"] == "abstained")
                elapsed = time.time() - total_start
                logger.info(
                    f"--- Progress: {i + 1}/{len(entries)} | "
                    f"correct={c} incorrect={x} abstained={a} | "
                    f"{elapsed / 60:.1f}min elapsed ---"
                )
    finally:
        await runner.shutdown()

    total_elapsed = time.time() - total_start

    # Compute scores
    scores = compute_scores(results)

    # Print summary
    cm = scores["confusion_matrix"]
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Entries: {scores['total']} | Decided: {scores['decided']}")
    logger.info(
        f"Correct: {scores['correct']} | Incorrect: {scores['incorrect']} | "
        f"Abstained: {scores['abstained']}"
    )
    logger.info(f"Accuracy: {scores['accuracy']:.1%} | F1: {scores['f1']:.3f}")
    logger.info(
        f"Precision: {scores['precision']:.3f} | Recall: {scores['recall']:.3f}"
    )
    logger.info(f"Confusion: TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
    logger.info(f"Time: {total_elapsed / 60:.1f} min")

    # Generate reports
    output_dir = PROJECT_ROOT / "benchmark_criteria"
    md_path, json_path = generate_report(results, scores, args, total_elapsed, output_dir)

    logger.info(f"Reports: {md_path}")
    logger.info(f"         {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
