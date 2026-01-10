#!/usr/bin/env python3
"""
Extraction Evaluation Script

Compares ground truth JSON with OCR extraction output using multiple metrics:
- Text: Normalized Edit Distance (NED), BLEU
- Hierarchy: TEDS, TEDS-S, Level Accuracy, Parent-Child F1
- Segmentation: Boundary Precision/Recall/F1

Usage:
    uv run python scripts/extraction_eval.py \
        --gt extraction_dataset/BZ_VR1_gt.json \
        --pred output/BZ_VR1/phase1_extraction.json \
        --output results/extraction_eval_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Tree Edit Distance
try:
    import zss
except ImportError:
    zss = None  # type: ignore

# Fuzzy matching
try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None  # type: ignore

# BLEU score
try:
    import sacrebleu
except ImportError:
    sacrebleu = None  # type: ignore


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Section:
    """Represents a document section."""

    section_id: str
    title: str
    content: str
    level: int
    path: str
    page_number: int = 0
    content_length: int = 0
    depth: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "Section":
        return cls(
            section_id=data.get("section_id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            level=data.get("level", 0),
            path=data.get("path", ""),
            page_number=data.get("page_number", 0),
            content_length=data.get("content_length", 0),
            depth=data.get("depth", 1),
        )


@dataclass
class Document:
    """Represents a parsed document."""

    document_id: str
    source_path: str
    num_sections: int
    hierarchy_depth: int
    num_roots: int
    sections: list[Section] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> "Document":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sections = [Section.from_dict(s) for s in data.get("sections", [])]

        return cls(
            document_id=data.get("document_id", ""),
            source_path=data.get("source_path", ""),
            num_sections=data.get("num_sections", len(sections)),
            hierarchy_depth=data.get("hierarchy_depth", 0),
            num_roots=data.get("num_roots", 0),
            sections=sections,
        )


@dataclass
class EvalResult:
    """Evaluation results."""

    hierarchy: dict = field(default_factory=dict)
    text: dict = field(default_factory=dict)
    segmentation: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    overall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "hierarchy": self.hierarchy,
            "text": self.text,
            "segmentation": self.segmentation,
            "metadata": self.metadata,
            "overall": self.overall,
        }


# =============================================================================
# Text Metrics
# =============================================================================


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_edit_distance(pred: str, ref: str) -> float:
    """
    Compute Normalized Edit Distance (NED).

    NED = EditDistance(pred, ref) / max(len(pred), len(ref))

    Returns:
        float: 0.0 (identical) to 1.0 (completely different)
    """
    if not pred and not ref:
        return 0.0
    if not pred or not ref:
        return 1.0

    distance = levenshtein_distance(pred, ref)
    max_len = max(len(pred), len(ref))
    return distance / max_len


def compute_bleu(pred: str, ref: str) -> float:
    """
    Compute BLEU score between prediction and reference.

    Returns:
        float: 0.0 to 100.0 (higher is better)
    """
    if sacrebleu is None:
        return 0.0

    if not pred or not ref:
        return 0.0

    # sacrebleu expects list of references
    try:
        result = sacrebleu.sentence_bleu(pred, [ref])
        return result.score
    except Exception:
        return 0.0


def normalize_text(text: str) -> str:
    """Normalize text for comparison (remove extra whitespace, citations, etc.)."""
    if not text:
        return ""

    # Remove citation markers like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text


def remove_bullets(text: str) -> str:
    """Remove bullet point lines from text for core content comparison."""
    if not text:
        return ""

    lines = text.split("\n")
    non_bullet_lines = [line for line in lines if not line.strip().startswith("•")]
    return "\n".join(non_bullet_lines)


# =============================================================================
# Section Matching
# =============================================================================


def match_sections_by_title(
    gt_sections: list[Section],
    pred_sections: list[Section],
    threshold: float = 0.8,
) -> list[tuple[Section, Section | None, float]]:
    """
    Match GT sections to predicted sections using fuzzy title matching.

    Args:
        gt_sections: Ground truth sections
        pred_sections: Predicted sections
        threshold: Minimum similarity ratio (0.0-1.0)

    Returns:
        List of (gt_section, matched_pred_section or None, similarity_score)
    """
    if fuzz is None:
        # Fallback to exact matching
        pred_by_title = {s.title.lower().strip(): s for s in pred_sections if s.title}
        matches = []
        for gt in gt_sections:
            gt_title = gt.title.lower().strip()
            pred = pred_by_title.get(gt_title)
            score = 1.0 if pred else 0.0
            matches.append((gt, pred, score))
        return matches

    pred_available = list(pred_sections)
    matches = []

    for gt in gt_sections:
        if not gt.title:
            # No title - try matching by path
            matches.append((gt, None, 0.0))
            continue

        best_match = None
        best_score = 0.0

        for pred in pred_available:
            if not pred.title:
                continue

            # Compute similarity
            score = fuzz.ratio(gt.title.lower(), pred.title.lower()) / 100.0

            if score > best_score:
                best_score = score
                best_match = pred

        if best_score >= threshold and best_match is not None:
            pred_available.remove(best_match)
            matches.append((gt, best_match, best_score))
        else:
            matches.append((gt, None, best_score))

    return matches


# =============================================================================
# Hierarchy Metrics
# =============================================================================


class TreeNode:
    """A node in the section tree for TEDS computation."""

    def __init__(self, label: str, children: list["TreeNode"] | None = None):
        self.label = label
        self.children = children or []

    @staticmethod
    def get_children(node: "TreeNode") -> list["TreeNode"]:
        return node.children

    @staticmethod
    def get_label(node: "TreeNode") -> str:
        return node.label

    def __repr__(self) -> str:
        return f"TreeNode({self.label}, children={len(self.children)})"


def build_section_tree(sections: list[Section], include_content: bool = True) -> TreeNode:
    """
    Build a tree structure from flat section list.

    Args:
        sections: List of sections with level information
        include_content: If True, include content in node labels (TEDS).
                        If False, use only structure (TEDS-S).

    Returns:
        Root TreeNode
    """
    if not sections:
        return TreeNode("root")

    root = TreeNode("root")
    stack: list[tuple[int, TreeNode]] = [(0, root)]

    for section in sections:
        level = section.level

        # Create node label
        if include_content:
            label = f"{section.title}|{normalize_text(section.content)[:100]}"
        else:
            # Structure only - just level and title
            label = f"L{level}:{section.title}" if section.title else f"L{level}"

        node = TreeNode(label)

        # Find parent node at appropriate level
        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            parent_node = stack[-1][1]
            parent_node.children.append(node)

        stack.append((level, node))

    return root


def tree_edit_distance(tree1: TreeNode, tree2: TreeNode) -> int:
    """
    Compute Tree Edit Distance using Zhang-Shasha algorithm.

    Args:
        tree1: First tree root
        tree2: Second tree root

    Returns:
        Edit distance (number of operations to transform tree1 to tree2)
    """
    if zss is None:
        raise ImportError("zss library required for tree edit distance. Install with: uv add zss")

    return zss.simple_distance(
        tree1,
        tree2,
        TreeNode.get_children,
        TreeNode.get_label,
        lambda x, y: 0 if x == y else 1,
    )


def count_nodes(node: TreeNode) -> int:
    """Count total nodes in a tree."""
    count = 1  # Current node
    for child in node.children:
        count += count_nodes(child)
    return count


def compute_teds(gt_sections: list[Section], pred_sections: list[Section]) -> float:
    """
    Compute TEDS (Tree Edit Distance Similarity) with content.

    TEDS = 1 - TED / max(|T1|, |T2|)

    Returns:
        float: 0.0 (completely different) to 1.0 (identical)
    """
    gt_tree = build_section_tree(gt_sections, include_content=True)
    pred_tree = build_section_tree(pred_sections, include_content=True)

    ted = tree_edit_distance(gt_tree, pred_tree)
    max_nodes = max(count_nodes(gt_tree), count_nodes(pred_tree))

    if max_nodes == 0:
        return 1.0

    return 1.0 - (ted / max_nodes)


def compute_teds_structure(gt_sections: list[Section], pred_sections: list[Section]) -> float:
    """
    Compute TEDS-S (structure only, ignoring content).

    Returns:
        float: 0.0 (completely different) to 1.0 (identical)
    """
    gt_tree = build_section_tree(gt_sections, include_content=False)
    pred_tree = build_section_tree(pred_sections, include_content=False)

    ted = tree_edit_distance(gt_tree, pred_tree)
    max_nodes = max(count_nodes(gt_tree), count_nodes(pred_tree))

    if max_nodes == 0:
        return 1.0

    return 1.0 - (ted / max_nodes)


def compute_level_accuracy(
    matches: list[tuple[Section, Section | None, float]],
) -> float:
    """
    Compute accuracy of level assignments for matched sections.

    Returns:
        float: 0.0 to 1.0 (percentage of correctly assigned levels)
    """
    total = 0
    correct = 0

    for gt, pred, _ in matches:
        if pred is not None:
            total += 1
            if gt.level == pred.level:
                correct += 1

    if total == 0:
        return 0.0

    return correct / total


def compute_parent_child_f1(
    gt_sections: list[Section], pred_sections: list[Section]
) -> dict[str, float]:
    """
    Compute F1 score for parent-child relationships.

    Returns:
        dict with precision, recall, f1
    """

    def extract_relationships(sections: list[Section]) -> set[tuple[str, str]]:
        """Extract parent-child pairs from sections."""
        relationships = set()
        stack: list[Section] = []

        for section in sections:
            level = section.level

            # Pop sections that are at same or higher level
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                parent = stack[-1]
                relationships.add((parent.title, section.title))

            stack.append(section)

        return relationships

    gt_rels = extract_relationships(gt_sections)
    pred_rels = extract_relationships(pred_sections)

    if not gt_rels and not pred_rels:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_rels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not gt_rels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(gt_rels & pred_rels)
    precision = true_positives / len(pred_rels)
    recall = true_positives / len(gt_rels)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# Segmentation Metrics
# =============================================================================


def compute_boundary_f1(
    gt_sections: list[Section], pred_sections: list[Section]
) -> dict[str, float]:
    """
    Compute Boundary Precision/Recall/F1.

    A boundary is defined as a section start (title).

    Returns:
        dict with precision, recall, f1
    """
    # Extract boundaries (normalized titles)
    gt_boundaries = {normalize_text(s.title).lower() for s in gt_sections if s.title}
    pred_boundaries = {normalize_text(s.title).lower() for s in pred_sections if s.title}

    if not gt_boundaries and not pred_boundaries:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_boundaries:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not gt_boundaries:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(gt_boundaries & pred_boundaries)
    precision = true_positives / len(pred_boundaries)
    recall = true_positives / len(gt_boundaries)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# Main Evaluation
# =============================================================================


def evaluate_extraction(gt_path: Path, pred_path: Path) -> EvalResult:
    """
    Main evaluation function.

    Args:
        gt_path: Path to ground truth JSON
        pred_path: Path to predicted/extracted JSON

    Returns:
        EvalResult with all metrics
    """
    # Load documents
    gt_doc = Document.from_json(gt_path)
    pred_doc = Document.from_json(pred_path)

    result = EvalResult()

    # Metadata
    result.metadata = {
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "gt_num_sections": gt_doc.num_sections,
        "pred_num_sections": pred_doc.num_sections,
        "gt_hierarchy_depth": gt_doc.hierarchy_depth,
        "pred_hierarchy_depth": pred_doc.hierarchy_depth,
        "gt_num_roots": gt_doc.num_roots,
        "pred_num_roots": pred_doc.num_roots,
    }

    # Filter sections with titles for matching
    gt_titled = [s for s in gt_doc.sections if s.title]
    pred_titled = [s for s in pred_doc.sections if s.title]

    # Section matching
    matches = match_sections_by_title(gt_titled, pred_titled, threshold=0.8)
    matched_count = sum(1 for _, pred, _ in matches if pred is not None)

    result.metadata["matched_sections"] = matched_count
    result.metadata["match_rate"] = matched_count / len(gt_titled) if gt_titled else 0.0

    # Text metrics (for matched sections)
    text_neds = []
    text_neds_core = []  # Without bullets
    text_bleus = []

    for gt, pred, _ in matches:
        if pred is not None:
            gt_content = normalize_text(gt.content)
            pred_content = normalize_text(pred.content)

            if gt_content or pred_content:
                ned = normalized_edit_distance(pred_content, gt_content)
                text_neds.append(ned)

                bleu = compute_bleu(pred_content, gt_content)
                text_bleus.append(bleu)

                # Core content (without merged bullets)
                gt_core = normalize_text(remove_bullets(gt.content))
                pred_core = normalize_text(remove_bullets(pred.content))
                if gt_core or pred_core:
                    ned_core = normalized_edit_distance(pred_core, gt_core)
                    text_neds_core.append(ned_core)

    result.text = {
        "avg_ned": sum(text_neds) / len(text_neds) if text_neds else 1.0,
        "avg_ned_core": sum(text_neds_core) / len(text_neds_core) if text_neds_core else 1.0,
        "avg_bleu": sum(text_bleus) / len(text_bleus) if text_bleus else 0.0,
        "num_compared": len(text_neds),
    }

    # Hierarchy metrics
    try:
        teds = compute_teds(gt_titled, pred_titled)
        teds_s = compute_teds_structure(gt_titled, pred_titled)
    except ImportError as e:
        print(f"Warning: {e}")
        teds = 0.0
        teds_s = 0.0

    level_acc = compute_level_accuracy(matches)
    parent_child = compute_parent_child_f1(gt_titled, pred_titled)

    result.hierarchy = {
        "teds": teds,
        "teds_structure": teds_s,
        "level_accuracy": level_acc,
        "parent_child_precision": parent_child["precision"],
        "parent_child_recall": parent_child["recall"],
        "parent_child_f1": parent_child["f1"],
    }

    # Segmentation metrics
    boundary = compute_boundary_f1(gt_doc.sections, pred_doc.sections)
    result.segmentation = {
        "boundary_precision": boundary["precision"],
        "boundary_recall": boundary["recall"],
        "boundary_f1": boundary["f1"],
    }

    # Overall score (weighted)
    # Hierarchy: 70%, Text: 30%
    hierarchy_score = (
        0.4 * teds_s + 0.3 * level_acc + 0.3 * parent_child["f1"]
    )
    # Use core NED (without bullets) for fairer comparison
    text_score = 1.0 - result.text["avg_ned_core"]

    result.overall = 0.7 * hierarchy_score + 0.3 * text_score

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate document extraction against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    uv run python scripts/extraction_eval.py \\
        --gt extraction_dataset/BZ_VR1_gt.json \\
        --pred output/BZ_VR1/phase1_extraction.json

    # Save results to file
    uv run python scripts/extraction_eval.py \\
        --gt extraction_dataset/BZ_VR1_gt.json \\
        --pred output/BZ_VR1/phase1_extraction.json \\
        --output results/eval_report.json
        """,
    )

    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Path to predicted/extracted JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save evaluation results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.gt.exists():
        print(f"Error: Ground truth file not found: {args.gt}")
        sys.exit(1)

    if not args.pred.exists():
        print(f"Error: Prediction file not found: {args.pred}")
        sys.exit(1)

    # Check dependencies
    missing = []
    if zss is None:
        missing.append("zss")
    if fuzz is None:
        missing.append("rapidfuzz")
    if sacrebleu is None:
        missing.append("sacrebleu")

    if missing:
        print(f"Warning: Missing optional dependencies: {', '.join(missing)}")
        print("Install with: uv add " + " ".join(missing))
        print()

    # Run evaluation
    print(f"Evaluating extraction...")
    print(f"  GT: {args.gt}")
    print(f"  Pred: {args.pred}")
    print()

    result = evaluate_extraction(args.gt, args.pred)

    # Print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 METADATA")
    print(f"  GT sections:     {result.metadata['gt_num_sections']}")
    print(f"  Pred sections:   {result.metadata['pred_num_sections']}")
    print(f"  Matched:         {result.metadata['matched_sections']} ({result.metadata['match_rate']:.1%})")
    print(f"  GT depth:        {result.metadata['gt_hierarchy_depth']}")
    print(f"  Pred depth:      {result.metadata['pred_hierarchy_depth']}")

    print("\n🌲 HIERARCHY (70% weight)")
    print(f"  TEDS:            {result.hierarchy['teds']:.3f}")
    print(f"  TEDS-S:          {result.hierarchy['teds_structure']:.3f}")
    print(f"  Level Accuracy:  {result.hierarchy['level_accuracy']:.3f}")
    print(f"  Parent-Child F1: {result.hierarchy['parent_child_f1']:.3f}")

    print("\n📝 TEXT (30% weight)")
    print(f"  Avg NED:         {result.text['avg_ned']:.3f} (lower is better)")
    print(f"  Avg NED (core):  {result.text['avg_ned_core']:.3f} (without bullets)")
    print(f"  Avg BLEU:        {result.text['avg_bleu']:.1f}")
    print(f"  Sections:        {result.text['num_compared']}")

    print("\n🔲 SEGMENTATION")
    print(f"  Boundary P:      {result.segmentation['boundary_precision']:.3f}")
    print(f"  Boundary R:      {result.segmentation['boundary_recall']:.3f}")
    print(f"  Boundary F1:     {result.segmentation['boundary_f1']:.3f}")

    print("\n" + "=" * 60)
    print(f"⭐ OVERALL SCORE: {result.overall:.3f}")
    print("=" * 60)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
