#!/usr/bin/env python3
"""
Tree Edit Distance (TED) Calculator for JSON Document Structures.

Implements the Zhang-Shasha algorithm for computing tree edit distance
between two hierarchical JSON document structures.

Usage:
    uv run python scripts/tree_edit_distance.py <file1.json> <file2.json>
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TreeNode:
    """A node in the tree structure for TED computation."""

    label: str  # Node label (e.g., section title or content hash)
    children: list["TreeNode"] = field(default_factory=list)
    node_id: str = ""  # Original section_id for reference

    def __repr__(self) -> str:
        return f"TreeNode({self.label[:30]}..., children={len(self.children)})"


class ZhangShashaAlgorithm:
    """
    Zhang-Shasha Tree Edit Distance Algorithm.

    Computes the minimum edit distance between two trees using
    dynamic programming. Operations:
    - Insert: Add a node (cost 1)
    - Delete: Remove a node (cost 1)
    - Relabel: Change a node's label (cost 1 if labels differ, 0 otherwise)
    """

    def __init__(self, tree1: TreeNode, tree2: TreeNode):
        self.tree1 = tree1
        self.tree2 = tree2

        # Build node lists in post-order
        self.nodes1: list[TreeNode] = []
        self.nodes2: list[TreeNode] = []

        # Leftmost leaf descendants (1-indexed in papers, 0-indexed here)
        self.l1: list[int] = []
        self.l2: list[int] = []

        # Key roots for the algorithm
        self.keyroots1: list[int] = []
        self.keyroots2: list[int] = []

        # Build auxiliary structures
        self._build_postorder(tree1, self.nodes1, self.l1)
        self._build_postorder(tree2, self.nodes2, self.l2)
        self._compute_keyroots(self.l1, self.keyroots1)
        self._compute_keyroots(self.l2, self.keyroots2)

    def _build_postorder(
        self,
        node: TreeNode,
        nodes: list[TreeNode],
        leftmost: list[int]
    ) -> int:
        """Build post-order traversal and compute leftmost leaf descendants."""
        if not node.children:
            # Leaf node
            idx = len(nodes)
            nodes.append(node)
            leftmost.append(idx)
            return idx

        # Process children first (post-order)
        first_child_idx = None
        for i, child in enumerate(node.children):
            child_idx = self._build_postorder(child, nodes, leftmost)
            if i == 0:
                first_child_idx = leftmost[child_idx]

        # Add current node
        idx = len(nodes)
        nodes.append(node)
        leftmost.append(first_child_idx if first_child_idx is not None else idx)
        return idx

    def _compute_keyroots(self, leftmost: list[int], keyroots: list[int]) -> None:
        """Compute key roots - nodes with unique leftmost leaf descendants."""
        seen = set()
        for i in range(len(leftmost) - 1, -1, -1):
            if leftmost[i] not in seen:
                keyroots.append(i)
                seen.add(leftmost[i])
        keyroots.sort()

    def _cost(self, node1: TreeNode | None, node2: TreeNode | None) -> int:
        """
        Compute the cost of relabeling/insert/delete.

        - Delete (node1 → None): cost 1
        - Insert (None → node2): cost 1
        - Relabel (node1 → node2): cost 0 if same label, 1 otherwise
        """
        if node1 is None:
            return 1  # Insert
        if node2 is None:
            return 1  # Delete
        return 0 if node1.label == node2.label else 1  # Relabel

    def compute_distance(self) -> int:
        """Compute the tree edit distance using Zhang-Shasha algorithm."""
        n = len(self.nodes1)
        m = len(self.nodes2)

        if n == 0:
            return m
        if m == 0:
            return n

        # Tree distance matrix
        td = [[0] * (m + 1) for _ in range(n + 1)]

        # Forest distance matrix (reused for each keyroot pair)
        for x in self.keyroots1:
            for y in self.keyroots2:
                self._compute_forest_distance(x, y, td)

        return td[n][m]

    def _compute_forest_distance(
        self,
        i: int,
        j: int,
        td: list[list[int]]
    ) -> None:
        """Compute forest distances for subtrees rooted at i and j."""
        l1, l2 = self.l1, self.l2
        m = len(self.nodes2)
        n = len(self.nodes1)

        # Forest distance for this subtree pair
        fd = defaultdict(lambda: defaultdict(int))

        # Base cases
        fd[l1[i] - 1][l2[j] - 1] = 0

        for x in range(l1[i], i + 1):
            fd[x][l2[j] - 1] = fd[x - 1][l2[j] - 1] + self._cost(self.nodes1[x], None)

        for y in range(l2[j], j + 1):
            fd[l1[i] - 1][y] = fd[l1[i] - 1][y - 1] + self._cost(None, self.nodes2[y])

        # Fill the matrix
        for x in range(l1[i], i + 1):
            for y in range(l2[j], j + 1):
                if l1[x] == l1[i] and l2[y] == l2[j]:
                    # Both nodes are in the same forest
                    fd[x][y] = min(
                        fd[x - 1][y] + self._cost(self.nodes1[x], None),  # Delete
                        fd[x][y - 1] + self._cost(None, self.nodes2[y]),  # Insert
                        fd[x - 1][y - 1] + self._cost(self.nodes1[x], self.nodes2[y])  # Match/Relabel
                    )
                    td[x + 1][y + 1] = fd[x][y]
                else:
                    # Use previously computed tree distances
                    fd[x][y] = min(
                        fd[x - 1][y] + self._cost(self.nodes1[x], None),  # Delete
                        fd[x][y - 1] + self._cost(None, self.nodes2[y]),  # Insert
                        fd[l1[x] - 1][l2[y] - 1] + td[x + 1][y + 1]  # Use tree distance
                    )


def parse_gt_json(data: dict) -> TreeNode:
    """
    Parse ground truth JSON format (sections as array with path hierarchy).

    Structure:
    - sections: array of {section_id, title, content, level, path, ...}
    - path contains hierarchy like "A > B > C"
    """
    sections = data.get("sections", [])
    if not sections:
        return TreeNode(label="empty_document")

    # Build tree from path hierarchy
    root = TreeNode(label="root", node_id="root")
    path_to_node: dict[str, TreeNode] = {"": root}

    for section in sections:
        section_id = section.get("section_id", "")
        title = section.get("title", "")
        path = section.get("path", "")
        level = section.get("level", 1)

        # Create label from title (normalized)
        label = _normalize_label(title if title else section.get("content", "")[:50])

        node = TreeNode(label=label, node_id=section_id)

        # Find parent from path
        path_parts = path.split(" > ") if path else []

        if len(path_parts) <= 1:
            # Root level section
            root.children.append(node)
            path_to_node[path] = node
        else:
            # Find parent path
            parent_path = " > ".join(path_parts[:-1])
            parent = path_to_node.get(parent_path, root)
            parent.children.append(node)
            path_to_node[path] = node

    return root


def parse_ingest_json(data: dict) -> TreeNode:
    """
    Parse ingest JSON format (sections as dict with parent_id/children).

    Structure:
    - sections: dict of section_id -> {title, level, parent_id, children, ...}
    """
    sections = data.get("sections", {})
    if not sections:
        return TreeNode(label="empty_document")

    # Build node map
    node_map: dict[str, TreeNode] = {}
    for sec_id, section in sections.items():
        title = section.get("title", "")
        # Clean markdown headers from title
        title = title.lstrip("#").strip()
        label = _normalize_label(title)
        node_map[sec_id] = TreeNode(label=label, node_id=sec_id)

    # Build tree structure
    root = TreeNode(label="root", node_id="root")

    for sec_id, section in sections.items():
        parent_id = section.get("parent_id")
        node = node_map[sec_id]

        if parent_id and parent_id in node_map:
            parent_node = node_map[parent_id]
            if node not in parent_node.children:
                parent_node.children.append(node)
        else:
            # Root level section
            if node not in root.children:
                root.children.append(node)

    return root


def _normalize_label(text: str, strip_numbers: bool = True) -> str:
    """
    Normalize text label for comparison.

    Args:
        text: The label text to normalize
        strip_numbers: If True, remove section numbering like "5.2.1 "
    """
    import re

    if not text:
        return "(empty)"

    # Remove markdown headers (##, ###, etc.)
    text = text.lstrip("#").strip()

    # Remove section numbering like "5.2.1 ", "1.2 ", "10. ", etc.
    if strip_numbers:
        text = re.sub(r"^[\d\.]+\s*", "", text)

    # Lowercase, strip whitespace
    text = text.lower().strip()

    # Remove common variations
    text = text.replace("  ", " ")

    # Truncate for comparison (long labels unlikely to match exactly anyway)
    return text[:100] if len(text) > 100 else text


def count_nodes(node: TreeNode) -> int:
    """Count total nodes in a tree."""
    return 1 + sum(count_nodes(child) for child in node.children)


def tree_depth(node: TreeNode) -> int:
    """Compute tree depth."""
    if not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)


def collect_labels(node: TreeNode, labels: set[str] | None = None) -> set[str]:
    """Collect all labels in a tree."""
    if labels is None:
        labels = set()
    labels.add(node.label)
    for child in node.children:
        collect_labels(child, labels)
    return labels


def compute_ted(file1: Path, file2: Path) -> dict[str, Any]:
    """
    Compute Tree Edit Distance between two JSON files.

    Returns a dictionary with:
    - distance: The TED value
    - tree1_stats: Statistics about tree 1
    - tree2_stats: Statistics about tree 2
    - normalized_distance: TED / max(nodes1, nodes2)
    """
    # Load files
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    # Detect format and parse
    if isinstance(data1.get("sections"), list):
        tree1 = parse_gt_json(data1)
        format1 = "gt (array)"
    else:
        tree1 = parse_ingest_json(data1)
        format1 = "ingest (dict)"

    if isinstance(data2.get("sections"), list):
        tree2 = parse_gt_json(data2)
        format2 = "gt (array)"
    else:
        tree2 = parse_ingest_json(data2)
        format2 = "ingest (dict)"

    # Compute statistics
    nodes1 = count_nodes(tree1)
    nodes2 = count_nodes(tree2)
    depth1 = tree_depth(tree1)
    depth2 = tree_depth(tree2)
    labels1 = collect_labels(tree1)
    labels2 = collect_labels(tree2)

    # Compute TED
    print(f"Computing TED between {nodes1} and {nodes2} nodes...")
    print("This may take a while for large trees...")

    algo = ZhangShashaAlgorithm(tree1, tree2)
    distance = algo.compute_distance()

    # Compute additional metrics
    max_nodes = max(nodes1, nodes2)
    normalized = distance / max_nodes if max_nodes > 0 else 0

    # Label overlap
    common_labels = labels1 & labels2
    all_labels = labels1 | labels2
    jaccard = len(common_labels) / len(all_labels) if all_labels else 0

    return {
        "distance": distance,
        "normalized_distance": normalized,
        "tree1": {
            "file": str(file1),
            "format": format1,
            "nodes": nodes1,
            "depth": depth1,
            "unique_labels": len(labels1),
        },
        "tree2": {
            "file": str(file2),
            "format": format2,
            "nodes": nodes2,
            "depth": depth2,
            "unique_labels": len(labels2),
        },
        "label_overlap": {
            "common": len(common_labels),
            "jaccard_similarity": jaccard,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Tree Edit Distance between two JSON document structures."
    )
    parser.add_argument("file1", type=Path, help="First JSON file")
    parser.add_argument("file2", type=Path, help="Second JSON file")
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file for detailed results"
    )

    args = parser.parse_args()

    if not args.file1.exists():
        print(f"Error: File not found: {args.file1}", file=sys.stderr)
        sys.exit(1)
    if not args.file2.exists():
        print(f"Error: File not found: {args.file2}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Tree Edit Distance (TED) Calculator")
    print(f"{'='*60}\n")

    results = compute_ted(args.file1, args.file2)

    # Print results
    print(f"\n{'─'*60}")
    print("RESULTS")
    print(f"{'─'*60}\n")

    print(f"Tree 1: {results['tree1']['file']}")
    print(f"  Format: {results['tree1']['format']}")
    print(f"  Nodes: {results['tree1']['nodes']}")
    print(f"  Depth: {results['tree1']['depth']}")
    print(f"  Unique labels: {results['tree1']['unique_labels']}")

    print(f"\nTree 2: {results['tree2']['file']}")
    print(f"  Format: {results['tree2']['format']}")
    print(f"  Nodes: {results['tree2']['nodes']}")
    print(f"  Depth: {results['tree2']['depth']}")
    print(f"  Unique labels: {results['tree2']['unique_labels']}")

    print(f"\n{'─'*60}")
    print(f"TREE EDIT DISTANCE: {results['distance']}")
    print(f"Normalized (TED / max_nodes): {results['normalized_distance']:.4f}")
    print(f"{'─'*60}")

    print(f"\nLabel Overlap:")
    print(f"  Common labels: {results['label_overlap']['common']}")
    print(f"  Jaccard similarity: {results['label_overlap']['jaccard_similarity']:.4f}")

    # Interpretation
    norm_dist = results['normalized_distance']
    print(f"\n{'─'*60}")
    print("INTERPRETATION")
    print(f"{'─'*60}")
    if norm_dist < 0.1:
        print("✓ Very similar structures (< 10% edits needed)")
    elif norm_dist < 0.3:
        print("○ Moderately similar structures (10-30% edits needed)")
    elif norm_dist < 0.5:
        print("△ Somewhat different structures (30-50% edits needed)")
    else:
        print("✗ Very different structures (> 50% edits needed)")

    # Save detailed results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    print()
    return results


if __name__ == "__main__":
    main()
