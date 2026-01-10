#!/usr/bin/env python3
"""
Normalize Ground Truth JSON by merging bullet points into parent sections.

Bullet points in GT are sections with empty titles. This script:
1. Identifies sections without titles (bullets)
2. Merges their content into the closest parent section with a title
3. Removes the bullet sections
4. Updates metadata (num_sections, etc.)

Usage:
    uv run python scripts/normalize_gt.py \
        --input extraction_dataset/BZ_VR1_gt.json \
        --output extraction_dataset/BZ_VR1_gt_normalized.json
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def find_parent_with_title(sections: list[dict], current_idx: int, current_level: int) -> int | None:
    """
    Find the index of the closest parent section with a title.

    Walks backwards through sections to find a section with:
    - A non-empty title
    - A lower level than current_level
    """
    for i in range(current_idx - 1, -1, -1):
        sec = sections[i]
        if sec.get("title", "").strip() and sec.get("level", 0) < current_level:
            return i
    return None


def merge_bullets_into_parents(sections: list[dict]) -> list[dict]:
    """
    Merge bullet point sections (no title) into their parent sections.

    Returns:
        List of sections with bullets merged
    """
    # Deep copy to avoid modifying original
    sections = deepcopy(sections)

    # Track which sections to keep
    keep_indices = set()

    # Track content to append to each parent
    parent_additions: dict[int, list[str]] = {}

    for i, sec in enumerate(sections):
        title = sec.get("title", "").strip()

        if title:
            # This is a titled section - keep it
            keep_indices.add(i)
        else:
            # This is a bullet - find parent and merge
            level = sec.get("level", 0)
            content = sec.get("content", "").strip()

            if content:
                parent_idx = find_parent_with_title(sections, i, level)

                if parent_idx is not None:
                    if parent_idx not in parent_additions:
                        parent_additions[parent_idx] = []
                    parent_additions[parent_idx].append(f"• {content}")

    # Apply additions to parents
    for parent_idx, additions in parent_additions.items():
        current_content = sections[parent_idx].get("content", "")
        bullet_text = "\n".join(additions)

        if current_content:
            sections[parent_idx]["content"] = f"{current_content}\n{bullet_text}"
        else:
            sections[parent_idx]["content"] = bullet_text

        # Update content length
        sections[parent_idx]["content_length"] = len(sections[parent_idx]["content"])

    # Filter to keep only titled sections
    result = [sections[i] for i in sorted(keep_indices)]

    # Re-assign section IDs
    for i, sec in enumerate(result):
        sec["section_id"] = f"sec_{i + 1}"

    return result


def compute_hierarchy_depth(sections: list[dict]) -> int:
    """Compute maximum hierarchy depth."""
    if not sections:
        return 0
    return max(s.get("level", 1) for s in sections)


def count_roots(sections: list[dict]) -> int:
    """Count root sections (level 1)."""
    return sum(1 for s in sections if s.get("level", 0) == 1)


def normalize_gt(input_path: Path, output_path: Path) -> dict:
    """
    Normalize ground truth JSON.

    Args:
        input_path: Path to original GT JSON
        output_path: Path to save normalized GT JSON

    Returns:
        Statistics about the normalization
    """
    # Load original
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_sections = data.get("sections", [])
    original_count = len(original_sections)

    # Count bullets
    bullets_count = sum(1 for s in original_sections if not s.get("title", "").strip())
    titled_count = original_count - bullets_count

    # Merge bullets
    normalized_sections = merge_bullets_into_parents(original_sections)

    # Update metadata
    normalized_data = {
        "document_id": data.get("document_id", ""),
        "source_path": data.get("source_path", ""),
        "num_sections": len(normalized_sections),
        "hierarchy_depth": compute_hierarchy_depth(normalized_sections),
        "num_roots": count_roots(normalized_sections),
        "num_tables": data.get("num_tables", 0),
        "sections": normalized_sections,
        "_normalization": {
            "original_sections": original_count,
            "bullets_merged": bullets_count,
            "final_sections": len(normalized_sections),
        }
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    return {
        "original_sections": original_count,
        "titled_sections": titled_count,
        "bullets_merged": bullets_count,
        "final_sections": len(normalized_sections),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Normalize GT JSON by merging bullet points into parents"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to original GT JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save normalized GT JSON",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    print(f"Normalizing: {args.input}")

    stats = normalize_gt(args.input, args.output)

    print(f"\n✅ Normalization complete!")
    print(f"   Original sections:  {stats['original_sections']}")
    print(f"   Titled sections:    {stats['titled_sections']}")
    print(f"   Bullets merged:     {stats['bullets_merged']}")
    print(f"   Final sections:     {stats['final_sections']}")
    print(f"   Output: {stats['output_path']}")

    return 0


if __name__ == "__main__":
    exit(main())
