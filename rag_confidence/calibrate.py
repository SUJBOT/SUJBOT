#!/usr/bin/env python3
"""
Calibrate Conformal Predictor using precomputed matrices.

Usage:
    # Full calibration with evaluation
    uv run python rag_confidence/calibrate.py --alpha 0.1 --evaluate

    # With train/test split (95% calibration, 5% test)
    uv run python rag_confidence/calibrate.py --alpha 0.1 --holdout 0.05

    # Different coverage levels
    uv run python rag_confidence/calibrate.py --alpha 0.05  # 95% coverage
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from conformal_predictor import ConformalPredictor, evaluate_coverage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_matrices(data_dir: Path):
    """Load precomputed similarity and relevance matrices."""
    sim_path = data_dir / "synthetic_similarity_matrix.npz"
    rel_path = data_dir / "synthetic_relevance_matrix.npz"

    logger.info(f"Loading similarity matrix from {sim_path}")
    sim_data = np.load(sim_path)

    # Check what keys are in the NPZ file
    logger.info(f"Similarity matrix keys: {list(sim_data.keys())}")

    # Load similarity matrix - handle different key names
    if "similarity" in sim_data:
        similarity_matrix = sim_data["similarity"]
    elif "arr_0" in sim_data:
        similarity_matrix = sim_data["arr_0"]
    else:
        # Use first available key
        key = list(sim_data.keys())[0]
        similarity_matrix = sim_data[key]
        logger.info(f"Using key '{key}' for similarity matrix")

    logger.info(f"Loading relevance matrix from {rel_path}")
    rel_data = np.load(rel_path)

    logger.info(f"Relevance matrix keys: {list(rel_data.keys())}")

    # Load relevance matrix
    if "relevance" in rel_data:
        relevance_matrix = rel_data["relevance"]
    elif "arr_0" in rel_data:
        relevance_matrix = rel_data["arr_0"]
    else:
        key = list(rel_data.keys())[0]
        relevance_matrix = rel_data[key]
        logger.info(f"Using key '{key}' for relevance matrix")

    # Load chunk IDs if available
    chunk_ids = None
    if "chunk_ids" in rel_data:
        chunk_ids = rel_data["chunk_ids"]
    elif "chunk_ids" in sim_data:
        chunk_ids = sim_data["chunk_ids"]

    # Load query IDs if available
    query_ids = None
    if "query_ids" in rel_data:
        query_ids = rel_data["query_ids"]
    elif "query_ids" in sim_data:
        query_ids = sim_data["query_ids"]

    logger.info(
        f"Loaded: {similarity_matrix.shape[0]} queries, {similarity_matrix.shape[1]} chunks"
    )
    logger.info(
        f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]"
    )
    logger.info(
        f"Relevant chunks per query: {(relevance_matrix > 0).sum(axis=1).mean():.2f} avg"
    )

    # Generate chunk_ids if not provided
    if chunk_ids is None:
        chunk_ids = np.array([f"chunk_{i}" for i in range(similarity_matrix.shape[1])])
        logger.warning("No chunk_ids found, using generated IDs")

    return similarity_matrix, relevance_matrix, chunk_ids, query_ids


def main():
    parser = argparse.ArgumentParser(description="Calibrate Conformal Predictor")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Error rate (0.1 = 90%% coverage guarantee)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing matrix files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for calibration result (default: calibration_result.json)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate coverage on calibration set (should be ~1-alpha)",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.0,
        help="Fraction to hold out for evaluation (0.0 = use all for calibration)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train/test split"
    )

    args = parser.parse_args()

    # Load data
    similarity_matrix, relevance_matrix, chunk_ids, query_ids = load_matrices(
        args.data_dir
    )

    # Optional train/test split
    n_queries = similarity_matrix.shape[0]
    if args.holdout > 0:
        np.random.seed(args.seed)
        n_train = int(n_queries * (1 - args.holdout))
        indices = np.random.permutation(n_queries)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_sim = similarity_matrix[train_idx]
        train_rel = relevance_matrix[train_idx]
        test_sim = similarity_matrix[test_idx]
        test_rel = relevance_matrix[test_idx]

        logger.info(f"Split: {len(train_idx)} train, {len(test_idx)} test")
    else:
        train_sim, train_rel = similarity_matrix, relevance_matrix
        test_sim, test_rel = similarity_matrix, relevance_matrix

    # Calibrate
    predictor = ConformalPredictor(alpha=args.alpha)
    result = predictor.calibrate_from_matrices(train_sim, train_rel)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULT")
    print("=" * 60)
    print(f"Alpha:              {result.alpha}")
    print(f"Coverage Guarantee: {result.coverage_guarantee:.0%}")
    print(f"Threshold:          {result.threshold:.6f}")
    print(f"N Calibration:      {result.n_calibration}")
    print(f"Calibrated At:      {result.calibrated_at}")
    print("\nScore Percentiles:")
    for key, val in result.score_percentiles.items():
        print(f"  {key}: {val:.6f}")
    print("=" * 60)

    # Evaluate on training set
    if args.evaluate or args.holdout > 0:
        logger.info("Evaluating coverage on training set...")
        train_metrics = evaluate_coverage(predictor, train_sim, train_rel, chunk_ids)

        print("\n" + "=" * 60)
        print("TRAINING SET EVALUATION")
        print("=" * 60)
        print(f"N Queries:          {train_metrics['n_test']}")
        print(f"Empirical Coverage: {train_metrics['empirical_coverage']:.2%}")
        print(f"Target Coverage:    {train_metrics['target_coverage']:.2%}")
        print(f"Coverage Satisfied: {train_metrics['coverage_satisfied']}")
        print(f"Recall:             {train_metrics['recall']:.2%}")
        print(f"Avg Retrieved:      {train_metrics['avg_retrieved']:.1f} chunks")
        print("=" * 60)

    # Evaluate on test set (held-out)
    if args.holdout > 0:
        logger.info("Evaluating coverage on held-out test set...")
        test_metrics = evaluate_coverage(predictor, test_sim, test_rel, chunk_ids)

        print("\n" + "=" * 60)
        print("TEST SET EVALUATION (HELD-OUT)")
        print("=" * 60)
        print(f"N Queries:          {test_metrics['n_test']}")
        print(f"Empirical Coverage: {test_metrics['empirical_coverage']:.2%}")
        print(f"Target Coverage:    {test_metrics['target_coverage']:.2%}")
        print(f"Coverage Satisfied: {test_metrics['coverage_satisfied']}")
        print(f"Recall:             {test_metrics['recall']:.2%}")
        print(f"Avg Retrieved:      {test_metrics['avg_retrieved']:.1f} chunks")
        print("=" * 60)

        # Highlight if test coverage matches guarantee
        if test_metrics["coverage_satisfied"]:
            print("\n*** CONFORMAL GUARANTEE VALIDATED ON TEST SET ***")
        else:
            print(
                f"\n*** WARNING: Test coverage ({test_metrics['empirical_coverage']:.2%}) "
                f"below target ({test_metrics['target_coverage']:.2%}) ***"
            )
            print("This may be due to statistical variance with small test set.")

    # Save
    output_path = args.output or args.data_dir / "calibration_result.json"
    predictor.save(output_path)
    print(f"\nSaved calibration to: {output_path}")


if __name__ == "__main__":
    main()
