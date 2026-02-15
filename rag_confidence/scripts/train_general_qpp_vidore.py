#!/usr/bin/env python3
"""
Train General QPP Model on ViDoRe V3 dataset.

Uses the research-validated optimal configuration:
- 24 language-agnostic GeneralQPP features
- MLP (128, 64) with alpha=0.1 (L2 regularization)
- Binary target: Recall@10 >= 0.5

Usage:
    uv run python rag_confidence/scripts/train_general_qpp_vidore.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_confidence.core.general_qpp_extractor import GeneralQPPExtractor
from rag_confidence.core.qpp_model import MLPModel
from rag_confidence.data.vidore_loader import VideoreDataset, DATASETS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "vidore_v3"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
OUTPUT_MODEL = Path(__file__).parent.parent / "models" / "general_qpp_model.pkl"
OUTPUT_CONFIG = Path(__file__).parent.parent / "models" / "general_qpp_config.json"

# English datasets used in research benchmarks
ENGLISH_DATASETS = [
    "vidore_v3_physics",
    "vidore_v3_energy",
    "vidore_v3_finance_en",
    "vidore_v3_hr",
    "vidore_v3_industrial",
    "vidore_v3_computer_science",
    "vidore_v3_pharmaceuticals",
    "esg_reports_eng_v2",
    "economics_reports_eng_v2",
    "biomedical_lectures_eng_v2",
]

# Optimal hyperparameters from research
HIDDEN_SIZES = (128, 64)
ALPHA = 0.1
MAX_ITER = 500
TAU = 0.711  # Conformal threshold


def load_embeddings() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict], List[Dict]]:
    """Load all embeddings and metadata."""
    logger.info("Loading embeddings...")

    # Load query embeddings
    query_data = np.load(EMBEDDINGS_DIR / "query_embeddings" / "all_queries.npz")
    query_embeddings = query_data["embeddings"]
    query_ids = query_data["query_ids"]

    # Load document embeddings
    doc_data = np.load(EMBEDDINGS_DIR / "document_embeddings" / "all_documents.npz")
    doc_embeddings = doc_data["embeddings"]
    doc_ids = doc_data["doc_ids"]

    # Load metadata
    with open(EMBEDDINGS_DIR / "metadata" / "query_metadata.json") as f:
        query_metadata = json.load(f)
    with open(EMBEDDINGS_DIR / "metadata" / "document_metadata.json") as f:
        doc_metadata = json.load(f)

    logger.info(f"Loaded {len(query_embeddings)} queries, {len(doc_embeddings)} documents")
    return query_embeddings, query_ids, doc_embeddings, doc_ids, query_metadata, doc_metadata


def compute_recall_at_k(
    similarities: np.ndarray,
    qrels: Dict,
    query_ids: List[str],
    doc_ids: List[str],
    k: int = 10,
) -> np.ndarray:
    """Compute Recall@K for each query."""
    recalls = []

    # Build doc_id lookup (convert to int for matching)
    doc_id_to_idx = {}
    for i, did in enumerate(doc_ids):
        try:
            doc_id_to_idx[int(did)] = i
        except (ValueError, TypeError):
            doc_id_to_idx[did] = i

    for i, qid in enumerate(query_ids):
        # Try both int and string versions
        try:
            qid_int = int(qid)
        except (ValueError, TypeError):
            qid_int = None

        qrel_entry = qrels.get(qid_int) or qrels.get(qid) or qrels.get(str(qid))

        if not qrel_entry:
            recalls.append(0.0)
            continue

        # Get relevant docs (convert to int for matching)
        relevant_doc_ids = set()
        for did, score in qrel_entry.items():
            if score > 0:
                try:
                    relevant_doc_ids.add(int(did))
                except (ValueError, TypeError):
                    relevant_doc_ids.add(did)

        if not relevant_doc_ids:
            recalls.append(0.0)
            continue

        # Get top-k retrieved doc indices
        top_k_indices = np.argsort(similarities[i])[::-1][:k]

        # Convert retrieved doc IDs to int for matching
        retrieved_doc_ids = set()
        for idx in top_k_indices:
            did = doc_ids[idx]
            try:
                retrieved_doc_ids.add(int(did))
            except (ValueError, TypeError):
                retrieved_doc_ids.add(did)

        # Compute recall
        hits = len(relevant_doc_ids & retrieved_doc_ids)
        recall = hits / len(relevant_doc_ids)
        recalls.append(recall)

    return np.array(recalls)


def process_dataset(
    dataset_name: str,
    query_embeddings: np.ndarray,
    query_ids: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_ids: np.ndarray,
    query_metadata: List[Dict],
    doc_metadata: List[Dict],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Process a single dataset: compute similarities and labels."""

    # Load dataset using vidore_loader
    try:
        dataset = VideoreDataset(dataset_name)
        qrels = dataset.get_qrels_dict()
    except Exception as e:
        logger.warning(f"Could not load qrels for {dataset_name}: {e}")
        return [], np.array([]), np.array([])

    if not qrels:
        logger.warning(f"No qrels found for {dataset_name}")
        return [], np.array([]), np.array([])

    # Filter queries and docs for this dataset
    query_mask = np.array([m["dataset"] == dataset_name for m in query_metadata])
    doc_mask = np.array([m["dataset"] == dataset_name for m in doc_metadata])

    if not query_mask.any() or not doc_mask.any():
        logger.warning(f"No embeddings for {dataset_name}")
        return [], np.array([]), np.array([])

    ds_query_emb = query_embeddings[query_mask]
    ds_query_ids = query_ids[query_mask]
    ds_doc_emb = doc_embeddings[doc_mask]
    ds_doc_ids = doc_ids[doc_mask]

    # Get local query IDs (used in qrels)
    ds_query_local_ids = [m["local_id"] for m in query_metadata if m["dataset"] == dataset_name]
    ds_doc_local_ids = [m["local_id"] for m in doc_metadata if m["dataset"] == dataset_name]

    # Normalize for cosine similarity
    query_norm = ds_query_emb / np.linalg.norm(ds_query_emb, axis=1, keepdims=True)
    doc_norm = ds_doc_emb / np.linalg.norm(ds_doc_emb, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = query_norm @ doc_norm.T

    # Compute recall using local IDs
    recalls = compute_recall_at_k(
        similarity_matrix, qrels, ds_query_local_ids, ds_doc_local_ids, k=10
    )

    # Get query texts
    queries_dict = dataset.get_queries()
    query_texts = [queries_dict.get(qid, f"query_{qid}") for qid in ds_query_local_ids]

    return query_texts, similarity_matrix, recalls


def extract_features(
    query_texts: List[str],
    similarity_matrix: np.ndarray,
    extractor: GeneralQPPExtractor,
) -> np.ndarray:
    """Extract GeneralQPP features for all queries."""
    features = []
    for i, query in enumerate(query_texts):
        sims = similarity_matrix[i]
        feat = extractor.extract(query, sims)
        features.append(feat.to_vector())
    return np.array(features)


def main():
    logger.info("=" * 60)
    logger.info("Training General QPP Model on ViDoRe V3")
    logger.info(f"Architecture: MLP{HIDDEN_SIZES}, alpha={ALPHA}")
    logger.info("=" * 60)

    # Load all embeddings
    query_emb, query_ids, doc_emb, doc_ids, query_meta, doc_meta = load_embeddings()

    # Initialize extractor
    extractor = GeneralQPPExtractor(tau=TAU)

    # Process each dataset
    all_features = []
    all_labels = []
    dataset_stats = []

    for dataset_name in ENGLISH_DATASETS:
        logger.info(f"\nProcessing {dataset_name}...")

        query_texts, sim_matrix, recalls = process_dataset(
            dataset_name, query_emb, query_ids, doc_emb, doc_ids, query_meta, doc_meta
        )

        if len(query_texts) == 0:
            logger.warning(f"Skipping {dataset_name} - no data")
            continue

        # Extract features
        logger.info(f"  Extracting features for {len(query_texts)} queries...")
        features = extract_features(query_texts, sim_matrix, extractor)

        # Binary labels: Recall@10 >= 0.5
        labels = (recalls >= 0.5).astype(int)

        all_features.append(features)
        all_labels.append(labels)

        pos_rate = labels.mean()
        logger.info(f"  {len(labels)} queries, {pos_rate:.1%} positive")
        dataset_stats.append({
            "dataset": dataset_name,
            "n_queries": len(labels),
            "positive_rate": float(pos_rate),
        })

    if not all_features:
        logger.error("No data collected! Check dataset paths and qrels.")
        sys.exit(1)

    # Combine all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    logger.info(f"\nTotal: {len(y)} queries, {y.mean():.1%} positive")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Train model
    logger.info(f"\nTraining MLP{HIDDEN_SIZES} with alpha={ALPHA}...")
    model = MLPModel(
        hidden_sizes=HIDDEN_SIZES,
        alpha=ALPHA,
        max_iter=MAX_ITER,
        early_stopping=True,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict_proba(X_test)

    auroc = roc_auc_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred)

    logger.info(f"\nTest Metrics:")
    logger.info(f"  AUROC: {auroc:.4f}")
    logger.info(f"  AUPRC: {auprc:.4f}")
    logger.info(f"  Brier: {brier:.4f}")

    # Save model
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_MODEL)
    logger.info(f"\nModel saved to {OUTPUT_MODEL}")

    # Save config
    config = {
        "model_type": "mlp",
        "hidden_sizes": list(HIDDEN_SIZES),
        "alpha": ALPHA,
        "n_features": extractor.n_features(),
        "feature_extractor": "GeneralQPPExtractor",
        "tau": TAU,
        "training_data": "vidore_v3",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_auroc": float(auroc),
        "test_auprc": float(auprc),
        "test_brier": float(brier),
        "dataset_stats": dataset_stats,
    }

    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {OUTPUT_CONFIG}")

    logger.info("\nDone!")
    return auroc


if __name__ == "__main__":
    main()
