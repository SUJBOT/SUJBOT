"""
RAG Confidence - Confidence scoring for RAG retrieval results.

This package provides confidence estimation for Retrieval-Augmented Generation (RAG)
systems using supervised QPP (Query Performance Prediction) features.

Quick Start:
    from rag_confidence import score_retrieval

    # Get confidence score
    result = score_retrieval(query_text, similarity_scores)
    print(f"Confidence: {result['confidence']:.2f}, Band: {result['band']}")

Vision RAG / Cross-Domain:
    from rag_confidence import score_retrieval_general

    # Works with any embeddings (text, vision, multi-modal)
    result = score_retrieval_general(query_text, similarity_scores)

Components:
    - score_retrieval: Main scoring function (v1 API, 32 features)
    - score_retrieval_general: Domain-agnostic scoring (23 features, for vision RAG)
    - score_retrieval_v2: Extended API with optional UQPP/SCA
    - V2Config: Configuration for v2 features
    - ConformalPredictor: Conformal prediction for uncertainty quantification
    - GeneralQPPExtractor: Feature extractor for cross-domain/vision RAG

Directory Structure:
    rag_confidence/
    ├── __init__.py      # This file - public API
    ├── scorer.py        # Main scoring logic
    ├── config.py        # Configuration
    ├── conformal_predictor.py  # Conformal prediction
    ├── core/            # Core feature extractors
    ├── models/          # Trained production models
    ├── evaluation/      # Evaluation scripts
    ├── data/            # Data files (matrices, datasets)
    ├── docs/            # Documentation
    ├── scripts/         # Training & utility scripts
    ├── tests/           # Test suite
    └── _experimental/   # Archived experimental code (UQPP, SCA)
"""

from .config import V2Config, DenseQPPConfig, StabilityConfig
from .scorer import (
    score_retrieval,
    score_retrieval_v1,
    score_retrieval_v2,
    score_retrieval_general,
)
from .conformal_predictor import ConformalPredictor
from .core.general_qpp_extractor import GeneralQPPExtractor

__all__ = [
    # Main API
    "score_retrieval",
    "score_retrieval_v1",
    "score_retrieval_v2",
    # Cross-domain / Vision RAG API
    "score_retrieval_general",
    "GeneralQPPExtractor",
    # Configuration
    "V2Config",
    "DenseQPPConfig",
    "StabilityConfig",
    # Conformal prediction
    "ConformalPredictor",
]

__version__ = "2.1.0"
