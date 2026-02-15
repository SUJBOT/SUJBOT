"""
Core feature extraction components for RAG Confidence.

This module contains the production feature extractors used by the scorer:
- ImprovedQPPExtractor: QPP (Query Performance Prediction) features from similarity scores
- QueryFeatureExtractor: Query-specific features (length, complexity, etc.)
- GeneralQPPExtractor: Language-agnostic QPP features for benchmarking
- QPPModelFactory: Pluggable model interface for QPP classifiers
- LLMSufficiencyAssessor: LLM-based sufficiency assessment
"""

from .qpp_extractor import ImprovedQPPExtractor, QPPFeatures
from .query_feature_extractor import QueryFeatureExtractor, QueryFeatures
from .general_qpp_extractor import GeneralQPPExtractor, GeneralQPPFeatures
from .qpp_model import QPPModelFactory, BaseQPPModel, LogisticRegressionModel, MLPModel
from .llm_assessor import LLMSufficiencyAssessor, AssessmentResult

__all__ = [
    # Production extractors
    "ImprovedQPPExtractor",
    "QPPFeatures",
    "QueryFeatureExtractor",
    "QueryFeatures",
    # Generalized benchmark components
    "GeneralQPPExtractor",
    "GeneralQPPFeatures",
    "QPPModelFactory",
    "BaseQPPModel",
    "LogisticRegressionModel",
    "MLPModel",
    "LLMSufficiencyAssessor",
    "AssessmentResult",
]
