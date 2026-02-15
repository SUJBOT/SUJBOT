"""
RAG Confidence Scorer - Main Entry Point.

Provides confidence scoring for RAG retrieval results using a supervised
MLP model trained on QPP (Query Performance Prediction) features.

Usage:
    from rag_confidence import score_retrieval, V2Config

    # Simple usage (v1 mode)
    result = score_retrieval(query_text, similarities)
    # {"confidence": 0.85, "band": "MEDIUM"}

    # With v2 config (UQPP/SCA disabled by default)
    config = V2Config()
    result = score_retrieval_v2(query_text, similarities, config=config)

Note: UQPP (Dense-QPP) was evaluated but showed no improvement over the
supervised model. It's disabled by default. See _experimental/uqpp/ for
research code.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import V2Config

logger = logging.getLogger(__name__)

# Singleton instances for model and extractors
_v1_model = None
_qpp_extractor = None
_query_extractor = None
_sca_assessor = None

# Singleton instances for general model (23 features, domain-agnostic)
_general_model = None
_general_extractor = None


def _load_v1_model():
    """Load v1 model and extractors (singleton pattern)."""
    global _v1_model, _qpp_extractor, _query_extractor

    if _v1_model is not None:
        return _v1_model, _qpp_extractor, _query_extractor

    from .core.qpp_extractor import ImprovedQPPExtractor
    from .core.query_feature_extractor import QueryFeatureExtractor

    # Load trained model
    model_path = Path(__file__).parent / "models" / "qpp_mlp_query_augmented_model.pkl"
    with open(model_path, "rb") as f:
        _v1_model = pickle.load(f)

    # Initialize extractors
    _qpp_extractor = ImprovedQPPExtractor(tau=0.711)
    _query_extractor = QueryFeatureExtractor()

    logger.info("Loaded v1 model and extractors")

    return _v1_model, _qpp_extractor, _query_extractor


def _get_sca_assessor(model: str = "claude-haiku-4-5"):
    """Get or create SCA assessor (singleton)."""
    global _sca_assessor

    if _sca_assessor is None:
        from ._experimental.sca import SufficiencyAssessor

        _sca_assessor = SufficiencyAssessor(model=model)

    return _sca_assessor


def _load_general_model():
    """Load general QPP model (23 features, domain-agnostic).

    This model uses only language-agnostic and domain-agnostic features,
    making it suitable for cross-domain transfer and vision RAG applications.
    """
    global _general_model, _general_extractor

    if _general_model is not None:
        return _general_model, _general_extractor

    from .core.general_qpp_extractor import GeneralQPPExtractor
    from .core.qpp_model import load_model

    # Load 23-feature model
    model_path = Path(__file__).parent / "models" / "general_qpp_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"General QPP model not found at {model_path}. "
            "Train it first with: uv run python rag_confidence/scripts/train_general_qpp.py"
        )

    _general_model = load_model(model_path)

    # Initialize extractor with default tau
    _general_extractor = GeneralQPPExtractor(tau=0.711)

    logger.info("Loaded general QPP model (23 features, domain-agnostic)")
    return _general_model, _general_extractor


def _compute_band(confidence: float) -> str:
    """Compute confidence band from confidence score."""
    if confidence >= 0.90:
        return "HIGH"
    elif confidence >= 0.75:
        return "MEDIUM"
    elif confidence >= 0.50:
        return "LOW"
    else:
        return "VERY_LOW"


def score_retrieval_v1(query_text: str, similarities: np.ndarray) -> Dict[str, Any]:
    """
    Score retrieval using v1 model (supervised QPP only).

    This is the original v1 function for backward compatibility.

    Args:
        query_text: Query text
        similarities: Similarity scores to all chunks, shape [n_chunks]

    Returns:
        {"confidence": float, "band": str}
    """
    model, qpp_extractor, query_extractor = _load_v1_model()

    # Extract features
    qpp_features = qpp_extractor.extract(similarities)
    query_features = query_extractor.extract(query_text, similarities)

    # Combine features (14 QPP + 18 query = 32 total)
    features = np.concatenate([qpp_features.to_vector(), query_features.to_vector()])

    # Predict
    confidence = float(np.clip(model.predict([features])[0], 0, 1))
    band = _compute_band(confidence)

    return {"confidence": confidence, "band": band}


def score_retrieval_general(
    query_text: str,
    similarities: np.ndarray,
    *,
    return_features: bool = False,
) -> Dict[str, Any]:
    """
    Score retrieval using general QPP model (23 features).

    Domain-agnostic and language-agnostic - works with any embedding type
    including vision embeddings (ColPali, SigLIP, etc.).

    This function is suitable for:
    - Cross-domain transfer (trained on legal, applied to medical)
    - Cross-language transfer (trained on Czech, applied to English)
    - Vision RAG (similarities from image embeddings instead of text)

    Args:
        query_text: Query text string (still text-based, even for vision RAG)
        similarities: Similarity scores between query and all documents/pages.
                      Shape: (n_docs,) or (n_pages,).
                      Can be from text embeddings OR vision embeddings.
        return_features: If True, include extracted features in response.

    Returns:
        {"confidence": float, "band": str}
        If return_features=True: {"confidence": float, "band": str, "features": dict}

    Example (text RAG):
        similarities = compute_cosine_similarity(query_emb, chunk_embs)
        result = score_retrieval_general("What are the limits?", similarities)

    Example (vision RAG with ColPali):
        # Colleague computes: page_embs = colpali.encode(page_images)
        # You compute: similarities = cosine_similarity(query_emb, page_embs)
        result = score_retrieval_general("What are the safety limits?", similarities)

    Features used (23 total):
        - Distribution features (14): top1_minus_p99, sim_std_top10, bimodal_gap, etc.
        - Query features (5): query_char_len, query_word_count, has_numbers, etc.
        - Global similarity (4): sim_mean_all, sim_median_all, sim_max, etc.
    """
    model, extractor = _load_general_model()

    # Extract 23 general features
    features = extractor.extract(query_text, similarities)
    feature_vector = features.to_vector()

    # Predict confidence
    confidence = float(model.predict_proba(feature_vector.reshape(1, -1))[0])
    band = _compute_band(confidence)

    result = {"confidence": confidence, "band": band}

    if return_features:
        result["features"] = features.to_dict()

    return result


def score_retrieval_v2(
    query_text: str,
    similarities: np.ndarray,
    *,
    topk_doc_ids: Optional[List[str]] = None,
    topk_doc_embeddings: Optional[np.ndarray] = None,
    topk_chunk_texts: Optional[List[str]] = None,
    query_embedding: Optional[np.ndarray] = None,
    retriever: Optional[Callable[[np.ndarray, int], List[str]]] = None,
    config: Optional[V2Config] = None,
) -> Dict[str, Any]:
    """
    Score retrieval using v2 system (v1 + UQPP + optional SCA).

    When config=None, behaves exactly like v1 (backward compatible).

    Args:
        query_text: Query text
        similarities: Similarity scores to all chunks, shape [n_chunks]

        # Optional v2 inputs
        topk_doc_ids: Top-k document/chunk IDs for stability computation
        topk_doc_embeddings: Top-k document embeddings [k, d] for coherence
        topk_chunk_texts: Top-k chunk texts for SCA
        query_embedding: Query embedding [d] for stability
        retriever: Callback function(embedding, k) -> list[str] for stability
        config: V2Config (None = v1 mode)

    Returns:
        v1 mode: {"confidence": float, "band": str}
        v2 mode: {"confidence": float, "band": str, "uqpp": {...}, "sca": {...}, "p_final": float}

    Example:
        # v1 mode
        result = score_retrieval_v2("What are the limits?", similarities)

        # v2 mode with UQPP only
        config = V2Config(enable_uqpp=True, enable_sca=False)
        result = score_retrieval_v2(
            query_text="What are the limits?",
            similarities=similarities,
            topk_doc_embeddings=chunk_embeddings,
            config=config,
        )
    """
    # =========================================================================
    # Step 1: Compute v1 supervised confidence (always)
    # =========================================================================
    v1_result = score_retrieval_v1(query_text, similarities)
    p_sup = v1_result["confidence"]
    band = v1_result["band"]

    # v1 mode: return immediately if no config
    if config is None:
        return {"confidence": p_sup, "band": band}

    # Check if any v2 features are enabled
    if not config.enable_uqpp and not config.enable_sca:
        return {"confidence": p_sup, "band": band}

    # Build output dict
    out = {"confidence": p_sup, "band": band}

    # =========================================================================
    # Step 2: UQPP signals (disabled by default - evaluation showed no improvement)
    # NOTE: Dense-QPP was evaluated but degraded AUROC by 0.008-0.055.
    #       Coherence was removed earlier (redundant with QPP features).
    #       UQPP stability kept for future research but disabled in production.
    # =========================================================================
    uqpp_result = {
        "u_stability": None,
        "u_score": None,
    }

    # UQPP stability disabled: Dense-QPP evaluation showed degraded performance.
    # See rag_confidence/evaluation/compare_classification.py results.
    # Keeping code for research but not using in p_final calculation.

    out["uqpp"] = uqpp_result

    # =========================================================================
    # Step 3: Compute SCA if enabled (conditional)
    # =========================================================================
    sca_result = {"p_suff": None, "p_chunk": None, "method": None}

    if config.enable_sca and topk_chunk_texts is not None:
        # Determine if SCA should be triggered
        # (Run SCA when uncertain or stability is low)
        trigger_sca = (
            config.T_LOW <= p_sup < config.T_HIGH
            or (uqpp_result["u_stability"] is not None and uqpp_result["u_stability"] <= config.U_LOW)
        )

        if trigger_sca:
            sca_assessor = _get_sca_assessor(config.sca_model)

            # Format chunks for SCA
            chunks = []
            for i, text in enumerate(topk_chunk_texts[: config.sca_max_chunks]):
                chunk_id = topk_doc_ids[i] if topk_doc_ids and i < len(topk_doc_ids) else f"chunk_{i}"
                chunks.append({"chunk_id": chunk_id, "content": text})

            # Run assessment
            assessment = sca_assessor.assess(
                query=query_text,
                chunks=chunks,
                triggered_by="low_confidence" if p_sup < config.T_MED else "low_uqpp",
            )

            sca_result = {
                "p_suff": assessment.p_suff,
                "p_chunk": assessment.p_chunk,
                "method": assessment.method,
            }

    out["sca"] = sca_result

    # =========================================================================
    # Step 4: Compute p_final (combined confidence)
    # =========================================================================
    p_final = _compute_p_final(
        p_sup=p_sup,
        u_score=uqpp_result["u_score"],
        p_suff=sca_result["p_suff"],
        config=config,
    )

    out["p_final"] = p_final

    return out


def _compute_p_final(
    p_sup: float,
    u_score: Optional[float],
    p_suff: Optional[float],
    config: V2Config,
) -> float:
    """
    Compute final combined confidence score.

    Combines supervised confidence (p_sup), UQPP score (u_score),
    and SCA sufficiency (p_suff) using configurable weights.

    Default formula:
    - Base: p_final = w_sup * p_sup
    - If u_score: p_final += w_uqpp * u_score
    - If p_suff: p_final = max(p_final, p_suff)

    Args:
        p_sup: Supervised confidence [0, 1]
        u_score: UQPP combined score [0, 1] or None
        p_suff: SCA sufficiency [0, 1] or None
        config: V2Config with weights

    Returns:
        Combined confidence score [0, 1]
    """
    # Start with supervised confidence
    if u_score is not None:
        # Normalize weights to sum to 1 when UQPP available
        w_total = config.p_final_w_sup + config.p_final_w_uqpp
        p_final = (config.p_final_w_sup * p_sup + config.p_final_w_uqpp * u_score) / w_total
    else:
        p_final = p_sup

    # SCA can override if it's higher (strong evidence of sufficiency)
    if p_suff is not None:
        p_final = max(p_final, p_suff)

    return float(np.clip(p_final, 0.0, 1.0))


# Convenience function for backward compatibility
def score_retrieval(query_text: str, similarities: np.ndarray) -> Dict[str, Any]:
    """
    Score retrieval (v1 API - backward compatible).

    Same as score_retrieval_v1(). Use score_retrieval_v2() for v2 features.
    """
    return score_retrieval_v1(query_text, similarities)
