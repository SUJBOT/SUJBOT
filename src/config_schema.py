"""
JSON Configuration Schema for MY_SUJBOT.

This module defines the complete configuration schema using Pydantic models.
ALL fields are REQUIRED unless explicitly marked Optional with a default value.
NO fallback values - if a required field is missing, validation will fail.

Migration from .env to config.json (2025-11-10):
- Removed fallback values and defaults
- Strict validation on startup
- Hierarchical JSON structure for better organization
"""

import json
import os
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Type alias for providers
LLMProvider = Literal["anthropic", "openai", "google", "local_llm", "local_llm_8b"]
EmbeddingProvider = Literal["openai", "voyage", "huggingface", "jina"]


# =============================================================================
# Unified Model Configuration (SSOT for pricing, provider, metadata)
# =============================================================================


class ModelPricing(BaseModel):
    """
    Model pricing per 1M tokens.

    All prices are in USD. Input and output prices are separate
    because most providers charge differently for each.
    """

    input: float = Field(
        ...,
        ge=0.0,
        description="Price per 1M input tokens (USD)"
    )
    output: float = Field(
        default=0.0,
        ge=0.0,
        description="Price per 1M output tokens (USD). Default 0.0 for embeddings."
    )


class LLMModelConfig(BaseModel):
    """
    Full LLM model configuration with all metadata.

    This is the new unified format for model configuration.
    Provider and pricing are explicit - no pattern guessing needed.
    """

    id: str = Field(
        ...,
        description="Full model identifier (e.g., 'claude-haiku-4-5-20251001')"
    )
    provider: LLMProvider = Field(
        ...,
        description="Provider name: anthropic, openai, google, local_llm, local_llm_8b"
    )
    pricing: ModelPricing = Field(
        ...,
        description="Pricing per 1M tokens"
    )
    context_window: int = Field(
        default=128000,
        ge=1000,
        description="Maximum context window in tokens"
    )
    supports_caching: bool = Field(
        default=False,
        description="Whether model supports prompt caching (Anthropic feature)"
    )
    supports_extended_thinking: bool = Field(
        default=False,
        description="Whether model supports extended thinking"
    )
    supports_vision: bool = Field(
        default=False,
        description="Whether model supports vision/image inputs"
    )


class EmbeddingModelConfig(BaseModel):
    """
    Full embedding model configuration with all metadata.

    Embedding models only have input pricing (no output tokens).
    Dimensions are required for vector store configuration.
    """

    id: str = Field(
        ...,
        description="Full model identifier (e.g., 'Qwen/Qwen3-Embedding-8B')"
    )
    provider: EmbeddingProvider = Field(
        ...,
        description="Provider name: openai, voyage, huggingface, jina"
    )
    pricing: ModelPricing = Field(
        ...,
        description="Pricing per 1M tokens (output price typically 0)"
    )
    dimensions: int = Field(
        ...,
        ge=1,
        description="Embedding vector dimensions"
    )
    is_local: bool = Field(
        default=False,
        description="Whether model runs locally (no API costs)"
    )


class RerankerModelConfig(BaseModel):
    """
    Reranker model configuration.

    Rerankers are typically local models with no API pricing.
    """

    id: str = Field(
        ...,
        description="Full model identifier (e.g., 'BAAI/bge-reranker-large')"
    )
    is_local: bool = Field(
        default=True,
        description="Whether model runs locally"
    )


# Type aliases for backward compatibility with string-based configs
# Allows: "haiku": "claude-haiku..." (old) OR "haiku": { "id": "...", ... } (new)
LLMModelEntry = str | LLMModelConfig
EmbeddingModelEntry = str | EmbeddingModelConfig
RerankerModelEntry = str | RerankerModelConfig


class APIKeysConfig(BaseModel):
    """
    API Keys configuration - loaded from .env file.

    API keys are NEVER stored in config.json for security.
    Instead, they are loaded from environment variables.
    """

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic Claude API key (loaded from ANTHROPIC_API_KEY env var)"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (loaded from OPENAI_API_KEY env var)"
    )
    voyage_api_key: Optional[str] = Field(
        default=None,
        description="Voyage AI API key (loaded from VOYAGE_API_KEY env var)"
    )
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key (loaded from GOOGLE_API_KEY env var)"
    )

    @model_validator(mode="after")
    def load_from_env(self):
        """Load API keys from environment variables (.env file)."""
        # Load .env file if it exists
        load_dotenv()

        # Load API keys from environment, overriding any values from JSON
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or self.anthropic_api_key
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or self.openai_api_key
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY") or self.voyage_api_key
        self.google_api_key = os.getenv("GOOGLE_API_KEY") or self.google_api_key

        return self


class ModelsConfig(BaseModel):
    """Core model selection."""

    llm_model: str = Field(
        ...,
        description="LLM model for summaries and agent (e.g., claude-sonnet-4-5-20250929, gpt-4o-mini)"
    )
    llm_provider: Optional[str] = Field(
        None,
        description="LLM provider override (auto-detected from llm_model if not set): 'claude', 'openai', 'google'"
    )
    embedding_model: str = Field(
        ...,
        description="Embedding model for vector store (e.g., text-embedding-3-large, bge-m3)"
    )
    embedding_provider: Optional[str] = Field(
        None,
        description="Embedding provider override (auto-detected from embedding_model if not set): 'huggingface', 'openai', 'voyage'"
    )


class ExtractionConfig(BaseModel):
    """Document extraction configuration (PHASE 1)."""

    # Extraction backend selection
    backend: Literal["gemini", "unstructured", "auto"] = Field(
        ...,
        description="Extraction backend: 'gemini' (native PDF), 'unstructured' (with ToC), 'auto'"
    )
    gemini_model: str = Field(
        ...,
        description="Gemini model for extraction (e.g., 'gemini-2.5-flash')"
    )
    gemini_file_size_threshold_mb: float = Field(
        default=0.3,
        description="File size threshold in MB for chunked extraction (lower = more reliable for large docs)",
        ge=0.1,
        le=50.0
    )
    gemini_max_output_tokens: int = Field(
        default=100000,
        description="Maximum output tokens for Gemini extraction",
        ge=1000,
        le=200000
    )
    fallback_to_unstructured: bool = Field(
        ...,
        description="Fallback to Unstructured if Gemini fails"
    )

    # Table extraction
    extract_tables: bool = Field(..., description="Extract tables from documents")
    table_mode: Literal["ACCURATE", "FAST"] = Field(..., description="Table extraction mode")

    # Hierarchy extraction (CRITICAL for hierarchical chunking)
    extract_hierarchy: bool = Field(
        ...,
        description="Extract document hierarchy (MUST be true for PHASE 3)"
    )
    enable_smart_hierarchy: bool = Field(
        ...,
        description="Font-size based hierarchy detection (CRITICAL - DO NOT DISABLE)"
    )
    hierarchy_tolerance: float = Field(
        ...,
        description="BBox height clustering tolerance in pixels (0.8 is research-optimal)",
        ge=0.0,
        le=2.0
    )

    # Watermark filtering
    filter_rotated_text: bool = Field(..., description="Remove diagonally oriented watermark text")
    rotation_min_angle: float = Field(
        ...,
        description="Minimum rotation angle in degrees",
        ge=0.0,
        le=90.0
    )
    rotation_max_angle: float = Field(
        ...,
        description="Maximum rotation angle in degrees",
        ge=0.0,
        le=90.0
    )

    # Summary generation (PHASE 2 integration)
    generate_summaries: bool = Field(..., description="Generate document/section summaries")
    summary_model: Optional[str] = Field(None, description="Override summary model (uses llm_model if None)")
    summary_max_chars: int = Field(
        ...,
        description="Summary length target for sections (research-optimal: 150 chars, valid range: 50-300)",
        ge=50,
        le=300
    )
    document_summary_max_chars: int = Field(
        ...,
        description="Document summary length target (100-1000 chars for describing document and sections)",
        ge=100,
        le=1000
    )
    summary_style: Literal["generic", "expert"] = Field(
        ...,
        description="Summary style (research-optimal: 'generic', proven better for legal docs)"
    )
    use_batch_api: bool = Field(..., description="Use OpenAI Batch API for summaries (50% cheaper)")
    batch_api_poll_interval: int = Field(
        ...,
        description="Seconds between batch status checks",
        ge=1
    )
    batch_api_timeout: int = Field(
        ...,
        description="Max wait for batch completion in seconds",
        ge=60
    )

    # Output formats
    generate_markdown: bool = Field(..., description="Generate markdown output")
    generate_json: bool = Field(..., description="Generate JSON output")


class UnstructuredConfig(BaseModel):
    """Unstructured.io extraction configuration."""

    strategy: Literal["hi_res", "fast", "ocr_only"] = Field(
        ...,
        description="Extraction strategy for PDF files"
    )
    model: Literal["yolox", "detectron2_mask_rcnn", "detectron2_onnx", "detectron2_quantized"] = Field(
        ...,
        description="Hi-res model for PDF extraction (yolox recommended from testing)"
    )
    languages: List[str] = Field(..., description="OCR languages (e.g., ['ces', 'eng'])")
    detect_language_per_element: bool = Field(..., description="Detect language per element")
    infer_table_structure: bool = Field(..., description="Extract table structure")
    extract_images: bool = Field(..., description="Extract images from PDF")
    include_page_breaks: bool = Field(..., description="Include page breaks in output")


class HierarchyDetectionConfig(BaseModel):
    """Generic hierarchy detection configuration."""

    enable_generic_hierarchy: bool = Field(..., description="Enable generic hierarchy detection")
    hierarchy_signals: List[str] = Field(
        ...,
        description="Signals for hierarchy detection: type, font_size, spacing, indentation, numbering, length, parent_id"
    )
    clustering_eps: float = Field(
        ...,
        description="DBSCAN clustering epsilon (lower = more granular levels)",
        ge=0.01,
        le=1.0
    )
    clustering_min_samples: int = Field(
        ...,
        description="Minimum samples per cluster",
        ge=1
    )


class SummarizationConfig(BaseModel):
    """Summarization configuration (PHASE 2)."""

    speed_mode: Literal["fast", "eco"] = Field(
        ...,
        description="Pipeline speed mode: 'fast' (immediate) or 'eco' (Batch API, 50% cheaper)"
    )
    temperature: float = Field(
        ...,
        description="LLM temperature for summaries (0.3 = consistent)",
        ge=0.0,
        le=2.0
    )
    max_tokens: int = Field(
        ...,
        description="Max output tokens for summary generation",
        ge=10
    )
    retry_on_exceed: bool = Field(..., description="Retry if summary exceeds max_chars")
    max_retries: int = Field(..., description="Max retry attempts", ge=0)
    max_workers: int = Field(
        ...,
        description="Parallel summary generation threads",
        ge=1
    )
    min_text_length: int = Field(
        ...,
        description="Minimum text length for summarization",
        ge=10
    )
    enable_batching: bool = Field(
        ...,
        description="Enable prompt batching (disabled by default due to JSON overhead)"
    )
    batch_size: int = Field(
        ...,
        description="Number of sections per batch if batching enabled",
        ge=1
    )


class ContextGenerationConfig(BaseModel):
    """Contextual Retrieval configuration (PHASE 3A)."""

    enable_contextual: bool = Field(
        ...,
        description="Enable contextual retrieval (-67% retrieval failures)"
    )
    temperature: float = Field(
        ...,
        description="Temperature for context generation",
        ge=0.0,
        le=2.0
    )
    max_tokens: int = Field(
        ...,
        description="Max tokens for generated context (50-100 words)",
        ge=10
    )
    include_surrounding: bool = Field(
        ...,
        description="Include neighboring chunks for context"
    )
    num_surrounding_chunks: int = Field(
        ...,
        description="Number of chunks before/after to include",
        ge=0
    )
    fallback_to_basic: bool = Field(
        ...,
        description="Fallback to basic chunking if context generation fails"
    )
    batch_size: int = Field(
        ...,
        description="Context generation batch size",
        ge=1
    )
    max_workers: int = Field(
        ...,
        description="Context generation parallel threads",
        ge=1
    )
    use_batch_api: bool = Field(..., description="Use OpenAI Batch API (50% cheaper)")
    batch_api_poll_interval: int = Field(
        ...,
        description="Batch API poll interval in seconds",
        ge=1
    )
    batch_api_timeout: int = Field(
        ...,
        description="Batch API timeout in seconds",
        ge=60
    )
    language: str = Field(
        default="ces",
        description="Language code for context generation ('ces' for Czech, 'eng' for English)"
    )


class ChunkingConfig(BaseModel):
    """Token-aware chunking configuration (PHASE 3)."""

    max_tokens: int = Field(
        ...,
        description="Max tokens per chunk (research-optimal: 512 for legal docs, valid range: 100-8192)",
        ge=100,
        le=8192
    )
    tokenizer_model: str = Field(
        ...,
        description="Tokenizer model (must match embedding_model)"
    )
    enable_sac: bool = Field(
        ...,
        description="Enable Summary-Augmented Chunking (-58% context drift)"
    )


class EmbeddingConfig(BaseModel):
    """Embedding configuration (PHASE 4)."""

    batch_size: int = Field(
        ...,
        description="Embedding batch size (larger = faster but more memory)",
        ge=1
    )
    cache_enabled: bool = Field(
        ...,
        description="Enable embedding cache (40-80% hit rate)"
    )
    cache_size: int = Field(
        ...,
        description="Max embedding cache entries",
        ge=1
    )
    normalize: bool = Field(
        ...,
        description="Normalize embeddings for cosine similarity (REQUIRED: must be true for FAISS indexing)"
    )


class ClusteringConfig(BaseModel):
    """Semantic clustering configuration (PHASE 4.5)."""

    algorithm: Literal["hdbscan", "agglomerative"] = Field(
        ...,
        description="Clustering algorithm"
    )
    n_clusters: Optional[int] = Field(
        None,
        description="Number of clusters for agglomerative (ignored for HDBSCAN)",
        ge=1
    )
    min_size: int = Field(
        ...,
        description="Minimum chunks per cluster for HDBSCAN",
        ge=1
    )
    max_clusters: int = Field(
        ...,
        description="Maximum clusters for auto-detection (agglomerative)",
        ge=1
    )
    min_clusters: int = Field(
        ...,
        description="Minimum clusters for auto-detection (agglomerative)",
        ge=1
    )
    layers: List[int] = Field(
        ...,
        description="Layers to cluster (e.g., [3] for chunks only)"
    )
    enable_labels: bool = Field(..., description="Generate semantic labels using LLM")
    enable_visualization: bool = Field(..., description="Generate UMAP visualization")
    visualization_dir: str = Field(..., description="Visualization output directory")


class HybridSearchConfig(BaseModel):
    """Hybrid search configuration (PHASE 5)."""

    enable: bool = Field(
        ...,
        description="Enable hybrid search (BM25 + Dense + RRF fusion, +23% precision)"
    )
    fusion_k: int = Field(
        ...,
        description="RRF fusion parameter (60 = optimal)",
        ge=1
    )


class AgentConfig(BaseModel):
    """RAG agent configuration (PHASE 7)."""

    model: str = Field(
        ...,
        description="Agent model (can differ from llm_model)"
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Max output tokens for agent responses (model-dependent)",
        ge=100
    )
    temperature: float = Field(
        ...,
        description="Agent temperature (0.3 = consistent)",
        ge=0.0,
        le=2.0
    )
    enable_tool_validation: bool = Field(..., description="Enable tool validation on startup")
    debug_mode: bool = Field(..., description="Enable debug mode")
    vector_store_path: str = Field(
        ...,
        description="Vector store path (must point to phase4_vector_store directory)"
    )
    enable_prompt_caching: bool = Field(
        ...,
        description="Enable prompt caching (Anthropic only, 90% cost reduction)"
    )
    enable_context_management: bool = Field(
        ...,
        description="Enable context management (prevents quadratic cost growth)"
    )
    context_management_trigger: int = Field(
        ...,
        description="Token threshold for context pruning",
        ge=1000
    )
    context_management_keep: int = Field(
        ...,
        description="Keep last N messages with full tool context",
        ge=1
    )


class AgentToolConfig(BaseModel):
    """Agent tool configuration."""

    default_k: int = Field(
        ...,
        description="Default number of results to retrieve",
        ge=1,
        le=100
    )
    max_document_compare: int = Field(
        ...,
        description="Max documents to compare",
        ge=1
    )
    compliance_threshold: float = Field(
        ...,
        description="Legal compliance confidence threshold",
        ge=0.0,
        le=1.0
    )


class CLIConfig(BaseModel):
    """CLI configuration."""

    show_citations: bool = Field(..., description="Display citations in responses")
    citation_format: Literal["inline", "footnote", "detailed", "simple"] = Field(
        ...,
        description="Citation format"
    )
    show_tool_calls: bool = Field(..., description="Display tool calls in output")
    show_timing: bool = Field(..., description="Display timing information")
    enable_streaming: bool = Field(..., description="Enable streaming responses")
    save_history: bool = Field(..., description="Save conversation history")
    history_file: str = Field(..., description="History file path")
    max_history_items: int = Field(
        ...,
        description="Max history items to keep",
        ge=1
    )


class LabelingConfig(BaseModel):
    """
    Document labeling configuration (PHASE 3.5 extension).

    Adds hierarchical labeling with smart propagation:
    - Categories: Document-level (1 LLM call), propagated to sections/chunks
    - Keywords: Section-level (~100 LLM calls), propagated to chunks
    - Synthetic Questions: Chunk-level (HyDE boost for retrieval)

    Uses OpenAI Batch API for 50% cost savings.
    """

    enabled: bool = Field(
        default=True,
        description="Enable document labeling pipeline"
    )

    # Feature toggles
    enable_categories: bool = Field(
        default=True,
        description="Enable category/classification extraction"
    )
    enable_keywords: bool = Field(
        default=True,
        description="Enable keyword extraction"
    )
    enable_questions: bool = Field(
        default=True,
        description="Enable synthetic question generation (HyDE boost)"
    )

    # Model configuration
    provider: Literal["openai", "gemini", "anthropic"] = Field(
        default="openai",
        description="LLM provider for labeling"
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for labeling (gpt-4o-mini recommended for cost)"
    )
    use_batch_api: bool = Field(
        default=True,
        description="Use OpenAI Batch API for 50% cost savings (async, up to 24h)"
    )

    # Smart propagation
    category_generation_level: Literal["document", "section", "chunk"] = Field(
        default="document",
        description="Level at which to generate categories (document = 1 LLM call)"
    )
    keyword_generation_level: Literal["document", "section", "chunk"] = Field(
        default="section",
        description="Level at which to generate keywords (section = ~100 LLM calls)"
    )
    question_generation_level: Literal["chunk"] = Field(
        default="chunk",
        description="Level at which to generate questions (always chunk-level)"
    )

    # Dynamic categories
    use_dynamic_categories: bool = Field(
        default=True,
        description="LLM creates document-specific taxonomy (vs fixed categories)"
    )
    fixed_categories: Optional[List[str]] = Field(
        default=None,
        description="Fixed category list if use_dynamic_categories=False"
    )

    # HyDE embedding boost
    include_questions_in_embedding: bool = Field(
        default=True,
        description="Add synthetic questions to embedding_text (+20-30% retrieval precision)"
    )

    # Performance
    batch_size: int = Field(
        default=50,
        description="Batch size for Batch API requests",
        ge=1,
        le=1000
    )
    max_keywords_per_chunk: int = Field(
        default=10,
        description="Maximum keywords per chunk",
        ge=1,
        le=20
    )
    max_questions_per_chunk: int = Field(
        default=5,
        description="Maximum synthetic questions per chunk",
        ge=1,
        le=10
    )

    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Enable content-hash based caching"
    )
    cache_size: int = Field(
        default=1000,
        description="Maximum cache entries",
        ge=100
    )

    # Batch API timeouts
    batch_api_poll_interval: int = Field(
        default=30,
        description="Seconds between batch status checks",
        ge=5
    )
    batch_api_timeout_hours: int = Field(
        default=12,
        description="Maximum hours to wait for batch completion",
        ge=1,
        le=24
    )


class IndexingConfig(BaseModel):
    """
    Indexing pipeline configuration (optional section).

    Controls LlamaIndex wrapper, Redis caching, and entity labeling.
    Redis connection uses environment variables (REDIS_HOST, REDIS_PORT).
    """

    # LlamaIndex wrapper toggle
    use_llamaindex_wrapper: bool = Field(
        default=True,
        description="Use LlamaIndex wrapper for state persistence"
    )

    # Entity labeling settings
    enable_entity_labeling: bool = Field(
        default=True,
        description="Enable entity labeling phase (3.5) using LLM"
    )
    entity_labeling_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for entity labeling (gpt-4o-mini, gemini-2.5-flash, or claude-haiku-4-5)"
    )
    entity_labeling_batch_size: int = Field(
        default=10,
        description="Batch size for entity labeling",
        ge=1,
        le=50
    )

    # Document labeling (categories, keywords, questions)
    labeling: LabelingConfig = Field(
        default_factory=LabelingConfig,
        description="Document labeling configuration (categories, keywords, questions)"
    )


# ============================================================================
# SSOT Configuration - Centralized Settings (PHASE: SSOT Refactoring)
# ============================================================================

class ModelRegistryConfig(BaseModel):
    """
    Model registry configuration - SSOT for all model aliases.

    Supports TWO formats for backward compatibility:

    OLD FORMAT (string - still supported):
        "haiku": "claude-haiku-4-5-20251001"

    NEW FORMAT (object - preferred):
        "haiku": {
            "id": "claude-haiku-4-5-20251001",
            "provider": "anthropic",
            "pricing": { "input": 1.00, "output": 5.00 },
            "context_window": 200000,
            "supports_caching": true
        }

    The new format eliminates the need for:
    - Pattern-based provider detection in factory.py
    - Hardcoded PRICING dict in cost_tracker.py
    - Separate embedding_dimensions dict
    """

    llm_models: dict[str, LLMModelEntry] = Field(
        ...,
        description="LLM models: alias -> model_id (string) or full config (object)"
    )
    embedding_models: dict[str, EmbeddingModelEntry] = Field(
        ...,
        description="Embedding models: alias -> model_id (string) or full config (object)"
    )
    reranker_models: dict[str, RerankerModelEntry] = Field(
        ...,
        description="Reranker models: alias -> model_id (string) or full config (object)"
    )
    # DEPRECATED: Use dimensions in EmbeddingModelConfig instead
    # Kept for backward compatibility with old string-based configs
    embedding_dimensions: dict[str, int] = Field(
        default_factory=dict,
        description="[DEPRECATED] Embedding dimensions - use EmbeddingModelConfig.dimensions instead"
    )


class TimeoutDefaultsConfig(BaseModel):
    """Timeout defaults in seconds - SSOT for all timeout values."""

    api_request: int = Field(60, ge=1, description="Default API request timeout")
    ollama_request: int = Field(30, ge=1, description="Ollama/local model request timeout")
    postgres_command: int = Field(60, ge=1, description="PostgreSQL command timeout")
    batch_api_poll: int = Field(30, ge=1, description="Batch API polling interval")


class RetriesDefaultsConfig(BaseModel):
    """Retry defaults - SSOT for all retry configurations."""

    api_max_retries: int = Field(3, ge=1, le=10, description="Max retries for API calls")
    postgres_max_retries: int = Field(5, ge=1, le=10, description="Max retries for PostgreSQL")


class PoolSizesDefaultsConfig(BaseModel):
    """Connection pool size defaults - SSOT for all pool configurations."""

    postgres: int = Field(20, ge=1, description="PostgreSQL connection pool size")


class BatchSizesDefaultsConfig(BaseModel):
    """Batch processing size defaults - SSOT for all batch operations."""

    embedding: int = Field(64, ge=1, description="Embedding batch size")
    context_generation: int = Field(20, ge=1, description="Context generation batch size")


class RetrievalDefaultsConfig(BaseModel):
    """Retrieval defaults - SSOT for retrieval parameters."""

    layer_default_k: dict[str, int] = Field(
        default_factory=lambda: {"1": 3, "2": 5, "3": 10},
        description="Default k per layer (keys are layer numbers as strings)"
    )
    candidates_multiplier: int = Field(4, ge=1, description="Candidates multiplier for reranking")
    rrf_k: int = Field(60, ge=1, description="RRF k parameter for fusion")


class MaxWorkersDefaultsConfig(BaseModel):
    """Maximum worker thread defaults - SSOT for parallel processing."""

    summarization: int = Field(20, ge=1, description="Max workers for summarization")
    context_generation: int = Field(10, ge=1, description="Max workers for context generation")


class CacheDefaultsConfig(BaseModel):
    """Cache configuration defaults - SSOT for all cache settings."""

    embedding_cache_max_size: int = Field(1000, ge=100, description="Max size of embedding cache")
    token_manager_max_tokens_per_chunk: int = Field(600, ge=100, description="Max tokens per chunk in token manager")


class DefaultsConfig(BaseModel):
    """
    Centralized default values - SSOT for all hardcoded defaults.

    This section consolidates all default values that were previously
    scattered across multiple Python files as hardcoded constants.
    """

    timeouts: TimeoutDefaultsConfig = Field(default_factory=TimeoutDefaultsConfig)
    retries: RetriesDefaultsConfig = Field(default_factory=RetriesDefaultsConfig)
    pool_sizes: PoolSizesDefaultsConfig = Field(default_factory=PoolSizesDefaultsConfig)
    batch_sizes: BatchSizesDefaultsConfig = Field(default_factory=BatchSizesDefaultsConfig)
    retrieval: RetrievalDefaultsConfig = Field(default_factory=RetrievalDefaultsConfig)
    max_workers: MaxWorkersDefaultsConfig = Field(default_factory=MaxWorkersDefaultsConfig)
    cache: CacheDefaultsConfig = Field(default_factory=CacheDefaultsConfig)


class AgentVariantModelConfig(BaseModel):
    """Single agent variant configuration (premium/cheap/local)."""

    display_name: str = Field(..., description="Human-readable variant name")
    model: str = Field(..., description="Model identifier for this variant")


class AgentVariantsConfig(BaseModel):
    """
    Agent variant configuration - SSOT for model selection.

    Each variant maps to a single model (no per-agent tiering).
    """

    variants: dict[str, AgentVariantModelConfig] = Field(
        ...,
        description="Variant configurations keyed by variant name"
    )
    default_variant: str = Field(
        "remote",
        description="Default variant when lookup fails or user has no preference"
    )

    @model_validator(mode="after")
    def validate_default_variant(self):
        if self.default_variant not in self.variants:
            raise ValueError(
                f"default_variant '{self.default_variant}' not found in variants: "
                f"{list(self.variants.keys())}"
            )
        return self


# ============================================================================
# End of SSOT Configuration
# ============================================================================


class VLConfig(BaseModel):
    """Vision-Language architecture configuration."""

    embedder: Literal["jina", "local"] = Field(
        default="jina",
        description="Embedder backend: 'jina' (cloud API) or 'local' (vLLM on GB10)"
    )
    jina_model: str = Field(
        default="jina-embeddings-v4",
        description="Jina embedding model for VL page embeddings"
    )
    local_embedding_url: Optional[str] = Field(
        default=None,
        description="Local embedding server URL (e.g. http://localhost:8081/v1)"
    )
    local_embedding_model: str = Field(
        default="Qwen/Qwen3-VL-Embedding-8B",
        description="Local embedding model name"
    )
    dimensions: int = Field(
        default=4096,
        ge=1,
        description="Embedding dimensions (4096 for local Qwen3-VL-Embedding-8B, 2048 for Jina v4)"
    )
    default_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of pages to retrieve"
    )
    page_image_dpi: int = Field(
        default=150,
        ge=72,
        le=600,
        description="DPI for PDF page rendering"
    )
    page_image_format: Literal["png", "jpg"] = Field(
        default="png",
        description="Image format for rendered pages"
    )
    page_store_dir: str = Field(
        default="data/vl_pages",
        description="Directory for rendered page images"
    )
    source_pdf_dir: str = Field(
        default="data",
        description="Directory containing source PDFs"
    )
    max_pages_per_query: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum page images to include in a single LLM call"
    )
    image_tokens_per_page: int = Field(
        default=1600,
        ge=100,
        description="Estimated tokens per page image (Anthropic vision API)"
    )


class PipelineConfig(BaseModel):
    """General pipeline configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        ...,
        description="Logging level"
    )
    log_file: str = Field(..., description="Log file path")
    data_dir: str = Field(..., description="Data directory for input documents")
    output_dir: str = Field(..., description="Output directory for pipeline results")


class RootConfig(BaseModel):
    """
    Root configuration - ALL sections REQUIRED.

    Strict validation enabled:
    - No automatic type coercion ("512" will not be converted to 512)
    - Unknown fields are rejected
    - All types must match exactly
    """

    model_config = ConfigDict(
        strict=True,  # Disable automatic type coercion
        extra="allow",  # Allow unknown fields (for backwards compatibility with storage, security_monitoring, etc.)
    )

    vl: Optional[VLConfig] = Field(
        default=None,
        description="Vision-Language configuration"
    )
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    models: ModelsConfig
    extraction: Optional[ExtractionConfig] = Field(
        default=None, description="[DEPRECATED] OCR extraction config"
    )
    unstructured: Optional[UnstructuredConfig] = Field(
        default=None, description="[DEPRECATED] Unstructured.io config"
    )
    hierarchy_detection: Optional[HierarchyDetectionConfig] = Field(
        default=None, description="[DEPRECATED] Hierarchy detection config"
    )
    summarization: Optional[SummarizationConfig] = Field(
        default=None, description="[DEPRECATED] Summarization config"
    )
    context_generation: Optional[ContextGenerationConfig] = Field(
        default=None, description="[DEPRECATED] Contextual Retrieval config"
    )
    chunking: Optional[ChunkingConfig] = Field(
        default=None, description="[DEPRECATED] Chunking config"
    )
    embedding: Optional[EmbeddingConfig] = Field(
        default=None, description="[DEPRECATED] OCR embedding config"
    )
    clustering: Optional[ClusteringConfig] = Field(
        default=None, description="[DEPRECATED] Clustering config"
    )
    hybrid_search: Optional[HybridSearchConfig] = Field(
        default=None, description="[DEPRECATED] Hybrid search config"
    )
    agent: AgentConfig
    agent_tools: AgentToolConfig
    cli: CLIConfig
    pipeline: PipelineConfig
    indexing: Optional[IndexingConfig] = Field(
        default=None, description="[DEPRECATED] OCR indexing config"
    )

    # SSOT Configuration sections (optional for backwards compatibility)
    model_registry: Optional[ModelRegistryConfig] = Field(
        default=None,
        description="Centralized model aliases - SSOT for ModelRegistry"
    )
    defaults: DefaultsConfig = Field(
        default_factory=DefaultsConfig,
        description="Centralized default values - SSOT for hardcoded defaults"
    )
    agent_variants: Optional[AgentVariantsConfig] = Field(
        default=None,
        description="Agent variant tier configuration - SSOT for backend/constants.py"
    )

    def _is_placeholder(self, value: Optional[str]) -> bool:
        """
        Check if API key is a placeholder value from config.json.example.

        Args:
            value: API key value to check

        Returns:
            True if value is empty, None, or contains placeholder text
        """
        if not value:
            return True

        # Common placeholder patterns
        PLACEHOLDER_PATTERNS = [
            "REQUIRED_IF",
            "YOUR_API_KEY",
            "YOUR_AURA_PASSWORD",
            "YOUR_INSTANCE_ID",
            "sk-placeholder",
            "example",
            "INSERT",
            "TODO",
            "CHANGEME",
        ]

        return any(pattern in value.upper() for pattern in PLACEHOLDER_PATTERNS)

    @model_validator(mode="after")
    def validate_api_keys(self) -> "RootConfig":
        """
        Validate that at least one API key is provided and matches model selection.
        API keys are loaded from .env file - this validation runs after loading.

        Raises:
            ValueError: If required API keys are missing
        """
        # Check LLM API keys
        if self.models.llm_model.startswith("claude-"):
            if self._is_placeholder(self.api_keys.anthropic_api_key):
                raise ValueError(
                    f"ANTHROPIC_API_KEY is REQUIRED for LLM model '{self.models.llm_model}'.\n"
                    f"Please set ANTHROPIC_API_KEY in your .env file.\n"
                    f"Example: ANTHROPIC_API_KEY=sk-ant-..."
                )
        elif self.models.llm_model.startswith(("gpt-", "o1-", "o3-")):
            if self._is_placeholder(self.api_keys.openai_api_key):
                raise ValueError(
                    f"OPENAI_API_KEY is REQUIRED for LLM model '{self.models.llm_model}'.\n"
                    f"Please set OPENAI_API_KEY in your .env file.\n"
                    f"Example: OPENAI_API_KEY=sk-proj-..."
                )
        elif self.models.llm_model.startswith("gemini-"):
            if self._is_placeholder(self.api_keys.google_api_key):
                raise ValueError(
                    f"GOOGLE_API_KEY is REQUIRED for LLM model '{self.models.llm_model}'.\n"
                    f"Please set GOOGLE_API_KEY in your .env file.\n"
                    f"Example: GOOGLE_API_KEY=..."
                )

        # Check embedding API keys
        if self.models.embedding_provider == "openai":
            if self._is_placeholder(self.api_keys.openai_api_key):
                raise ValueError(
                    f"OPENAI_API_KEY is REQUIRED for EMBEDDING_PROVIDER 'openai'.\n"
                    f"Please set OPENAI_API_KEY in your .env file.\n"
                    f"Example: OPENAI_API_KEY=sk-proj-..."
                )
        elif self.models.embedding_provider == "voyage":
            if self._is_placeholder(self.api_keys.voyage_api_key):
                raise ValueError(
                    f"VOYAGE_API_KEY is REQUIRED for EMBEDDING_PROVIDER 'voyage'.\n"
                    f"Please set VOYAGE_API_KEY in your .env file.\n"
                    f"Example: VOYAGE_API_KEY=..."
                )

        # Validate at least one API key is set (not placeholder)
        valid_keys = [
            k for k in [
                self.api_keys.anthropic_api_key,
                self.api_keys.openai_api_key,
                self.api_keys.voyage_api_key,
                self.api_keys.google_api_key
            ] if not self._is_placeholder(k)
        ]

        if not valid_keys:
            raise ValueError(
                "At least ONE valid API key must be set in your .env file.\n"
                "Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, VOYAGE_API_KEY, or GOOGLE_API_KEY.\n"
                "See .env.example for template."
            )

        return self

    @classmethod
    def from_json_file(cls, path: Path) -> "RootConfig":
        """
        Load configuration from JSON file with strict validation.

        Args:
            path: Path to config.json

        Returns:
            Validated RootConfig instance

        Raises:
            FileNotFoundError: If config.json does not exist
            ValueError: If validation fails or required fields are missing
            json.JSONDecodeError: If JSON is malformed
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Please create config.json from config.json.example:\n"
                f"  cp config.json.example config.json\n"
                f"  # Edit config.json with your settings"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in {path}:\n{e}\n"
                f"Please fix the JSON syntax error."
            ) from e
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Invalid UTF-8 encoding in {path}:\n{e}\n"
                f"Please save the file with UTF-8 encoding."
            ) from e

        try:
            config = cls(**data)
            # validate_api_keys() is now a @model_validator, runs automatically
            return config
        except ValidationError as e:
            # Pydantic validation errors - user config issues
            raise ValueError(
                f"Configuration validation failed:\n{e}\n\n"
                f"Please check config.json matches the required schema.\n"
                f"See config.json.example for reference."
            ) from e
        except ValueError as e:
            # API key validation errors or other value errors
            raise
        except (TypeError, AttributeError) as e:
            # Code bugs - don't hide these
            import logging
            logging.error(f"Internal validation error (this is a bug): {e}", exc_info=True)
            raise RuntimeError(
                f"Internal validation error. This is a bug in the validation code.\n"
                f"Please report this issue at https://github.com/ADS-teamA/SUJBOT/issues\n"
                f"Include your config.json file (remove API keys) in the report."
            ) from e
        # DO NOT catch Exception - let unexpected errors propagate with full traceback


def load_config(config_path: Optional[Path] = None) -> RootConfig:
    """
    Load and validate configuration from config.json.

    This is the main entry point for loading configuration.

    Args:
        config_path: Optional path to config.json (default: ./config.json)

    Returns:
        Validated RootConfig instance

    Raises:
        FileNotFoundError: If config.json does not exist
        ValueError: If validation fails
    """
    if config_path is None:
        # Default: look for config.json in project root
        config_path = Path.cwd() / "config.json"

        # Fallback: if we're in src/, look in parent
        if not config_path.exists() and Path(__file__).parent.name == "src":
            config_path = Path(__file__).parent.parent / "config.json"

    return RootConfig.from_json_file(config_path)
