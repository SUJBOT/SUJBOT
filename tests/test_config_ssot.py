"""
SSOT (Single Source of Truth) Tests for config.json consolidation.

These tests verify that:
1. ModelRegistry reads model aliases from config.json
2. backend/constants reads agent variants from config.json
3. layer_default_k reads from config.json
4. No duplicate MODEL_PRICING exists
5. Embedding dimensions are consistently 4096 (Qwen3-Embedding-8B)

SSOT Architecture:
- config.json is the single source of truth for all configuration values
- Modules use lazy loading with fallbacks for backward compatibility
- Tests verify config values propagate correctly to runtime
"""

import json
from pathlib import Path

import pytest


class TestConfigSchema:
    """Test config.json schema and structure."""

    @pytest.fixture
    def config_data(self):
        """Load config.json."""
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path) as f:
            return json.load(f)

    def test_model_registry_section_exists(self, config_data):
        """Verify model_registry section exists in config.json."""
        assert "model_registry" in config_data
        assert "llm_models" in config_data["model_registry"]
        assert "embedding_models" in config_data["model_registry"]

    def test_defaults_section_exists(self, config_data):
        """Verify defaults section exists in config.json."""
        assert "defaults" in config_data
        assert "timeouts" in config_data["defaults"]
        assert "pool_sizes" in config_data["defaults"]
        assert "retrieval" in config_data["defaults"]

    def test_agent_variants_section_exists(self, config_data):
        """Verify agent_variants section exists in config.json."""
        assert "agent_variants" in config_data
        assert "variants" in config_data["agent_variants"]
        assert "default_variant" in config_data["agent_variants"]

    def test_layer_default_k_values(self, config_data):
        """Verify layer_default_k has correct values."""
        layer_k = config_data["defaults"]["retrieval"]["layer_default_k"]
        assert layer_k["1"] == 3  # Documents
        assert layer_k["2"] == 5  # Sections
        assert layer_k["3"] == 10  # Chunks

    def test_embedding_dimensions_qwen3(self, config_data):
        """Verify Qwen3-Embedding-8B has 4096 dimensions (NOT 3072)."""
        embedding_models = config_data["model_registry"]["embedding_models"]
        # Dimensions are now inside each model definition
        qwen_model = embedding_models.get("qwen3-embedding", {})
        assert qwen_model.get("dimensions") == 4096
        # OpenAI is 3072
        openai_model = embedding_models.get("text-embedding-3-large", {})
        assert openai_model.get("dimensions") == 3072

    def test_embedding_models_have_dimensions(self, config_data):
        """Verify all embedding models have dimensions defined."""
        embedding_models = config_data["model_registry"]["embedding_models"]
        for model_name, model_config in embedding_models.items():
            assert "dimensions" in model_config, f"Missing dimensions for {model_name}"
            assert model_config["dimensions"] > 0, f"Invalid dimensions for {model_name}"

    def test_variant_models_are_valid(self, config_data):
        """Verify all variant models reference real model IDs."""
        variants = config_data["agent_variants"]["variants"]
        for variant_name, variant_config in variants.items():
            assert "model" in variant_config, f"Missing model in {variant_name}"
            assert "display_name" in variant_config, f"Missing display_name in {variant_name}"
            # Models should have version suffixes (not just aliases)
            if "claude" in variant_config["model"]:
                assert (
                    "-" in variant_config["model"]
                ), f"Model should have version: {variant_config['model']}"


class TestModelRegistrySSoT:
    """Test that ModelRegistry reads from config.json."""

    def test_resolve_llm_from_config(self):
        """Verify ModelRegistry.resolve_llm uses config values."""
        from src.utils.model_registry import ModelRegistry

        # Test alias resolution
        resolved = ModelRegistry.resolve_llm("haiku")
        assert "claude-haiku" in resolved or "claude-haiku-4" in resolved

        resolved = ModelRegistry.resolve_llm("sonnet")
        assert "claude-sonnet" in resolved

        resolved = ModelRegistry.resolve_llm("opus")
        assert "claude-opus" in resolved

    def test_get_embedding_model_dimensions(self):
        """Verify embedding dimensions are retrieved correctly."""
        from src.utils.model_registry import ModelRegistry

        # Qwen3 should be 4096
        dims = ModelRegistry.get_embedding_dimensions("Qwen/Qwen3-Embedding-8B")
        assert dims == 4096

        # OpenAI should be 3072
        dims = ModelRegistry.get_embedding_dimensions("text-embedding-3-large")
        assert dims == 3072

    def test_registry_reload(self):
        """Verify registry can be reloaded from config."""
        from src.utils.model_registry import reload_registry

        # Should not raise
        reload_registry()


class TestBackendConstantsSSoT:
    """Test that backend/constants reads from config.json."""

    def test_variant_config_from_config(self):
        """Verify VARIANT_CONFIG matches config.json."""
        from backend.constants import get_variant_config, reload_constants

        reload_constants()  # Ensure fresh load
        variants = get_variant_config()

        # Check expected variants exist
        assert "remote" in variants
        assert "local" in variants

        # Check each variant has required fields (single model per variant)
        for name, config in variants.items():
            assert "display_name" in config
            assert "model" in config

    def test_get_variant_model_function(self):
        """Verify get_variant_model returns correct models."""
        from backend.constants import get_variant_model, reload_constants

        reload_constants()
        model = get_variant_model("remote")
        assert "haiku" in model.lower() or "claude" in model.lower()

        model = get_variant_model("local")
        assert "Qwen" in model or "qwen" in model.lower()

    def test_get_variant_model_unknown_falls_back(self):
        """Unknown variants (e.g., legacy 'premium') fall back to default."""
        from backend.constants import get_variant_model, get_default_variant, reload_constants

        reload_constants()
        default_model = get_variant_model(get_default_variant())

        # Legacy variant names should fallback, not raise
        assert get_variant_model("premium") == default_model
        assert get_variant_model("cheap") == default_model
        assert get_variant_model("nonexistent") == default_model


class TestNoDuplicatePricing:
    """Verify MODEL_PRICING is not duplicated."""

    def test_no_model_pricing_in_toc_retrieval(self):
        """Verify ToC_retrieval.py uses central PRICING, not local MODEL_PRICING."""
        import inspect

        from src import ToC_retrieval

        # Check LLMAgent class doesn't have MODEL_PRICING as class attribute
        # (it should only have _DEFAULT_PRICING as fallback)
        agent_class = ToC_retrieval.LLMAgent

        # MODEL_PRICING should NOT be a class attribute
        assert not hasattr(agent_class, "MODEL_PRICING") or agent_class.MODEL_PRICING is None

    def test_toc_retrieval_uses_central_pricing(self):
        """Verify ToC_retrieval imports PRICING from cost_tracker."""
        import importlib
        import sys

        # Check import statement exists
        spec = importlib.util.find_spec("src.ToC_retrieval")
        source_path = spec.origin

        with open(source_path, "r") as f:
            content = f.read()

        assert "from src.cost_tracker import PRICING" in content


class TestDimensionsConsistency:
    """Verify embedding dimensions are consistently 4096."""

    def test_runner_dimensions_not_3072(self):
        """Verify runner.py doesn't fallback to 3072."""
        runner_path = Path(__file__).parent.parent / "src" / "multi_agent" / "runner.py"

        with open(runner_path, "r") as f:
            content = f.read()

        # Should NOT have dimensions=3072 (was a bug)
        # Should have dimensions=4096 as fallback
        assert "dimensions=3072" not in content or "dimensions, 3072" not in content
        # Should have correct fallback
        assert "4096" in content

    def test_enable_reranking_default_false(self):
        """Verify enable_reranking default is False (matches config.json)."""
        from src.agent.config import ToolConfig

        # Default should be False (matches config.json)
        # enable_reranking is in ToolConfig, not AgentConfig
        config = ToolConfig()
        assert config.enable_reranking is False


class TestLayerDefaultK:
    """Test layer_default_k centralization."""

    def test_utils_loads_layer_k_from_config(self):
        """Verify _utils.py loads layer_default_k from config."""
        from src.agent.tools._utils import _ensure_config_loaded, _layer_default_k

        _ensure_config_loaded()

        # Should have expected values
        assert _layer_default_k.get(1) == 3  # Documents
        assert _layer_default_k.get(2) == 5  # Sections
        assert _layer_default_k.get(3) == 10  # Chunks


class TestArchitectureConfig:
    """Test architecture and VL configuration sections."""

    @pytest.fixture
    def config_data(self):
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path) as f:
            return json.load(f)

    def test_architecture_field_exists(self, config_data):
        """Verify 'architecture' field is present and valid."""
        assert "architecture" in config_data
        assert config_data["architecture"] in ("ocr", "vl")

    def test_vl_section_exists(self, config_data):
        """Verify 'vl' section exists with required fields."""
        assert "vl" in config_data
        vl = config_data["vl"]
        required = [
            "jina_model",
            "dimensions",
            "default_k",
            "page_image_dpi",
            "page_image_format",
            "page_store_dir",
            "source_pdf_dir",
            "max_pages_per_query",
            "image_tokens_per_page",
        ]
        for field in required:
            assert field in vl, f"Missing VL field: {field}"

    def test_vl_jina_dimensions_2048(self, config_data):
        """Jina v4 must produce 2048-dim embeddings (NOT 4096 like Qwen3)."""
        assert config_data["vl"]["dimensions"] == 2048

    def test_vl_jina_model_is_v4(self, config_data):
        """Jina model must be v4."""
        assert "v4" in config_data["vl"]["jina_model"]

    def test_vl_page_image_dpi_reasonable(self, config_data):
        """DPI should be between 72 and 600."""
        dpi = config_data["vl"]["page_image_dpi"]
        assert 72 <= dpi <= 600

    def test_jina_v4_in_model_registry(self, config_data):
        """Jina v4 must be registered in model_registry.embedding_models."""
        emb = config_data["model_registry"]["embedding_models"]
        assert "jina-v4" in emb
        assert emb["jina-v4"]["dimensions"] == 2048
        assert emb["jina-v4"]["provider"] == "jina"


class TestSingleAgentConfig:
    """Test single_agent configuration section."""

    @pytest.fixture
    def config_data(self):
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path) as f:
            return json.load(f)

    def test_single_agent_section_exists(self, config_data):
        """Verify 'single_agent' section exists."""
        assert "single_agent" in config_data

    def test_single_agent_required_fields(self, config_data):
        """Verify single_agent has all required fields."""
        sa = config_data["single_agent"]
        required = ["model", "max_tokens", "temperature", "max_iterations", "prompt_file"]
        for field in required:
            assert field in sa, f"Missing single_agent field: {field}"

    def test_single_agent_prompt_file_exists(self, config_data):
        """Verify prompt file referenced in single_agent config exists."""
        prompt_path = Path(__file__).parent.parent / config_data["single_agent"]["prompt_file"]
        assert prompt_path.exists(), f"Prompt file not found: {prompt_path}"

    def test_single_agent_model_is_valid(self, config_data):
        """Model should reference a known model ID."""
        model = config_data["single_agent"]["model"]
        assert model  # Not empty
        # Should be resolvable (full ID or alias)
        llm_models = config_data["model_registry"]["llm_models"]
        all_model_ids = {v["id"] if isinstance(v, dict) else v for v in llm_models.values()}
        assert model in all_model_ids or model in llm_models

    def test_single_agent_max_iterations_bounded(self, config_data):
        """Max iterations should be reasonable (1-50)."""
        iters = config_data["single_agent"]["max_iterations"]
        assert 1 <= iters <= 50
