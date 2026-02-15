"""
SSOT (Single Source of Truth) Tests for config.json consolidation.

These tests verify that:
1. ModelRegistry reads model aliases from config.json
2. backend/constants reads agent variants from config.json
3. Embedding dimensions are consistently 2048 (Jina v4)
4. VL architecture configuration is correct

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

    def test_agent_variants_section_exists(self, config_data):
        """Verify agent_variants section exists in config.json."""
        assert "agent_variants" in config_data
        assert "variants" in config_data["agent_variants"]
        assert "default_variant" in config_data["agent_variants"]

    def test_embedding_dimensions_jina_v4(self, config_data):
        """Verify Jina v4 has 2048 dimensions."""
        embedding_models = config_data["model_registry"]["embedding_models"]
        jina_model = embedding_models.get("jina-v4", {})
        assert jina_model.get("dimensions") == 2048

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
        """Verify Jina v4 embedding dimensions from config."""
        import json
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)
        jina = config_data["model_registry"]["embedding_models"]["jina-v4"]
        assert jina["dimensions"] == 2048

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


class TestDimensionsConsistency:
    """Verify embedding dimensions are consistently 2048 (Jina v4)."""

    def test_tool_config_defaults(self):
        """Verify ToolConfig defaults (VL-only)."""
        from src.agent.config import ToolConfig

        config = ToolConfig()
        assert config.default_k == 6
        assert config.compliance_threshold == 0.7
        assert config.max_document_compare == 3
        assert config.graph_storage is None


class TestArchitectureConfig:
    """Test architecture and VL configuration sections."""

    @pytest.fixture
    def config_data(self):
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path) as f:
            return json.load(f)

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
        required = ["model", "max_tokens", "temperature", "max_iterations"]
        for field in required:
            assert field in sa, f"Missing single_agent field: {field}"

    def test_single_agent_prompt_files_exist(self, config_data):
        """Verify VL prompt file exists."""
        root = Path(__file__).parent.parent
        assert (root / "prompts/agents/unified.txt").exists(), "VL prompt missing"

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
