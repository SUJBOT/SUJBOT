# Configuration Guide

## Two-File System

Configuration is split between two files:

| File | Contents | Version Control |
|------|----------|-----------------|
| `.env` | Secrets (API keys, passwords, JWT secret) | Gitignored |
| `config.json` | Application settings (models, retrieval, storage) | Tracked |

**Rule:** API keys and credentials go in `.env` ONLY. Never in `config.json` or code.

## Environment Variables (`.env`)

Copy `.env.example` to `.env` and fill in your values.

### Required API Keys

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude models (Haiku, Sonnet, Opus) |
| `OPENAI_API_KEY` | OpenAI models (GPT-4o, GPT-4o-mini) |
| `DEEPINFRA_API_KEY` | DeepInfra models (Qwen3-VL, Qwen embedding) |
| `JINA_API_KEY` | Jina v4 embeddings (VL mode) |

### Optional API Keys

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` | Google Gemini (extraction backend) |
| `VOYAGE_API_KEY` | Voyage embedding models (not currently used) |

### Database

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_USER` | `postgres` | PostgreSQL username |
| `POSTGRES_PASSWORD` | — | PostgreSQL password (required) |
| `POSTGRES_DB` | `sujbot` | Database name |
| `DATABASE_URL` | — | Full connection string (overrides individual vars) |

The `DATABASE_URL` format is `postgresql://USER:PASSWORD@HOST:PORT/DATABASE`. In production, the backend container overrides this to use `sujbot_postgres:5432` (Docker DNS).

### Authentication

| Variable | Purpose |
|----------|---------|
| `AUTH_SECRET_KEY` | JWT signing key (min 32 chars). Generate with `python -c "import secrets; print(secrets.token_urlsafe(64))"` |

### LangSmith

| Variable | Purpose |
|----------|---------|
| `LANGSMITH_API_KEY` | LangSmith API key (`lsv2_pt_...`) |
| `LANGSMITH_PROJECT_NAME` | Project name (default: `sujbot-multi-agent`) |
| `LANGSMITH_ENDPOINT` | API endpoint. Use `https://eu.api.smith.langchain.com` for EU workspaces |

## Application Settings (`config.json`)

### Architecture Selection

```json
{
  "architecture": "vl"
}
```

Set to `"vl"` for Vision-Language mode (page images) or `"ocr"` for text chunk mode. See [ARCHITECTURE.md](ARCHITECTURE.md) for details on each mode.

### VL Configuration

```json
{
  "vl": {
    "jina_model": "jina-embeddings-v4",
    "dimensions": 2048,
    "default_k": 5,
    "page_image_dpi": 150,
    "page_image_format": "png",
    "page_store_dir": "data/vl_pages",
    "source_pdf_dir": "data",
    "max_pages_per_query": 8,
    "image_tokens_per_page": 1600
  }
}
```

| Field | Description |
|-------|-------------|
| `default_k` | Number of pages returned per search query |
| `page_image_dpi` | DPI for rendering PDF pages to images |
| `max_pages_per_query` | Maximum pages sent to LLM per query |
| `image_tokens_per_page` | Estimated token cost per page image |

### Single Agent

```json
{
  "single_agent": {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 4096,
    "temperature": 0.3,
    "max_iterations": 10,
    "enable_prompt_caching": true
  }
}
```

The `model` field is the fallback model. In practice, the model is determined by the user's agent variant selection.

### Agent Variants

```json
{
  "agent_variants": {
    "variants": {
      "remote": {
        "display_name": "Remote (Sonnet 4.5)",
        "model": "claude-sonnet-4-5-20250929"
      },
      "local": {
        "display_name": "Local (Qwen3 VL 235B)",
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct"
      }
    },
    "default_variant": "remote",
    "deepinfra_supported_models": [
      "Qwen/Qwen2.5-VL-32B-Instruct",
      "Qwen/Qwen3-VL-235B-A22B-Instruct"
    ]
  }
}
```

Users select their variant in the web UI settings. The backend resolves variant to model via `backend/constants.py:get_variant_model()`.

### Model Registry

The `model_registry` section defines all available LLM and embedding models with their pricing and capabilities:

```json
{
  "model_registry": {
    "llm_models": {
      "haiku": {
        "id": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "pricing": { "input": 1.00, "output": 5.00 },
        "context_window": 200000,
        "supports_caching": true
      }
    },
    "embedding_models": {
      "jina-v4": {
        "id": "jina-embeddings-v4",
        "provider": "jina",
        "pricing": { "input": 0.02, "output": 0.00 },
        "dimensions": 2048
      }
    }
  }
}
```

Pricing is in USD per 1M tokens.

### Adding a New Model

1. Check the provider's current pricing
2. Add an entry to `config.json` → `model_registry.llm_models`:
   ```json
   "new-model": {
     "id": "provider/model-name",
     "provider": "provider-name",
     "pricing": { "input": X.XX, "output": X.XX },
     "context_window": 128000
   }
   ```
3. If using DeepInfra, also add the model ID to `agent_variants.deepinfra_supported_models`

### Retrieval (OCR Mode)

```json
{
  "retrieval": {
    "method": "hyde_expansion_fusion",
    "original_weight": 0.5,
    "hyde_weight": 0.25,
    "expansion_weight": 0.25,
    "default_k": 16,
    "candidates_multiplier": 3
  }
}
```

These weights control the HyDE + Expansion Fusion retrieval pipeline used in OCR mode. Not applicable to VL mode.

### Storage

```json
{
  "storage": {
    "backend": "postgresql",
    "storage_layers": [1, 2, 3],
    "postgresql": {
      "connection_string_env": "DATABASE_URL",
      "pool_size": 20,
      "dimensions": 4096
    }
  }
}
```

The `dimensions` field (4096) applies to OCR mode embeddings. VL mode uses 2048 dimensions configured in the `vl` section.

### Chunking (OCR Mode)

```json
{
  "chunking": {
    "max_tokens": 512,
    "tokenizer_model": "Qwen/Qwen3-Embedding-8B",
    "enable_sac": true
  }
}
```

- `max_tokens: 512` — Changing this invalidates ALL existing vector stores
- `enable_sac` — Summary-Augmented Chunking: prepends document summary during embedding

### Extraction

```json
{
  "extraction": {
    "backend": "gemini",
    "gemini_model": "gemini-2.5-flash",
    "generate_summaries": true,
    "summary_style": "generic"
  }
}
```

The extraction backend (`EXTRACTION_BACKEND` env var or `extraction.backend`) can be `auto`, `gemini`, or `unstructured`. This controls PDF text extraction for the OCR pipeline.

### Security Monitoring

The `security_monitoring` section configures automated health checks and alerting:

```json
{
  "security_monitoring": {
    "enabled": true,
    "schedule": {
      "regular_checks": ["6:00", "10:00", "14:00", "18:00"],
      "daily_summary": "22:00",
      "timezone": "Europe/Prague"
    },
    "checks": {
      "docker_health": { "enabled": true },
      "log_analysis": { "enabled": true },
      "system_resources": { "enabled": true },
      "health_endpoints": { "enabled": true },
      "ssl_certificate": { "enabled": true }
    }
  }
}
```

Notifications can be sent to Microsoft Teams via webhook (`notifications.teams`).
