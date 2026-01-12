#!/usr/bin/env python3
"""
Fetch current pricing from DeepInfra API.

This script fetches model pricing from DeepInfra and outputs in two formats:
1. config.json format (NEW - SSOT) - use --config-format
2. cost_tracker.py format (LEGACY) - default

Usage:
    # Update config.json (recommended)
    uv run python scripts/fetch_deepinfra_pricing.py --config-format

    # Specific model
    uv run python scripts/fetch_deepinfra_pricing.py --model Qwen/Qwen2.5-72B-Instruct --config-format

    # All models
    uv run python scripts/fetch_deepinfra_pricing.py --all --json

    # Update config.json in-place
    uv run python scripts/fetch_deepinfra_pricing.py --config-format --update

Sources:
    - DeepInfra API: https://api.deepinfra.com/models/list
    - DeepInfra Pricing: https://deepinfra.com/pricing
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: uv add httpx")
    sys.exit(1)


def fetch_deepinfra_models() -> list[dict]:
    """Fetch all models from DeepInfra API."""
    try:
        response = httpx.get(
            "https://api.deepinfra.com/models/list",
            timeout=30.0,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        print("ERROR: Request to DeepInfra API timed out after 30s. Try again later.")
        return []
    except httpx.ConnectError as e:
        print(f"ERROR: Cannot connect to DeepInfra API (network/DNS issue): {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"ERROR: DeepInfra API returned HTTP {e.response.status_code}: {e.response.text[:200]}")
        return []
    except httpx.HTTPError as e:
        print(f"ERROR: HTTP error fetching from DeepInfra API: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON response from DeepInfra API at position {e.pos}: {e.msg}")
        return []


def extract_pricing(models: list[dict], filter_models: list[str] | None = None) -> dict[str, dict]:
    """
    Extract pricing from model list.

    Args:
        models: List of model dicts from API
        filter_models: Optional list of model names to filter

    Returns:
        Dict mapping model name to pricing dict
    """
    pricing = {}

    for model in models:
        name = model.get("model_name", "")
        if not name:
            continue

        # Apply filter if specified
        if filter_models and name not in filter_models:
            continue

        # Extract pricing (cents per token -> USD per 1M tokens)
        model_pricing = model.get("pricing", {})
        if model_pricing:
            # DeepInfra returns cents per token, convert to USD per 1M tokens
            input_cents = model_pricing.get("cents_per_input_token", 0)
            output_cents = model_pricing.get("cents_per_output_token", 0)

            # Convert: cents/token * 1M tokens / 100 cents = USD/1M tokens
            pricing[name] = {
                "input": round(input_cents * 10000, 2),  # cents * 10000 = USD/1M
                "output": round(output_cents * 10000, 2),
            }

    return pricing


def format_for_cost_tracker(pricing: dict[str, dict]) -> str:
    """Format pricing as Python code for cost_tracker.py (LEGACY)."""
    lines = []
    for model, prices in sorted(pricing.items()):
        lines.append(f'        "{model}": {{"input": {prices["input"]}, "output": {prices["output"]}}},')
    return "\n".join(lines)


def format_for_config_json(pricing: dict[str, dict]) -> dict:
    """
    Format pricing as config.json model_registry format (NEW SSOT).

    Returns dict structure ready to merge into config.json model_registry.llm_models
    """
    result = {}
    for model, prices in pricing.items():
        # Create alias from model name (e.g., "Qwen/Qwen2.5-72B-Instruct" -> "minimax-m2")
        alias = model.split("/")[-1].lower().replace("_", "-")

        result[alias] = {
            "id": model,
            "provider": "deepinfra",
            "pricing": {
                "input": prices["input"],
                "output": prices["output"],
            },
        }

    return result


def update_config_json(pricing: dict[str, dict]) -> bool:
    """
    Update config.json with new pricing data.

    Merges pricing into model_registry.llm_models section.
    """
    config_path = Path(__file__).parent.parent / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found at {config_path}")
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Get or create model_registry.llm_models
        if "model_registry" not in config:
            config["model_registry"] = {"llm_models": {}, "embedding_models": {}, "reranker_models": {}}
        if "llm_models" not in config["model_registry"]:
            config["model_registry"]["llm_models"] = {}

        # Update pricing for existing entries or add new ones
        for model, prices in pricing.items():
            alias = model.split("/")[-1].lower().replace("_", "-")

            # Check if model already exists (by alias or by ID)
            existing_alias = None
            for key, entry in config["model_registry"]["llm_models"].items():
                entry_id = entry["id"] if isinstance(entry, dict) else entry
                if entry_id == model:
                    existing_alias = key
                    break

            target_alias = existing_alias or alias

            if target_alias in config["model_registry"]["llm_models"]:
                # Update existing entry
                entry = config["model_registry"]["llm_models"][target_alias]
                if isinstance(entry, dict):
                    entry["pricing"] = {"input": prices["input"], "output": prices["output"]}
                else:
                    # Convert string to dict format
                    config["model_registry"]["llm_models"][target_alias] = {
                        "id": entry,
                        "provider": "deepinfra",
                        "pricing": {"input": prices["input"], "output": prices["output"]},
                    }
                print(f"  Updated: {target_alias}")
            else:
                # Add new entry
                config["model_registry"]["llm_models"][target_alias] = {
                    "id": model,
                    "provider": "deepinfra",
                    "pricing": {"input": prices["input"], "output": prices["output"]},
                }
                print(f"  Added: {target_alias}")

        # Write back
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nUpdated {config_path}")
        return True

    except json.JSONDecodeError as e:
        print(f"ERROR: config.json has invalid JSON at line {e.lineno}: {e.msg}")
        return False
    except KeyError as e:
        print(f"ERROR: config.json missing expected key: {e}")
        return False
    except OSError as e:
        print(f"ERROR: Cannot write to {config_path}: {e}")
        return False


def get_configured_models() -> list[str]:
    """Get models from config.json that need pricing."""
    config_path = Path(__file__).parent.parent / "config.json"
    if not config_path.exists():
        print(f"WARNING: config.json not found at {config_path}")
        return []

    try:
        with open(config_path) as f:
            config = json.load(f)

        models = set()

        # Get deepinfra_supported_models
        agent_variants = config.get("agent_variants", {})
        models.update(agent_variants.get("deepinfra_supported_models", []))

        # Get variant models
        for variant in agent_variants.get("variants", {}).values():
            for key in ["opus_model", "default_model"]:
                model = variant.get(key, "")
                if "/" in model:  # HuggingFace-style path = DeepInfra model
                    models.add(model)

        return list(models)
    except json.JSONDecodeError as e:
        print(f"WARNING: config.json has invalid JSON at line {e.lineno}: {e.msg}")
        return []
    except KeyError as e:
        print(f"WARNING: config.json missing expected key: {e}")
        return []
    except OSError as e:
        print(f"WARNING: Cannot read config.json: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Fetch DeepInfra pricing")
    parser.add_argument("--model", "-m", help="Specific model to fetch pricing for")
    parser.add_argument("--all", "-a", action="store_true", help="Fetch all available models")
    parser.add_argument("--configured", "-c", action="store_true", help="Fetch only models from config.json")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--config-format", action="store_true", help="Output in config.json model_registry format (NEW SSOT)"
    )
    parser.add_argument(
        "--update", action="store_true", help="Update config.json in-place (use with --config-format)"
    )
    args = parser.parse_args()

    print("Fetching models from DeepInfra API...")
    models = fetch_deepinfra_models()

    if not models:
        print("No models found or API error")
        sys.exit(1)

    print(f"Found {len(models)} models")

    # Determine filter
    filter_models = None
    if args.model:
        filter_models = [args.model]
    elif args.configured:
        filter_models = get_configured_models()
        print(f"Filtering to {len(filter_models)} configured models")
    elif not args.all:
        # Default: show only LLM models we commonly use
        filter_models = [
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen3-Embedding-8B",
        ]

    pricing = extract_pricing(models, filter_models)

    if not pricing:
        print("No pricing found for specified models")
        sys.exit(1)

    # Handle config.json format output
    if args.config_format:
        if args.update:
            print("\nUpdating config.json...")
            if update_config_json(pricing):
                print("✓ Config updated successfully")
            else:
                print("✗ Failed to update config")
                sys.exit(1)
        else:
            # Output config.json format
            config_format = format_for_config_json(pricing)
            print("\n" + "=" * 60)
            print("CONFIG.JSON FORMAT (merge into model_registry.llm_models):")
            print("=" * 60)
            print(json.dumps(config_format, indent=2))
            print("=" * 60)
    elif args.json:
        print(json.dumps(pricing, indent=2))
    else:
        print("\n" + "=" * 60)
        print("PRICING (copy to src/cost_tracker.py PRICING['deepinfra']):")
        print("=" * 60)
        print(format_for_cost_tracker(pricing))
        print("=" * 60)

    # Show summary
    print("\nSummary:")
    for model, prices in sorted(pricing.items()):
        print(f"  {model}: ${prices['input']}/1M input, ${prices['output']}/1M output")


if __name__ == "__main__":
    main()
