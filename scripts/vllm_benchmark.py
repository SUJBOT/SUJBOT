#!/usr/bin/env python3
"""
vLLM throughput benchmark for Qwen3-VL models on GB10.

Measures TTFT, decode throughput, and e2e latency for realistic RAG workloads.
Uses streaming to capture per-token timing. Supports A/B comparison between
two endpoints (baseline vs optimized), including different models.

Usage:
    # Single endpoint (default model: Qwen3-VL-30B)
    python scripts/vllm_benchmark.py --url http://localhost:8080/v1

    # Benchmark 8B model
    python scripts/vllm_benchmark.py \
        --url http://localhost:8082/v1 \
        --model Qwen/Qwen3-VL-8B-Instruct

    # A/B comparison: 30B vs 8B (different models on different endpoints)
    python scripts/vllm_benchmark.py \
        --url http://localhost:8080/v1 \
        --compare-url http://localhost:8082/v1 \
        --compare-model Qwen/Qwen3-VL-8B-Instruct

    # Quick test (1 iteration, no warmup)
    python scripts/vllm_benchmark.py --url http://localhost:8080/v1 --quick

    # Specific profile only
    python scripts/vllm_benchmark.py --url http://localhost:8080/v1 --profile rag_5pages

    # Via SSH tunnel (e.g., from sujbot)
    python scripts/vllm_benchmark.py --url http://localhost:8080/v1
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Thinking"

# ---------------------------------------------------------------------------
# Realistic tool definitions (subset matching production SUJBOT tools)
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the document corpus for pages relevant to a query. Returns page images (base64 PNG) ranked by cosine similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in Czech or English. Use legal terminology and identifiers.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of pages to retrieve (default 5, max 20).",
                        "default": 5,
                    },
                    "filter_category": {
                        "type": "string",
                        "enum": ["legislation", "documentation"],
                        "description": "Filter by document category.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_context",
            "description": "Get neighboring pages around a page_id from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Page identifier (format: {doc_id}_p{NNN}).",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of pages before and after (default 2).",
                        "default": 2,
                    },
                },
                "required": ["page_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_info",
            "description": "Get document summary, metadata, or section list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document identifier.",
                    }
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_list",
            "description": "List all indexed documents in the corpus.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": "Search the knowledge graph for entities and their relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for entities.",
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "REGULATION",
                            "STANDARD",
                            "ORGANIZATION",
                            "PERSON",
                            "CONCEPT",
                            "REQUIREMENT",
                            "FACILITY",
                        ],
                        "description": "Filter by entity type.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_context",
            "description": "Get multi-hop neighborhood of an entity in the knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Entity name to explore.",
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Number of hops (default 2).",
                        "default": 2,
                    },
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compliance_check",
            "description": "Assess compliance by finding requirements in legislation and checking evidence in documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regulation_query": {
                        "type": "string",
                        "description": "Regulation or requirement to check compliance against.",
                    },
                    "target_document": {
                        "type": "string",
                        "description": "Document to check for compliance evidence.",
                    },
                },
                "required": ["regulation_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current information not in the document corpus. Last resort only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt (truncated for benchmarking — keeps realistic token count)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a document analysis agent for Czech legal and nuclear regulatory documents. You answer questions by searching a document corpus and reading retrieved page images.

You operate in the context of SÚJB (Czech State Office for Nuclear Safety) regulatory compliance. Your role is to help compliance officers, inspectors, and legal professionals find and interpret regulatory information.

## Retrieval system
You operate in VL (Vision-Language) mode. The `search` tool returns page images — full scanned pages as PNG images.
- The query is embedded and matched via cosine similarity against page embeddings in PostgreSQL.
- Results are page images injected into the conversation. You can read the text directly from the images.
- Each page has a `page_id` (format: `{doc_id}_p{NNN}`).
- Default: 5 pages per search (~1600 tokens each).

## Available tools
| Tool | When to use |
|------|------------|
| `search` | Find pages relevant to a question. Primary tool for every factual question. |
| `expand_context` | Get neighboring pages around a page_id from search results. |
| `get_document_info` | Get document summary, metadata, or section list. |
| `get_document_list` | List all indexed documents. |
| `graph_search` | Search knowledge graph for entities. |
| `graph_context` | Get multi-hop neighborhood of an entity. |
| `compliance_check` | Assess compliance: finds requirements in legislation, checks evidence. |
| `web_search` | Search the internet. Last resort only. |

## Search strategy
1. Always search before answering factual questions. Plan 3–8 tool calls per query.
2. Search in Czech with correct declension and legal terminology.
3. Cite every factual claim with `\\cite{page_id}` immediately after the statement.

## Response structure
Organize findings by topic using ## headings with inline citations. End with confidence assessment:
- **Vysoká**: multiple corroborating sources
- **Střední**: limited sources or partial coverage
- **Nízká**: significant gaps after thorough search"""

# ---------------------------------------------------------------------------
# Test profiles — realistic RAG workloads
# ---------------------------------------------------------------------------

# Simulate page image tokens with placeholder text (vLLM processes text tokens
# for the purpose of throughput measurement — actual image tokens would be
# similar in count but different in embedding; this is sufficient for
# measuring decode speed and prefill latency)
PAGE_PLACEHOLDER = (
    "This is a simulated page image placeholder representing approximately "
    "1600 tokens of visual content from a scanned document page. In production, "
    "this would be a base64-encoded PNG image processed by the vision encoder. "
    "The page contains regulatory text in Czech language about nuclear safety "
    "requirements, definitions of key terms, organizational responsibilities, "
    "and compliance criteria. Typical content includes paragraphs referencing "
    "specific sections of Czech legislation (zákon č. 263/2016 Sb.), technical "
    "standards, safety analysis reports, and administrative procedures. "
) * 8  # ~1280 tokens of filler text to approximate ~1600 image tokens


def make_rag_result_content(n_pages: int) -> str:
    """Simulate a search tool result with n_pages of content."""
    pages = []
    for i in range(n_pages):
        pages.append(
            f"## Page {i+1}: DOC_p{i+1:03d}\n"
            f"Similarity: 0.{85 - i*3}\n"
            f"Document: Bezpečnostní zpráva VR1\n\n"
            f"{PAGE_PLACEHOLDER[:800]}"
        )
    return "\n\n---\n\n".join(pages)


PROFILES = {
    "text_only": {
        "description": "System prompt + short query (no RAG context)",
        "messages": [
            {"role": "user", "content": "Jaká je definice jaderného zařízení podle zákona č. 263/2016 Sb.?"},
        ],
        "include_tools": True,
        "max_tokens": 1024,
    },
    "rag_5pages": {
        "description": "Typical RAG query with 5 retrieved pages",
        "messages": [
            {
                "role": "user",
                "content": "Jaké jsou požadavky na zabezpečení jaderného materiálu při přepravě?",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps(
                                {"query": "zabezpečení jaderného materiálu přeprava požadavky"}
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": make_rag_result_content(5),
            },
        ],
        "include_tools": True,
        "max_tokens": 2048,
    },
    "rag_8pages": {
        "description": "Heavy RAG query with 8 retrieved pages",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Proveďte srovnání požadavků vyhlášky 378/2016 Sb. a bezpečnostní zprávy "
                    "reaktoru VR1 v oblasti radiační ochrany. Kde jsou případné nesrovnalosti?"
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_002",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps(
                                {
                                    "query": "vyhláška 378/2016 radiační ochrana požadavky",
                                    "k": 8,
                                }
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "content": make_rag_result_content(8),
            },
        ],
        "include_tools": True,
        "max_tokens": 4096,
    },
    "tool_call": {
        "description": "Tool-calling scenario (expects short tool call output)",
        "messages": [
            {
                "role": "user",
                "content": "Vyhledej v dokumentech informace o kategorizaci zdrojů ionizujícího záření.",
            },
        ],
        "include_tools": True,
        "max_tokens": 512,
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    profile: str
    ttft_s: float  # Time to first token (seconds)
    total_tokens: int  # Total tokens generated (including think tokens)
    visible_tokens: int  # Tokens after stripping <think> blocks
    decode_tps: float  # Decode throughput (total tokens / decode time)
    e2e_latency_s: float  # End-to-end latency (seconds)
    prefill_time_s: float  # Approximate prefill time (≈ TTFT)
    decode_time_s: float  # Decode time (e2e - TTFT)
    think_ratio: float  # Fraction of tokens that were <think> content


@dataclass
class ProfileSummary:
    profile: str
    n_runs: int
    ttft_mean: float
    ttft_std: float
    decode_tps_mean: float
    decode_tps_std: float
    e2e_mean: float
    e2e_std: float
    think_ratio_mean: float
    total_tokens_mean: float
    visible_tokens_mean: float


# ---------------------------------------------------------------------------
# Streaming benchmark runner
# ---------------------------------------------------------------------------


def count_think_tokens(text: str) -> tuple[int, int]:
    """Count total tokens (approx by words) and visible tokens after stripping <think>."""
    import re

    # Rough word-based token estimate (good enough for throughput measurement)
    total_words = len(text.split())

    # Strip <think>...</think> blocks
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    visible_words = len(visible.split())

    return total_words, visible_words


def run_single_benchmark(
    client: httpx.Client,
    base_url: str,
    profile_name: str,
    profile: dict,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
) -> BenchmarkResult:
    """Run a single benchmark request with streaming and measure timing."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + profile["messages"]
    tools = TOOL_DEFS if profile.get("include_tools") else None
    max_tokens = profile.get("max_tokens", 2048)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    url = f"{base_url}/chat/completions"

    full_text = ""
    first_token_time = None
    start_time = time.perf_counter()
    token_count = 0

    with client.stream("POST", url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # Count content tokens
            content = delta.get("content", "")
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += content
                token_count += 1  # Each SSE chunk ≈ 1 token

            # Count tool call tokens
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "")
                    if args:
                        token_count += max(1, len(args) // 4)

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time  # No tokens produced

    ttft = first_token_time - start_time
    e2e = end_time - start_time
    decode_time = e2e - ttft

    # Token counting
    total_words, visible_words = count_think_tokens(full_text)
    # Use SSE chunk count as primary, fall back to word count
    total_tokens = max(token_count, total_words)
    visible_tokens = visible_words

    think_ratio = 1.0 - (visible_tokens / total_tokens) if total_tokens > 0 else 0.0
    decode_tps = total_tokens / decode_time if decode_time > 0 else 0.0

    return BenchmarkResult(
        profile=profile_name,
        ttft_s=ttft,
        total_tokens=total_tokens,
        visible_tokens=visible_tokens,
        decode_tps=decode_tps,
        e2e_latency_s=e2e,
        prefill_time_s=ttft,
        decode_time_s=decode_time,
        think_ratio=think_ratio,
    )


def run_profile_benchmark(
    base_url: str,
    profile_name: str,
    profile: dict,
    model: str = DEFAULT_MODEL,
    n_warmup: int = 2,
    n_iterations: int = 5,
    timeout: float = 120.0,
) -> ProfileSummary:
    """Run a complete benchmark for one profile with warmup and iterations."""
    client = httpx.Client(timeout=timeout)

    # Warmup runs (discarded)
    for i in range(n_warmup):
        print(f"  Warmup {i+1}/{n_warmup}...", end=" ", flush=True)
        try:
            result = run_single_benchmark(client, base_url, profile_name, profile, model, timeout)
            print(f"OK ({result.total_tokens} tok, {result.e2e_latency_s:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Measured runs
    results: list[BenchmarkResult] = []
    for i in range(n_iterations):
        print(f"  Run {i+1}/{n_iterations}...", end=" ", flush=True)
        try:
            result = run_single_benchmark(client, base_url, profile_name, profile, model, timeout)
            results.append(result)
            print(
                f"TTFT={result.ttft_s:.2f}s  "
                f"decode={result.decode_tps:.1f} tok/s  "
                f"e2e={result.e2e_latency_s:.1f}s  "
                f"tokens={result.total_tokens} (visible={result.visible_tokens})  "
                f"think={result.think_ratio:.0%}"
            )
        except Exception as e:
            print(f"FAILED: {e}")

    client.close()

    if not results:
        return ProfileSummary(
            profile=profile_name,
            n_runs=0,
            ttft_mean=0,
            ttft_std=0,
            decode_tps_mean=0,
            decode_tps_std=0,
            e2e_mean=0,
            e2e_std=0,
            think_ratio_mean=0,
            total_tokens_mean=0,
            visible_tokens_mean=0,
        )

    return ProfileSummary(
        profile=profile_name,
        n_runs=len(results),
        ttft_mean=statistics.mean(r.ttft_s for r in results),
        ttft_std=statistics.stdev(r.ttft_s for r in results) if len(results) > 1 else 0,
        decode_tps_mean=statistics.mean(r.decode_tps for r in results),
        decode_tps_std=statistics.stdev(r.decode_tps for r in results) if len(results) > 1 else 0,
        e2e_mean=statistics.mean(r.e2e_latency_s for r in results),
        e2e_std=statistics.stdev(r.e2e_latency_s for r in results) if len(results) > 1 else 0,
        think_ratio_mean=statistics.mean(r.think_ratio for r in results),
        total_tokens_mean=statistics.mean(r.total_tokens for r in results),
        visible_tokens_mean=statistics.mean(r.visible_tokens for r in results),
    )


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------


def print_summary_table(label: str, summaries: list[ProfileSummary]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(
        f"{'Profile':<15} {'TTFT (s)':<12} {'Decode (t/s)':<14} {'E2E (s)':<12} "
        f"{'Tokens':<10} {'Visible':<10} {'Think%':<8} {'Runs':<5}"
    )
    print("-" * 86)
    for s in summaries:
        print(
            f"{s.profile:<15} "
            f"{s.ttft_mean:>5.2f}±{s.ttft_std:<4.2f} "
            f"{s.decode_tps_mean:>6.1f}±{s.decode_tps_std:<5.1f} "
            f"{s.e2e_mean:>5.1f}±{s.e2e_std:<4.1f} "
            f"{s.total_tokens_mean:>7.0f}   "
            f"{s.visible_tokens_mean:>7.0f}   "
            f"{s.think_ratio_mean:>5.0%}   "
            f"{s.n_runs:>3}"
        )
    print()


def print_comparison(a_label: str, a: list[ProfileSummary], b_label: str, b: list[ProfileSummary]):
    """Print side-by-side comparison of two benchmark runs."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: {a_label} vs {b_label}")
    print(f"{'='*80}")
    print(
        f"{'Profile':<15} {'TTFT Δ':<12} {'Decode Δ':<14} {'E2E Δ':<12} {'Think% Δ':<10}"
    )
    print("-" * 63)

    a_map = {s.profile: s for s in a}
    b_map = {s.profile: s for s in b}

    for profile in PROFILES:
        sa = a_map.get(profile)
        sb = b_map.get(profile)
        if not sa or not sb or sa.n_runs == 0 or sb.n_runs == 0:
            continue

        ttft_delta = sb.ttft_mean - sa.ttft_mean
        ttft_pct = (ttft_delta / sa.ttft_mean * 100) if sa.ttft_mean > 0 else 0
        decode_delta = sb.decode_tps_mean - sa.decode_tps_mean
        decode_pct = (decode_delta / sa.decode_tps_mean * 100) if sa.decode_tps_mean > 0 else 0
        e2e_delta = sb.e2e_mean - sa.e2e_mean
        e2e_pct = (e2e_delta / sa.e2e_mean * 100) if sa.e2e_mean > 0 else 0
        think_delta = sb.think_ratio_mean - sa.think_ratio_mean

        # Color-code: negative TTFT/e2e is good (↓), positive decode is good (↑)
        ttft_sign = "↓" if ttft_delta < 0 else "↑"
        decode_sign = "↑" if decode_delta > 0 else "↓"
        e2e_sign = "↓" if e2e_delta < 0 else "↑"

        print(
            f"{profile:<15} "
            f"{ttft_sign}{abs(ttft_delta):>4.2f}s ({ttft_pct:+.0f}%)  "
            f"{decode_sign}{abs(decode_delta):>4.1f} ({decode_pct:+.0f}%)   "
            f"{e2e_sign}{abs(e2e_delta):>4.1f}s ({e2e_pct:+.0f}%)  "
            f"{think_delta:+.0%}"
        )
    print()


# ---------------------------------------------------------------------------
# Server health check
# ---------------------------------------------------------------------------


def check_server(base_url: str, label: str = "Server") -> bool:
    """Check if vLLM server is responding."""
    try:
        r = httpx.get(f"{base_url}/models", timeout=10)
        r.raise_for_status()
        models = r.json().get("data", [])
        model_ids = [m["id"] for m in models]
        print(f"  {label}: OK — models: {', '.join(model_ids)}")
        return True
    except Exception as e:
        print(f"  {label}: FAILED — {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="vLLM throughput benchmark for Qwen3-VL models on GB10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Base URL of vLLM server (e.g., http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name for primary endpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--compare-url",
        help="Second URL for A/B comparison",
    )
    parser.add_argument(
        "--compare-model",
        help="Model name for compare endpoint (defaults to --model value)",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=5,
        help="Number of measured iterations per profile (default: 5)",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=2,
        help="Number of warmup iterations (discarded, default: 2)",
    )
    parser.add_argument(
        "--profile", "-p",
        choices=list(PROFILES.keys()),
        help="Run only a specific profile",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1 iteration, no warmup",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Request timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (in addition to table)",
    )
    args = parser.parse_args()

    if args.quick:
        args.iterations = 1
        args.warmup = 0

    compare_model = args.compare_model or args.model
    profiles_to_run = {args.profile: PROFILES[args.profile]} if args.profile else PROFILES

    # Health check
    print("\nChecking server(s)...")
    if not check_server(args.url, "Primary"):
        print("Primary server not responding. Exiting.")
        sys.exit(1)
    if args.compare_url and not check_server(args.compare_url, "Compare"):
        print("Compare server not responding. Exiting.")
        sys.exit(1)

    # Run primary benchmarks
    primary_label = f"{args.url} ({args.model})"
    print(f"\n--- Benchmarking: {primary_label} ---")
    print(f"    Iterations: {args.iterations}, Warmup: {args.warmup}")
    primary_results = []
    for name, profile in profiles_to_run.items():
        print(f"\n[{name}] {profile['description']}")
        summary = run_profile_benchmark(
            args.url, name, profile,
            model=args.model,
            n_warmup=args.warmup,
            n_iterations=args.iterations,
            timeout=args.timeout,
        )
        primary_results.append(summary)

    print_summary_table(primary_label, primary_results)

    # Run comparison benchmarks
    compare_results = None
    if args.compare_url:
        compare_label = f"{args.compare_url} ({compare_model})"
        print(f"\n--- Benchmarking: {compare_label} ---")
        print(f"    Iterations: {args.iterations}, Warmup: {args.warmup}")
        compare_results = []
        for name, profile in profiles_to_run.items():
            print(f"\n[{name}] {profile['description']}")
            summary = run_profile_benchmark(
                args.compare_url, name, profile,
                model=compare_model,
                n_warmup=args.warmup,
                n_iterations=args.iterations,
                timeout=args.timeout,
            )
            compare_results.append(summary)

        print_summary_table(compare_label, compare_results)
        print_comparison(primary_label, primary_results, compare_label, compare_results)

    # JSON output
    if args.json:
        output = {
            "primary": {
                "url": args.url,
                "model": args.model,
                "results": [
                    {
                        "profile": s.profile,
                        "n_runs": s.n_runs,
                        "ttft_mean": round(s.ttft_mean, 3),
                        "ttft_std": round(s.ttft_std, 3),
                        "decode_tps_mean": round(s.decode_tps_mean, 1),
                        "decode_tps_std": round(s.decode_tps_std, 1),
                        "e2e_mean": round(s.e2e_mean, 2),
                        "e2e_std": round(s.e2e_std, 2),
                        "think_ratio_mean": round(s.think_ratio_mean, 3),
                        "total_tokens_mean": round(s.total_tokens_mean, 0),
                        "visible_tokens_mean": round(s.visible_tokens_mean, 0),
                    }
                    for s in primary_results
                ],
            }
        }
        if compare_results:
            output["compare"] = {
                "url": args.compare_url,
                "model": compare_model,
                "results": [
                    {
                        "profile": s.profile,
                        "n_runs": s.n_runs,
                        "ttft_mean": round(s.ttft_mean, 3),
                        "ttft_std": round(s.ttft_std, 3),
                        "decode_tps_mean": round(s.decode_tps_mean, 1),
                        "decode_tps_std": round(s.decode_tps_std, 1),
                        "e2e_mean": round(s.e2e_mean, 2),
                        "e2e_std": round(s.e2e_std, 2),
                        "think_ratio_mean": round(s.think_ratio_mean, 3),
                        "total_tokens_mean": round(s.total_tokens_mean, 0),
                        "visible_tokens_mean": round(s.visible_tokens_mean, 0),
                    }
                    for s in compare_results
                ],
            }
        print("\n--- JSON Output ---")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
