# Agent Orchestration System

**Last Updated:** 2026-01-13
**Version:** PHASE 7 - Multi-Agent Orchestration with Autonomous Tool Loop

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Components](#2-core-components)
3. [Agent System](#3-agent-system)
4. [State Management](#4-state-management)
5. [Autonomous Tool Loop](#5-autonomous-tool-loop)
6. [Tool Adapter System](#6-tool-adapter-system)
7. [Prompt System](#7-prompt-system)
8. [Provider Architecture](#8-provider-architecture)
9. [Observability](#9-observability)
10. [Configuration Reference](#10-configuration-reference)
11. [End-to-End Execution Flow](#11-end-to-end-execution-flow)
12. [Error Handling](#12-error-handling)
13. [Human-in-the-Loop (HITL)](#13-human-in-the-loop-hitl)

---

## 1. Architecture Overview

### 1.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (Dual-Phase)                                │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │ PHASE 1: ROUTING            │    │ PHASE 2: SYNTHESIS                  │ │
│  │ • Query complexity (0-100)  │    │ • Agent output aggregation          │ │
│  │ • Query type classification │    │ • Citation validation               │ │
│  │ • Agent sequence planning   │    │ • Final answer generation           │ │
│  │ • Follow-up detection       │    │ • Language matching                 │ │
│  │ • Vagueness scoring         │    │ • Cost tracking                     │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
│                      src/multi_agent/agents/orchestrator.py:1-1045           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  EXTRACTOR AGENT     │ │  COMPLIANCE AGENT    │ │  RISK VERIFIER       │
│  Tier: SPECIALIST    │ │  Tier: SPECIALIST    │ │  Tier: SPECIALIST    │
│  Role: EXTRACT       │ │  Role: VERIFY        │ │  Role: VERIFY        │
│  extractor.py:1-160  │ │  compliance.py:1-177 │ │  risk_verifier.py    │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOOL ADAPTER LAYER                                   │
│                    src/multi_agent/tools/adapter.py:1-525                    │
│  • Hallucination detection (line 99-114)                                     │
│  • Input validation (lines 193-222)                                          │
│  • Schema generation (lines 440-496)                                         │
│  • Execution tracking & metrics                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG TOOL REGISTRY                                    │
│                      src/agent/tools/registry.py                             │
│  ┌─────────────┐ ┌───────────────┐ ┌─────────────────────┐ ┌──────────────┐ │
│  │ search      │ │ graph_search  │ │ hierarchical_search │ │ expand_ctx   │ │
│  │ bm25_search │ │ hybrid_search │ │ get_document_info   │ │ compare_docs │ │
│  └─────────────┘ └───────────────┘ └─────────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL LAYER                                      │
│  ┌──────────────────────────────┐    ┌──────────────────────────────────┐   │
│  │ HyDE + Expansion Fusion     │    │ PostgreSQL pgvector              │   │
│  │ fusion_retriever.py:1-794   │    │ vectors.layer1/2/3 tables        │   │
│  │ 4-signal weighted fusion    │    │ 4096-dim Qwen3-Embedding-8B      │   │
│  └──────────────────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Workflow Engine | LangGraph | Agent graph construction and execution |
| LLM Providers | Anthropic, OpenAI, Google, DeepInfra | Multi-provider support |
| State Management | Pydantic + LangGraph Reducers | Type-safe state with parallel merge |
| Vector Storage | PostgreSQL + pgvector | Embedding storage and similarity search |
| Knowledge Graph | Neo4j + Graphiti | Temporal entity relationships |
| Observability | LangSmith, Sentry, EventBus | Tracing, error tracking, real-time events |
| Caching | LRU + TTL Caches | Performance optimization |

### 1.3 Design Principles

1. **Autonomous Agents (CLAUDE.md Constraint #0)**
   - LLM decides tool calling sequence, NOT hardcoded workflows
   - System prompts guide behavior, code does NOT dictate steps
   - Exception: Orchestrator has routing logic (complexity → agent sequence)

2. **Single Source of Truth (SSOT)**
   - One canonical implementation per feature
   - Agent initialization via `initialize_agent()` only
   - Prompts loaded from `prompts/agents/*.txt`
   - API keys in `.env` only, never in code

3. **Dual-Phase Orchestration**
   - Phase 1: Query analysis and agent routing
   - Phase 2: Output synthesis after agents complete
   - Separation prevents context overflow

---

## 2. Core Components

### 2.1 BaseAgent Class

**File:** `src/multi_agent/core/agent_base.py:83-1104`

The abstract base class that all 8 agents inherit from. Defines standard interface, lifecycle, and autonomous tool calling patterns.

#### Class Hierarchy

```
BaseAgent (ABC)
├── AgentConfig (dataclass, lines 43-81)
├── AgentTier (enum, lines 25-30)
├── AgentRole (enum, lines 32-41)
└── Methods
    ├── execute() - Template method with error handling (lines 144-186)
    ├── execute_impl() - Abstract, implemented by subclasses (lines 126-142)
    ├── _run_autonomous_tool_loop() - Core autonomous pattern (lines 522-1040)
    ├── _build_agent_context() - State to context conversion (lines 321-392)
    ├── _should_stop_early() - Intelligent stopping (lines 413-477)
    └── compress_output_for_downstream() - Token optimization (lines 1042-1104)
```

#### AgentTier Enum (`agent_base.py:25-30`)

```python
class AgentTier(str, Enum):
    """Agent execution tier for organizational purposes."""
    ORCHESTRATOR = "orchestrator"  # Root coordinator
    SPECIALIST = "specialist"      # Domain-specific agent
    WORKER = "worker"              # Tool-executing agent
```

#### AgentRole Enum (`agent_base.py:32-41`)

```python
class AgentRole(str, Enum):
    """Agent responsibilities in the workflow."""
    ORCHESTRATE = "orchestrate"    # Coordinate other agents
    EXTRACT = "extract"            # Extract info from docs
    CLASSIFY = "classify"          # Classify queries/docs
    VERIFY = "verify"              # Verify compliance/risk
    AUDIT = "audit"                # Audit citations
    SYNTHESIZE = "synthesize"      # Synthesize gaps
    REPORT = "report"              # Generate reports
```

#### AgentConfig Dataclass (`agent_base.py:43-81`)

```python
@dataclass
class AgentConfig:
    """Per-agent configuration (loaded from config.json per-agent section)."""

    name: str                       # Agent name (e.g., 'extractor')
    role: AgentRole                 # Agent role/responsibility
    tier: AgentTier                 # Execution tier
    model: str                      # LLM model for this agent
    api_key: str = ""               # API key for LLM provider
    max_tokens: int = 4096          # Max output tokens
    temperature: float = 0.3        # LLM temperature
    tools: Set[str] = field(default_factory=set)  # Tool names
    timeout_seconds: int = 30       # Tool execution timeout
    retry_count: int = 2            # Retry failed tool calls
    enable_prompt_caching: bool = True
    enable_cost_tracking: bool = True
    parent_agent: Optional[str] = None  # For hierarchical agents
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2.2 Agent Initializer (SSOT)

**File:** `src/multi_agent/core/agent_initializer.py:33-95`

The single source of truth for agent initialization. All agents use this function instead of duplicating initialization code.

```python
def initialize_agent(
    config: Any,
    agent_name: str,
    prompt_name: Optional[str] = None
) -> AgentComponents:
    """
    Initialize common agent components (provider, prompts, tools).

    Returns:
        AgentComponents with provider, system_prompt, and tool_adapter
    """
    from src.agent.providers.factory import create_provider
    from ..prompts.loader import get_prompt_loader
    from ..tools.adapter import get_tool_adapter

    # 1. Initialize provider (auto-detects from model name: claude/gpt/gemini)
    provider = create_provider(model=config.model)

    # 2. Load system prompt
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_prompt(prompt_name or agent_name)

    # 3. Initialize tool adapter
    tool_adapter = get_tool_adapter()

    return AgentComponents(
        provider=provider,
        system_prompt=system_prompt,
        tool_adapter=tool_adapter
    )
```

#### AgentComponents Dataclass (`agent_initializer.py:24-31`)

```python
@dataclass
class AgentComponents:
    """Container for initialized agent components."""
    provider: Any           # BaseProvider
    system_prompt: str      # Loaded from prompts/agents/
    tool_adapter: Any       # ToolAdapter instance
```

---

## 3. Agent System

### 3.1 Agent Overview Table

| Agent | File | Role | Tier | Primary Tools | Lines |
|-------|------|------|------|---------------|-------|
| **Orchestrator** | `orchestrator.py` | ORCHESTRATE | ORCHESTRATOR | get_document_list | 1-1045 |
| **Extractor** | `extractor.py` | EXTRACT | SPECIALIST | search, hierarchical_search, expand_context | 1-160 |
| **Classifier** | `classifier.py` | CLASSIFY | SPECIALIST | search, get_document_info | 1-104 |
| **RequirementExtractor** | `requirement_extractor.py` | EXTRACT | SPECIALIST | hierarchical_search, graph_search | 1-156 |
| **Compliance** | `compliance.py` | VERIFY | SPECIALIST | graph_search, assess_confidence | 1-177 |
| **RiskVerifier** | `risk_verifier.py` | VERIFY | SPECIALIST | similarity_search, compare_documents | 1-104 |
| **CitationAuditor** | `citation_auditor.py` | AUDIT | SPECIALIST | search, get_document_info | 1-105 |
| **GapSynthesizer** | `gap_synthesizer.py` | SYNTHESIZE | SPECIALIST | hierarchical_search, graph_search | 1-105 |

### 3.2 Orchestrator Agent (Dual-Phase)

**File:** `src/multi_agent/agents/orchestrator.py:1-1045`

The orchestrator is the ONLY agent that communicates with the user. It operates in two phases:

#### Phase 1: Routing (`_route_query`, lines 127-249)

```python
async def _route_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    PHASE 1: Route query to appropriate agents.

    Uses unified LLM analysis for:
    - Follow-up detection and query rewriting
    - Complexity scoring (0-100)
    - Vagueness scoring (0.0-1.0)
    - Query type classification
    """
```

**Routing Decision Output:**
```python
{
    "complexity_score": 65,           # 0-100 complexity
    "query_type": "compliance",        # simple_search, cross_doc, compliance, risk, synthesis
    "agent_sequence": ["extractor", "requirement_extractor", "compliance"],
    "analysis": {
        "is_follow_up": False,         # Follow-up detection
        "follow_up_rewrite": None,     # Rewritten standalone query
        "vagueness_score": 0.3,        # 0.0 = specific, 1.0 = vague
        "needs_clarification": False,  # HITL trigger
        "semantic_type": "compliance_check"
    }
}
```

#### Phase 2: Synthesis (`_synthesize_final_answer`, lines 487-725)

```python
async def _synthesize_final_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    PHASE 2: Synthesize final answer from agent outputs.

    Features:
    - Citation validation (lines 607-629)
    - Language matching (Czech query → Czech answer)
    - Cost aggregation
    - Iteration request handling (for multi-step queries)
    """
```

#### Valid Agents List (`orchestrator.py:829-837`)

```python
VALID_AGENTS = {
    "extractor",
    "classifier",
    "requirement_extractor",
    "compliance",
    "risk_verifier",
    "citation_auditor",
    "gap_synthesizer",
}
```

### 3.3 Extractor Agent

**File:** `src/multi_agent/agents/extractor.py:1-160`

Primary document retrieval agent using hybrid search (BM25 + Dense + RRF).

```python
@register_agent("extractor")
class ExtractorAgent(BaseAgent):
    """
    Responsibilities:
    1. Hybrid search for document retrieval
    2. Context expansion around relevant chunks
    3. Document metadata and summary retrieval
    4. Citation preservation (chunk_id tracking)
    """

    # Retrieval parameters (lines 42-43)
    default_k = 6   # Default chunks to retrieve
    max_k = 10      # Maximum for complex queries
```

**Output Structure (lines 122-131):**
```python
extraction_output = {
    "analysis": final_answer,                    # LLM analysis
    "tool_calls_made": ["search", "expand_context"],
    "chunk_ids": ["BZ_VR1_L3_42", ...],          # PRIMARY: for \cite{chunk_id}
    "chunks_data": [{...}, ...],                 # Full chunk data
    "citations": ["1 > 2 > 3", ...],            # Breadcrumb citations
    "iterations": 3,
    "retrieval_method": "autonomous_llm_driven",
    "total_tool_cost_usd": 0.002
}
```

### 3.4 Requirement Extractor Agent

**File:** `src/multi_agent/agents/requirement_extractor.py:1-156`

Extracts atomic legal requirements for compliance checking.

```python
@register_agent("requirement_extractor")
class RequirementExtractorAgent(BaseAgent):
    """
    Based on Legal AI Research (2024): Requirement-First Compliance Checking

    Responsibilities:
    1. Extract atomic legal requirements from legal texts
    2. Decompose complex provisions into verifiable obligations
    3. Classify requirements by granularity, severity, applicability
    4. Generate structured compliance checklist
    """
```

**Output: Structured Checklist (JSON)**
```json
{
    "checklist": [
        {
            "id": "REQ-001",
            "requirement": "Operator shall maintain radiation monitoring...",
            "source_section": "§15 odst. 2",
            "severity": "mandatory",
            "applicability": "nuclear_facility"
        }
    ],
    "target_law": "Atomový zákon 263/2016 Sb."
}
```

### 3.5 Compliance Agent

**File:** `src/multi_agent/agents/compliance.py:1-177`

Verifies compliance using checklist from RequirementExtractor.

```python
@register_agent("compliance")
class ComplianceAgent(BaseAgent):
    """
    REQUIRES: RequirementExtractor output (checklist)

    Responsibilities:
    1. GDPR, CCPA, HIPAA, SOX compliance verification
    2. Bidirectional checking (Contract → Law, Law → Contract)
    3. Violation identification
    4. Gap analysis for missing requirements
    """
```

**Dependency Validation (lines 56-70):**
```python
# CRITICAL: ComplianceAgent REQUIRES RequirementExtractor output
requirement_extractor_output = state.get("agent_outputs", {}).get("requirement_extractor", {})

if not requirement_extractor_output:
    error_msg = (
        "ComplianceAgent error: Missing requirement_extractor output. "
        "Ensure orchestrator routes: extractor → requirement_extractor → compliance."
    )
    logger.error(error_msg)
    state["errors"].append(error_msg)
    return state
```

### 3.6 Risk Verifier Agent

**File:** `src/multi_agent/agents/risk_verifier.py:1-104`

```python
@register_agent("risk_verifier")
class RiskVerifierAgent(BaseAgent):
    """
    Responsibilities:
    1. Risk identification (Legal, Financial, Operational, Compliance, Reputational)
    2. Severity and likelihood assessment
    3. Comparison with industry standards
    4. Mitigation recommendations
    """
```

### 3.7 Citation Auditor Agent

**File:** `src/multi_agent/agents/citation_auditor.py:1-105`

```python
@register_agent("citation_auditor")
class CitationAuditorAgent(BaseAgent):
    """
    Responsibilities:
    1. Citation existence verification
    2. Citation accuracy checking (text matches)
    3. Citation completeness validation
    4. Citation format standardization
    5. Broken reference detection
    """
```

### 3.8 Gap Synthesizer Agent

**File:** `src/multi_agent/agents/gap_synthesizer.py:1-105`

```python
@register_agent("gap_synthesizer")
class GapSynthesizerAgent(BaseAgent):
    """
    5 Gap Types:
    1. Regulatory gaps (missing required clauses)
    2. Coverage gaps (topics not fully addressed)
    3. Consistency gaps (contradictions)
    4. Citation gaps (claims without evidence)
    5. Temporal gaps (outdated information)
    """
```

### 3.9 Classifier Agent

**File:** `src/multi_agent/agents/classifier.py:1-104`

```python
@register_agent("classifier")
class ClassifierAgent(BaseAgent):
    """
    Classification Dimensions:
    1. Document type (Contract, Policy, Report, etc.)
    2. Domain (Legal, Technical, Financial, etc.)
    3. Complexity assessment
    4. Language detection
    5. Sensitivity classification
    """
```

---

## 4. State Management

### 4.1 MultiAgentState

**File:** `src/multi_agent/core/state.py:292-437`

Pydantic model with LangGraph reducer annotations for parallel execution support.

```python
class MultiAgentState(BaseModel):
    """
    Comprehensive state for multi-agent workflow.

    IMPORTANT: Fields use Annotated with reducer functions to support parallel execution.
    When multiple agents run in parallel (fan-out/fan-in), LangGraph needs to know how
    to merge their state updates. Reducers define the merge strategy per field.
    """
```

#### State Fields by Category

**INPUT (immutable after initial set):**
```python
query: Annotated[str, keep_first] = ""  # Original user query
```

**ROUTING:**
```python
query_type: Annotated[QueryType, keep_first] = QueryType.UNKNOWN
complexity_score: Annotated[int, take_max] = 0  # 0-100, keep highest
execution_phase: Annotated[ExecutionPhase, keep_first] = ExecutionPhase.ROUTING
agent_sequence: Annotated[List[str], merge_lists_unique] = []
```

**EXECUTION:**
```python
current_agent: Annotated[Optional[str], keep_first] = None
agent_outputs: Annotated[Dict[str, Any], merge_dicts] = {}
tool_executions: Annotated[List[ToolExecution], operator.add] = []
```

**RETRIEVAL:**
```python
documents: Annotated[List[DocumentMetadata], operator.add] = []
retrieved_text: Annotated[str, operator.add] = ""
```

**RESULTS:**
```python
final_answer: Annotated[Optional[str], keep_first] = None
structured_output: Annotated[Dict[str, Any], merge_dicts] = {}
citations: Annotated[List[str], operator.add] = []
confidence_score: Annotated[Optional[float], take_max] = None
```

**HITL (Human-in-the-Loop):**
```python
quality_check_required: Annotated[bool, operator.or_] = False
quality_issues: Annotated[List[str], operator.add] = []
clarifying_questions: Annotated[List[Dict[str, Any]], operator.add] = []
awaiting_user_input: Annotated[bool, operator.or_] = False
clarification_round: Annotated[int, take_max] = 0
```

**CONVERSATION HISTORY:**
```python
conversation_history: Annotated[List[Dict[str, str]], keep_first] = []
# Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

**UNIFIED ANALYSIS:**
```python
unified_analysis: Annotated[Optional[Dict[str, Any]], keep_first] = None
# Contains: is_follow_up, follow_up_rewrite, vagueness_score, needs_clarification, semantic_type
```

### 4.2 Reducer Functions

**File:** `src/multi_agent/core/state.py:16-91`

| Reducer | Purpose | Usage |
|---------|---------|-------|
| `keep_first` | Keep first non-empty value | Immutable fields (query, session_id) |
| `take_max` | Return maximum value | Numeric fields (complexity_score, confidence) |
| `merge_dicts` | Merge dicts (new overrides existing) | agent_outputs, shared_context |
| `merge_lists_unique` | Merge lists, remove duplicates | agent_sequence |
| `operator.add` | Concatenate lists | tool_executions, errors, citations |
| `operator.or_` | Boolean OR | quality_check_required, awaiting_user_input |

#### keep_first Implementation (`state.py:20-41`)

```python
def keep_first(existing: Any, new: Any) -> Any:
    """
    Reducer that keeps the first non-empty value.
    Treats None, empty strings, and default Enum values as "no value".
    """
    if isinstance(existing, EnumBase):
        if existing.value in ('unknown', 'routing'):
            return new if new != existing else existing
        return existing

    if not existing:
        return new
    return existing
```

### 4.3 QueryType Enum (`state.py:93-102`)

```python
class QueryType(str, Enum):
    SIMPLE_SEARCH = "simple_search"       # Single doc lookup
    CROSS_DOC_ANALYSIS = "cross_doc"      # Compare multiple docs
    COMPLIANCE_CHECK = "compliance"        # Regulatory compliance
    RISK_ASSESSMENT = "risk"               # Risk analysis
    SYNTHESIS = "synthesis"                # Knowledge synthesis
    REPORTING = "reporting"                # Generate report
    UNKNOWN = "unknown"                    # Uncategorized
```

### 4.4 ExecutionPhase Enum (`state.py:104-116`)

```python
class ExecutionPhase(str, Enum):
    ROUTING = "routing"                    # Determine complexity
    AGENT_EXECUTION = "agent_execution"    # Agent executing task
    EXTRACTION = "extraction"              # Extract from docs
    CLASSIFICATION = "classification"      # Classify content
    REQUIREMENT_EXTRACTION = "requirement_extraction"
    VERIFICATION = "verification"          # Verify accuracy
    SYNTHESIS = "synthesis"                # Synthesize results
    REPORTING = "reporting"                # Generate report
    COMPLETE = "complete"                  # Workflow complete
    ERROR = "error"                        # Error state
```

---

## 5. Autonomous Tool Loop

### 5.1 Overview

**File:** `src/multi_agent/core/agent_base.py:522-1040`

The `_run_autonomous_tool_loop()` method is the core of autonomous agent behavior. It implements a loop where:

1. LLM sees state/query + available tools
2. LLM decides to call tools OR provide final answer
3. Tool results are fed back to LLM
4. Loop continues until LLM provides final answer or max iterations

### 5.2 Method Signature (`agent_base.py:522-547`)

```python
async def _run_autonomous_tool_loop(
    self,
    system_prompt: str,
    state: Dict[str, Any],
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Run autonomous tool calling loop where LLM decides which tools to call.

    Returns:
        Dict with:
            - final_answer: LLM's final response
            - tool_calls: List of tools called
            - tool_executions: List[ToolExecution] for state reducer
            - iterations: Number of iterations
            - reasoning: LLM's reasoning trace
            - total_tool_cost_usd: Total API cost
            - trajectory: Dict for evaluation
            - trajectory_metrics: Computed metrics
    """
```

### 5.3 Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS TOOL LOOP                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Build initial context from state                             │
│     └── _build_agent_context() (lines 321-392)                  │
│                                                                  │
│  2. Get tool schemas                                             │
│     └── _get_available_tool_schemas() (lines 394-411)           │
│                                                                  │
│  3. Initialize trajectory capture                                │
│     └── AgentTrajectory for evaluation (line 580)               │
│                                                                  │
│  FOR each iteration (max 10):                                    │
│  │                                                               │
│  │  4. Call LLM with tools                                       │
│  │     └── provider.create_message() (lines 604-611)            │
│  │                                                               │
│  │  5. Track LLM usage and cost (lines 626-656)                 │
│  │                                                               │
│  │  6. Check if LLM wants tools                                  │
│  │     ├── stop_reason == 'tool_use' → Execute tools            │
│  │     └── else → Return final answer                           │
│  │                                                               │
│  │  7. Execute requested tools (lines 674-790)                  │
│  │     ├── Emit TOOL_CALL_START event                           │
│  │     ├── tool_adapter.execute()                               │
│  │     ├── Emit TOOL_CALL_COMPLETE event                        │
│  │     └── Capture OBSERVATION in trajectory                    │
│  │                                                               │
│  │  8. Check early stopping conditions (lines 807-876)          │
│  │     └── _should_stop_early() (lines 413-477)                 │
│  │                                                               │
│  └── Continue loop or return                                     │
│                                                                  │
│  9. Max iterations reached → Final synthesis (lines 919-1040)   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Context Building (`agent_base.py:321-392`)

```python
def _build_agent_context(self, state: Dict[str, Any]) -> str:
    """
    Build context string from state for agent.

    Token-optimized: Filters debug/metadata fields and uses compact JSON.

    Includes:
    - Conversation history (last 3 turns, compact format)
    - Original query
    - Previous agent outputs (filtered, compact JSON)
    - Retrieved documents summary
    """
    # Excluded fields (line 316-319)
    _EXCLUDE_FIELDS = {
        "iterations", "retrieval_method", "total_tool_cost_usd",
        "tool_calls_made", "chunks", "expanded_results", "_internal"
    }
```

### 5.5 Early Stopping Logic (`agent_base.py:413-477`)

```python
def _should_stop_early(
    self,
    tool_call_history: List[Dict[str, Any]],
    iteration: int
) -> tuple[bool, str]:
    """
    Check if agent should stop early based on tool results.

    Prevents over-searching by detecting:
    1. expand_context returned 0 expansions (no more context available)
    2. Multiple consecutive searches returned no results

    NOTE: Does NOT use hardcoded score thresholds.
    Per CLAUDE.md "Autonomous Agents" principle, LLM decides when it has
    enough information using REFLECTION blocks, not hardcoded rules.
    """
```

### 5.6 Tool Result Summarization (`agent_base.py:479-520`)

```python
def _summarize_tool_result_for_context(self, content: str, max_length: int = 1500) -> str:
    """
    Summarize tool result content to reduce token usage in conversation context.

    Truncates long results while preserving structure:
    - Keeps beginning (usually most relevant) - 60%
    - Keeps end (may contain summaries/stats) - 35%
    - Adds truncation indicator in middle
    """
```

---

## 6. Tool Adapter System

### 6.1 Overview

**File:** `src/multi_agent/tools/adapter.py:1-525`

The Tool Adapter bridges LangGraph agents with the existing tool registry. It provides:

- Tool lookup and execution
- **Hallucination detection** (when LLM calls non-existent tools)
- **Input validation** (against Pydantic schemas)
- Execution tracking and metrics
- Schema generation for LLM tool calling

### 6.2 ToolAdapter Class (`adapter.py:35-437`)

```python
class ToolAdapter:
    """
    Adapts existing tool infrastructure for LangGraph agents.

    Handles:
    - Tool lookup in existing registry
    - Input validation (already done by existing tools)
    - Provider selection (already handled by existing providers)
    - Error handling
    - Result formatting
    - Execution tracking
    """

    def __init__(self, tool_registry=None):
        self.registry = tool_registry or get_old_registry()
        self.execution_history: List[ToolExecution] = []
        self.usage_metrics = ToolUsageMetrics()
```

### 6.3 Execute Method with Hallucination Detection (`adapter.py:68-191`)

```python
async def execute(
    self,
    tool_name: str,
    inputs: Dict[str, Any],
    agent_name: str
) -> Dict[str, Any]:
    """
    Execute a tool with LangGraph interface.

    Includes hallucination detection and input validation:
    1. Check if tool exists (hallucination detection)
    2. Validate inputs against Pydantic schema
    3. Execute tool
    4. Track metrics for evaluation
    """
    # Step 1: HALLUCINATION DETECTION (lines 99-114)
    tool = self.registry.get_tool(tool_name)

    if tool is None:
        logger.warning(
            f"HALLUCINATION DETECTED: Agent '{agent_name}' called "
            f"non-existent tool '{tool_name}'"
        )
        return self._format_error_result(
            error_type=ToolErrorType.HALLUCINATION,
            was_hallucinated=True
        )

    # Step 2: INPUT VALIDATION (lines 116-129)
    validation_error = self._validate_inputs(tool, inputs)
    if validation_error:
        return self._format_error_result(
            error_type=ToolErrorType.VALIDATION,
            validation_error=validation_error
        )

    # Step 3: EXECUTE (lines 131-168)
    result: ToolResult = tool.execute(**inputs)
```

### 6.4 Tool Error Types (`adapter.py:26-33`)

```python
class ToolErrorType(str, Enum):
    HALLUCINATION = "hallucination"    # Tool doesn't exist in registry
    VALIDATION = "validation"          # Tool exists but input invalid
    EXECUTION = "execution"            # Tool executed but failed
    TIMEOUT = "timeout"                # Tool execution timed out
    SUCCESS = "success"                # No error
```

### 6.5 Schema Generation (`adapter.py:440-496`)

```python
def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get tool schema in LLM-compatible format (Anthropic/OpenAI).

    Converts Pydantic input schema to tool calling format:
    {
        "name": "search",
        "description": "Search for documents...",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    """
```

### 6.6 Usage Metrics (`state.py:163-259`)

```python
class ToolUsageMetrics(BaseModel):
    """
    Aggregated tool usage metrics for evaluation.
    """
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    hallucinated_calls: int = 0  # Tool name doesn't exist
    validation_errors: int = 0   # Tool exists but input invalid

    tool_stats: Dict[str, ToolStats] = {}

    def hallucination_rate(self) -> float:
        """Critical metric for agent quality evaluation."""
        if self.total_calls == 0:
            return 0.0
        return self.hallucinated_calls / self.total_calls
```

---

## 7. Prompt System

### 7.1 Overview

**File:** `src/multi_agent/prompts/loader.py:1-260`

All LLM system prompts are loaded from the `prompts/` directory. This ensures:
- Easy iteration and version control
- Hot-reload in development
- No hardcoded prompts in Python code

### 7.2 Directory Structure

```
prompts/
├── agents/                    # Multi-agent system prompts
│   ├── orchestrator.txt       # Orchestrator routing/synthesis
│   ├── extractor.txt          # Extractor agent
│   ├── classifier.txt         # Classifier agent
│   ├── compliance.txt         # Compliance agent
│   ├── risk_verifier.txt      # Risk verifier agent
│   ├── requirement_extractor.txt
│   ├── citation_auditor.txt
│   └── gap_synthesizer.txt
├── document_summary.txt       # Document summary generation
├── section_summary.txt        # Section summary generation
├── hyde_expansion.txt         # HyDE query expansion
├── entity_extraction.txt      # Entity extraction (legacy)
└── relationship_extraction.txt # Relationship extraction (legacy)
```

### 7.3 PromptLoader Class (`loader.py:18-223`)

```python
class PromptLoader:
    """
    Load agent prompts from folder.

    Features:
    - Caching for performance
    - Prompt validation
    - Dynamic formatting with context
    - Hot-reloading in development
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent.parent.parent / "prompts" / "agents"

        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
        self._load_all_prompts()  # Load on startup
```

### 7.4 Loading Prompts

```python
# For multi-agent system (agents in prompts/agents/)
from src.multi_agent.prompts.loader import load_prompt, get_prompt_loader

system_prompt = load_prompt("extractor")  # → prompts/agents/extractor.txt

# With context formatting
loader = get_prompt_loader()
formatted = loader.format_prompt("extractor", context={"max_tokens": 1000})

# Hot-reload in development
loader.reload_all()
```

### 7.5 Prompt Injection for Testing (`loader.py:190-202`)

```python
def inject_prompts(self, prompts: Dict[str, str]) -> None:
    """
    Inject prompts directly into cache (for testing/optimization).

    Allows TextGrad optimization to test modified prompts without
    writing to disk.
    """
    for agent_name, prompt_text in prompts.items():
        self._cache[agent_name] = prompt_text
```

---

## 8. Provider Architecture

### 8.1 Overview

The provider system supports multiple LLM providers through a unified interface.

**Key Files:**
- `src/agent/providers/factory.py` - Provider creation and model detection
- `src/agent/providers/base.py` - BaseProvider interface
- `src/agent/providers/anthropic.py` - Anthropic Claude provider
- `src/agent/providers/openai.py` - OpenAI GPT provider
- `src/agent/providers/deepinfra.py` - DeepInfra provider

### 8.2 Provider Detection

```python
from src.agent.providers.factory import detect_provider_from_model, create_provider

# Auto-detect provider from model name
provider = detect_provider_from_model("claude-sonnet-4-5")  # → "anthropic"
provider = detect_provider_from_model("gpt-4o-mini")         # → "openai"
provider = detect_provider_from_model("gemini-2.0-flash")    # → "google"

# Create provider instance
provider = create_provider(model="claude-sonnet-4-5")
```

### 8.3 Unified Provider Interface

```python
class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3
    ) -> ProviderResponse:
        """Create a message with optional tool calling."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (anthropic, openai, google, deepinfra)."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model name being used."""
        pass
```

### 8.4 Model Selection Guidelines

| Use Case | Model | Provider |
|----------|-------|----------|
| Production | `claude-sonnet-4-5` | anthropic |
| Development | `gpt-4o-mini` | openai |
| Budget | `claude-haiku-4-5` | anthropic |
| Embeddings | `Qwen/Qwen3-Embedding-8B` | deepinfra |

---

## 9. Observability

### 9.1 EventBus

**File:** `src/multi_agent/core/event_bus.py:1-313`

Thread-safe event bus for real-time progress streaming.

```python
class EventBus:
    """
    Thread-safe event bus for multi-agent system.

    Features:
    - Async-first design (asyncio.Queue)
    - Bounded queue to prevent memory leaks
    - Event validation via Pydantic
    - Subscriber pattern for logging/debugging
    - Non-blocking batch retrieval
    """
```

#### Event Types (`event_bus.py:31-52`)

```python
class EventType(str, Enum):
    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"

    # Agent events
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"

    # Workflow events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"

    # Error events
    ERROR = "error"
```

#### Usage Example

```python
event_bus = EventBus(max_queue_size=1000)

# Emit event
await event_bus.emit(
    EventType.TOOL_CALL_START,
    {"agent": "extractor", "tool": "search"},
    agent_name="extractor",
    tool_name="search"
)

# Consume events
events = await event_bus.get_pending_events()
for event in events:
    print(f"{event.event_type}: {event.data}")
```

### 9.2 Trajectory Tracking

**File:** `src/multi_agent/observability/trajectory.py`

Captures agent execution trajectory for evaluation:

```python
class AgentTrajectory:
    """
    Captures agent execution steps for evaluation.

    Steps:
    - THOUGHT: LLM reasoning before action
    - ACTION: Tool call with parameters
    - OBSERVATION: Tool result
    """

    def add_thought(self, content: str, iteration: int): ...
    def add_action(self, tool_name: str, tool_input: dict, iteration: int): ...
    def add_observation(self, content: str, success: bool, ...): ...
    def compute_metrics(self) -> TrajectoryMetrics: ...
```

### 9.3 Error Tracking

**File:** `src/multi_agent/core/error_tracker.py`

Centralized error tracking with Sentry integration:

```python
from ..core.error_tracker import track_error, ErrorSeverity

error_id = track_error(
    error=exception,
    severity=ErrorSeverity.HIGH,
    agent_name="orchestrator",
    context={"query": query[:200], "phase": "routing"}
)

# Returns unique error ID for user-facing messages
# Example: "ORG-2026-01-13-12345"
```

### 9.4 Cost Tracking

**File:** `src/cost_tracker.py`

Global cost tracker for LLM and tool usage:

```python
from src.cost_tracker import get_global_tracker

tracker = get_global_tracker()

# Track LLM call
cost = tracker.track_llm(
    provider="anthropic",
    model="claude-sonnet-4-5",
    input_tokens=1500,
    output_tokens=800,
    operation="agent_extractor",
    cache_creation_tokens=0,
    cache_read_tokens=500,
    response_time_ms=1250.5
)

# Get total cost
total = tracker.get_total_cost()
```

### 9.5 LangSmith Integration

**Configuration:**
```bash
# .env
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_PROJECT_NAME=sujbot-multi-agent
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com  # EU endpoint
```

**Key Metrics to Monitor:**
- Latency per agent: extractor, orchestrator_synthesis, compliance
- Token usage: prompt_tokens, completion_tokens
- Tool calls: which tools called, how many iterations
- Error rate: failed runs, timeout patterns

---

## 10. Configuration Reference

### 10.1 config.json Multi-Agent Section

```json
{
    "multi_agent": {
        "enabled": true,
        "orchestrator": {
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "temperature": 0.3,
            "complexity_threshold_low": 30,
            "complexity_threshold_high": 70
        },
        "agents": {
            "extractor": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "temperature": 0.3,
                "tools": ["search", "hierarchical_search", "expand_context", "get_document_info"]
            },
            "classifier": {
                "model": "claude-haiku-4-5",
                "max_tokens": 2048,
                "temperature": 0.2,
                "tools": ["search", "get_document_info"]
            },
            "requirement_extractor": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "temperature": 0.2,
                "tools": ["hierarchical_search", "graph_search", "definition_aligner"]
            },
            "compliance": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "temperature": 0.2,
                "tools": ["graph_search", "assess_confidence"]
            },
            "risk_verifier": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "temperature": 0.3,
                "tools": ["similarity_search", "compare_documents"]
            },
            "citation_auditor": {
                "model": "claude-haiku-4-5",
                "max_tokens": 2048,
                "temperature": 0.2,
                "tools": ["search", "get_document_info"]
            },
            "gap_synthesizer": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "temperature": 0.3,
                "tools": ["hierarchical_search", "graph_search"]
            }
        }
    }
}
```

### 10.2 Environment Variables

```bash
# LLM Providers
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
GOOGLE_API_KEY=xxx
DEEPINFRA_API_KEY=xxx

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/sujbot

# LangSmith Observability
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_PROJECT_NAME=sujbot-multi-agent
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com

# Sentry Error Tracking
SENTRY_DSN=https://xxx@sentry.io/xxx
```

---

## 11. End-to-End Execution Flow

### 11.1 Compliance Check Example

**Query:** "Zkontroluj soulad provozní dokumentace s vyhláškou 359/2016 Sb."

```
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: ORCHESTRATOR PHASE 1 (ROUTING)                                     │
│ File: orchestrator.py:127-249                                              │
├────────────────────────────────────────────────────────────────────────────┤
│ Input: query = "Zkontroluj soulad provozní dokumentace..."                 │
│                                                                            │
│ LLM Analysis:                                                              │
│ ├── complexity_score: 75                                                   │
│ ├── query_type: "compliance"                                               │
│ ├── agent_sequence: ["extractor", "requirement_extractor", "compliance"]   │
│ └── analysis:                                                              │
│     ├── is_follow_up: false                                                │
│     ├── vagueness_score: 0.2                                               │
│     └── semantic_type: "compliance_check"                                  │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: EXTRACTOR AGENT                                                    │
│ File: extractor.py:45-159                                                  │
├────────────────────────────────────────────────────────────────────────────┤
│ _run_autonomous_tool_loop():                                               │
│ ├── Iteration 1: hierarchical_search("vyhláška 359/2016", k=6)            │
│ │   └── Result: 6 chunks from legal documents                              │
│ ├── Iteration 2: search("provozní dokumentace", k=6)                       │
│ │   └── Result: 4 chunks from operational docs                             │
│ └── Iteration 3: LLM provides final answer (no more tools)                 │
│                                                                            │
│ Output:                                                                    │
│ ├── chunk_ids: ["VYH359_L3_42", "VYH359_L3_43", "BZ_VR1_L3_156", ...]     │
│ ├── chunks_data: [{chunk_id, content, score}, ...]                        │
│ └── analysis: "Nalezeno 10 relevantních úseků..."                         │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: REQUIREMENT EXTRACTOR AGENT                                        │
│ File: requirement_extractor.py:43-155                                      │
├────────────────────────────────────────────────────────────────────────────┤
│ _run_autonomous_tool_loop(max_iterations=15):                              │
│ ├── Reads extractor output (chunk_ids, chunks_data)                        │
│ ├── Uses graph_search to find related legal provisions                     │
│ └── Generates structured checklist (JSON)                                  │
│                                                                            │
│ Output (checklist):                                                        │
│ {                                                                          │
│   "checklist": [                                                           │
│     {                                                                      │
│       "id": "REQ-001",                                                     │
│       "requirement": "Provozovatel musí vést evidenci...",                │
│       "source_section": "§15 odst. 2",                                     │
│       "severity": "mandatory"                                              │
│     },                                                                     │
│     ...                                                                    │
│   ],                                                                       │
│   "target_law": "Vyhláška 359/2016 Sb."                                   │
│ }                                                                          │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: COMPLIANCE AGENT                                                   │
│ File: compliance.py:41-176                                                 │
├────────────────────────────────────────────────────────────────────────────┤
│ Validates requirement_extractor output (lines 56-116):                     │
│ ├── Parses JSON checklist                                                  │
│ └── Validates structure: checklist array, target_law                       │
│                                                                            │
│ _run_autonomous_tool_loop():                                               │
│ ├── For each requirement in checklist:                                     │
│ │   ├── graph_search to find evidence in operational docs                  │
│ │   └── Assess compliance status (COMPLIANT/GAP/VIOLATION)                │
│ └── Generates compliance report                                            │
│                                                                            │
│ Output:                                                                    │
│ {                                                                          │
│   "analysis": "## Compliance Report\n\n### REQ-001: COMPLIANT...",        │
│   "violations": [],                                                        │
│   "gaps": ["REQ-003: Missing documentation for..."]                       │
│ }                                                                          │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: ORCHESTRATOR PHASE 2 (SYNTHESIS)                                   │
│ File: orchestrator.py:487-725                                              │
├────────────────────────────────────────────────────────────────────────────┤
│ _synthesize_final_answer():                                                │
│ ├── Builds synthesis context from all agent outputs (lines 733-792)        │
│ ├── Provides available chunk_ids for citation validation (lines 954-991)  │
│ ├── LLM generates final answer with \cite{chunk_id} citations             │
│ └── Validates citations (lines 993-1044)                                   │
│                                                                            │
│ Output:                                                                    │
│ {                                                                          │
│   "final_answer": "## Kontrola souladu s vyhláškou 359/2016 Sb.\n\n..."   │
│ }                                                                          │
│                                                                            │
│ Citation Validation:                                                       │
│ ├── used_citations: ["VYH359_L3_42", "BZ_VR1_L3_156"]                     │
│ ├── invalid_citations: []                                                  │
│ └── valid: true                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Error Handling

### 12.1 Exception Hierarchy

**File:** `src/exceptions.py`

```python
SujbotError (base)
├── ExtractionError      # Document extraction failures
├── ValidationError      # Input/schema validation failures
├── ProviderError        # LLM provider errors
│   └── APIKeyError      # Missing/invalid API key
├── ToolExecutionError   # Tool execution failures
├── AgentError           # Agent-level errors
│   └── AgentInitializationError
├── StorageError         # Database/storage errors
└── RetrievalError       # Vector retrieval errors
```

### 12.2 Agent Error Handling (`agent_base.py:187-225`)

```python
async def handle_error(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle errors gracefully.

    Implements fallback strategy:
    1. Track error with unique ID for Sentry
    2. Add error to state for user notification
    3. Mark execution phase as error
    """
    from .error_tracker import track_error, ErrorSeverity

    error_id = track_error(
        error=error,
        severity=ErrorSeverity.HIGH,
        agent_name=self.config.name,
        context={"query": state.get("query", ""), "phase": state.get("execution_phase", "")}
    )

    error_message = f"[{error_id}] {self.config.name}: {type(error).__name__}: {str(error)}"
    state["errors"].append(error_message)
    state["execution_phase"] = "error"

    return state
```

### 12.3 Critical Tool Failure Surfacing (`agent_base.py:724-752`)

```python
# Critical tools that must succeed for valid results
critical_tools = {
    "hierarchical_search", "similarity_search", "graph_search",
    "get_document_info", "bm25_search", "hybrid_search"
}

if tool_name in critical_tools and not result.get("success"):
    # Add error to state for user notification
    state["errors"].append(
        f"Critical tool '{tool_name}' failed: {error_msg}. "
        f"Results may be incomplete or unreliable."
    )
```

---

## 13. Human-in-the-Loop (HITL)

### 13.1 State Fields (`state.py:352-361`)

```python
# === HUMAN-IN-THE-LOOP (CLARIFICATIONS) ===
quality_check_required: Annotated[bool, operator.or_] = False
quality_issues: Annotated[List[str], operator.add] = []
quality_metrics: Annotated[Optional[Dict[str, float]], merge_dicts] = None
clarifying_questions: Annotated[List[Dict[str, Any]], operator.add] = []
original_query: Annotated[Optional[str], keep_first] = None
user_clarification: Annotated[Optional[str], keep_first] = None
enriched_query: Annotated[Optional[str], keep_first] = None
clarification_round: Annotated[int, take_max] = 0
awaiting_user_input: Annotated[bool, operator.or_] = False
```

### 13.2 Vagueness Detection

The orchestrator's unified analysis includes vagueness scoring:

```python
unified_analysis = {
    "vagueness_score": 0.7,          # 0.0 = specific, 1.0 = vague
    "needs_clarification": True,      # Triggers HITL flow
    "semantic_type": "analytical"
}
```

**Threshold:** `vagueness_score > 0.6` AND `needs_clarification=True` triggers HITL

### 13.3 Clarification Flow

```
┌───────────────────────────────────────────────────────────────┐
│ 1. Orchestrator detects vague query                           │
│    └── vagueness_score: 0.7, needs_clarification: true       │
├───────────────────────────────────────────────────────────────┤
│ 2. State updated:                                             │
│    ├── awaiting_user_input: true                             │
│    ├── clarifying_questions: [                                │
│    │     {                                                    │
│    │       "question": "Jaký konkrétní zákon?",              │
│    │       "context": "Query mentions compliance...",         │
│    │       "suggestions": ["Atomový zákon", "Vyhláška 359"]  │
│    │     }                                                    │
│    │   ]                                                      │
│    └── original_query: "Zkontroluj soulad..."                │
├───────────────────────────────────────────────────────────────┤
│ 3. Frontend pauses workflow, shows clarification UI          │
├───────────────────────────────────────────────────────────────┤
│ 4. User provides clarification                                │
│    └── user_clarification: "Myslím vyhlášku 359/2016 Sb."    │
├───────────────────────────────────────────────────────────────┤
│ 5. Workflow resumes with enriched query                       │
│    └── enriched_query: "Zkontroluj soulad ... s 359/2016"    │
└───────────────────────────────────────────────────────────────┘
```

---

## Summary

The SUJBOT multi-agent orchestration system implements:

1. **8 Specialized Agents** with autonomous tool calling
2. **Dual-Phase Orchestration** (routing → synthesis)
3. **LangGraph State Management** with parallel merge reducers
4. **Hallucination Detection** in tool adapter
5. **SSOT Principles** throughout (single initialization, prompts from files)
6. **Real-time Observability** via EventBus, LangSmith, Sentry
7. **Human-in-the-Loop** for vague queries

The system processes complex legal/technical queries by:
1. Analyzing complexity and routing to appropriate agents
2. Executing agents with autonomous tool calling
3. Synthesizing outputs into final answer with citations
4. Tracking costs, metrics, and errors throughout
