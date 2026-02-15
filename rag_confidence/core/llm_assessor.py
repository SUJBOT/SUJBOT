"""
LLM Sufficiency Assessor for RAG Confidence Benchmarking.

Implements binary sufficiency assessment matching the ICLR 2025 paper methodology:
- Context-set level binary classification (NOT per-chunk)
- Binary output: 0 (insufficient) or 1 (sufficient)
- 1-shot prompting with chain-of-thought reasoning

Reference: "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems"
           Joren et al., ICLR 2025

This module provides a clean interface for benchmarking purposes.
"""

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


def _create_llm_client(model: str):
    """Create an LLM client based on model name."""
    if "claude" in model or "anthropic" in model:
        import anthropic
        return anthropic.Anthropic()
    elif "gpt" in model or "o1" in model or "o3" in model:
        import openai
        return openai.OpenAI()
    else:
        # Default to Anthropic
        import anthropic
        return anthropic.Anthropic()


def _extract_text(response) -> str:
    """Extract text from an Anthropic Messages API response."""
    if hasattr(response, "content") and response.content:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
    return str(response)


@dataclass
class AssessmentResult:
    """Result of a single sufficiency assessment."""

    is_sufficient: bool  # Binary classification: True = sufficient
    score: float  # 1.0 if sufficient, 0.0 if insufficient
    reasoning: Optional[str] = None  # LLM explanation (if available)
    latency_ms: float = 0.0  # Time taken for assessment
    tokens_used: int = 0  # Approximate token count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_sufficient": self.is_sufficient,
            "score": self.score,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
        }


@dataclass
class BatchAssessmentResult:
    """Result of batch sufficiency assessment."""

    scores: List[float]  # List of scores (0.0 or 1.0)
    results: List[AssessmentResult]  # Individual results
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    @property
    def success_rate(self) -> float:
        """Fraction of queries classified as sufficient."""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


# Vision-specific prompt template (ICLR 2025 paper adapted for images)
VISION_PROMPT_TEMPLATE = """You are an expert LLM evaluator that excels at evaluating a QUESTION and DOCUMENT IMAGES.

Consider the following criteria:
Sufficient Context: 1 IF the DOCUMENT IMAGES contain sufficient information to infer the answer to the question
                   0 IF they cannot be used to infer the answer to the question

Here is an EXAMPLE that demonstrates the task and the output format:
### EXAMPLE QUESTION
What is the total revenue reported in Q3?
### EXAMPLE DOCUMENT IMAGES
[Images of financial report pages showing quarterly data]
### EXAMPLE EXPLANATION
The document images show a financial report with quarterly breakdown. Page 2 contains a table with Q3 revenue figures clearly visible. The information needed to answer the question is present.
### EXAMPLE EVALUATION
{{"Sufficient Context": 1}}

Now evaluate the following:

### QUESTION
{query}

### DOCUMENT IMAGES
[{num_images} document page images shown above]

First, provide your step-by-step reasoning in the ### EXPLANATION section.
Then, provide your evaluation in the ### EVALUATION section as JSON.

### EXPLANATION
"""

# Paper-faithful 1-shot prompt template
PAPER_PROMPT_TEMPLATE = """You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES.
Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question

First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION.
Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON)

EXAMPLE:
### QUESTION
In which year did the publisher of Roald Dahl's Guide to Railway Safety cease to exist?
### References
Roald Dahl's Guide to Railway Safety was published in 1991 by the British Railways Board. The British Railways Board had asked Roald Dahl to write the text of the booklet, and Quentin Blake to illustrate it, to help young people enjoy using the railways safely. The British Railways Board (BRB) was a nationalised industry in the United Kingdom that operated from 1963 to 2001. Until 1997 it was responsible for most railway services in Great Britain, trading under the brand name British Railways and, from 1965, British Rail. It did not operate railways in Northern Ireland, where railways were the responsibility of the Government of Northern Ireland.
### EXPLANATION
The context mentions that Roald Dahl's Guide to Railway Safety was published by the British Railways Board. It also states that the British Railways Board operated from 1963 to 2001, meaning the year it ceased to exist was 2001. Therefore, the context does provide a precise answer to the question.
### JSON
{{"Sufficient Context": 1}}

Remember the instructions: You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES. Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question

First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION.
Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON)

### QUESTION
{query}
### REFERENCES
{context}
"""


class LLMSufficiencyAssessor:
    """
    Binary sufficiency assessor for RAG confidence benchmarking.

    Implements the ICLR 2025 paper methodology:
    - Evaluates entire context set (not per-chunk)
    - Returns binary classification (0 or 1)
    - Uses 1-shot prompting with chain-of-thought

    Example:
        assessor = LLMSufficiencyAssessor(model="claude-haiku-4-5")
        result = assessor.assess(
            query="What year did the British Railways Board cease to exist?",
            chunks=[{"content": "The BRB operated from 1963 to 2001..."}]
        )
        print(result.is_sufficient)  # True
        print(result.score)  # 1.0
    """

    # Default model for cost efficiency
    DEFAULT_MODEL = "claude-haiku-4-5"

    # Max context length (characters) to avoid context window issues
    MAX_CONTEXT_CHARS = 24000

    # Max chunks to include in context
    MAX_CHUNKS = 10

    # Max images to include in vision context
    MAX_IMAGES = 10

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prompt_template: Optional[str] = None,
        vision_prompt_template: Optional[str] = None,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        max_chunks: int = MAX_CHUNKS,
        max_images: int = MAX_IMAGES,
    ):
        """
        Initialize LLM sufficiency assessor.

        Args:
            model: LLM model to use (default: claude-haiku-4-5)
            prompt_template: Custom prompt template for text (uses paper template if None)
            vision_prompt_template: Custom prompt template for vision (uses vision template if None)
            max_context_chars: Maximum context length in characters
            max_chunks: Maximum number of chunks to include
            max_images: Maximum number of images to include in vision assessment
        """
        self._model = model
        self._provider = None  # Lazy initialization
        self._prompt_template = prompt_template or PAPER_PROMPT_TEMPLATE
        self._vision_prompt_template = vision_prompt_template or VISION_PROMPT_TEMPLATE
        self._max_context_chars = max_context_chars
        self._max_chunks = max_chunks
        self._max_images = max_images

        logger.info(f"LLMSufficiencyAssessor initialized (model={model})")

    @property
    def client(self):
        """Lazy-load LLM client on first access."""
        if self._provider is None:
            self._provider = _create_llm_client(self._model)
        return self._provider

    @property
    def model(self) -> str:
        """Return model name."""
        return self._model

    def assess(
        self,
        query: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List["Image.Image"]] = None,
        max_chunks: Optional[int] = None,
        max_images: Optional[int] = None,
    ) -> AssessmentResult:
        """
        Assess whether context (chunks or images) is sufficient to answer query.

        Args:
            query: User query
            chunks: List of chunks with 'content' key (for text assessment)
            images: List of PIL Image objects (for vision assessment)
            max_chunks: Override max chunks (optional)
            max_images: Override max images (optional)

        Returns:
            AssessmentResult with binary classification

        Note:
            If both chunks and images are provided, images take precedence.
            If neither is provided, returns insufficient result.
        """
        # Use vision assessment if images provided
        if images is not None:
            return self._assess_vision(query, images, max_images)

        # Fall back to text assessment
        return self._assess_text(query, chunks or [], max_chunks)

    def _assess_text(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_chunks: Optional[int] = None,
    ) -> AssessmentResult:
        """
        Assess whether text chunks are sufficient to answer query.

        Args:
            query: User query
            chunks: List of chunks with 'content' key
            max_chunks: Override max chunks (optional)

        Returns:
            AssessmentResult with binary classification
        """
        start_time = time.time()

        # Build context from chunks
        max_chunks = max_chunks or self._max_chunks
        context = self._build_context(chunks, max_chunks)

        if not context or not context.strip():
            return AssessmentResult(
                is_sufficient=False,
                score=0.0,
                reasoning="Empty context provided",
                latency_ms=0.0,
            )

        # Build prompt
        prompt = self._prompt_template.format(
            query=query,
            context=context,
        )

        # Call LLM
        try:
            response = self._call_llm(prompt)
            is_sufficient, reasoning = self._parse_response(response)
        except Exception as e:
            logger.error(f"LLM assessment failed: {e}", exc_info=True)
            # Default to insufficient on error
            return AssessmentResult(
                is_sufficient=False,
                score=0.0,
                reasoning=f"Assessment failed: {e}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        latency_ms = (time.time() - start_time) * 1000

        return AssessmentResult(
            is_sufficient=is_sufficient,
            score=1.0 if is_sufficient else 0.0,
            reasoning=reasoning,
            latency_ms=latency_ms,
        )

    def _assess_vision(
        self,
        query: str,
        images: List["Image.Image"],
        max_images: Optional[int] = None,
    ) -> AssessmentResult:
        """
        Assess whether document images are sufficient to answer query.

        Uses Claude's vision capability to analyze document images directly,
        avoiding OCR artifacts and preserving visual context.

        Args:
            query: User query
            images: List of PIL Image objects
            max_images: Override max images (optional)

        Returns:
            AssessmentResult with binary classification
        """
        start_time = time.time()

        max_images = max_images or self._max_images
        images_to_use = images[:max_images]

        if not images_to_use:
            return AssessmentResult(
                is_sufficient=False,
                score=0.0,
                reasoning="No images provided",
                latency_ms=0.0,
            )

        # Build prompt for vision
        prompt = self._vision_prompt_template.format(
            query=query,
            num_images=len(images_to_use),
        )

        # Call LLM with vision
        try:
            response = self._call_llm_vision(prompt, images_to_use)
            is_sufficient, reasoning = self._parse_response(response)
        except Exception as e:
            logger.error(f"Vision assessment failed: {e}", exc_info=True)
            return AssessmentResult(
                is_sufficient=False,
                score=0.0,
                reasoning=f"Vision assessment failed: {e}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        latency_ms = (time.time() - start_time) * 1000

        return AssessmentResult(
            is_sufficient=is_sufficient,
            score=1.0 if is_sufficient else 0.0,
            reasoning=reasoning,
            latency_ms=latency_ms,
        )

    def assess_batch(
        self,
        queries: List[str],
        chunk_lists: List[List[Dict[str, Any]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchAssessmentResult:
        """
        Assess multiple query-context pairs.

        Args:
            queries: List of queries
            chunk_lists: List of chunk lists (one per query)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            BatchAssessmentResult with all scores
        """
        if len(queries) != len(chunk_lists):
            raise ValueError("queries and chunk_lists must have same length")

        results = []
        total_latency = 0.0
        total_tokens = 0

        for i, (query, chunks) in enumerate(zip(queries, chunk_lists)):
            result = self.assess(query, chunks)
            results.append(result)
            total_latency += result.latency_ms
            total_tokens += result.tokens_used

            if progress_callback:
                progress_callback(i + 1, len(queries))

            if (i + 1) % 50 == 0:
                logger.info(f"SCA progress: {i + 1}/{len(queries)}")

        scores = [r.score for r in results]

        return BatchAssessmentResult(
            scores=scores,
            results=results,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
        )

    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: int,
    ) -> str:
        """Build context string from chunks."""
        chunks_to_use = chunks[:max_chunks]
        context_parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks_to_use):
            content = chunk.get("content", chunk.get("raw_content", ""))
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")

            # Check if adding this chunk exceeds limit
            chunk_text = f"[{chunk_id}]\n{content}"
            if total_chars + len(chunk_text) > self._max_context_chars:
                # Truncate this chunk
                remaining = self._max_context_chars - total_chars - 50
                if remaining > 100:
                    content = content[:remaining] + "...[truncated]"
                    chunk_text = f"[{chunk_id}]\n{content}"
                    context_parts.append(chunk_text)
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text) + 2  # +2 for newlines

        return "\n\n".join(context_parts)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response text."""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self._model,
            messages=messages,
            system="You are an expert evaluator of document relevance and sufficiency.",
            max_tokens=1000,
            temperature=0.0,
        )

        return _extract_text(response)

    def _call_llm_vision(self, prompt: str, images: List["Image.Image"]) -> str:
        """
        Call LLM with vision input (images + text prompt).

        Args:
            prompt: Text prompt for evaluation
            images: List of PIL Image objects to analyze

        Returns:
            Response text from the LLM
        """
        # Build multimodal content: images first, then text prompt
        content_blocks = []

        for img in images:
            base64_data = self._image_to_base64(img)
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_data,
                },
            })

        content_blocks.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content_blocks}]

        response = self.client.messages.create(
            model=self._model,
            messages=messages,
            system="You are an expert evaluator of document relevance and sufficiency.",
            max_tokens=1000,
            temperature=0.0,
        )

        return _extract_text(response)

    def _image_to_base64(self, img: "Image.Image", max_size: int = 1568) -> str:
        """
        Convert PIL Image to base64-encoded JPEG.

        Args:
            img: PIL Image object
            max_size: Maximum dimension (Anthropic recommends max 1568px)

        Returns:
            Base64-encoded JPEG string
        """
        # Resize if needed (Anthropic recommends max 1568px on any side)
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size)

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Encode to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    def _parse_response(self, response: str) -> tuple[bool, Optional[str]]:
        """
        Parse LLM response to extract binary classification.

        Returns:
            (is_sufficient: bool, explanation: Optional[str])
        """
        explanation = None
        is_sufficient = False

        # Try to extract explanation
        explanation_match = re.search(
            r"###\s*EXPLANATION\s*\n(.*?)(?=###|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if explanation_match:
            explanation = explanation_match.group(1).strip()

        # Try to extract JSON with "Sufficient Context" key
        # Handle various output formats from the LLM
        json_patterns = [
            # Standard JSON formats
            r'\{\s*"Sufficient Context"\s*:\s*(\d)\s*\}',
            r'\{\s*"Sufficient_Context"\s*:\s*(\d)\s*\}',
            r'\{\s*"sufficient_context"\s*:\s*(\d)\s*\}',
            # JSON with true/false
            r'\{\s*"Sufficient Context"\s*:\s*(true|false)\s*\}',
            r'\{\s*"Sufficient_Context"\s*:\s*(true|false)\s*\}',
            # After ### EVALUATION or ### JSON header
            r"###\s*(?:EVALUATION|JSON)\s*(?:\(JSON\))?\s*\n\s*\{[^}]*[\"']?Sufficient[_ ]?Context[\"']?\s*:\s*(\d|true|false)",
            # Inline patterns
            r'"Sufficient Context"\s*:\s*(\d|true|false)',
            r'"Sufficient_Context"\s*:\s*(\d|true|false)',
            r"Sufficient Context:\s*(\d|true|false)",
            # Simpler patterns - just look for the value after Sufficient Context
            r"Sufficient\s*Context[\"']?\s*:\s*[\"']?(\d|true|false)",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value_str = match.group(1).lower()
                # Handle both numeric (0/1) and boolean (true/false)
                if value_str in ("1", "true"):
                    is_sufficient = True
                elif value_str in ("0", "false"):
                    is_sufficient = False
                else:
                    continue  # Try next pattern
                return is_sufficient, explanation

        # Fallback: look for explicit keywords
        lower_response = response.lower()
        if "insufficient" in lower_response or "not sufficient" in lower_response:
            is_sufficient = False
        elif "sufficient" in lower_response:
            is_sufficient = True

        # Last resort: try to parse any JSON
        try:
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                parsed = json.loads(json_match.group())
                for key in ["Sufficient Context", "Sufficient_Context", "sufficient_context", "sufficient"]:
                    if key in parsed:
                        is_sufficient = bool(int(parsed[key]))
                        return is_sufficient, explanation
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        logger.warning("Could not parse SCA response, defaulting to insufficient")
        return is_sufficient, explanation


def estimate_cost(
    n_queries: int,
    model: str = "claude-haiku-4-5",
    avg_input_tokens: int = 1500,
    avg_output_tokens: int = 200,
    vision: bool = False,
    images_per_query: int = 10,
) -> Dict[str, float]:
    """
    Estimate LLM API cost for assessment.

    Args:
        n_queries: Number of queries to assess
        model: Model name
        avg_input_tokens: Average input tokens per query (text only)
        avg_output_tokens: Average output tokens per query
        vision: Whether using vision (images) instead of text
        images_per_query: Number of images per query (if vision=True)

    Returns:
        Dict with input_cost, output_cost, total_cost
    """
    # Pricing per 1M tokens (as of late 2024)
    pricing = {
        "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
    }

    model_pricing = pricing.get(model, pricing["claude-haiku-4-5"])

    # For vision: ~1000 tokens per image (Anthropic estimates 1,000-1,600 for ~1000x750px)
    if vision:
        tokens_per_image = 1000
        avg_input_tokens = (images_per_query * tokens_per_image) + 500  # 500 for prompt text

    input_cost = (n_queries * avg_input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (n_queries * avg_output_tokens / 1_000_000) * model_pricing["output"]

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "cost_per_query": (input_cost + output_cost) / max(n_queries, 1),
        "vision": vision,
    }


if __name__ == "__main__":
    # Quick test (requires API key)
    print("LLM Sufficiency Assessor")
    print("=" * 50)

    # Estimate costs
    for n in [10, 100, 1000]:
        for model in ["claude-haiku-4-5", "claude-sonnet-4"]:
            cost = estimate_cost(n, model)
            print(f"{model} ({n} queries): ${cost['total_cost']:.4f} total, "
                  f"${cost['cost_per_query']:.6f}/query")

    print("\nTo run actual assessment, set ANTHROPIC_API_KEY environment variable.")
