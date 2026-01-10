"""
Multi-Metric Loss Function for TextGrad optimization.

Combines 3 LLM-as-judge evaluation metrics:
- semantic_correctness (0.40 weight)
- factual_accuracy (0.35 weight)
- completeness (0.25 weight)

Generates textual feedback for TextGrad backward pass.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
import textgrad as tg

logger = logging.getLogger(__name__)

# Retry configuration for API calls
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0

# Prompts directory (SSOT: prompts/evaluation/)
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "evaluation"


def _load_evaluation_prompt(metric_name: str) -> str:
    """Load evaluation prompt from prompts/evaluation/ directory."""
    prompt_file = PROMPTS_DIR / f"{metric_name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Evaluation prompt not found: {prompt_file}. "
            f"Please ensure prompts/evaluation/{metric_name}.txt exists."
        )
    return prompt_file.read_text(encoding="utf-8")


@dataclass
class EvaluationResult:
    """Result from a single metric evaluation."""

    metric: str
    score: int  # 0 or 1, -1 indicates error
    rationale: str
    raw_response: Optional[str] = None
    is_error: bool = False  # True if evaluation failed (score is not meaningful)


@dataclass
class LossResult:
    """Complete loss result with all metrics."""

    weighted_score: float
    scores: Dict[str, int] = field(default_factory=dict)
    rationales: Dict[str, str] = field(default_factory=dict)
    feedback_text: str = ""
    evaluations: List[EvaluationResult] = field(default_factory=list)


class MultiMetricLoss:
    """
    Multi-metric loss function for TextGrad optimization.

    Evaluates predictions using 3 metrics and generates textual feedback
    for backward pass.
    """

    # Metric weights (sum to 1.0)
    METRIC_WEIGHTS = {
        "semantic_correctness": 0.40,
        "factual_accuracy": 0.35,
        "completeness": 0.25,
    }

    def __init__(
        self,
        judge_model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.0,
    ):
        """
        Initialize the loss function.

        Args:
            judge_model: Anthropic model for evaluation (judge)
            temperature: Temperature for evaluation calls
        """
        self.judge_model = judge_model
        self.temperature = temperature
        self.client = anthropic.Anthropic()
        self.last_result: Optional[LossResult] = None

        # Load evaluation prompts from prompts/evaluation/ (SSOT)
        self._prompt_cache: Dict[str, str] = {}
        for metric in self.METRIC_WEIGHTS:
            self._prompt_cache[metric] = _load_evaluation_prompt(metric)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON object from text that may have extra content."""
        import re

        # First try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        json_pattern = r'\{[^{}]*"rationale"[^{}]*"score"[^{}]*\}|\{[^{}]*"score"[^{}]*"rationale"[^{}]*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try finding any JSON-like structure
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found", text, 0)

    def _extract_score_regex(self, text: str) -> tuple:
        """Extract score using regex as fallback."""
        import re

        # Look for score patterns
        score_patterns = [
            r'"score"\s*:\s*(\d+)',
            r'score\s*[:=]\s*(\d+)',
            r'\bscore\b.*?(\d+)',
        ]

        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if score in (0, 1):
                    # Try to extract rationale
                    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
                    rationale = rationale_match.group(1) if rationale_match else None
                    return score, rationale

        return None, None

    def _evaluate_metric(
        self,
        metric: str,
        question: str,
        predicted: str,
        reference: str,
    ) -> EvaluationResult:
        """
        Evaluate a single metric using LLM-as-judge.

        Args:
            metric: Metric name (semantic_correctness, factual_accuracy, completeness)
            question: Original question
            predicted: System's answer
            reference: Reference (ground truth) answer

        Returns:
            EvaluationResult with score and rationale
        """
        prompt_template = self._prompt_cache.get(metric)
        if not prompt_template:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(self._prompt_cache.keys())}")

        prompt = prompt_template.format(
            question=question,
            predicted=predicted,
            reference=reference,
        )

        response_text = ""
        last_error = None

        # Retry loop for transient API errors
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.judge_model,
                    max_tokens=256,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )

                response_text = response.content[0].text.strip()

                # Try to extract JSON from response (may have extra text before/after)
                result = self._extract_json(response_text)
                score = int(result.get("score", 0))
                rationale = result.get("rationale", "No rationale provided")

                return EvaluationResult(
                    metric=metric,
                    score=score,
                    rationale=rationale,
                    raw_response=response_text,
                )

            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Rate limited on {metric} evaluation (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                logger.error(f"Rate limit exceeded for {metric} evaluation after {MAX_RETRIES} retries")

            except anthropic.APIConnectionError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                    logger.warning(
                        f"API connection error on {metric} (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                logger.error(f"API connection failed for {metric} after {MAX_RETRIES} retries: {e}")

            except anthropic.AuthenticationError as e:
                # Don't retry authentication errors - they won't resolve
                logger.error(f"Authentication error evaluating {metric}: {e}")
                return EvaluationResult(
                    metric=metric,
                    score=-1,
                    rationale=f"Authentication error: API key invalid or missing",
                    is_error=True,
                )

            except anthropic.APIError as e:
                # Other API errors - log and fail
                logger.error(f"API error evaluating {metric}: {e}")
                return EvaluationResult(
                    metric=metric,
                    score=-1,
                    rationale=f"API error: {e}",
                    is_error=True,
                )

            except json.JSONDecodeError as e:
                # Try regex extraction as fallback
                score, rationale = self._extract_score_regex(response_text)
                if score is not None:
                    return EvaluationResult(
                        metric=metric,
                        score=score,
                        rationale=rationale or "Extracted via regex",
                        raw_response=response_text,
                    )
                logger.warning(f"Failed to parse {metric} evaluation: {e}")
                return EvaluationResult(
                    metric=metric,
                    score=-1,
                    rationale=f"Evaluation parse error: {e}",
                    raw_response=response_text,
                    is_error=True,
                )

        # All retries exhausted
        return EvaluationResult(
            metric=metric,
            score=-1,
            rationale=f"Evaluation failed after {MAX_RETRIES} retries: {last_error}",
            is_error=True,
        )

    def compute_loss(
        self,
        question: str,
        predicted: str,
        reference: str,
        agent_trace: Optional[Dict[str, Any]] = None,
    ) -> tg.Variable:
        """
        Compute loss as TextGrad Variable for backpropagation.

        Args:
            question: Original question
            predicted: System's predicted answer
            reference: Reference (ground truth) answer
            agent_trace: Optional dict with agent_sequence and agent_outputs

        Returns:
            TextGrad Variable containing loss feedback text
        """
        evaluations = []
        scores = {}
        rationales = {}
        has_errors = False

        # Evaluate each metric
        for metric in self.METRIC_WEIGHTS:
            result = self._evaluate_metric(metric, question, predicted, reference)
            evaluations.append(result)
            if result.is_error:
                has_errors = True
                scores[metric] = 0  # Treat errors as 0 for scoring but flag it
                logger.warning(f"Evaluation error for {metric}: {result.rationale}")
            else:
                scores[metric] = result.score
            rationales[metric] = result.rationale

        # Compute weighted score (0.0 to 1.0, higher is better)
        # If any metric had errors, weighted_score may be unreliable
        weighted_score = sum(
            scores[m] * w for m, w in self.METRIC_WEIGHTS.items()
        )
        if has_errors:
            logger.warning(
                f"Some metrics failed to evaluate - weighted_score {weighted_score:.2f} may be unreliable"
            )

        # Generate feedback text for TextGrad
        feedback_parts = []

        # Overall status
        if weighted_score >= 0.8:
            feedback_parts.append("OVERALL: Answer is GOOD. Minor improvements possible.")
        elif weighted_score >= 0.5:
            feedback_parts.append("OVERALL: Answer is PARTIAL. Significant improvements needed.")
        else:
            feedback_parts.append("OVERALL: Answer FAILED. Major improvements required.")

        feedback_parts.append(f"\nWeighted Score: {weighted_score:.2f}")
        feedback_parts.append(f"Question: {question[:200]}...")

        # Per-metric feedback
        for metric, weight in self.METRIC_WEIGHTS.items():
            score = scores[metric]
            rationale = rationales[metric]
            status = "PASS" if score == 1 else "FAIL"

            feedback_parts.append(
                f"\n[{metric.upper()}] {status} (weight={weight:.2f}): {rationale}"
            )

        # Agent trace info (for credit assignment)
        if agent_trace:
            agent_sequence = agent_trace.get("agent_sequence", [])
            if agent_sequence:
                feedback_parts.append(f"\nAgents executed: {', '.join(agent_sequence)}")

        # Improvement suggestions based on failures
        failed_metrics = [m for m, s in scores.items() if s == 0]
        if failed_metrics:
            feedback_parts.append("\n\nIMPROVEMENT SUGGESTIONS:")
            for metric in failed_metrics:
                if metric == "semantic_correctness":
                    feedback_parts.append(
                        "- Improve routing logic to select correct agents for query type"
                    )
                    feedback_parts.append(
                        "- Enhance synthesis to capture the main point of the question"
                    )
                elif metric == "factual_accuracy":
                    feedback_parts.append(
                        "- Improve retrieval to find exact values and specifications"
                    )
                    feedback_parts.append(
                        "- Add explicit instructions to verify numbers before answering"
                    )
                elif metric == "completeness":
                    feedback_parts.append(
                        "- Ensure all relevant chunks are retrieved"
                    )
                    feedback_parts.append(
                        "- Add instructions to cover all aspects of multi-part questions"
                    )

        feedback_text = "\n".join(feedback_parts)

        # Store result for later access
        self.last_result = LossResult(
            weighted_score=weighted_score,
            scores=scores,
            rationales=rationales,
            feedback_text=feedback_text,
            evaluations=evaluations,
        )

        # Create TextGrad Variable for backward pass
        # Note: Lower score = higher loss, so we invert for gradient direction
        # TextGrad will use this feedback to suggest improvements
        return tg.Variable(
            feedback_text,
            requires_grad=False,
            role_description="Evaluation feedback from LLM-as-judge for prompt optimization",
        )

    def get_last_scores(self) -> Dict[str, int]:
        """Get scores from last evaluation."""
        return self.last_result.scores if self.last_result else {}

    def get_last_weighted_score(self) -> float:
        """Get weighted score from last evaluation."""
        return self.last_result.weighted_score if self.last_result else 0.0
