#!/usr/bin/env python3
"""
TextGrad Prompt Optimization for SUJBOT Multi-Agent System.

Optimizes 8 agent prompts using TextGrad's automatic "differentiation" via text.
Uses LLM-as-judge evaluation with 3 metrics: semantic_correctness, factual_accuracy, completeness.

Usage:
    # Full optimization (20 iterations)
    uv run python scripts/textgrad_optimize.py --iterations 20

    # Quick test (1 iteration, dry-run)
    uv run python scripts/textgrad_optimize.py --iterations 1 --dry-run

    # Resume from checkpoint
    uv run python scripts/textgrad_optimize.py --resume-from prompts/agents/versions/checkpoints/checkpoint_10.json

    # Optimize specific agents only
    uv run python scripts/textgrad_optimize.py --agents orchestrator,extractor --iterations 10
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
import textgrad as tg
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Apply nest_asyncio early for Neo4j/graphiti compatibility
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import optimization components
from src.prompt_optimization.variables import PromptVariableManager, AGENT_NAMES
from src.prompt_optimization.loss import MultiMetricLoss
from src.prompt_optimization.credit_assignment import CreditAssigner
from src.prompt_optimization.versioning import PromptVersionManager

# Paths
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "agents"
DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset_exp_ver_2.json"
CONFIG_PATH = PROJECT_ROOT / "config.json"


# =============================================================================
# Structure Preservation Constraint
# =============================================================================

STRUCTURE_CONSTRAINT = """
When suggesting improvements for this agent prompt, you MUST PRESERVE:
1. Section headers (##, ===, ---)
2. Code block formatting (```)
3. Tool lists (AVAILABLE TOOLS:, AVAILABLE AGENTS:)
4. Example blocks (EXAMPLE 1:, Tool #1:, etc.)
5. Reflection format (**REFLECTION [...]:**)
6. Citation format instructions (\\cite{chunk_id})
7. Czech language support (prompts must work with Czech queries)
8. JSON output format specifications

ONLY modify INSTRUCTIONAL CONTENT to improve agent behavior.
DO NOT remove or restructure existing sections.
"""


class TextGradPromptOptimizer:
    """
    Main optimizer class for TextGrad prompt optimization.

    Coordinates:
    - Loading prompts as TextGrad Variables
    - Running forward pass through multi-agent system
    - Computing loss using LLM-as-judge
    - Credit assignment for multi-agent blame
    - Backward pass with textual gradients
    - Optimizer step to update prompts
    """

    def __init__(
        self,
        forward_model: str = "MiniMaxAI/MiniMax-M2",
        backward_engine: str = "claude-sonnet-4-5-20250929",
        judge_model: str = "claude-sonnet-4-5-20250929",
        agents: Optional[List[str]] = None,
        dry_run: bool = False,
    ):
        """
        Initialize the optimizer.

        Args:
            forward_model: Model for multi-agent execution
            backward_engine: Model for TextGrad gradient generation
            judge_model: Model for LLM-as-judge evaluation
            agents: Optional list of agents to optimize (defaults to all 8)
            dry_run: If True, don't save optimized prompts
        """
        self.forward_model = forward_model
        self.backward_engine = backward_engine
        self.judge_model = judge_model
        self.agents = agents or AGENT_NAMES
        self.dry_run = dry_run

        # Load config
        self.config = json.loads(CONFIG_PATH.read_text())

        # Initialize components
        self.variable_manager = PromptVariableManager(PROMPTS_DIR, agents=self.agents)
        self.loss_fn = MultiMetricLoss(judge_model=judge_model)
        self.credit_assigner = CreditAssigner(all_agents=self.agents)
        self.version_manager = PromptVersionManager(PROMPTS_DIR)

        # Load dataset
        self.dataset = json.loads(DATASET_PATH.read_text())
        logger.info(f"Loaded dataset with {len(self.dataset)} examples")

        # Multi-agent runner (initialized async)
        self.runner = None

        # Metrics tracking
        self.metrics_history: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize async components."""
        from src.multi_agent.runner import MultiAgentRunner
        from src.multi_agent.prompts.loader import get_prompt_loader

        # Initialize multi-agent runner
        self.runner = MultiAgentRunner(self.config)
        await self.runner.initialize()
        logger.info("MultiAgentRunner initialized for optimization")

        # Get prompt loader for injection
        self.prompt_loader = get_prompt_loader()

        # Setup TextGrad backward engine
        # Use litellm experimental engine for Anthropic
        logger.info(f"Setting backward engine: {self.backward_engine}")
        try:
            tg.set_backward_engine(
                tg.get_engine(f"experimental:anthropic/{self.backward_engine}"),
                override=True
            )
        except anthropic.AuthenticationError as e:
            logger.error(f"Anthropic authentication failed: {e}")
            raise RuntimeError(
                "ANTHROPIC_API_KEY is invalid or missing. "
                "Please set a valid API key in .env file."
            ) from e
        except Exception as e:
            # Log warning but try fallback - user should be aware
            logger.warning(
                f"Failed to set Anthropic backward engine ({e}). "
                f"Falling back to gpt-4o - this will use OpenAI API and may incur different costs."
            )
            try:
                tg.set_backward_engine("gpt-4o", override=True)
                logger.info("Successfully set fallback engine: gpt-4o")
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to set both primary ({self.backward_engine}) and fallback (gpt-4o) engines. "
                    f"Primary error: {e}, Fallback error: {fallback_error}"
                ) from fallback_error

    async def run_forward_pass(
        self,
        question: str,
        prompt_variables: Dict[str, tg.Variable],
    ) -> Dict[str, Any]:
        """
        Run multi-agent system with current prompt variables.

        Args:
            question: User question
            prompt_variables: Current prompt variables

        Returns:
            Dict with answer, agent_sequence, etc.
        """
        # Inject current prompts into loader
        self.variable_manager.inject_into_loader(self.prompt_loader)

        # Run query through multi-agent system
        result = {
            "answer": "",
            "agent_sequence": [],
            "agent_outputs": {},
            "success": False,
            "errors": [],
        }

        try:
            async for event in self.runner.run_query(question):
                if event.get("type") == "final":
                    result["answer"] = event.get("final_answer", "")
                    result["agent_sequence"] = event.get("agent_sequence", [])
                    result["agent_outputs"] = event.get("agent_outputs", {})
                    result["success"] = True
                    break

        except KeyboardInterrupt:
            logger.info("Forward pass interrupted by user")
            raise  # Re-raise to allow clean shutdown
        except anthropic.AuthenticationError as e:
            logger.error(f"Anthropic authentication error: {e}")
            raise RuntimeError("API authentication failed - check ANTHROPIC_API_KEY") from e
        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            result["errors"].append(f"Rate limit: {e}")
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            result["errors"].append(f"API error: {e}")
        except asyncio.TimeoutError as e:
            logger.error(f"Forward pass timeout: {e}")
            result["errors"].append(f"Timeout: {e}")
        except Exception as e:
            logger.error(f"Forward pass error: {type(e).__name__}: {e}", exc_info=True)
            result["errors"].append(str(e))

        return result

    async def optimize(
        self,
        num_iterations: int = 20,
        batch_size: int = 5,
        save_every: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Main optimization loop.

        Args:
            num_iterations: Number of optimization iterations
            batch_size: Examples per iteration
            save_every: Save checkpoint every N iterations

        Returns:
            Metrics history
        """
        # Load prompts as variables
        prompt_variables = self.variable_manager.load_all_prompts()
        logger.info(f"Loaded {len(prompt_variables)} prompt variables")

        # Backup originals
        if not self.dry_run:
            self.version_manager.backup_originals(list(prompt_variables.keys()))

        # Create TGD optimizer
        optimizer = tg.TGD(parameters=list(prompt_variables.values()))

        # Track total examples
        total_examples = 0

        for iteration in range(num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*60}")

            # Sample batch from dataset
            batch = random.sample(self.dataset, min(batch_size, len(self.dataset)))
            total_examples += len(batch)

            iteration_scores = {
                "semantic_correctness": [],
                "factual_accuracy": [],
                "completeness": [],
            }
            iteration_losses = []

            for i, example in enumerate(batch):
                question = example["question"]
                reference = example["answer"]

                logger.info(f"  Example {i+1}/{len(batch)}: {question[:80]}...")

                # Forward pass
                result = await self.run_forward_pass(question, prompt_variables)
                predicted = result["answer"]

                logger.info(f"    Answer: {predicted[:100]}...")

                # Compute loss
                loss = self.loss_fn.compute_loss(
                    question=question,
                    predicted=predicted,
                    reference=reference,
                    agent_trace={
                        "agent_sequence": result["agent_sequence"],
                        "agent_outputs": result.get("agent_outputs", {}),
                    },
                )

                iteration_losses.append(loss)

                # Track scores
                scores = self.loss_fn.get_last_scores()
                for metric in iteration_scores:
                    iteration_scores[metric].append(scores.get(metric, 0))

                logger.info(
                    f"    Scores: sem={scores.get('semantic_correctness', 0)}, "
                    f"fact={scores.get('factual_accuracy', 0)}, "
                    f"comp={scores.get('completeness', 0)}"
                )

            # Compute average scores
            avg_scores = {
                metric: sum(vals) / len(vals) if vals else 0
                for metric, vals in iteration_scores.items()
            }

            # Combined weighted score
            combined_score = sum(
                avg_scores[m] * self.loss_fn.METRIC_WEIGHTS[m]
                for m in avg_scores
            )

            logger.info(f"\nIteration {iteration + 1} Average Scores:")
            logger.info(f"  semantic_correctness: {avg_scores['semantic_correctness']:.3f}")
            logger.info(f"  factual_accuracy: {avg_scores['factual_accuracy']:.3f}")
            logger.info(f"  completeness: {avg_scores['completeness']:.3f}")
            logger.info(f"  Combined (weighted): {combined_score:.3f}")

            # Credit assignment
            failed_metrics = [m for m, s in avg_scores.items() if s < 0.6]
            blame = self.credit_assigner.assign_credit(
                failed_metrics=failed_metrics,
                agent_sequence=list(prompt_variables.keys()),
            )
            logger.info(f"  Credit assignment: {blame.reasoning}")

            # Backward pass with aggregated loss
            if iteration_losses:
                # Create aggregated loss text
                aggregated_feedback = self._aggregate_losses(iteration_losses, blame)

                # Create loss variable for backward
                loss_var = tg.Variable(
                    aggregated_feedback,
                    requires_grad=False,
                    role_description="Aggregated evaluation feedback for prompt optimization",
                )

                # Backward pass
                logger.info("  Running backward pass...")
                try:
                    loss_var.backward()
                except anthropic.AuthenticationError as e:
                    logger.error(f"  Backward pass authentication error: {e}")
                    raise RuntimeError(
                        "API authentication failed during backward pass - check ANTHROPIC_API_KEY"
                    ) from e
                except anthropic.RateLimitError as e:
                    logger.warning(f"  Backward pass rate limited: {e}. Skipping optimizer step.")
                    # Skip optimizer step for this iteration
                    self.metrics_history.append({
                        "iteration": iteration + 1,
                        "scores": avg_scores,
                        "combined_score": combined_score,
                        "failed_metrics": failed_metrics,
                        "blame": blame.agent_weights,
                        "timestamp": datetime.now().isoformat(),
                        "backward_pass_failed": True,
                    })
                    continue  # Skip to next iteration
                except anthropic.APIError as e:
                    logger.error(f"  Backward pass API error: {e}")
                    # Non-recoverable API error, skip this iteration
                    continue
                except Exception as e:
                    logger.warning(
                        f"  Backward pass error ({type(e).__name__}: {e}). "
                        f"Skipping optimizer step for this iteration."
                    )

                # Scale gradients by credit assignment
                self._scale_gradients_by_blame(prompt_variables, blame)

                # Optimizer step
                logger.info("  Applying optimizer step...")
                optimizer.step()

            # Track metrics
            self.metrics_history.append({
                "iteration": iteration + 1,
                "scores": avg_scores,
                "combined_score": combined_score,
                "failed_metrics": failed_metrics,
                "blame": blame.agent_weights,
                "timestamp": datetime.now().isoformat(),
            })

            # Save checkpoint
            if not self.dry_run and (iteration + 1) % save_every == 0:
                logger.info(f"  Saving checkpoint at iteration {iteration + 1}...")
                self.version_manager.save_checkpoint(
                    iteration=iteration + 1,
                    prompt_variables=prompt_variables,
                    metrics_history=self.metrics_history,
                    total_examples=total_examples,
                )

                # Save versions for each agent
                for agent_name, var in prompt_variables.items():
                    self.version_manager.save_version(
                        agent_name=agent_name,
                        prompt_text=var.value,
                        iteration=iteration + 1,
                        metrics=avg_scores,
                    )

        # Save final results
        if not self.dry_run:
            logger.info("\nSaving final results...")
            results_dir = self.version_manager.save_final_results(
                prompt_variables=prompt_variables,
                metrics_history=self.metrics_history,
                original_prompts=self.variable_manager._original_prompts,
            )
            logger.info(f"Results saved to: {results_dir}")

            # Update active prompts
            for agent_name, var in prompt_variables.items():
                self.version_manager.save_as_active(agent_name, var.value)

        return self.metrics_history

    def _aggregate_losses(
        self,
        losses: List[tg.Variable],
        blame: Any,
    ) -> str:
        """Aggregate multiple loss feedbacks into one."""
        parts = [
            "AGGREGATED EVALUATION FEEDBACK",
            f"Examples evaluated: {len(losses)}",
            "",
            STRUCTURE_CONSTRAINT,
            "",
            "=" * 50,
            "",
        ]

        # Add individual loss feedbacks
        for i, loss in enumerate(losses):
            parts.append(f"--- Example {i+1} ---")
            parts.append(loss.value[:1000])  # Truncate if too long
            parts.append("")

        # Add credit assignment info
        parts.append("=" * 50)
        parts.append("CREDIT ASSIGNMENT:")
        parts.append(blame.reasoning)
        parts.append("")

        # Add priority guidance
        high_blame = [
            (agent, weight)
            for agent, weight in blame.agent_weights.items()
            if weight > 0.2
        ]
        if high_blame:
            parts.append("FOCUS IMPROVEMENTS ON:")
            for agent, weight in sorted(high_blame, key=lambda x: -x[1]):
                parts.append(f"  - {agent} (blame={weight:.2f})")

        return "\n".join(parts)

    def _scale_gradients_by_blame(
        self,
        prompt_variables: Dict[str, tg.Variable],
        blame: Any,
    ) -> None:
        """Scale gradients based on credit assignment."""
        scaling_prefixes = self.credit_assigner.get_gradient_scaling(blame)

        for agent_name, var in prompt_variables.items():
            if hasattr(var, 'grad') and var.grad is not None:
                prefix = scaling_prefixes.get(agent_name, "")
                if prefix:
                    var.grad = prefix + "\n\n" + str(var.grad)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TextGrad Prompt Optimization for SUJBOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of optimization iterations (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Examples per iteration (default: 5)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N iterations (default: 5)",
    )
    parser.add_argument(
        "--backward-engine",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model for backward pass (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model for LLM-as-judge (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated list of agents to optimize (default: all)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save optimized prompts",
    )

    args = parser.parse_args()

    # Parse agents
    agents = None
    if args.agents:
        agents = [a.strip() for a in args.agents.split(",")]

    # Create optimizer
    optimizer = TextGradPromptOptimizer(
        backward_engine=args.backward_engine,
        judge_model=args.judge_model,
        agents=agents,
        dry_run=args.dry_run,
    )

    # Initialize
    logger.info("Initializing TextGrad optimizer...")
    await optimizer.initialize()

    # Resume from checkpoint if specified
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if checkpoint_path.exists():
            checkpoint = optimizer.version_manager.load_checkpoint(checkpoint_path)
            optimizer.metrics_history = checkpoint.metrics_history
            # Restore prompt values from checkpoint
            if checkpoint.prompt_values:
                optimizer.variable_manager.restore_from_checkpoint(checkpoint.prompt_values)
                logger.info(f"Restored {len(checkpoint.prompt_values)} prompt values from checkpoint")
            logger.info(f"Resumed from checkpoint at iteration {checkpoint.iteration}")
        else:
            logger.warning(f"Checkpoint file not found: {checkpoint_path}, starting fresh")

    # Run optimization
    logger.info(f"Starting optimization: {args.iterations} iterations, batch_size={args.batch_size}")
    metrics_history = await optimizer.optimize(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )

    # Print final summary
    if metrics_history:
        first = metrics_history[0]
        last = metrics_history[-1]

        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)

        logger.info("\nMetrics Improvement:")
        for metric in ["semantic_correctness", "factual_accuracy", "completeness"]:
            before = first["scores"].get(metric, 0)
            after = last["scores"].get(metric, 0)
            change = after - before
            logger.info(f"  {metric}: {before:.3f} -> {after:.3f} ({change:+.3f})")

        logger.info(f"\nCombined score: {first['combined_score']:.3f} -> {last['combined_score']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
