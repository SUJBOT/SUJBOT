"""
Prompt Versioning for TextGrad Optimization.

Manages versioning, checkpoints, and rollback for optimized prompts.
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import textgrad as tg

logger = logging.getLogger(__name__)


@dataclass
class VersionMetadata:
    """Metadata for a prompt version."""

    agent_name: str
    iteration: int
    timestamp: str
    metrics: Dict[str, float]
    prompt_length: int
    diff_from_original: int


@dataclass
class CheckpointData:
    """Data stored in a checkpoint."""

    iteration: int
    timestamp: str
    metrics_history: List[Dict[str, Any]]
    prompt_values: Dict[str, str]
    total_examples_processed: int


class PromptVersionManager:
    """
    Manages versioning of optimized prompts.

    Directory structure:
    prompts/
    └── agents/
        ├── orchestrator.txt           # Current active prompt
        └── versions/
            ├── orchestrator_v0.txt    # Original (before optimization)
            ├── orchestrator_v5.txt    # After iteration 5
            ├── orchestrator_v10.txt   # After iteration 10
            └── checkpoints/
                ├── checkpoint_5.json
                └── checkpoint_10.json
    """

    def __init__(self, prompts_dir: Path, output_dir: Optional[Path] = None):
        """
        Initialize the version manager.

        Args:
            prompts_dir: Path to prompts/agents/ directory
            output_dir: Optional output directory for optimization results
        """
        self.prompts_dir = Path(prompts_dir)
        self.versions_dir = self.prompts_dir / "versions"
        self.checkpoints_dir = self.versions_dir / "checkpoints"
        self.output_dir = output_dir or Path("output/textgrad_optimization")

        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track metadata
        self._version_metadata: Dict[str, List[VersionMetadata]] = {}

    def backup_originals(self, agent_names: List[str]) -> None:
        """
        Backup original prompts before optimization.

        Args:
            agent_names: List of agent names to backup
        """
        for agent_name in agent_names:
            original_file = self.prompts_dir / f"{agent_name}.txt"
            backup_file = self.versions_dir / f"{agent_name}_v0.txt"

            if original_file.exists() and not backup_file.exists():
                content = original_file.read_text(encoding="utf-8")

                # Add metadata header
                header = self._create_header(
                    agent_name=agent_name,
                    iteration=0,
                    metrics={"original": True},
                    note="Original prompt before TextGrad optimization",
                )

                backup_file.write_text(header + content, encoding="utf-8")
                logger.info(f"Backed up original: {backup_file.name}")

    def save_version(
        self,
        agent_name: str,
        prompt_text: str,
        iteration: int,
        metrics: Dict[str, float],
    ) -> Path:
        """
        Save a versioned prompt with metadata.

        Args:
            agent_name: Name of the agent
            prompt_text: Current prompt text
            iteration: Optimization iteration number
            metrics: Current evaluation metrics

        Returns:
            Path to saved version file
        """
        version_file = self.versions_dir / f"{agent_name}_v{iteration}.txt"

        # Create header with metadata
        header = self._create_header(
            agent_name=agent_name,
            iteration=iteration,
            metrics=metrics,
        )

        version_file.write_text(header + prompt_text, encoding="utf-8")

        # Track metadata
        if agent_name not in self._version_metadata:
            self._version_metadata[agent_name] = []

        self._version_metadata[agent_name].append(
            VersionMetadata(
                agent_name=agent_name,
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
                prompt_length=len(prompt_text),
                diff_from_original=0,  # Would need original to compute
            )
        )

        logger.info(f"Saved version: {version_file.name}")
        return version_file

    def save_as_active(self, agent_name: str, prompt_text: str) -> Path:
        """
        Save prompt as the active version.

        Args:
            agent_name: Name of the agent
            prompt_text: Optimized prompt text

        Returns:
            Path to active prompt file
        """
        active_file = self.prompts_dir / f"{agent_name}.txt"
        active_file.write_text(prompt_text, encoding="utf-8")
        logger.info(f"Updated active prompt: {active_file.name}")
        return active_file

    def save_checkpoint(
        self,
        iteration: int,
        prompt_variables: Dict[str, tg.Variable],
        metrics_history: List[Dict[str, Any]],
        total_examples: int,
    ) -> Path:
        """
        Save a full checkpoint.

        Args:
            iteration: Current iteration number
            prompt_variables: Dict of agent_name -> TextGrad Variable
            metrics_history: List of metrics from all iterations
            total_examples: Total examples processed

        Returns:
            Path to checkpoint file
        """
        checkpoint_file = self.checkpoints_dir / f"checkpoint_{iteration}.json"

        # Extract prompt values
        prompt_values = {
            name: var.value for name, var in prompt_variables.items()
        }

        checkpoint = CheckpointData(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            metrics_history=metrics_history,
            prompt_values=prompt_values,
            total_examples_processed=total_examples,
        )

        checkpoint_file.write_text(
            json.dumps(asdict(checkpoint), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(f"Saved checkpoint: {checkpoint_file.name}")
        return checkpoint_file

    def load_checkpoint(self, checkpoint_path: Path) -> CheckpointData:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            CheckpointData with all checkpoint information
        """
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return CheckpointData(**data)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint file."""
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None

        # Sort by iteration number
        def get_iter(p: Path) -> int:
            try:
                return int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                return 0

        return max(checkpoints, key=get_iter)

    def rollback(self, agent_name: str, version: int) -> str:
        """
        Rollback to a specific version.

        Args:
            agent_name: Name of the agent
            version: Version number to rollback to

        Returns:
            The restored prompt text
        """
        version_file = self.versions_dir / f"{agent_name}_v{version}.txt"
        if not version_file.exists():
            raise FileNotFoundError(f"Version {version} not found for {agent_name}")

        # Read version (skip header)
        content = version_file.read_text(encoding="utf-8")
        prompt_text = self._strip_header(content)

        # Save as active
        self.save_as_active(agent_name, prompt_text)

        logger.info(f"Rolled back {agent_name} to version {version}")
        return prompt_text

    def save_final_results(
        self,
        prompt_variables: Dict[str, tg.Variable],
        metrics_history: List[Dict[str, Any]],
        original_prompts: Dict[str, str],
    ) -> Path:
        """
        Save final optimization results.

        Args:
            prompt_variables: Final optimized prompts
            metrics_history: Complete metrics history
            original_prompts: Original prompts for comparison

        Returns:
            Path to results directory
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = self.output_dir / f"run_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save optimized prompts
        prompts_dir = results_dir / "optimized_prompts"
        prompts_dir.mkdir(exist_ok=True)

        for agent_name, var in prompt_variables.items():
            prompt_file = prompts_dir / f"{agent_name}.txt"
            prompt_file.write_text(var.value, encoding="utf-8")

        # Save metrics history
        metrics_file = results_dir / "metrics_history.json"
        metrics_file.write_text(
            json.dumps(metrics_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Generate summary report
        report = self._generate_summary_report(
            prompt_variables, metrics_history, original_prompts
        )
        report_file = results_dir / "summary_report.md"
        report_file.write_text(report, encoding="utf-8")

        logger.info(f"Saved final results to: {results_dir}")
        return results_dir

    def _create_header(
        self,
        agent_name: str,
        iteration: int,
        metrics: Dict[str, Any],
        note: str = "",
    ) -> str:
        """Create metadata header for versioned file."""
        lines = [
            "# TextGrad Optimized Prompt",
            f"# Agent: {agent_name}",
            f"# Iteration: {iteration}",
            f"# Timestamp: {datetime.now().isoformat()}",
            f"# Metrics: {json.dumps(metrics)}",
        ]
        if note:
            lines.append(f"# Note: {note}")
        lines.append("# " + "=" * 50)
        lines.append("")
        return "\n".join(lines)

    def _strip_header(self, content: str) -> str:
        """Strip metadata header from versioned file."""
        lines = content.split("\n")
        # Find the separator line
        for i, line in enumerate(lines):
            if line.startswith("# " + "="):
                return "\n".join(lines[i + 2:])  # Skip separator and blank line
        return content  # No header found

    def _generate_summary_report(
        self,
        prompt_variables: Dict[str, tg.Variable],
        metrics_history: List[Dict[str, Any]],
        original_prompts: Dict[str, str],
    ) -> str:
        """Generate markdown summary report."""
        lines = [
            "# TextGrad Optimization Summary",
            "",
            f"**Run Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Iterations**: {len(metrics_history)}",
            f"**Agents Optimized**: {len(prompt_variables)}",
            "",
        ]

        # Metrics improvement
        if metrics_history:
            first = metrics_history[0].get("scores", {})
            last = metrics_history[-1].get("scores", {})

            lines.append("## Metrics Improvement")
            lines.append("")
            lines.append("| Metric | Before | After | Change |")
            lines.append("|--------|--------|-------|--------|")

            for metric in ["semantic_correctness", "factual_accuracy", "completeness"]:
                before = first.get(metric, 0)
                after = last.get(metric, 0)
                if isinstance(before, list):
                    before = sum(before) / len(before) if before else 0
                if isinstance(after, list):
                    after = sum(after) / len(after) if after else 0
                change = after - before
                change_str = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
                lines.append(f"| {metric} | {before:.2f} | {after:.2f} | {change_str} |")

            lines.append("")

        # Per-agent changes
        lines.append("## Agent Prompt Changes")
        lines.append("")

        for agent_name, var in prompt_variables.items():
            original = original_prompts.get(agent_name, "")
            current = var.value
            diff_chars = len(current) - len(original)

            lines.append(f"### {agent_name}")
            lines.append(f"- Original length: {len(original)} chars")
            lines.append(f"- Optimized length: {len(current)} chars")
            lines.append(f"- Difference: {diff_chars:+d} chars")
            lines.append("")

        return "\n".join(lines)
