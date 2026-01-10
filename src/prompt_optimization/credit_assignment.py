"""
Credit Assignment for Multi-Agent TextGrad Optimization.

Assigns blame weights to specific agents based on evaluation failures.
This enables targeted gradient updates to the agents that caused problems.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class BlameAssignment:
    """Result of credit assignment for a single evaluation."""

    agent_weights: Dict[str, float]  # agent_name -> blame weight (0.0 to 1.0)
    failed_metrics: List[str]
    reasoning: str
    total_blame: float = 1.0


class CreditAssigner:
    """
    Assigns credit/blame to specific agents in multi-agent execution.

    Strategy:
    1. Trace-based: Use agent_outputs from state to see which agents ran
    2. Error propagation: If extractor returns bad chunks, downstream agents fail
    3. Metric-specific: Different metrics implicate different agents
    """

    # Agent dependency graph
    # Key: agent, Value: agents that must succeed for this agent to work
    DEPENDENCIES = {
        "orchestrator": [],  # No dependencies (root)
        "extractor": [],  # No dependencies (retrieval only)
        "classifier": ["extractor"],
        "requirement_extractor": ["extractor"],
        "compliance": ["extractor", "requirement_extractor"],
        "risk_verifier": ["extractor"],
        "citation_auditor": ["extractor"],
        "gap_synthesizer": ["extractor", "compliance"],
    }

    # Which agents are primarily responsible for each metric
    # Values: Dict[agent_name, base_weight]
    METRIC_BLAME_MAP = {
        "semantic_correctness": {
            # Primarily orchestrator (synthesis) and extractor (retrieval)
            "orchestrator": 0.50,  # Responsible for final synthesis
            "extractor": 0.30,  # Responsible for retrieval quality
            "_specialized": 0.20,  # Split among other agents that ran
        },
        "factual_accuracy": {
            # Primarily extractor (wrong chunks) and citation-related
            "extractor": 0.50,  # Wrong or missing chunks
            "orchestrator": 0.25,  # Bad synthesis of facts
            "citation_auditor": 0.15,  # Should verify citations
            "_specialized": 0.10,
        },
        "completeness": {
            # Extractor (incomplete retrieval) and gap synthesizer
            "extractor": 0.40,  # Didn't retrieve all relevant chunks
            "gap_synthesizer": 0.25,  # Should identify gaps
            "orchestrator": 0.25,  # Incomplete synthesis
            "_specialized": 0.10,
        },
    }

    def __init__(self, all_agents: Optional[List[str]] = None):
        """
        Initialize the credit assigner.

        Args:
            all_agents: List of all agent names (defaults to DEPENDENCIES keys)
        """
        self.all_agents = all_agents or list(self.DEPENDENCIES.keys())

    def assign_credit(
        self,
        failed_metrics: List[str],
        agent_sequence: List[str],
        agent_outputs: Optional[Dict[str, Any]] = None,
        scores: Optional[Dict[str, int]] = None,
    ) -> BlameAssignment:
        """
        Assign blame weights to each agent for evaluation failures.

        Args:
            failed_metrics: List of metrics that scored 0
            agent_sequence: List of agents that were executed
            agent_outputs: Optional dict of agent outputs for analysis
            scores: Optional dict of metric scores

        Returns:
            BlameAssignment with weights per agent
        """
        # Initialize blame weights for agents that actually ran
        blame_weights = {agent: 0.0 for agent in agent_sequence}

        if not failed_metrics:
            # No failures - minimal blame
            reasoning = "All metrics passed. No significant blame assigned."
            return BlameAssignment(
                agent_weights=blame_weights,
                failed_metrics=[],
                reasoning=reasoning,
            )

        # Calculate blame for each failed metric
        for metric in failed_metrics:
            metric_blame = self.METRIC_BLAME_MAP.get(metric, {})

            for agent_name, base_weight in metric_blame.items():
                if agent_name == "_specialized":
                    # Distribute among specialized agents that ran
                    specialized = [
                        a for a in agent_sequence
                        if a not in ["orchestrator", "extractor"]
                    ]
                    if specialized:
                        per_agent = base_weight / len(specialized)
                        for agent in specialized:
                            blame_weights[agent] = blame_weights.get(agent, 0.0) + per_agent
                elif agent_name in blame_weights:
                    blame_weights[agent_name] += base_weight

        # Normalize to sum to 1.0
        total = sum(blame_weights.values())
        if total > 0:
            blame_weights = {k: v / total for k, v in blame_weights.items()}

        # Generate reasoning
        reasoning_parts = [f"Failed metrics: {', '.join(failed_metrics)}"]

        # Top blamed agents
        sorted_blame = sorted(blame_weights.items(), key=lambda x: x[1], reverse=True)
        top_agents = [(a, w) for a, w in sorted_blame if w > 0.1]

        if top_agents:
            top_str = ", ".join(f"{a}={w:.2f}" for a, w in top_agents[:3])
            reasoning_parts.append(f"Primary blame: {top_str}")

        reasoning = " | ".join(reasoning_parts)

        return BlameAssignment(
            agent_weights=blame_weights,
            failed_metrics=failed_metrics,
            reasoning=reasoning,
        )

    def get_gradient_scaling(
        self,
        blame_assignment: BlameAssignment,
        threshold: float = 0.1,
    ) -> Dict[str, str]:
        """
        Get gradient scaling prefixes for each agent based on blame.

        These prefixes are added to gradients to control optimization priority.

        Args:
            blame_assignment: Result from assign_credit()
            threshold: Minimum blame to receive gradient updates

        Returns:
            Dict mapping agent_name -> priority prefix string
        """
        prefixes = {}

        for agent, weight in blame_assignment.agent_weights.items():
            if weight < threshold:
                prefixes[agent] = (
                    f"[LOW PRIORITY - blame={weight:.2f}] "
                    "This agent had minimal impact on failures. "
                    "Only make changes if clearly beneficial."
                )
            elif weight < 0.3:
                prefixes[agent] = (
                    f"[MEDIUM PRIORITY - blame={weight:.2f}] "
                    "This agent contributed to some failures. "
                    "Consider improvements that could help."
                )
            else:
                prefixes[agent] = (
                    f"[HIGH PRIORITY - blame={weight:.2f}] "
                    "This agent is a PRIMARY cause of failures. "
                    "Focus improvements on addressing: {}.".format(
                        ", ".join(blame_assignment.failed_metrics)
                    )
                )

        return prefixes

    def should_update_agent(
        self,
        agent_name: str,
        blame_assignment: BlameAssignment,
        threshold: float = 0.05,
    ) -> bool:
        """
        Determine if an agent should receive gradient updates.

        Args:
            agent_name: Name of the agent
            blame_assignment: Result from assign_credit()
            threshold: Minimum blame to receive updates

        Returns:
            True if agent should be updated
        """
        weight = blame_assignment.agent_weights.get(agent_name, 0.0)
        return weight >= threshold

    def analyze_failure_chain(
        self,
        agent_sequence: List[str],
        agent_outputs: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Analyze the failure chain to identify root cause.

        This looks at agent outputs to determine where things went wrong.

        Args:
            agent_sequence: Agents that were executed
            agent_outputs: Outputs from each agent

        Returns:
            Dict mapping agent_name -> failure analysis
        """
        analysis = {}

        for agent in agent_sequence:
            output = agent_outputs.get(agent, {})

            # Check for common failure patterns
            if isinstance(output, dict):
                # No results from retrieval
                if agent == "extractor":
                    documents = output.get("documents", [])
                    if not documents:
                        analysis[agent] = "RETRIEVAL_FAILURE: No documents retrieved"
                    elif len(documents) < 3:
                        analysis[agent] = f"LOW_RECALL: Only {len(documents)} documents"
                    else:
                        analysis[agent] = "OK"

                # Check for errors
                errors = output.get("errors", [])
                if errors:
                    analysis[agent] = f"ERRORS: {errors[:2]}"

            elif output is None:
                analysis[agent] = "NO_OUTPUT: Agent returned None"

        return analysis
