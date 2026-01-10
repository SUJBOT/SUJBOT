"""
Prompt Variable Management for TextGrad optimization.

Loads agent prompts from prompts/agents/ as TextGrad Variables with requires_grad=True.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import textgrad as tg

logger = logging.getLogger(__name__)

# Agent names matching files in prompts/agents/
AGENT_NAMES = [
    "orchestrator",
    "extractor",
    "classifier",
    "compliance",
    "risk_verifier",
    "requirement_extractor",
    "citation_auditor",
    "gap_synthesizer",
]

# Role descriptions for each agent (helps TextGrad understand variable purpose)
AGENT_ROLE_DESCRIPTIONS = {
    "orchestrator": (
        "System prompt for the Orchestrator agent that routes queries to specialized agents "
        "and synthesizes final answers. Controls multi-agent coordination in Czech legal RAG system."
    ),
    "extractor": (
        "System prompt for the Extractor agent that retrieves documents using hybrid search "
        "(HyDE + Expansion Fusion). Primary retrieval agent with 10 tool call limit."
    ),
    "classifier": (
        "System prompt for the Classifier agent that categorizes documents across 5 dimensions: "
        "Document Type, Domain, Complexity, Language, Sensitivity."
    ),
    "compliance": (
        "System prompt for the Compliance agent that verifies regulatory compliance against "
        "legal requirements. Classifies gaps as REGULATORY_GAP or SCOPE_GAP."
    ),
    "risk_verifier": (
        "System prompt for the Risk Verifier agent that performs risk assessment across "
        "6 categories: Legal, Financial, Operational, Compliance, Safety, Reputational."
    ),
    "requirement_extractor": (
        "System prompt for the Requirement Extractor agent that extracts atomic legal obligations "
        "from documents for compliance checking."
    ),
    "citation_auditor": (
        "System prompt for the Citation Auditor agent that verifies citations across 4 dimensions: "
        "EXISTS, ACCURATE, COMPLETE, FORMATTED."
    ),
    "gap_synthesizer": (
        "System prompt for the Gap Synthesizer agent that identifies gaps across multiple documents. "
        "Handles REGULATORY_GAP, SCOPE_GAP, Coverage Gap, Consistency Gap, Citation Gap, Temporal Gap."
    ),
}


class PromptVariableManager:
    """
    Manages TextGrad Variables for agent prompts.

    Loads prompts from prompts/agents/ directory and creates TextGrad Variables
    with requires_grad=True for optimization.
    """

    def __init__(self, prompts_dir: Path, agents: Optional[List[str]] = None):
        """
        Initialize the prompt variable manager.

        Args:
            prompts_dir: Path to prompts/agents/ directory
            agents: Optional list of agent names to load (defaults to all 8)
        """
        self.prompts_dir = Path(prompts_dir)
        self.agents = agents or AGENT_NAMES
        self.variables: Dict[str, tg.Variable] = {}
        self._original_prompts: Dict[str, str] = {}  # Store originals for comparison

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

    def load_all_prompts(self) -> Dict[str, tg.Variable]:
        """
        Load all agent prompts as TextGrad Variables.

        Returns:
            Dict mapping agent_name -> tg.Variable with requires_grad=True
        """
        for agent_name in self.agents:
            prompt_file = self.prompts_dir / f"{agent_name}.txt"

            if not prompt_file.exists():
                logger.warning(f"Prompt file not found: {prompt_file}, skipping {agent_name}")
                continue

            prompt_text = prompt_file.read_text(encoding="utf-8")
            self._original_prompts[agent_name] = prompt_text

            # Create TextGrad Variable with gradient tracking
            role_desc = AGENT_ROLE_DESCRIPTIONS.get(
                agent_name,
                f"System prompt for {agent_name} agent in multi-agent RAG system"
            )

            self.variables[agent_name] = tg.Variable(
                prompt_text,
                requires_grad=True,
                role_description=role_desc
            )

            logger.info(
                f"Loaded prompt variable: {agent_name} "
                f"({len(prompt_text)} chars, {len(prompt_text.split())} words)"
            )

        logger.info(f"Loaded {len(self.variables)} prompt variables")
        return self.variables

    def get_variable(self, agent_name: str) -> Optional[tg.Variable]:
        """Get a specific agent's prompt variable."""
        return self.variables.get(agent_name)

    def get_original_prompt(self, agent_name: str) -> Optional[str]:
        """Get the original (pre-optimization) prompt text."""
        return self._original_prompts.get(agent_name)

    def get_current_prompt(self, agent_name: str) -> Optional[str]:
        """Get the current (possibly optimized) prompt text."""
        var = self.variables.get(agent_name)
        return var.value if var else None

    def get_all_parameters(self) -> List[tg.Variable]:
        """
        Get all variables as a list for optimizer.

        Returns:
            List of all prompt Variables for TGD optimizer
        """
        return list(self.variables.values())

    def get_prompt_diff(self, agent_name: str) -> Optional[Dict[str, str]]:
        """
        Get diff between original and current prompt.

        Returns:
            Dict with 'original', 'current', 'changed' keys
        """
        original = self._original_prompts.get(agent_name)
        var = self.variables.get(agent_name)

        if not original or not var:
            return None

        current = var.value
        return {
            "original": original,
            "current": current,
            "changed": original != current,
            "original_len": len(original),
            "current_len": len(current),
            "diff_chars": len(current) - len(original),
        }

    def inject_into_loader(self, prompt_loader) -> None:
        """
        Inject current prompt values into PromptLoader cache.

        This patches the prompt loader so MultiAgentRunner uses optimized prompts.

        Args:
            prompt_loader: PromptLoader instance from src.multi_agent.prompts.loader
        """
        for agent_name, variable in self.variables.items():
            prompt_loader._cache[agent_name] = variable.value

        logger.debug(f"Injected {len(self.variables)} prompts into loader cache")

    def __len__(self) -> int:
        return len(self.variables)

    def __repr__(self) -> str:
        return f"PromptVariableManager({len(self.variables)} agents: {list(self.variables.keys())})"
