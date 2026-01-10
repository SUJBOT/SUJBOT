"""
TextGrad Prompt Optimization Module for SUJBOT.

This module provides tools for optimizing multi-agent system prompts using
TextGrad's automatic "differentiation" via text.

Components:
    - PromptVariableManager: Load/manage prompts as TextGrad Variables
    - MultiMetricLoss: Combined loss from 3 evaluation metrics
    - CreditAssigner: Attribute blame to specific agents
    - PromptVersionManager: Version control for optimized prompts
"""

from src.prompt_optimization.variables import PromptVariableManager
from src.prompt_optimization.loss import MultiMetricLoss
from src.prompt_optimization.credit_assignment import CreditAssigner
from src.prompt_optimization.versioning import PromptVersionManager

__all__ = [
    "PromptVariableManager",
    "MultiMetricLoss",
    "CreditAssigner",
    "PromptVersionManager",
]
