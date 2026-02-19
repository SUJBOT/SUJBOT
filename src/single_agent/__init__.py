"""
Single-agent runner for SUJBOT.

SingleAgentRunner: autonomous tool loop with unified prompt.
RoutingAgentRunner: 8B router → 30B thinking worker architecture.
"""

from .runner import SingleAgentRunner
from .routing_runner import RoutingAgentRunner

__all__ = ["SingleAgentRunner", "RoutingAgentRunner"]
