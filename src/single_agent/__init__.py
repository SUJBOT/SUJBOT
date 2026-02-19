"""
Single-agent runner for SUJBOT.

SingleAgentRunner: autonomous tool loop with unified prompt.
RoutingAgentRunner: 8B router classifies queries, delegates complex ones to 30B.
"""

from .runner import SingleAgentRunner
from .routing_runner import RoutingAgentRunner

__all__ = ["SingleAgentRunner", "RoutingAgentRunner"]
