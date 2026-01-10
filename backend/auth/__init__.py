"""
Authentication package for SUJBOT.

Provides JWT token management and password hashing using Argon2.
"""

from .manager import AuthManager

__all__ = ["AuthManager"]
