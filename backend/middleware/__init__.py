"""
Middleware package for FastAPI.

Provides authentication, rate limiting, and security headers.
"""

from .auth import get_current_user, get_current_admin_user, set_auth_instances
from .rate_limit import RateLimitMiddleware
from .security_headers import SecurityHeadersMiddleware

__all__ = [
    "get_current_user",
    "get_current_admin_user",
    "set_auth_instances",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware"
]
