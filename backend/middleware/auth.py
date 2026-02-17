"""
Authentication Dependencies - JWT Token Validation

Protects FastAPI routes by validating JWT tokens from httpOnly cookies or Authorization headers.

Usage:
    @app.get("/protected")
    async def protected_route(user: Dict = Depends(get_current_user)):
        return {"user_id": user["id"], "email": user["email"]}
"""

from typing import Dict
from fastapi import Request, HTTPException, status
import logging

logger = logging.getLogger(__name__)


# =========================================================================
# Legacy bridge — delegates to backend.deps (SSOT)
# =========================================================================

def set_auth_instances(auth_manager, auth_queries):
    """Set global auth instances — delegates to deps.py (SSOT)."""
    from backend.deps import set_auth_dependencies
    set_auth_dependencies(auth_manager, auth_queries)


async def get_current_user(request: Request) -> Dict:
    """
    FastAPI dependency for getting authenticated user.

    Validates JWT token from cookie or header, loads user from database.
    Uses backend.deps as single source of truth for auth instances.
    """
    from backend.deps import get_auth_manager, get_auth_queries, extract_token_from_request

    auth_manager = get_auth_manager()
    auth_queries = get_auth_queries()

    token = extract_token_from_request(request)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required (missing token)"
        )

    # Validate token
    payload = auth_manager.validate_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    # Load user from database
    user = await auth_queries.get_active_user_by_id(payload.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    return user


async def get_current_active_user(request: Request) -> Dict:
    """
    FastAPI dependency for getting active user (extra validation).

    Similar to get_current_user but enforces is_active check.

    Usage:
        @app.post("/admin/users")
        async def create_user(user: Dict = Depends(get_current_active_user)):
            # Only active users can access this route
            pass

    Args:
        request: FastAPI request object

    Returns:
        Active user dict

    Raises:
        HTTPException 401: If user not authenticated or inactive
    """
    user = await get_current_user(request)

    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    return user

async def get_current_admin_user(request: Request) -> Dict:
    """
    FastAPI dependency for getting admin user (admin-only routes).

    Validates authentication AND admin privileges.

    Usage:
        @app.post("/auth/register")
        async def register_user(admin: Dict = Depends(get_current_admin_user)):
            # Only admins can create new users
            pass

    Args:
        request: FastAPI request object

    Returns:
        Admin user dict

    Raises:
        HTTPException 401: If user not authenticated
        HTTPException 403: If user is not admin or inactive
    """
    user = await get_current_user(request)

    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    if not user.get("is_admin", False):
        logger.warning(f"Non-admin user {user['id']} attempted to access admin endpoint")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )

    return user
