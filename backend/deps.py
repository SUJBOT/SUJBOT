"""
Shared Backend Dependencies (SSOT)

Centralizes dependency injection for modules shared across multiple route files.
All route modules import from here instead of defining their own globals.

Dependencies:
- AuthManager: JWT token operations
- AuthQueries: User database queries
- PostgreSQLStorageAdapter: Conversation/message storage
- VL + Graph components: Vision-Language pipeline and knowledge graph services
- Auth helpers: Cookie setting, login logic, token extraction
- PDF scan cache: Cached filesystem listing for document endpoints
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.auth.manager import AuthManager
from backend.database.auth_queries import AuthQueries
from src.storage.postgres_adapter import PostgreSQLStorageAdapter
from src.utils.cache import TTLCache

logger = logging.getLogger(__name__)

# =============================================================================
# Auth Dependencies
# =============================================================================

_auth_manager: Optional[AuthManager] = None
_auth_queries: Optional[AuthQueries] = None


def set_auth_dependencies(auth_manager: AuthManager, auth_queries: AuthQueries) -> None:
    """Set global auth dependencies (called from main.py lifespan)."""
    global _auth_manager, _auth_queries
    _auth_manager = auth_manager
    _auth_queries = auth_queries


def get_auth_manager() -> AuthManager:
    """Dependency injection for AuthManager."""
    if _auth_manager is None:
        raise RuntimeError("AuthManager not initialized. Call set_auth_dependencies() first.")
    return _auth_manager


def get_auth_queries() -> AuthQueries:
    """Dependency injection for AuthQueries."""
    if _auth_queries is None:
        raise RuntimeError("AuthQueries not initialized. Call set_auth_dependencies() first.")
    return _auth_queries


# =============================================================================
# PostgreSQL Adapter
# =============================================================================

_postgres_adapter: Optional[PostgreSQLStorageAdapter] = None


def set_postgres_adapter(adapter: PostgreSQLStorageAdapter) -> None:
    """Set the global PostgreSQL adapter instance (called from main.py lifespan)."""
    global _postgres_adapter
    _postgres_adapter = adapter


def get_postgres_adapter() -> PostgreSQLStorageAdapter:
    """Dependency for getting PostgreSQL adapter."""
    if _postgres_adapter is None:
        raise RuntimeError("PostgreSQLStorageAdapter not initialized")
    return _postgres_adapter


# =============================================================================
# VL Components
# =============================================================================

_vl_components: Dict[str, Any] = {}


def set_vl_components(
    jina_client: Any,
    page_store: Any,
    vector_store: Any,
    summary_provider: Any = None,
    entity_extractor: Any = None,
    graph_storage: Any = None,
    community_detector: Any = None,
    community_summarizer: Any = None,
    graph_embedder: Any = None,
    dedup_provider: Any = None,
) -> None:
    """Set VL components for document management (called from main.py lifespan)."""
    _vl_components["jina_client"] = jina_client
    _vl_components["page_store"] = page_store
    _vl_components["vector_store"] = vector_store
    _vl_components["summary_provider"] = summary_provider
    _vl_components["entity_extractor"] = entity_extractor
    _vl_components["graph_storage"] = graph_storage
    _vl_components["community_detector"] = community_detector
    _vl_components["community_summarizer"] = community_summarizer
    _vl_components["graph_embedder"] = graph_embedder
    _vl_components["dedup_provider"] = dedup_provider


def get_vl_components() -> Dict[str, Any]:
    """Get VL components dict. Returns the shared dict (may be empty if not initialized)."""
    return _vl_components


# =============================================================================
# Auth Cookie Helper
# =============================================================================


def set_auth_cookie(response, token: str, is_production: Optional[bool] = None) -> None:
    """Set JWT httpOnly cookie with consistent settings across all login endpoints.

    Args:
        response: FastAPI Response object
        token: JWT token string
        is_production: Override production detection (defaults to BUILD_TARGET env var)
    """
    if is_production is None:
        is_production = os.getenv("BUILD_TARGET", "development") == "production"
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=is_production,
        samesite="lax",
        max_age=86400,  # 24 hours
        path="/",
    )


# =============================================================================
# Login Helper
# =============================================================================


async def authenticate_user(
    email: str,
    password: str,
    auth_manager: AuthManager,
    auth_queries: AuthQueries,
    require_admin: bool = False,
) -> Dict:
    """Shared login logic for auth and admin login endpoints.

    Validates credentials, checks active status, optionally checks admin flag.

    Args:
        email: User email
        password: Plain text password
        auth_manager: AuthManager for password verification
        auth_queries: AuthQueries for user lookup
        require_admin: If True, also check is_admin flag

    Returns:
        User dict from database

    Raises:
        HTTPException 401: Invalid credentials
        HTTPException 403: Inactive or not admin
    """
    from fastapi import HTTPException, status

    user = await auth_queries.get_user_by_email(email)
    if not user:
        logger.warning(f"Login attempt for non-existent user: {email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Verify password FIRST (prevent timing attacks on is_admin check)
    if not auth_manager.verify_password(password, user["password_hash"]):
        logger.warning(f"Failed login attempt for user: {email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user["is_active"]:
        logger.warning(f"Login attempt for inactive user: {email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Contact administrator.",
        )

    if require_admin and not user.get("is_admin", False):
        logger.warning(f"Non-admin login attempt on admin endpoint: {email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return user


# =============================================================================
# Token Extraction Helper
# =============================================================================


def extract_token_from_request(request) -> Optional[str]:
    """Extract JWT token from cookie or Authorization header.

    Priority:
    1. httpOnly cookie "access_token" (XSS-safe)
    2. Authorization header "Bearer <token>" (API clients)

    Args:
        request: FastAPI/Starlette Request object

    Returns:
        JWT token string or None if not found
    """
    token = request.cookies.get("access_token")
    if token:
        return token

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


# =============================================================================
# PDF Filesystem Scan Cache
# =============================================================================

# TTL cache for PDF directory listing (avoids repeated glob + stat on every request)
_pdf_scan_cache: TTLCache[List[Tuple[str, str, int]]] = TTLCache(
    ttl_seconds=30,
    max_size=1,
    name="pdf_scan",
)

_PDF_SCAN_KEY = "pdf_list"


def get_pdf_file_list(pdf_base_dir: Path) -> List[Tuple[str, str, int]]:
    """Return cached list of (stem, filename, size_bytes) for all PDFs in the data dir.

    Caches the filesystem scan result for 30 seconds to avoid repeated
    glob + stat calls on every /documents or /admin/documents request.
    """
    cached = _pdf_scan_cache.get(_PDF_SCAN_KEY)
    if cached is not None:
        return cached

    result = []
    for pdf_path in pdf_base_dir.glob("*.pdf"):
        result.append((pdf_path.stem, pdf_path.name, pdf_path.stat().st_size))
    result.sort(key=lambda t: t[1])

    _pdf_scan_cache.set(_PDF_SCAN_KEY, result)
    return result


def invalidate_pdf_scan_cache() -> None:
    """Invalidate the PDF scan cache (call after upload/delete)."""
    _pdf_scan_cache.delete(_PDF_SCAN_KEY)
