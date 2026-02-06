"""
User settings API endpoints.

Provides endpoints for managing user-specific settings:
- Agent variant preference (remote/local)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from ..models import AgentVariantRequest, AgentVariantResponse, SpendingResponse
from ..middleware.auth import get_current_user
from ..constants import get_variant_config, get_default_variant, is_valid_variant
from .auth import get_auth_queries

router = APIRouter(prefix="/settings", tags=["settings"])
logger = logging.getLogger(__name__)


def _build_variant_response(variant: str) -> AgentVariantResponse:
    """Build AgentVariantResponse from variant config."""
    config = get_variant_config()[variant]
    return AgentVariantResponse(
        variant=variant,
        display_name=config["display_name"],
        model=config["model"],
    )


@router.get("/agent-variant", response_model=AgentVariantResponse)
async def get_agent_variant(current_user: dict = Depends(get_current_user)):
    """
    Get current user's agent variant preference.

    Returns:
        AgentVariantResponse with variant, display_name, and model

    Example response:
        {
            "variant": "remote",
            "display_name": "Remote (Haiku 4.5)",
            "model": "claude-haiku-4-5-20251001"
        }
    """
    queries = get_auth_queries()
    variant = await queries.get_agent_variant(current_user["id"])

    # Defensive check: fall back to default if variant unknown
    if not is_valid_variant(variant):
        logger.warning(f"Unknown variant '{variant}' for user {current_user['id']}, using default")
        variant = get_default_variant()

    return _build_variant_response(variant)


@router.post("/agent-variant", response_model=AgentVariantResponse)
async def update_agent_variant(
    request: AgentVariantRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user's agent variant preference.

    Args:
        request: AgentVariantRequest with variant field

    Returns:
        AgentVariantResponse with updated variant information

    Raises:
        HTTPException(400): If variant is invalid

    Example request:
        {"variant": "remote"}

    Example response:
        {
            "variant": "remote",
            "display_name": "Remote (Haiku 4.5)",
            "model": "claude-haiku-4-5-20251001"
        }
    """
    queries = get_auth_queries()

    try:
        await queries.update_agent_variant(current_user["id"], request.variant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _build_variant_response(request.variant)


@router.get("/spending", response_model=SpendingResponse)
async def get_spending(current_user: dict = Depends(get_current_user)):
    """
    Get current user's spending information.

    Returns:
        SpendingResponse with spending details in CZK

    Example response:
        {
            "total_spent_czk": 123.45,
            "spending_limit_czk": 500.00,
            "remaining_czk": 376.55,
            "reset_at": "2024-12-01T00:00:00+00:00"
        }
    """
    queries = get_auth_queries()
    spending = await queries.get_user_spending(current_user["id"])

    return SpendingResponse(
        total_spent_czk=spending["total_spent_czk"],
        spending_limit_czk=spending["spending_limit_czk"],
        remaining_czk=spending["remaining_czk"],
        reset_at=spending["reset_at"]
    )
