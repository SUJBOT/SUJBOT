"""
Pydantic models for API request/response validation.

These models define the contract between frontend and backend.
"""

import base64
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    """Single chat message (user or assistant)."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


class SelectedContext(BaseModel):
    """Selected text from PDF for agent context.

    When user selects text in the PDF viewer, this context is passed
    to the agent to provide additional relevant information.
    """

    text: str = Field(
        ...,
        max_length=10000,
        description="Selected text content (max 10K chars)",
    )
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Human-readable document name")
    page_start: int = Field(..., ge=1, description="Starting page number (1-indexed)")
    page_end: int = Field(..., ge=1, description="Ending page number (1-indexed)")


ALLOWED_ATTACHMENT_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain",  # .txt
    "text/markdown",  # .md
    "text/html",  # .html
    "application/x-tex",  # .tex
    "text/x-tex",  # .tex (alt)
    "application/x-latex",  # .latex
}

MAX_ATTACHMENT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ATTACHMENTS_PER_MESSAGE = 5


class AttachmentData(BaseModel):
    """File attachment for a chat message (image or PDF).

    Attachments are sent as base64-encoded data in the JSON body.
    They become part of the agent's multimodal context but are NOT indexed.
    """

    filename: str = Field(..., max_length=255)
    mime_type: str = Field(...)
    base64_data: str = Field(...)

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        if v not in ALLOWED_ATTACHMENT_MIME_TYPES:
            raise ValueError(
                f"Unsupported file type: {v}. "
                f"Allowed: {', '.join(sorted(ALLOWED_ATTACHMENT_MIME_TYPES))}"
            )
        return v

    @field_validator("base64_data")
    @classmethod
    def validate_base64_size(cls, v: str) -> str:
        try:
            decoded = base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 data")
        if len(decoded) > MAX_ATTACHMENT_SIZE_BYTES:
            size_mb = len(decoded) / (1024 * 1024)
            raise ValueError(
                f"File too large: {size_mb:.1f} MB (max {MAX_ATTACHMENT_SIZE_BYTES // (1024 * 1024)} MB)"
            )
        return v


class ChatRequest(BaseModel):
    """Request to send a message to the agent."""

    message: str = Field(
        ...,
        max_length=50000,
        description="User message (max 50K chars to prevent abuse)",
    )
    conversation_id: Optional[str] = Field(None, description="Conversation ID for history")
    skip_save_user_message: bool = Field(
        False,
        description="Skip saving user message to database (for regenerate where message already exists)",
    )
    messages: Optional[List[ChatMessage]] = Field(
        None,
        description="Conversation history for context (last N messages)",
    )
    selected_context: Optional[SelectedContext] = Field(
        None,
        description="Selected text from PDF viewer for additional context",
    )
    attachments: Optional[List[AttachmentData]] = Field(
        None,
        description="File attachments (images/PDFs) for multimodal context",
    )
    web_search_enabled: bool = Field(
        False,
        description="Enable web search tool for this request",
    )

    @field_validator("attachments")
    @classmethod
    def validate_attachment_count(
        cls, v: Optional[List[AttachmentData]],
    ) -> Optional[List[AttachmentData]]:
        if v and len(v) > MAX_ATTACHMENTS_PER_MESSAGE:
            raise ValueError(
                f"Too many attachments: {len(v)} (max {MAX_ATTACHMENTS_PER_MESSAGE})"
            )
        return v

    @model_validator(mode="after")
    def validate_message_or_attachments(self) -> "ChatRequest":
        if not self.message.strip() and not self.attachments:
            raise ValueError("Either message or attachments must be provided")
        return self


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "error"]
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ClarificationRequest(BaseModel):
    """Request to provide clarification for interrupted workflow."""

    thread_id: str = Field(..., description="Thread ID from clarification_needed event")
    response: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User's free-form clarification response",
    )


class AgentVariantRequest(BaseModel):
    """Request to update agent variant preference."""

    variant: Literal["remote", "local"] = Field(
        ...,
        description="Agent variant: 'remote' (Haiku) or 'local' (Qwen3 VL 235B via DeepInfra)"
    )


class AgentVariantResponse(BaseModel):
    """Response with agent variant information."""

    variant: Literal["remote", "local"] = Field(..., description="Current variant")
    display_name: str = Field(..., description="Human-readable variant name")
    model: str = Field(..., description="Default model identifier for this variant")


class SpendingResponse(BaseModel):
    """User spending information."""

    total_spent_czk: float = Field(..., ge=0, description="Total amount spent in CZK")
    spending_limit_czk: float = Field(..., ge=0, description="Spending limit in CZK")
    remaining_czk: float = Field(..., ge=0, description="Remaining budget in CZK")
    reset_at: Optional[str] = Field(None, description="ISO timestamp of last reset")


# =============================================================================
# Admin API Models
# =============================================================================


class AdminUserResponse(BaseModel):
    """Admin view of user (includes all fields)."""

    id: int
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    agent_variant: Optional[str] = None
    created_at: str
    updated_at: str
    last_login_at: Optional[str] = None
    # Spending tracking
    spending_limit_czk: Optional[float] = Field(default=500.0, ge=0)
    total_spent_czk: float = Field(default=0.0, ge=0)
    spending_reset_at: Optional[str] = None


class AdminUserListResponse(BaseModel):
    """Paginated list of users for admin."""

    users: List[AdminUserResponse]
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1, le=1000)
    offset: int = Field(..., ge=0)


class AdminUserCreateRequest(BaseModel):
    """Request to create a new user (admin endpoint)."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password (min 8 chars)")
    full_name: Optional[str] = Field(None, max_length=100, description="Display name")
    is_admin: bool = Field(False, description="Grant admin privileges")
    is_active: bool = Field(True, description="Account active status")


class AdminUserUpdateRequest(BaseModel):
    """Request to update user (admin endpoint)."""

    email: Optional[EmailStr] = Field(None, description="New email address")
    password: Optional[str] = Field(None, min_length=8, max_length=128, description="New password (min 8 chars)")
    full_name: Optional[str] = Field(None, max_length=100, description="Display name")
    is_admin: Optional[bool] = Field(None, description="Admin privileges")
    is_active: Optional[bool] = Field(None, description="Account active status")
    agent_variant: Optional[Literal["remote", "local"]] = Field(
        None, description="Model preference"
    )
    spending_limit_czk: Optional[float] = Field(
        None, ge=0, description="Spending limit in CZK"
    )


class AdminLoginRequest(BaseModel):
    """Admin login credentials."""

    email: EmailStr = Field(..., description="Admin email address")
    password: str = Field(..., min_length=8, description="Admin password")


class ServiceHealthDetail(BaseModel):
    """Individual service health status."""

    name: str = Field(..., min_length=1)
    status: Literal["healthy", "degraded", "unhealthy"]
    latency_ms: Optional[float] = Field(None, ge=0)
    message: Optional[str] = None


class AdminHealthResponse(BaseModel):
    """Detailed health check for admin dashboard."""

    status: Literal["healthy", "degraded", "unhealthy"]
    services: List[ServiceHealthDetail]
    timestamp: str


class AdminStatsResponse(BaseModel):
    """System statistics for admin dashboard."""

    total_users: int = Field(..., ge=0)
    active_users: int = Field(..., ge=0)
    admin_users: int = Field(..., ge=0)
    total_conversations: int = Field(..., ge=0)
    total_messages: int = Field(..., ge=0)
    users_last_24h: int = Field(..., ge=0)
    # Spending statistics
    total_spent_czk: float = Field(default=0.0, ge=0)
    avg_spent_per_message_czk: float = Field(default=0.0, ge=0)
    avg_spent_per_conversation_czk: float = Field(default=0.0, ge=0)
    timestamp: str


# =============================================================================
# Admin Conversation Viewing Models (Read-Only)
# =============================================================================


class AdminConversationResponse(BaseModel):
    """Admin view of a user conversation (read-only)."""

    id: str = Field(..., description="Conversation UUID")
    title: str = Field(..., description="Conversation title")
    message_count: int = Field(..., ge=0, description="Number of messages")
    created_at: str = Field(..., description="ISO timestamp of creation")
    updated_at: str = Field(..., description="ISO timestamp of last update")


class AdminMessageResponse(BaseModel):
    """Admin view of a conversation message (read-only)."""

    id: int = Field(..., description="Message ID")
    role: Literal["user", "assistant", "system"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Message metadata")
    created_at: str = Field(..., description="ISO timestamp of creation")
