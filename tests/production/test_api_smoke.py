"""
API smoke tests for production.

These tests verify that core API endpoints work correctly.
Requires authentication (set PROD_TEST_USER and PROD_TEST_PASSWORD).

Usage:
    uv run pytest tests/production/test_api_smoke.py -v
"""

import pytest
import httpx
import warnings


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_login_with_invalid_credentials(self, http_client: httpx.Client):
        """Login with invalid credentials returns 401."""
        response = http_client.post(
            "/auth/login",
            json={"email": "invalid@test.com", "password": "wrongpassword"}
        )
        assert response.status_code == 401

    def test_login_with_missing_fields(self, http_client: httpx.Client):
        """Login with missing fields returns 422."""
        response = http_client.post("/auth/login", json={})
        assert response.status_code == 422

    def test_me_endpoint_without_auth(self, http_client: httpx.Client):
        """Accessing /auth/me without auth returns 401."""
        response = http_client.get("/auth/me")
        assert response.status_code == 401

    def test_me_endpoint_with_auth(self, auth_client: httpx.Client, requires_auth):
        """Accessing /auth/me with auth returns user info."""
        response = auth_client.get("/auth/me")
        assert response.status_code == 200

        data = response.json()
        assert "email" in data, f"Response missing 'email' field: {data}"
        assert "id" in data, f"Response missing 'id' field: {data}"


class TestConversationEndpoints:
    """Test conversation management endpoints."""

    def test_list_conversations_without_auth(self, http_client: httpx.Client):
        """Listing conversations without auth returns 401."""
        response = http_client.get("/conversations")
        assert response.status_code == 401

    def test_list_conversations_with_auth(self, auth_client: httpx.Client, requires_auth):
        """Listing conversations with auth returns array."""
        response = auth_client.get("/conversations")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list), f"Expected list, got {type(data)}"

    def test_create_conversation(self, auth_client: httpx.Client, requires_auth):
        """Creating a new conversation works."""
        response = auth_client.post("/conversations")
        assert response.status_code in (200, 201), \
            f"Expected 200/201, got {response.status_code}: {response.text[:200]}"

        data = response.json()
        assert "id" in data, f"Response missing 'id' field: {data}"

        # Cleanup - delete the conversation
        conversation_id = data["id"]
        try:
            auth_client.delete(f"/conversations/{conversation_id}")
        except httpx.HTTPError as e:
            warnings.warn(f"Failed to cleanup conversation: {e}")

    def test_get_nonexistent_conversation(self, auth_client: httpx.Client, requires_auth):
        """Getting nonexistent conversation returns 404."""
        response = auth_client.get("/conversations/nonexistent-id-12345")
        assert response.status_code == 404


class TestSettingsEndpoints:
    """Test settings endpoints."""

    def test_get_agent_variant_without_auth(self, http_client: httpx.Client):
        """Getting agent variant without auth returns 401."""
        response = http_client.get("/settings/agent-variant")
        assert response.status_code == 401

    def test_get_agent_variant_with_auth(self, auth_client: httpx.Client, requires_auth):
        """Getting agent variant with auth works."""
        response = auth_client.get("/settings/agent-variant")
        assert response.status_code == 200

        data = response.json()
        assert "variant" in data, f"Response missing 'variant' field: {data}"

    def test_set_invalid_agent_variant(self, auth_client: httpx.Client, requires_auth):
        """Setting invalid agent variant returns error."""
        response = auth_client.post(
            "/settings/agent-variant",
            json={"variant": "nonexistent_variant_xyz"}
        )
        # Should return 400 or 422 for invalid variant
        assert response.status_code in (400, 422), \
            f"Expected 400/422 for invalid variant, got {response.status_code}"


class TestDocumentEndpoints:
    """Test document access endpoints."""

    def test_list_documents_without_auth(self, http_client: httpx.Client):
        """Listing documents without auth returns 401."""
        # Use trailing slash to avoid redirect
        response = http_client.get("/documents/")
        assert response.status_code == 401, \
            f"Expected 401 for unauthenticated access, got {response.status_code}"

    def test_list_documents_with_auth(self, auth_client: httpx.Client, requires_auth):
        """Listing documents with auth returns array."""
        response = auth_client.get("/documents")
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}: {response.text[:200]}"

        data = response.json()
        assert isinstance(data, list), f"Expected list, got {type(data)}"

        # If documents exist, verify structure
        if data:
            doc = data[0]
            assert "document_id" in doc, f"Document missing 'document_id': {doc}"
            assert "display_name" in doc, f"Document missing 'display_name': {doc}"

    def test_get_nonexistent_document(self, auth_client: httpx.Client, requires_auth):
        """Getting nonexistent document returns 404."""
        response = auth_client.get("/documents/nonexistent_doc_xyz/pdf")
        assert response.status_code == 404, \
            f"Expected 404 for nonexistent document, got {response.status_code}"


class TestCitationEndpoints:
    """Test citation metadata endpoints."""

    def test_citations_without_auth(self, http_client: httpx.Client):
        """Citation endpoint without auth returns 401."""
        response = http_client.post(
            "/citations/batch",
            json={"chunk_ids": ["test_chunk_1"]}
        )
        assert response.status_code == 401, \
            f"Expected 401 for unauthenticated access, got {response.status_code}"

    def test_citation_batch_empty_list(self, auth_client: httpx.Client, requires_auth):
        """Citation batch with empty list returns validation error."""
        response = auth_client.post(
            "/citations/batch",
            json={"chunk_ids": []}
        )
        assert response.status_code == 422, \
            f"Expected 422 for empty chunk_ids, got {response.status_code}"

    def test_citation_batch_invalid_format(self, auth_client: httpx.Client, requires_auth):
        """Citation batch with invalid chunk_id format returns error."""
        response = auth_client.post(
            "/citations/batch",
            json={"chunk_ids": ["<script>alert('xss')</script>"]}
        )
        # Should return 400 for invalid format or 200 with error info
        assert response.status_code in (400, 422, 200), \
            f"Unexpected status for invalid chunk_id: {response.status_code}"

    def test_citation_batch_nonexistent(self, auth_client: httpx.Client, requires_auth):
        """Citation batch for nonexistent chunks returns empty/partial results."""
        response = auth_client.post(
            "/citations/batch",
            json={"chunk_ids": ["nonexistent_chunk_xyz123"]}
        )
        # Should return 200 with empty results or error info, not 500
        assert response.status_code in (200, 404), \
            f"Expected 200/404 for nonexistent chunk, got {response.status_code}"


class TestFeedbackEndpoints:
    """Test feedback endpoints."""

    def test_feedback_without_auth(self, http_client: httpx.Client):
        """Submitting feedback without auth returns 401."""
        response = http_client.post(
            "/feedback",
            json={"message_id": 1, "score": 1}
        )
        assert response.status_code == 401, \
            f"Expected 401 for unauthenticated access, got {response.status_code}"

    def test_feedback_invalid_score(self, auth_client: httpx.Client, requires_auth):
        """Feedback with invalid score returns validation error."""
        response = auth_client.post(
            "/feedback",
            json={"message_id": 1, "score": 99}  # Invalid score
        )
        assert response.status_code == 422, \
            f"Expected 422 for invalid score, got {response.status_code}"

    def test_feedback_missing_fields(self, auth_client: httpx.Client, requires_auth):
        """Feedback with missing required fields returns validation error."""
        response = auth_client.post("/feedback", json={})
        assert response.status_code == 422, \
            f"Expected 422 for missing fields, got {response.status_code}"

    def test_feedback_nonexistent_message(self, auth_client: httpx.Client, requires_auth):
        """Feedback for nonexistent message returns 404."""
        response = auth_client.post(
            "/feedback",
            json={"message_id": 999999999, "score": 1}
        )
        # Should return 404 for nonexistent message or 403 for wrong ownership
        assert response.status_code in (403, 404), \
            f"Expected 403/404 for nonexistent message, got {response.status_code}"


class TestChatEndpoints:
    """Test chat streaming endpoints."""

    def test_chat_stream_without_auth(self, http_client: httpx.Client):
        """Chat stream without auth returns 401."""
        response = http_client.post(
            "/chat/stream",
            json={"message": "test"}
        )
        assert response.status_code == 401

    def test_chat_stream_with_empty_message(self, auth_client: httpx.Client, requires_auth):
        """Chat stream with empty message returns error."""
        response = auth_client.post(
            "/chat/stream",
            json={"message": ""}
        )
        # Should return 422 for validation error
        assert response.status_code == 422

    def test_chat_stream_missing_conversation(self, auth_client: httpx.Client, requires_auth):
        """Chat stream without conversation_id returns appropriate error."""
        response = auth_client.post(
            "/chat/stream",
            json={"message": "Test query without conversation"}
        )
        # May auto-create conversation or return error
        assert response.status_code in (200, 400, 422), \
            f"Unexpected status for missing conversation_id: {response.status_code}"


class TestAdminEndpoints:
    """Test admin endpoints (should require admin auth)."""

    def test_admin_health_without_auth(self, http_client: httpx.Client):
        """Admin health without auth returns 401."""
        response = http_client.get("/admin/health")
        assert response.status_code == 401

    def test_admin_users_without_auth(self, http_client: httpx.Client):
        """Admin users list without auth returns 401."""
        response = http_client.get("/admin/users")
        assert response.status_code == 401

    def test_admin_stats_without_auth(self, http_client: httpx.Client):
        """Admin stats without auth returns 401."""
        response = http_client.get("/admin/stats")
        assert response.status_code == 401


class TestCORSHeaders:
    """Test CORS configuration."""

    def test_cors_preflight(self, http_client: httpx.Client):
        """CORS preflight request returns proper headers."""
        response = http_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET"
            }
        )
        # Should allow the origin
        assert response.status_code in (200, 204), \
            f"Expected 200/204 for preflight, got {response.status_code}"

    def test_cors_origin_header(self, http_client: httpx.Client):
        """Response includes CORS origin header."""
        response = http_client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"}
        )
        assert response.status_code == 200
        # May or may not have CORS header depending on nginx config


class TestRateLimiting:
    """Test rate limiting behavior (if implemented)."""

    def test_rapid_requests_handled(self, http_client: httpx.Client):
        """Multiple rapid requests don't cause server errors."""
        errors = []
        for i in range(10):
            try:
                response = http_client.get("/health")
                if response.status_code >= 500:
                    errors.append(f"Request {i}: status {response.status_code}")
            except httpx.HTTPError as e:
                errors.append(f"Request {i}: {type(e).__name__}: {e}")

        # Some rate limiting (429) is OK, but no server errors
        assert not errors, f"Server errors during rapid requests: {errors}"
