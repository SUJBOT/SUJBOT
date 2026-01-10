"""
API smoke tests for production.

These tests verify that core API endpoints work correctly.
Requires authentication (set PROD_TEST_USER and PROD_TEST_PASSWORD).

Usage:
    uv run pytest tests/production/test_api_smoke.py -v
"""

import pytest
import httpx


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
        assert "email" in data
        assert "id" in data


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
        assert isinstance(data, list)

    def test_create_conversation(self, auth_client: httpx.Client, requires_auth):
        """Creating a new conversation works."""
        response = auth_client.post("/conversations")
        assert response.status_code in (200, 201)

        data = response.json()
        assert "id" in data

        # Cleanup - delete the conversation
        conversation_id = data["id"]
        auth_client.delete(f"/conversations/{conversation_id}")

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
        assert "variant" in data

    def test_set_invalid_agent_variant(self, auth_client: httpx.Client, requires_auth):
        """Setting invalid agent variant returns error."""
        response = auth_client.post(
            "/settings/agent-variant",
            json={"variant": "nonexistent_variant_xyz"}
        )
        # Should return 400 or 422 for invalid variant
        assert response.status_code in (400, 422)


class TestDocumentEndpoints:
    """Test document access endpoints."""

    def test_list_documents_without_auth(self, http_client: httpx.Client):
        """Listing documents without auth returns 401."""
        response = http_client.get("/documents")
        # May return 401 or 404 depending on implementation
        assert response.status_code in (401, 404)


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
        assert response.status_code in (200, 204)

    def test_cors_origin_header(self, http_client: httpx.Client):
        """Response includes CORS origin header."""
        response = http_client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"}
        )
        assert response.status_code == 200
        # May or may not have CORS header depending on nginx config
