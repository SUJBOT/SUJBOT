"""
End-to-end RAG query tests for production.

These tests verify that the full RAG pipeline works correctly.
Requires authentication and indexed documents.

Usage:
    uv run pytest tests/production/test_rag_e2e.py -v

Note: These tests are slower as they make actual LLM calls.
"""

import pytest
import httpx
import json
import time
from typing import Generator


class TestRAGQueryFlow:
    """Test complete RAG query flow."""

    @pytest.fixture
    def conversation_id(self, auth_client: httpx.Client, requires_auth) -> Generator[str, None, None]:
        """Create a conversation for testing, cleanup after."""
        response = auth_client.post("/conversations")
        assert response.status_code in (200, 201)

        data = response.json()
        conversation_id = data["id"]

        yield conversation_id

        # Cleanup
        auth_client.delete(f"/conversations/{conversation_id}")

    def test_simple_query_streaming(self, auth_client: httpx.Client, conversation_id: str):
        """Simple query returns streaming response."""
        # Use a simple greeting to test basic flow
        with auth_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "Ahoj, co umíš?",
                "conversation_id": conversation_id
            },
            timeout=60.0
        ) as response:
            assert response.status_code == 200

            events = []
            for line in response.iter_lines():
                if line.startswith("data:"):
                    try:
                        event_data = json.loads(line[5:].strip())
                        events.append(event_data)
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("event:"):
                    event_type = line[6:].strip()
                    events.append({"event_type": event_type})

            # Should have received some events
            assert len(events) > 0

    @pytest.mark.slow
    def test_rag_query_with_search(self, auth_client: httpx.Client, conversation_id: str):
        """RAG query that triggers document search."""
        # Use a domain-specific query that should trigger search
        query = "Jaké jsou požadavky na bezpečnost jaderných zařízení?"

        collected_text = ""
        tool_calls = []
        cost_info = None

        with auth_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": query,
                "conversation_id": conversation_id
            },
            timeout=120.0
        ) as response:
            assert response.status_code == 200

            current_event = None
            for line in response.iter_lines():
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:") and current_event:
                    try:
                        data = json.loads(line[5:].strip())
                        if current_event == "text_delta":
                            collected_text += data.get("content", "")
                        elif current_event == "tool_call":
                            tool_calls.append(data)
                        elif current_event == "cost_summary":
                            cost_info = data
                    except json.JSONDecodeError:
                        pass

        # Should have generated some response
        assert len(collected_text) > 50, "Response should contain meaningful text"

        # Cost tracking should work
        if cost_info:
            assert "total_cost" in cost_info
            assert cost_info["total_cost"] >= 0

    @pytest.mark.slow
    def test_conversation_history_preserved(self, auth_client: httpx.Client, conversation_id: str):
        """Conversation history is preserved between messages."""
        # First message
        first_response = ""
        with auth_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "Moje jméno je Testovací Uživatel.",
                "conversation_id": conversation_id
            },
            timeout=60.0
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        first_response += data.get("content", "")
                    except json.JSONDecodeError:
                        pass

        # Give backend time to save
        time.sleep(1)

        # Second message - should remember the name
        second_response = ""
        with auth_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "Jak se jmenuji?",
                "conversation_id": conversation_id
            },
            timeout=60.0
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        second_response += data.get("content", "")
                    except json.JSONDecodeError:
                        pass

        # Response should mention the name
        assert "Testovací" in second_response or "testovací" in second_response.lower(), \
            f"Assistant should remember the name. Response: {second_response[:200]}"


class TestCostTracking:
    """Test cost tracking functionality."""

    @pytest.fixture
    def conversation_id(self, auth_client: httpx.Client, requires_auth) -> Generator[str, None, None]:
        """Create a conversation for testing."""
        response = auth_client.post("/conversations")
        data = response.json()
        yield data["id"]
        auth_client.delete(f"/conversations/{data['id']}")

    def test_cost_summary_event_present(self, auth_client: httpx.Client, conversation_id: str):
        """Cost summary event is included in response."""
        cost_summary = None

        with auth_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "Řekni mi krátce co jsi.",
                "conversation_id": conversation_id
            },
            timeout=60.0
        ) as response:
            current_event = None
            for line in response.iter_lines():
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:") and current_event == "cost_summary":
                    try:
                        cost_summary = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        pass

        assert cost_summary is not None, "Cost summary event should be present"
        assert "total_cost" in cost_summary
        assert cost_summary["total_cost"] > 0, "Cost should be greater than 0"


class TestErrorHandling:
    """Test error handling in production."""

    def test_invalid_conversation_id(self, auth_client: httpx.Client, requires_auth):
        """Invalid conversation ID is handled gracefully."""
        with auth_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "Test",
                "conversation_id": "invalid-uuid-format"
            },
            timeout=30.0
        ) as response:
            # Should return error status or handle gracefully
            # Implementation may vary - could be 400, 404, or stream error event
            pass  # Just verify no crash

    def test_very_long_message(self, auth_client: httpx.Client, requires_auth):
        """Very long message is handled (may be truncated or rejected)."""
        # Create conversation first
        conv_response = auth_client.post("/conversations")
        if conv_response.status_code not in (200, 201):
            pytest.skip("Cannot create conversation")
        conversation_id = conv_response.json()["id"]

        try:
            long_message = "Test " * 10000  # ~50k characters

            response = auth_client.post(
                "/chat/stream",
                json={
                    "message": long_message,
                    "conversation_id": conversation_id
                },
                timeout=30.0
            )
            # Should either process or return validation error
            assert response.status_code in (200, 400, 422, 413)
        finally:
            auth_client.delete(f"/conversations/{conversation_id}")
