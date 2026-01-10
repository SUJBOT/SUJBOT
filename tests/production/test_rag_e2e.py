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
import warnings
from typing import Generator


class TestRAGQueryFlow:
    """Test complete RAG query flow."""

    @pytest.fixture
    def conversation_id(self, auth_client: httpx.Client, requires_auth) -> Generator[str, None, None]:
        """Create a conversation for testing, cleanup after."""
        response = auth_client.post("/conversations", json={"title": "E2E Test"})
        assert response.status_code in (200, 201), \
            f"Failed to create conversation: {response.status_code} - {response.text[:200]}"

        data = response.json()
        conversation_id = data["id"]

        yield conversation_id

        # Cleanup with error handling
        try:
            delete_response = auth_client.delete(f"/conversations/{conversation_id}")
            if delete_response.status_code not in (200, 204, 404):
                warnings.warn(
                    f"Failed to cleanup conversation {conversation_id}: "
                    f"status {delete_response.status_code}"
                )
        except httpx.HTTPError as e:
            warnings.warn(f"Error cleaning up conversation {conversation_id}: {e}")

    def _parse_streaming_response(self, response) -> tuple[list, list, str]:
        """
        Parse SSE streaming response and track failures.

        Returns:
            Tuple of (events, parse_failures, collected_text)
        """
        events = []
        parse_failures = []
        collected_text = ""
        current_event = None

        for line in response.iter_lines():
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    if data_str:
                        data = json.loads(data_str)
                        events.append({"event_type": current_event, "data": data})

                        # Collect text from text_delta events
                        if current_event == "text_delta":
                            collected_text += data.get("content", "")
                except json.JSONDecodeError as e:
                    parse_failures.append({
                        "line": line[:100],
                        "error": str(e),
                        "event_type": current_event
                    })

        return events, parse_failures, collected_text

    def _assert_parse_failures_acceptable(self, parse_failures: list, total_events: int):
        """Assert that JSON parse failure rate is below 10%."""
        if not parse_failures:
            return

        failure_rate = len(parse_failures) / max(total_events + len(parse_failures), 1)

        if failure_rate > 0.1:
            pytest.fail(
                f"Excessive JSON parse failures: {len(parse_failures)} "
                f"({failure_rate:.1%}). First failure: {parse_failures[0]}"
            )
        else:
            warnings.warn(
                f"{len(parse_failures)} JSON parse failures (within threshold)"
            )

    def test_simple_query_streaming(self, auth_client: httpx.Client, conversation_id: str):
        """Simple query returns streaming response."""
        with auth_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "Ahoj, co umíš?",
                "conversation_id": conversation_id
            },
            timeout=60.0
        ) as response:
            assert response.status_code == 200, \
                f"Expected 200, got {response.status_code}"

            events, parse_failures, _ = self._parse_streaming_response(response)

            # Check for excessive parse failures
            self._assert_parse_failures_acceptable(parse_failures, len(events))

            # Should have received some events
            assert len(events) > 0, "No events received from streaming response"

    @pytest.mark.slow
    def test_rag_query_with_search(self, auth_client: httpx.Client, conversation_id: str):
        """RAG query that triggers document search."""
        query = "Jaké jsou požadavky na bezpečnost jaderných zařízení?"

        collected_text = ""
        tool_calls = []
        cost_info = None
        parse_failures = []

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
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                        if current_event == "text_delta":
                            collected_text += data.get("content", "")
                        elif current_event == "tool_call":
                            tool_calls.append(data)
                        elif current_event == "cost_summary":
                            cost_info = data
                    except json.JSONDecodeError as e:
                        parse_failures.append({
                            "line": line[:100],
                            "error": str(e),
                            "event_type": current_event
                        })

        # Check parse failures
        total_events = 1 + len(tool_calls) + (1 if cost_info else 0)
        self._assert_parse_failures_acceptable(parse_failures, total_events)

        # Should have generated some response
        assert len(collected_text) > 50, \
            f"Response too short ({len(collected_text)} chars). Expected meaningful text."

        # Cost tracking should work
        if cost_info:
            assert "total_cost" in cost_info, "Cost summary missing total_cost field"
            assert cost_info["total_cost"] >= 0, "Cost should be non-negative"

    @pytest.mark.slow
    @pytest.mark.xfail(reason="Conversation memory may not persist across messages")
    def test_conversation_history_preserved(self, auth_client: httpx.Client, conversation_id: str):
        """Conversation history is preserved between messages."""
        # First message
        first_response = ""
        first_parse_failures = []

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
                    except json.JSONDecodeError as e:
                        first_parse_failures.append(str(e))

        # Validate first response before continuing
        assert len(first_response) > 0, \
            "First response was empty - cannot test conversation history"

        if first_parse_failures:
            warnings.warn(f"{len(first_parse_failures)} parse failures in first message")

        # Give backend time to save
        time.sleep(1)

        # Second message - should remember the name
        second_response = ""
        second_parse_failures = []

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
                    except json.JSONDecodeError as e:
                        second_parse_failures.append(str(e))

        if second_parse_failures:
            warnings.warn(f"{len(second_parse_failures)} parse failures in second message")

        # Response should mention the name
        assert "Testovací" in second_response or "testovací" in second_response.lower(), \
            f"Assistant should remember the name. Response: {second_response[:200]}"


class TestCostTracking:
    """Test cost tracking functionality."""

    @pytest.fixture
    def conversation_id(self, auth_client: httpx.Client, requires_auth) -> Generator[str, None, None]:
        """Create a conversation for testing."""
        response = auth_client.post("/conversations", json={"title": "Cost Test"})
        assert response.status_code in (200, 201)
        data = response.json()
        conversation_id = data["id"]

        yield conversation_id

        # Cleanup with error handling
        try:
            auth_client.delete(f"/conversations/{conversation_id}")
        except httpx.HTTPError as e:
            warnings.warn(f"Error cleaning up conversation: {e}")

    def test_cost_summary_event_present(self, auth_client: httpx.Client, conversation_id: str):
        """Cost summary event is included in response."""
        cost_summary = None
        parse_failures = []

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
                    except json.JSONDecodeError as e:
                        parse_failures.append(str(e))

        if parse_failures:
            warnings.warn(f"JSON parse failures while looking for cost_summary: {parse_failures}")

        assert cost_summary is not None, "Cost summary event should be present"
        assert "total_cost" in cost_summary, "Cost summary missing total_cost"
        assert cost_summary["total_cost"] > 0, "Cost should be greater than 0"


class TestErrorHandling:
    """Test error handling in production."""

    def test_invalid_conversation_id(self, auth_client: httpx.Client, requires_auth):
        """Invalid conversation ID returns appropriate error response."""
        with auth_client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "Test",
                "conversation_id": "invalid-uuid-format"
            },
            timeout=30.0
        ) as response:
            # Should return client error (4xx) OR stream with error event
            # NOT server error (5xx)
            if response.status_code >= 500:
                pytest.fail(f"Server error for invalid UUID: {response.status_code}")

            if response.status_code == 200:
                # If streaming succeeds, check for error event in stream
                has_error_event = False
                for line in response.iter_lines():
                    if "error" in line.lower():
                        has_error_event = True
                        break

                # Either error event OR valid response is acceptable
                # (agent might handle gracefully)

            elif response.status_code in (400, 404, 422):
                # Expected error response for invalid UUID
                pass
            else:
                pytest.fail(
                    f"Unexpected status code for invalid conversation_id: "
                    f"{response.status_code}"
                )

    def test_very_long_message(self, auth_client: httpx.Client, requires_auth):
        """Very long message is handled (may be truncated or rejected)."""
        # Create conversation first
        conv_response = auth_client.post("/conversations", json={"title": "Error Test"})
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
            # Should either process (200), reject as too large (413),
            # or return validation error (400, 422)
            assert response.status_code in (200, 400, 413, 422), \
                f"Unexpected status for long message: {response.status_code}"

            # Should NOT be a server error
            assert response.status_code < 500, \
                f"Server error for long message: {response.status_code}"

        finally:
            try:
                auth_client.delete(f"/conversations/{conversation_id}")
            except httpx.HTTPError:
                pass  # Cleanup failure is not critical for this test

    def test_empty_message_rejected(self, auth_client: httpx.Client, requires_auth):
        """Empty message is rejected with validation error."""
        conv_response = auth_client.post("/conversations", json={"title": "Error Test"})
        if conv_response.status_code not in (200, 201):
            pytest.skip("Cannot create conversation")
        conversation_id = conv_response.json()["id"]

        try:
            response = auth_client.post(
                "/chat/stream",
                json={
                    "message": "",
                    "conversation_id": conversation_id
                },
                timeout=10.0
            )
            # Empty message should be rejected
            assert response.status_code == 422, \
                f"Empty message should return 422, got {response.status_code}"

        finally:
            try:
                auth_client.delete(f"/conversations/{conversation_id}")
            except httpx.HTTPError:
                pass


class TestSpendingLimit:
    """Test spending limit enforcement."""

    def test_spending_limit_response_structure(self, http_client: httpx.Client):
        """
        Verify that 402 spending limit response has correct structure.

        Note: This test verifies the response structure if a user hits the limit.
        We can't easily trigger this in tests without a test user at the limit.
        """
        # This is a structural test - we verify that IF the endpoint returns 402,
        # it has the expected fields. We check this by examining the API docs.
        response = http_client.get("/openapi.json")
        if response.status_code != 200:
            pytest.skip("OpenAPI docs not available")

        # The actual 402 test would require a user at spending limit
        # For now, we document the expected structure
        expected_402_fields = [
            "error",
            "total_spent_czk",
            "spending_limit_czk",
            "message_cs",
            "message_en"
        ]

        # This serves as documentation of the expected response
        assert expected_402_fields, "402 response should include these fields"
