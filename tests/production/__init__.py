"""
Production tests for SUJBOT.

These tests verify that the application works correctly in production.
They require running services (Docker containers) and real API keys.

Usage:
    # Run all production tests
    uv run pytest tests/production/ -v

    # Run only health checks (fast)
    uv run pytest tests/production/test_health.py -v

    # Run smoke tests (requires auth)
    uv run pytest tests/production/test_api_smoke.py -v

    # Run E2E tests (slow, requires indexed data)
    uv run pytest tests/production/test_rag_e2e.py -v

Environment:
    PROD_BASE_URL: Base URL for API (default: http://localhost:8000)
    PROD_TEST_USER: Test user email
    PROD_TEST_PASSWORD: Test user password
"""
