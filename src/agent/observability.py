"""
LangSmith observability setup.

Extracted from multi_agent/observability/langsmith_integration.py.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional
import uuid

logger = logging.getLogger(__name__)


class LangSmithIntegration:
    """LangSmith integration for agent observability."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.api_key = os.getenv("LANGSMITH_API_KEY") or config.get("api_key")
        self.project_name = (
            os.getenv("LANGSMITH_PROJECT_NAME")
            or config.get("project_name", "sujbot-multi-agent")
        )
        self.endpoint = (
            os.getenv("LANGSMITH_ENDPOINT")
            or config.get("endpoint", "https://api.smith.langchain.com")
        )
        self.trace_logging_level = config.get("trace_logging_level", "INFO")
        self.sample_rate = config.get("sample_rate", 1.0)

        self._tracing_enabled = False
        self._client = None
        self._current_run_id: Optional[str] = None

    def setup(self) -> bool:
        if not self.enabled:
            logger.info("LangSmith tracing disabled in configuration")
            return False

        if not self.api_key:
            logger.warning(
                "LangSmith API key not provided, tracing disabled. "
                "Set LANGSMITH_API_KEY environment variable to enable."
            )
            return False

        try:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint

            if self.sample_rate < 1.0:
                os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = str(self.sample_rate)

            self._tracing_enabled = True

            try:
                from langsmith import Client

                self._client = Client()
            except ImportError:
                logger.warning("langsmith package not installed, feedback submission disabled.")
            except Exception as e:
                logger.warning(f"LangSmith client initialization failed: {e}")

            logger.info(
                f"LangSmith tracing enabled: project={self.project_name}, "
                f"endpoint={self.endpoint}, sample_rate={self.sample_rate}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to set up LangSmith: {e}", exc_info=True)
            return False

    def is_enabled(self) -> bool:
        return self._tracing_enabled

    @contextmanager
    def trace_workflow(self, workflow_name: str, metadata: Optional[Dict] = None):
        if not self._tracing_enabled:
            yield None
            return

        run_id = str(uuid.uuid4())
        self._current_run_id = run_id
        os.environ["LANGCHAIN_RUN_ID"] = run_id

        try:
            yield run_id
        finally:
            os.environ.pop("LANGCHAIN_RUN_ID", None)

    def get_current_run_id(self) -> Optional[str]:
        return self._current_run_id

    def disable(self) -> None:
        if self._tracing_enabled:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
            os.environ.pop("LANGCHAIN_PROJECT", None)
            os.environ.pop("LANGCHAIN_TRACING_SAMPLING_RATE", None)
            self._tracing_enabled = False


def setup_langsmith(config: Dict[str, Any]) -> Optional[LangSmithIntegration]:
    """
    Set up LangSmith integration from configuration.

    Args:
        config: Config dict (expects a "langsmith" sub-key or direct langsmith config)

    Returns:
        LangSmithIntegration instance or None if disabled
    """
    langsmith_config = config if "enabled" in config else config.get("langsmith", {})

    try:
        integration = LangSmithIntegration(langsmith_config)
        if integration.setup():
            logger.info("LangSmith integration set up successfully")
            return integration
        else:
            return None
    except Exception as e:
        logger.error(f"Failed to set up LangSmith integration: {e}", exc_info=True)
        return None
