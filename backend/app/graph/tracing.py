"""LangSmith tracing wiring.

LangChain/LangGraph read `LANGCHAIN_*` / `LANGSMITH_*` from the *process environment*, but pydantic
settings only read `.env` into the Settings object. `configure_tracing` bridges the two, and
`run_config` attaches session-level tags/metadata to every graph run so traces are filterable.
"""
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def configure_tracing(settings) -> dict:
    """Export tracing settings to os.environ. Returns {"enabled": bool, "project": str, "reason": str}."""
    requested = str(settings.LANGCHAIN_TRACING_V2).strip().lower() in _TRUTHY
    api_key = settings.LANGCHAIN_API_KEY or os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY", "")

    enabled, reason = requested, "enabled"
    if requested and not api_key:
        enabled, reason = False, "LANGCHAIN_TRACING_V2 is true but no LANGCHAIN_API_KEY is set"
        logger.warning("LangSmith tracing disabled: %s", reason)
    elif not requested:
        reason = "disabled by configuration"

    flag = "true" if enabled else "false"
    env: dict[str, str] = {
        "LANGCHAIN_TRACING_V2": flag,
        "LANGSMITH_TRACING": flag,
        "LANGCHAIN_PROJECT": settings.LANGCHAIN_PROJECT,
        "LANGSMITH_PROJECT": settings.LANGCHAIN_PROJECT,
    }
    if api_key:
        env["LANGCHAIN_API_KEY"] = api_key
        env["LANGSMITH_API_KEY"] = api_key
    if settings.LANGCHAIN_ENDPOINT:
        env["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
        env["LANGSMITH_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ.update(env)
    return {"enabled": enabled, "project": settings.LANGCHAIN_PROJECT, "reason": reason}


def run_config(session_id: str, **metadata: Any) -> dict:
    """RunnableConfig for one graph run: thread isolation + trace tags/metadata."""
    return {
        "configurable": {"thread_id": session_id},
        "run_name": "smartlearn-agent",
        "tags": ["smartlearn", f"session:{session_id}"],
        "metadata": {"session_id": session_id, "thread_id": session_id, **metadata},
        "recursion_limit": 40,
    }
