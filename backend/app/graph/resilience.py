"""Graph-level resilience: turn node failures into state, then route on it.

Nodes decorated with `@guarded(name)` never raise. A failure is written to `state["error"]`
(classified as rate_limit / malformed / transient / fatal) and the node's conditional edge sends the
run to `retry_gate`, which either loops back to the failed node -- on the same model, or on the
fallback model -- or hands over to `error_handler` once retries are exhausted. This keeps the
recovery policy visible in the graph (and in LangSmith traces) instead of buried in try/except blocks.
"""
import asyncio
import logging
from functools import wraps
from typing import Awaitable, Callable, Literal

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END
from langgraph.types import Command

from backend.app.graph.common import emit_text, tagged_ai, turn_has_output
from backend.app.graph.llm import classify_error, fallback_available, learn_from_error, retry_after_seconds
from backend.config import get_settings

logger = logging.getLogger(__name__)

NodeFn = Callable[[dict, dict], Awaitable[dict]]

USER_MESSAGES = {
    "too_large": "That request was too large for the AI service to handle. Try a shorter question or a smaller request.",
    "rate_limit": "The AI service is rate-limited right now. Please try again in a moment.",
    "malformed": "I couldn't produce a well-formed answer this time. Please try again or rephrase your request.",
    "transient": "I couldn't reach the AI service. Please try again shortly.",
    "fatal": "Something went wrong while processing your request. Please try again.",
}


def guarded(node_name: str, retry_at: str | None = None) -> Callable[[NodeFn], NodeFn]:
    """Convert exceptions raised by a node into an `error` state update.

    `retry_at` names the node the retry loop should re-enter when it differs from the failing node
    (e.g. the notes merger retries the whole fan-out from `notes_prepare`).
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        async def wrapper(state: dict, config: RunnableConfig) -> dict:
            try:
                return await fn(state, config)
            except GraphBubbleUp:
                raise  # interrupts / parent commands are control flow, not failures
            except Exception as exc:  # noqa: BLE001 - deliberate catch-all; classified below
                learn_from_error(exc)
                kind = classify_error(exc)
                logger.warning("node %s failed (%s): %s", node_name, kind, exc)
                if kind == "fatal":
                    logger.exception("fatal error in node %s", node_name)
                return {
                    "error": {
                        "node": retry_at or node_name,
                        "kind": kind,
                        "message": str(exc)[:500],
                        "retry_after": retry_after_seconds(exc),
                    }
                }

        return wrapper

    return decorator


def has_error(state: dict) -> bool:
    return bool(state.get("error"))


def route_end_or_retry(state: dict) -> str:
    """Conditional edge for nodes with nothing after them: finish, or hand a failure to the retry gate."""
    return "retry_gate" if has_error(state) else END


# After the human gate has been passed, retries must not re-trigger the interrupt.
RETRY_ALIASES = {"viva_score": "viva_score_auto"}

RETRY_TARGETS = Literal[
    "supervisor", "doubt_agent", "viva_ask", "viva_evaluate", "viva_score", "viva_score_auto",
    "test_generate", "notes_prepare", "error_handler",
]


async def retry_gate(state: dict, config: RunnableConfig) -> Command[RETRY_TARGETS]:
    """Decide whether a failed node is retried (same model / fallback model) or the run gives up."""
    settings = get_settings()
    err = state.get("error") or {}
    kind = err.get("kind", "fatal")
    retries = int(state.get("retry_count") or 0)

    if kind == "fatal" or retries >= settings.LLM_MAX_RETRIES:
        return Command(goto="error_handler")

    if kind == "too_large":
        # max_tokens has just been lowered to fit the limit the API reported: retry straight away.
        use_fallback, delay = bool(state.get("use_fallback")), 0.0
    elif kind == "rate_limit" and fallback_available():
        # A different model has its own quota bucket, so there's nothing to wait for.
        use_fallback, delay = True, 0.0
    else:
        # Same model again after a backoff (or the API's Retry-After); later attempts prefer the fallback.
        use_fallback = bool(state.get("use_fallback")) or (retries >= 1 and fallback_available())
        delay = err.get("retry_after")
        if delay is None:
            delay = settings.LLM_RETRY_BACKOFF * (2 ** retries)
        delay = min(float(delay), 30.0)

    await adispatch_custom_event(
        "retry",
        {"node": err.get("node"), "kind": kind, "attempt": retries + 1, "fallback": use_fallback},
        config=config,
    )
    if delay > 0:
        await asyncio.sleep(delay)

    return Command(
        goto=RETRY_ALIASES.get(err["node"], err["node"]),
        update={"retry_count": retries + 1, "use_fallback": use_fallback, "error": None},
    )


async def error_handler(state: dict, config: RunnableConfig) -> dict:
    """Terminal node: tell the user what happened, in plain language, and clear the error."""
    err = state.get("error") or {}
    kind = err.get("kind", "fatal")
    text = USER_MESSAGES.get(kind, USER_MESSAGES["fatal"])
    await emit_text(config, "error_handler", text, separate=turn_has_output(state))
    agent = state.get("active_agent") or "doubt"
    return {
        "messages": [tagged_ai(text, agent, error=kind, failed_node=err.get("node"))],
        "error": None,
    }
