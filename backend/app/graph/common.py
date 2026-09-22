"""Helpers shared by the graph nodes."""
import re
from typing import Any, Optional

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from backend.app.graph.llm import message_text

# Tag on LLM calls whose tokens should be streamed to the end user (vs. internal JSON/classification calls).
USER_STREAM_TAG = "user_stream"

_TOKEN_SPLIT = re.compile(r"\S+\s*|\s+")


def last_human(state: dict) -> Optional[HumanMessage]:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m
    return None


def last_human_text(state: dict) -> str:
    m = last_human(state)
    return message_text(m) if m else ""


def turn_messages(state: dict) -> list[AnyMessage]:
    """Messages belonging to the current turn (from the latest human message onward)."""
    msgs = list(state.get("messages") or [])
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            return msgs[i:]
    return msgs


def turn_has_output(state: dict) -> bool:
    """True if an earlier node in this turn already produced user-visible text."""
    return any(
        isinstance(m, AIMessage) and not m.tool_calls and message_text(m).strip()
        for m in turn_messages(state)
    )


def agent_history(state: dict, agent: str, limit: int = 8) -> list[AnyMessage]:
    """Recent messages this agent took part in (excluding the current turn), oldest first."""
    prior = [m for m in (state.get("messages") or [])]
    turn = turn_messages(state)
    prior = prior[: len(prior) - len(turn)]
    mine = [
        m for m in prior
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
        and m.additional_kwargs.get("agent") == agent
        and message_text(m).strip()
    ]
    return mine[-limit:]


def tagged_ai(content: str, agent: str, **extra: Any) -> AIMessage:
    return AIMessage(content=content, additional_kwargs={"agent": agent, **extra})


def transcript_context(state: dict, max_chars: int = 15000) -> str:
    transcript = state.get("transcript") or ""
    if not transcript:
        return "No transcript available."
    if len(transcript) <= max_chars:
        return transcript
    return transcript[:max_chars] + "\n[... transcript truncated ...]"


def system(text: str) -> SystemMessage:
    return SystemMessage(content=text)


async def emit_text(config: dict, node: str, text: str, *, separate: bool = False) -> None:
    """Stream already-known text to the client as word-sized token events.

    LLM output streams itself (through `on_chat_model_stream`); this is for templated text
    (evaluation feedback, summaries) so the client sees one uniform token stream.
    """
    if separate:
        await adispatch_custom_event("token", {"text": "\n\n", "node": node}, config=config)
    for chunk in _TOKEN_SPLIT.findall(text):
        await adispatch_custom_event("token", {"text": chunk, "node": node}, config=config)


def is_tool_or_empty(m: AnyMessage) -> bool:
    return isinstance(m, ToolMessage) or bool(getattr(m, "tool_calls", None)) or not message_text(m).strip()
