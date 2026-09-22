"""Unified state schema for the SmartLearn multi-agent graph.

One `AgentState` flows through every node. It is persisted by the LangGraph checkpointer under
`thread_id == session_id`, so each learning session owns an isolated, durable copy of it.
"""
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_worker_errors(current: Optional[dict], update: Optional[dict]) -> dict:
    """Reducer for errors reported by the parallel notes workers.

    Workers run in the same superstep and each writes only its own key. The `__reset__` marker
    (written by `notes_prepare`) starts a fresh notes run.
    """
    if not update:
        return dict(current or {})
    if update.get("__reset__"):
        return {k: v for k, v in update.items() if k != "__reset__"}
    merged = dict(current or {})
    merged.update(update)
    return merged


class VivaScore(TypedDict, total=False):
    asked: int              # questions that have been scored
    correct: int            # questions scored as (mostly) correct
    points: float           # sum of per-question points (0..1 each)
    active: bool            # a viva session is in progress
    pending_question: str   # the question currently awaiting the student's answer
    history: list[dict]     # [{question, answer, verdict, points}]


class AgentState(TypedDict, total=False):
    # --- conversation ---------------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str

    # --- active context metadata (loaded once per thread from the session record) ---
    transcript: str
    frame_urls: list[str]
    meta: dict                      # {video_id, title, transcript_chars, frame_count}

    # --- per-turn request + routing --------------------------------------
    request: dict                   # {action, agent, params} supplied by the API for this turn
    active_agent: str               # doubt | viva | test | notes
    route: dict                     # supervisor decision {agent, reason, target, ...}

    # --- viva -----------------------------------------------------------
    viva_score: VivaScore           # persists across agent handoffs
    viva_pending_eval: Optional[dict]   # provisional evaluation awaiting the human gate
    viva_elaboration: str           # the student's clarification, injected on resume
    viva_continue: bool             # after scoring, ask the next question
    audio_url: str                  # TTS rendering of the latest viva question

    # --- test -----------------------------------------------------------
    current_quiz: Optional[dict]    # {questions, answer_key, difficulty}
    test_result: Optional[dict]

    # --- notes (fan-out / fan-in) ----------------------------------------
    notes_draft: str                # merged, final notes markdown
    notes_segments: list[dict]      # transcript windows shared by the parallel workers
    notes_summary: Optional[dict]   # SummarizerNode output
    notes_latex: Optional[dict]     # LaTeXNode output
    notes_diagrams: Optional[dict]  # DiagramMapperNode output
    worker_errors: Annotated[dict, merge_worker_errors]

    # --- resilience -------------------------------------------------------
    error: Optional[dict]           # {node, kind, message, retry_after}
    retry_count: int
    use_fallback: bool
    tool_rounds: int


def viva_of(state: dict[str, Any]) -> VivaScore:
    """Current viva score with defaults filled in (never mutates state)."""
    v = dict(state.get("viva_score") or {})
    v.setdefault("asked", 0)
    v.setdefault("correct", 0)
    v.setdefault("points", 0.0)
    v.setdefault("active", False)
    v.setdefault("pending_question", "")
    v.setdefault("history", [])
    return v  # type: ignore[return-value]
