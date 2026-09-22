"""Master supervisor: decides which specialist handles each turn.

Routing is *per turn*, not per session, which is what makes mid-conversation handoffs work: a
question asked during a viva is classified as a doubt and sent to DoubtAgent, while `viva_score`
(untouched by that agent) stays in state so the viva resumes exactly where it left off.

Decision order:
  1. Explicit `action` from the API (generate_notes, submit_test, ...) -> deterministic, no LLM call.
  2. Viva in progress, request hinted at the viva tab, and the message isn't question-shaped
     -> it's an answer; skip the LLM call.
  3. Otherwise an LLM intent classifier returns a schema-validated `RouteDecision`.
"""
import re
from typing import Literal, Optional

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, field_validator

from backend.app.graph.common import last_human, last_human_text
from backend.app.graph.llm import ainvoke_json, get_llm, message_text
from backend.app.graph.resilience import guarded, has_error
from backend.app.graph.state import viva_of

ACTION_ROUTES: dict[str, tuple[str, str]] = {
    "generate_notes": ("notes", "notes_prepare"),
    "generate_test": ("test", "test_generate"),
    "submit_test": ("test", "test_grade"),
    "viva_start": ("viva", "viva_ask"),
    "viva_end": ("viva", "viva_end"),
}

_QUESTIONISH = re.compile(
    r"^\s*(what|why|how|when|where|who|which|can|could|would|should|is|are|does|do|explain|define|"
    r"clarify|tell me|help)\b",
    re.IGNORECASE,
)


def looks_like_question(text: str) -> bool:
    return "?" in text or bool(_QUESTIONISH.match(text))


class RouteDecision(BaseModel):
    agent: Literal["doubt", "viva", "test", "notes"]
    viva_action: Optional[Literal["start", "answer", "end"]] = None
    reason: str = ""
    num_questions: Optional[int] = None
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None

    @field_validator("num_questions", mode="before")
    @classmethod
    def _clip_count(cls, v):
        try:
            return None if v is None else max(1, min(30, int(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("difficulty", "viva_action", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return None if v in ("", "none", "null") else v


def _viva_target(state: dict, viva_action: Optional[str]) -> tuple[str, bool]:
    """Concrete viva node for this turn, and whether it starts a fresh session."""
    viva = viva_of(state)
    in_progress = bool(viva["active"] and viva["pending_question"])
    if viva_action == "end":
        return "viva_end", False
    if viva_action == "start":
        return "viva_ask", True
    if viva_action == "answer" or (viva_action is None and in_progress):
        return ("viva_evaluate", False) if in_progress else ("viva_ask", True)
    return ("viva_ask", True)


def _classifier_prompt(state: dict, text: str, hint: Optional[str]) -> str:
    viva = viva_of(state)
    last_ai = ""
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", "") == "ai" and message_text(m).strip():
            last_ai = message_text(m).strip()[:300]
            break
    return f"""You are the router for SmartLearn.AI, a platform where a student studies a YouTube lecture.
Pick the single specialist that should handle the student's latest message.

Specialists:
- "doubt": questions about the lecture or related concepts, explanations, calculations, greetings, thanks, anything else.
- "viva": oral-exam practice. viva_action is "start" (begin/restart a viva), "answer" (the student is answering the examiner's current question), or "end" (stop the viva / give the summary).
- "test": create a multiple-choice quiz. Extract num_questions (1-30) and difficulty (easy|medium|hard) if stated.
- "notes": generate lecture notes / a summary document.

Context:
- Viva in progress: {viva["active"]}
- Examiner's pending question: {viva["pending_question"] or "none"}
- The student's active tab (a hint, not binding): {hint or "none"}
- Tutor's previous reply: {last_ai or "none"}

IMPORTANT: if a viva is in progress but the student asks their own question (e.g. "what does backpropagation mean?"), route to "doubt" -- the viva stays paused and resumes later. Only route to "viva"/"answer" when the message is an attempt to answer the pending question.

Student's message: {text!r}

Respond with ONLY a JSON object: {{"agent": "doubt|viva|test|notes", "viva_action": "start|answer|end|null", "num_questions": null, "difficulty": null, "reason": "<short reason>"}}"""


@guarded("supervisor")
async def supervisor_node(state: dict, config: RunnableConfig) -> dict:
    request = state.get("request") or {}
    text = last_human_text(state)
    action = request.get("action")
    hint = request.get("agent")
    viva = viva_of(state)

    route: dict
    if action in ACTION_ROUTES:
        agent, target = ACTION_ROUTES[action]
        route = {"agent": agent, "target": target, "source": "action", "reason": f"action={action}",
                 "fresh": action == "viva_start"}
    elif viva["active"] and viva["pending_question"] and hint == "viva" and not looks_like_question(text):
        route = {"agent": "viva", "target": "viva_evaluate", "source": "heuristic",
                 "reason": "answer to the pending viva question", "fresh": False}
    else:
        llm = get_llm("supervisor", temperature=0.0, max_tokens=200,
                      fallback=bool(state.get("use_fallback")), json_mode=True)
        decision = await ainvoke_json(llm, [HumanMessage(content=_classifier_prompt(state, text, hint))], RouteDecision, config)
        if decision.agent == "viva":
            target, fresh = _viva_target(state, decision.viva_action)
        else:
            target, fresh = {"doubt": "doubt_agent", "test": "test_generate", "notes": "notes_prepare"}[decision.agent], False
        route = {"agent": decision.agent, "target": target, "source": "llm", "reason": decision.reason,
                 "fresh": fresh, "num_questions": decision.num_questions, "difficulty": decision.difficulty}

    update: dict = {"active_agent": route["agent"], "route": route}

    # Tag the user's message with the agent that handled it so per-agent history can be reconstructed.
    orig = last_human(state)
    if orig is not None and orig.id:
        update["messages"] = [HumanMessage(
            content=orig.content, id=orig.id,
            additional_kwargs={**orig.additional_kwargs, "agent": route["agent"]},
        )]

    await adispatch_custom_event(
        "route", {"agent": route["agent"], "target": route["target"], "reason": route["reason"], "source": route["source"]},
        config=config,
    )
    return update


SUPERVISOR_TARGETS = (
    "doubt_agent", "viva_ask", "viva_evaluate", "viva_end", "test_generate", "test_grade", "notes_prepare",
)


def route_supervisor(state: dict) -> str:
    if has_error(state):
        return "retry_gate"
    return (state.get("route") or {}).get("target", "doubt_agent")
