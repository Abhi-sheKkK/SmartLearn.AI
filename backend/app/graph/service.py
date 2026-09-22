"""Runs the compiled graph for one API request and translates it into client-facing events.

Responsibilities kept out of the router and out of the graph nodes:
  * per-thread serialisation (two requests for the same session never interleave on one checkpoint)
  * human-in-the-loop plumbing: detect a run paused at the scoring gate, and apply a `resume`
  * mapping LangGraph `astream_events` to a small SSE event vocabulary

Event types: start | route | token | tool_start | tool_end | retry | audio | notes_progress |
             interrupt | error | final   (the stream always ends after `final` or `error`)
"""
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.graph.common import USER_STREAM_TAG
from backend.app.graph.llm import classify_error, message_text
from backend.app.graph.quiz import public_quiz
from backend.app.graph.tracing import run_config
from backend.app.models import AgentRequest

GATE_NODE = "viva_score"

ACTION_TEXT = {
    "generate_notes": "Generate comprehensive lecture notes",
    "generate_test": "Generate an MCQ test",
    "submit_test": "Submit my test answers",
    "viva_start": "Start a viva session",
    "viva_end": "End the viva session and give me a summary",
}


class AgentRequestError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class RunPlan:
    config: dict
    graph_input: Optional[dict]
    pre_updates: list[tuple[dict, Optional[str]]] = field(default_factory=list)
    start_count: int = 0
    resumed: bool = False


def _truncate(value: Any, limit: int = 600) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def join_contents(parts: list[str]) -> str:
    out = ""
    for part in parts:
        if not part.strip():
            continue
        out += part if (not out or part.startswith("\n")) else "\n\n" + part
    return out.strip()


def map_event(ev: dict) -> Optional[dict]:
    """LangGraph astream_events (v2) -> client event, or None if the client shouldn't see it."""
    kind, name = ev["event"], ev.get("name", "")
    if kind == "on_chat_model_stream" and USER_STREAM_TAG in (ev.get("tags") or []):
        text = message_text(ev["data"]["chunk"])
        if text:
            return {"type": "token", "text": text, "node": (ev.get("metadata") or {}).get("langgraph_node")}
    elif kind == "on_custom_event":
        if name == "token":
            return {"type": "token", **ev["data"]}
        if name in ("route", "retry", "audio", "notes_progress"):
            return {"type": name, **ev["data"]}
    elif kind == "on_tool_start":
        return {"type": "tool_start", "name": name, "input": _truncate(ev["data"].get("input"))}
    elif kind == "on_tool_end":
        out = ev["data"].get("output")
        return {"type": "tool_end", "name": name, "output": _truncate(getattr(out, "content", out))}
    return None


class AgentService:
    def __init__(self, graph):
        self.graph = graph
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    # ------------------------------------------------------------------ planning

    async def prepare(self, session_id: str, req: AgentRequest) -> RunPlan:
        """Validate the request against the thread's current state. Raises AgentRequestError."""
        config = run_config(session_id)
        snap = await self.graph.aget_state(config)
        values = snap.values or {}
        prior = values.get("messages") or []
        paused = bool(snap.next) and GATE_NODE in snap.next
        params = req.params.model_dump()

        elaboration: Optional[str] = None
        if req.resume is not None:
            if not paused:
                raise AgentRequestError(409, "Nothing to resume: this session is not waiting for input.")
            elaboration = req.resume.elaboration
        elif paused and req.message and req.action is None and req.agent in (None, "viva"):
            elaboration = req.message  # the student answered the clarification prompt in free text

        if elaboration is not None:
            elaboration = elaboration.strip()
            update: dict = {"viva_elaboration": elaboration}
            if elaboration:
                update["messages"] = [HumanMessage(content=elaboration, id=uuid.uuid4().hex, additional_kwargs={"agent": "viva"})]
            return RunPlan(config, None, [(update, "viva_evaluate")], start_count=len(prior), resumed=True)

        pre: list[tuple[dict, Optional[str]]] = []
        if paused:
            # New, unrelated request while the gate is open: abandon the pending evaluation.
            pre.append(({"viva_pending_eval": None, "viva_elaboration": "", "viva_continue": False}, GATE_NODE))

        text = req.message or ACTION_TEXT[req.action]  # validator guarantees one of them
        graph_input = {
            "messages": [HumanMessage(content=text, id=uuid.uuid4().hex)],
            "session_id": session_id,
            "request": {"action": req.action, "agent": req.agent, "params": params},
        }
        return RunPlan(config, graph_input, pre, start_count=len(prior))

    # ----------------------------------------------------------------- execution

    async def stream(self, session_id: str, req: AgentRequest) -> AsyncIterator[dict]:
        async with self._lock(session_id):
            try:
                plan = await self.prepare(session_id, req)
                for values, as_node in plan.pre_updates:
                    await self.graph.aupdate_state(plan.config, values, as_node=as_node)

                yield {"type": "start", "thread_id": session_id, "resumed": plan.resumed}
                async for ev in self.graph.astream_events(plan.graph_input, plan.config, version="v2"):
                    mapped = map_event(ev)
                    if mapped:
                        yield mapped

                snap = await self.graph.aget_state(plan.config)
                for event in self._closing_events(session_id, snap, plan.start_count):
                    yield event
            except AgentRequestError as e:
                yield {"type": "error", "message": e.detail, "kind": "request", "status": e.status}
            except Exception as e:  # noqa: BLE001 - never let a stream die without telling the client
                yield {"type": "error", "message": str(e)[:300], "kind": classify_error(e)}

    async def run_once(self, session_id: str, req: AgentRequest) -> dict:
        """Non-streaming variant: same execution path, returns the terminal event."""
        last: dict = {}
        async for event in self.stream(session_id, req):
            last = event
        return last

    # ----------------------------------------------------------------- projections

    def _closing_events(self, session_id: str, snap, start_count: int) -> list[dict]:
        values = snap.values or {}
        paused = bool(snap.next) and GATE_NODE in snap.next
        new = (values.get("messages") or [])[start_count:]
        assistant = [m for m in new if isinstance(m, AIMessage) and not m.tool_calls and message_text(m).strip()]
        messages = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant",
             "content": message_text(m), "agent": m.additional_kwargs.get("agent")}
            for m in new
            if isinstance(m, HumanMessage) or m in assistant
        ]
        error = next((m.additional_kwargs["error"] for m in reversed(assistant) if m.additional_kwargs.get("error")), None)
        events: list[dict] = []
        interrupt = self._interrupt_payload(values) if paused else None
        if interrupt:
            events.append({"type": "interrupt", **interrupt})
        events.append({
            "type": "final",
            "thread_id": session_id,
            "agent": values.get("active_agent"),
            "content": interrupt["prompt"] if interrupt and not assistant else join_contents([message_text(m) for m in assistant]),
            "messages": messages,
            "interrupted": bool(interrupt),
            "interrupt": interrupt,
            "error": error,
            "state": self.state_view(values, interrupt),
        })
        return events

    @staticmethod
    def _interrupt_payload(values: dict) -> Optional[dict]:
        pending = values.get("viva_pending_eval")
        if not pending:
            return None
        return {
            "node": GATE_NODE,
            "prompt": pending.get("clarification_prompt") or "Would you like to elaborate before I finalize your score?",
            "question": pending.get("question", ""),
            "answer": pending.get("answer", ""),
            "evaluation": {k: pending.get(k) for k in ("verdict", "points", "feedback", "correct_answer")},
        }

    @staticmethod
    def state_view(values: dict, interrupt: Optional[dict] = None) -> dict:
        return {
            "viva_score": values.get("viva_score") or {},
            "quiz": public_quiz(values.get("current_quiz")),
            "test_result": values.get("test_result"),
            "notes_draft": values.get("notes_draft") or "",
            "audio_url": values.get("audio_url") or "",
            "active_agent": values.get("active_agent"),
            "interrupt": interrupt,
        }

    async def snapshot(self, session_id: str) -> dict:
        snap = await self.graph.aget_state(run_config(session_id))
        values = snap.values or {}
        paused = bool(snap.next) and GATE_NODE in snap.next
        return self.state_view(values, self._interrupt_payload(values) if paused else None)

    async def history(self, session_id: str, agent: Optional[str] = None) -> list[dict]:
        snap = await self.graph.aget_state(run_config(session_id))
        out = []
        for m in (snap.values or {}).get("messages") or []:
            if not isinstance(m, (HumanMessage, AIMessage)) or getattr(m, "tool_calls", None):
                continue
            text = message_text(m)
            tag = m.additional_kwargs.get("agent")
            if not text.strip() or (agent and tag != agent):
                continue
            out.append({"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": text, "agent": tag})
        return out
