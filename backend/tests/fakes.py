"""Scripted stand-in for the Groq chat models.

`Brain` answers every LLM call by `LLMSpec.purpose` with deterministic, content-driven output, records
every call (so tests can assert which model / tools were used), and can inject failures.
"""
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx
from groq import APIStatusError, AuthenticationError, InternalServerError, NotFoundError, RateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from backend.app.graph.llm import LLMSpec


def _resp(status: int, headers: Optional[dict] = None) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.groq.test/x"), headers=headers or {})


def rate_limit_error(retry_after: str = "0") -> Exception:
    return RateLimitError("Rate limit reached for model", response=_resp(429, {"retry-after": retry_after}), body=None)


def server_error() -> Exception:
    return InternalServerError("upstream exploded", response=_resp(503), body=None)


def auth_error() -> Exception:
    return AuthenticationError("Invalid API key", response=_resp(401), body=None)


def too_large_error(model: str = "primary-model", limit: int = 1000, requested: int = 1200) -> Exception:
    return RateLimitError(
        f"Error code: 429 - Request too large for model `{model}` in organization `org_x` service tier `on_demand` "
        f"on output tokens per minute (OTPM): Limit {limit}, Requested {requested}. Reduce max_tokens.",
        response=_resp(429), body=None)


def model_missing_error(model: str = "fallback-model") -> Exception:
    return NotFoundError(f"The model `{model}` does not exist or you do not have access to it.", response=_resp(404), body=None)


def tool_use_failed() -> Exception:
    return APIStatusError("tool_use_failed: Failed to call a function", response=_resp(400), body=None)


@dataclass
class Call:
    purpose: str
    fallback: bool
    tools: bool
    json_mode: bool
    max_tokens: int
    text: str          # concatenated prompt text (for "which transcript did this call see?" assertions)
    started: float = 0.0
    ended: float = 0.0


@dataclass
class Brain:
    calls: list[Call] = field(default_factory=list)
    delays: dict[str, float] = field(default_factory=dict)
    _failures: dict[str, list] = field(default_factory=dict)
    _overrides: dict[str, Callable[[str], str]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---------------------------------------------------------------- scripting
    def fail(self, purpose: str, *excs: Exception, on_fallback: Optional[bool] = None) -> None:
        """Raise these exceptions (in order) on the next calls for `purpose`.
        on_fallback=False -> only fail the primary model; True -> only the fallback; None -> either."""
        self._failures.setdefault(purpose, []).extend((e, on_fallback) for e in excs)

    def fail_always(self, purpose: str, exc_factory: Callable[[], Exception]) -> None:
        self._failures[purpose] = [(exc_factory, None)] * 50

    def override(self, purpose: str, fn: Callable[[str], str]) -> None:
        self._overrides[purpose] = fn

    def calls_for(self, purpose: str) -> list[Call]:
        return [c for c in self.calls if c.purpose == purpose]

    def factory(self, spec: LLMSpec) -> BaseChatModel:
        return ScriptedChat(brain=self, spec=spec)

    # ---------------------------------------------------------------- behaviour
    def respond(self, spec: LLMSpec, messages: list[BaseMessage], kwargs: dict) -> AIMessage:
        text = "\n".join(str(m.content) for m in messages)
        call = Call(spec.purpose, spec.fallback, bool(kwargs.get("tools")), spec.json_mode, spec.max_tokens, text, started=time.time())
        with self._lock:
            self.calls.append(call)
            queue = self._failures.get(spec.purpose, [])
            for i, (exc, only_fb) in enumerate(queue):
                if only_fb is None or only_fb == spec.fallback:
                    queue.pop(i)
                    raise exc() if callable(exc) else exc
        delay = self.delays.get(spec.purpose, 0)
        if delay:
            time.sleep(delay)
        try:
            if spec.purpose in self._overrides:
                return AIMessage(content=self._overrides[spec.purpose](text))
            return getattr(self, f"_p_{spec.purpose}")(messages, text, kwargs)
        finally:
            call.ended = time.time()

    # -- per-purpose defaults -------------------------------------------------
    def _p_supervisor(self, messages, text, kwargs) -> AIMessage:
        m = re.search(r"Student's message: (.*)", text)
        msg = (m.group(1) if m else "").lower()
        out = {"agent": "doubt", "viva_action": None, "num_questions": None, "difficulty": None, "reason": "default"}
        if "viva" in msg and "end" in msg:
            out.update(agent="viva", viva_action="end")
        elif "start viva" in msg:
            out.update(agent="viva", viva_action="start")
        elif msg.strip("' \"").startswith("answer:"):
            out.update(agent="viva", viva_action="answer")
        elif "quiz" in msg or "test me" in msg:
            out.update(agent="test", num_questions=3, difficulty="hard")
        elif "notes" in msg:
            out.update(agent="notes")
        return AIMessage(content=json.dumps(out))

    def _p_doubt(self, messages, text, kwargs) -> AIMessage:
        human = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        tool_result = next((str(m.content) for m in reversed(messages) if isinstance(m, ToolMessage)), None)
        if kwargs.get("tools") and "calculate" in human.lower() and tool_result is None:
            return AIMessage(content="", tool_calls=[{"name": "python_repl", "args": {"code": "print(6*7)"}, "id": "call_1", "type": "tool_call"}])
        if tool_result is not None:
            return AIMessage(content=f"I verified it with Python: the result is {tool_result.strip()}.")
        return AIMessage(content=f"Here is a clear explanation for: {human}")

    def _p_viva_question(self, messages, text, kwargs) -> AIMessage:
        m = re.search(r"Do not repeat these earlier questions: \[(.*?)\]", text, re.S)
        n = (m.group(1).count("Question") if m else 0) + 1
        return AIMessage(content=f"Question {n}: explain concept number {n}?")

    def _p_viva_eval(self, messages, text, kwargs) -> AIMessage:
        ans = re.search(r"STUDENT'S ANSWER: (.*)", text)
        answer = (ans.group(1) if ans else "").lower()
        if "elaborated" in text:
            out = dict(verdict="correct", points=0.9, feedback="Thanks, that clears it up.", correct_answer="", needs_clarification=False)
        elif "partial" in answer:
            out = dict(verdict="partial", points=0.5, feedback="You have part of it.", correct_answer="The full idea is X.",
                       needs_clarification=True, clarification_prompt="Can you say more about X?")
        elif "great" in answer:
            out = dict(verdict="correct", points=1.0, feedback="Excellent answer.", correct_answer="", needs_clarification=False)
        else:
            out = dict(verdict="incorrect", points=0.1, feedback="Not quite.", correct_answer="The answer is Y.", needs_clarification=False)
        return AIMessage(content=json.dumps(out))

    def _p_quiz(self, messages, text, kwargs) -> AIMessage:
        n = int(re.search(r"Generate (\d+) multiple-choice", text).group(1))
        qs = [{"question": f"Fake question {i}-{time.time_ns()}?", "options": ["A", "B", "C", "D"],
               "correct_answer": i % 4, "explanation": f"Because {i}."} for i in range(n)]
        return AIMessage(content=json.dumps({"questions": qs}))

    def _p_notes_summary(self, messages, text, kwargs) -> AIMessage:
        ids = [int(i) for i in re.findall(r"\[Section (\d+) \|", text)]
        blocks = "\n".join(
            f"=== SECTION {i} ===\nHEADING: Heading {i}\nKEY_TERMS: term{i}\nCONTENT:\n- point about part {i}" for i in ids)
        return AIMessage(content=f"TITLE: Test Lecture\nSUMMARY: An overall summary.\n{blocks}")

    def _p_notes_latex(self, messages, text, kwargs) -> AIMessage:
        return AIMessage(content="@@ EQUATION section=1\nLATEX: E = mc^2\nEXPLAIN: Mass-energy equivalence.")


class ScriptedChat(BaseChatModel):
    brain: Any
    spec: Any

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[getattr(t, "name", str(t)) for t in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.brain.respond(self.spec, messages, kwargs)
        # fresh object every time: the add_messages reducer keys on message id
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=msg.content, tool_calls=msg.tool_calls))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.brain.respond(self.spec, messages, kwargs)
        if msg.tool_calls:
            chunks = [AIMessageChunk(content="", tool_call_chunks=[
                {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                for i, tc in enumerate(msg.tool_calls)])]
        else:
            chunks = [AIMessageChunk(content=tok) for tok in re.findall(r"\S+\s*|\s+", str(msg.content))]
        for c in chunks:
            yield ChatGenerationChunk(message=c)
