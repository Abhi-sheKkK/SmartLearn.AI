"""Error edges: 429 / malformed JSON / server errors reroute to retry, fallback model, or a graceful stop."""
import pytest

from backend.app.graph.llm import MalformedOutputError, classify_error
from backend.tests.conftest import final, run, types
from backend.tests.fakes import auth_error, model_missing_error, rate_limit_error, server_error, tool_use_failed, too_large_error


async def test_rate_limit_switches_to_the_fallback_model(service, brain, make_session):
    make_session("s")
    brain.fail("doubt", rate_limit_error(), on_fallback=False)
    events = await run(service, "s", message="What is a gradient?", agent="doubt")
    f = final(events)
    retry = next(e for e in events if e["type"] == "retry")
    assert retry == {"type": "retry", "node": "doubt_agent", "kind": "rate_limit", "attempt": 1, "fallback": True}
    assert [c.fallback for c in brain.calls_for("doubt")] == [False, True]
    assert "explanation" in f["content"].lower() and f["error"] is None


async def test_malformed_json_retries_same_model_then_falls_back(service, brain, make_session):
    make_session("s")
    calls = iter(["not json", "{still: broken", None])
    real = brain._p_supervisor
    brain.override("supervisor", lambda t: next(calls) or real([], t, {}).content)
    events = await run(service, "s", message="What is a gradient?")
    f = final(events)
    retries = [e for e in events if e["type"] == "retry"]
    assert [(r["kind"], r["fallback"]) for r in retries] == [("malformed", False), ("malformed", True)]
    assert [c.fallback for c in brain.calls_for("supervisor")] == [False, False, True]
    assert f["error"] is None and f["content"]


async def test_transient_5xx_retries_on_the_same_model_first(service, brain, make_session):
    make_session("s")
    brain.fail("doubt", server_error(), on_fallback=None)
    events = await run(service, "s", message="Explain?", agent="doubt")
    retry = next(e for e in events if e["type"] == "retry")
    assert retry["kind"] == "transient" and retry["fallback"] is False
    assert [c.fallback for c in brain.calls_for("doubt")] == [False, False]
    assert final(events)["error"] is None


async def test_exhausted_retries_end_gracefully_with_a_user_message(service, brain, make_session):
    make_session("s")
    brain.fail_always("doubt", rate_limit_error)
    events = await run(service, "s", message="Explain?", agent="doubt")
    f = final(events)
    assert f["type"] == "final" and f["error"] == "rate_limit"
    assert "rate-limited" in f["content"]
    assert len(brain.calls_for("doubt")) == 3                          # 1 attempt + LLM_MAX_RETRIES retries
    assert len([e for e in events if e["type"] == "retry"]) == 2
    # the thread is healthy afterwards: the next request works
    brain._failures.clear()
    assert final(await run(service, "s", message="Explain again?", agent="doubt"))["error"] is None


async def test_fatal_errors_are_not_retried(service, brain, make_session):
    make_session("s")
    brain.fail("doubt", auth_error())
    events = await run(service, "s", message="Explain?", agent="doubt")
    f = final(events)
    assert "retry" not in types(events) and len(brain.calls_for("doubt")) == 1
    assert f["error"] == "fatal"


async def test_supervisor_failure_is_recoverable_too(service, brain, make_session):
    make_session("s")
    brain.fail("supervisor", rate_limit_error(), on_fallback=False)
    events = await run(service, "s", message="Explain?")
    assert final(events)["error"] is None
    assert [c.fallback for c in brain.calls_for("supervisor")] == [False, True]


async def test_tools_are_dropped_on_the_last_attempt_after_malformed_tool_calls(service, brain, make_session):
    make_session("s")
    brain.fail("doubt", tool_use_failed(), tool_use_failed())
    events = await run(service, "s", message="Explain?", agent="doubt")
    assert final(events)["error"] is None
    assert [c.tools for c in brain.calls_for("doubt")] == [True, True, False]


async def test_viva_and_quiz_nodes_use_the_same_recovery_path(service, brain, make_session):
    make_session("s")
    brain.fail("viva_question", server_error())
    ev = await run(service, "s", action="viva_start")
    assert "retry" in types(ev) and final(ev)["state"]["viva_score"]["active"]

    brain.fail("quiz", rate_limit_error(), on_fallback=False)
    ev = await run(service, "s", action="generate_test", params={"num_questions": 2})
    assert "retry" in types(ev) and len(final(ev)["state"]["quiz"]["questions"]) == 2


async def test_quiz_with_only_invalid_questions_counts_as_malformed(service, brain, make_session):
    import json
    make_session("s")
    bad = json.dumps({"questions": [{"question": "q", "options": ["only", "two"], "correct_answer": 9}]})
    brain.override("quiz", lambda t: bad)
    events = await run(service, "s", action="generate_test", params={"num_questions": 2})
    f = final(events)
    assert f["error"] == "malformed" and len([e for e in events if e["type"] == "retry"]) == 2


@pytest.mark.parametrize("exc,kind", [
    (rate_limit_error(), "rate_limit"),
    (server_error(), "transient"),
    (auth_error(), "fatal"),
    (tool_use_failed(), "malformed"),
    (MalformedOutputError("bad"), "malformed"),
    (TimeoutError("slow"), "transient"),
    (ValueError("programmer error"), "fatal"),
])
def test_error_classification(exc, kind):
    assert classify_error(exc) == kind


# ------------------------------------------------------- limits learned from the API's own error messages

async def test_output_token_limit_is_learned_and_applied_to_the_retry_and_later_calls(service, brain, make_session):
    make_session("s")
    brain.fail("doubt", too_large_error(limit=1000, requested=1200))
    events = await run(service, "s", message="Explain?", agent="doubt")
    retry = next(e for e in events if e["type"] == "retry")
    assert retry["kind"] == "too_large" and retry["fallback"] is False     # same model, immediately, smaller request
    first, second = brain.calls_for("doubt")
    assert first.max_tokens == 1200 and second.max_tokens == 800           # 80% of the reported limit
    assert final(events)["error"] is None

    await run(service, "s", message="And again?", agent="doubt")
    assert brain.calls_for("doubt")[-1].max_tokens == 800                  # remembered: no more failed first attempts
    assert brain.calls_for("viva_question") == []                          # (other purposes unaffected until used)
    await run(service, "s", action="generate_notes")
    assert brain.calls_for("notes_summary")[-1].max_tokens <= 800


async def test_configured_token_ceiling_applies_up_front(service, brain, make_session, monkeypatch):
    from backend.config import get_settings
    monkeypatch.setattr(get_settings(), "LLM_MAX_OUTPUT_TOKENS", 700)
    make_session("s")
    await run(service, "s", message="Explain?", agent="doubt")
    assert brain.calls_for("doubt")[0].max_tokens == 700


async def test_missing_fallback_model_is_dropped_and_the_primary_is_retried(service, brain, make_session):
    make_session("s")
    brain.fail("doubt", rate_limit_error(), on_fallback=False)             # -> switch to fallback...
    brain.fail("doubt", model_missing_error(), on_fallback=True)           # ...which doesn't exist for this account
    events = await run(service, "s", message="Explain?", agent="doubt")
    f = final(events)
    assert f["error"] is None and "explanation" in f["content"].lower()
    assert [c.fallback for c in brain.calls_for("doubt")] == [False, True, False]   # ended up back on the primary

    # from now on there is no fallback to switch to: a 429 waits out Retry-After on the primary instead
    brain.calls.clear()
    brain.fail("doubt", rate_limit_error(), on_fallback=None)
    events = await run(service, "s", message="Again?", agent="doubt")
    assert [c.fallback for c in brain.calls_for("doubt")] == [False, False]
    assert next(e for e in events if e["type"] == "retry")["fallback"] is False


async def test_missing_primary_model_is_fatal(service, brain, make_session):
    make_session("s")
    brain.fail("doubt", model_missing_error("primary-model"))
    events = await run(service, "s", message="Explain?", agent="doubt")
    assert final(events)["error"] == "fatal" and len(brain.calls_for("doubt")) == 1


async def test_rate_limit_honours_retry_after_when_no_fallback_exists(service, brain, make_session, monkeypatch):
    import asyncio
    from backend.config import get_settings
    monkeypatch.setattr(get_settings(), "GROQ_FALLBACK_MODEL", "")
    slept = []
    real_sleep = asyncio.sleep
    monkeypatch.setattr("backend.app.graph.resilience.asyncio.sleep", lambda d: (slept.append(d), real_sleep(0))[1])
    make_session("s")
    brain.fail("doubt", rate_limit_error(retry_after="7"))
    events = await run(service, "s", message="Explain?", agent="doubt")
    assert slept == [7.0] and final(events)["error"] is None


async def test_quiz_pool_is_sized_to_the_budget_and_progress_survives_a_failure(service, brain, make_session, monkeypatch):
    from backend.config import get_settings
    from backend.tests.fakes import server_error
    monkeypatch.setattr(get_settings(), "LLM_MAX_OUTPUT_TOKENS", 330)     # -> 2 questions per call
    make_session("s")
    calls_seen = {"n": 0}
    real = brain._p_quiz

    def flaky(messages, text, kwargs):
        calls_seen["n"] += 1
        if calls_seen["n"] == 2:
            raise server_error()
        return real(messages, text, kwargs)

    brain._p_quiz = flaky
    events = await run(service, "s", action="generate_test", params={"num_questions": 4, "difficulty": "easy"})
    f = final(events)
    assert f["error"] is None and len(f["state"]["quiz"]["questions"]) == 4
    counts = [int(__import__("re").search(r"Generate (\d+) multiple", c.text).group(1)) for c in brain.calls_for("quiz")]
    assert counts == [2, 2, 2]                                             # 3 calls, never more than the budget allows
    # the failed second call did NOT throw away the first call's questions
    assert len(brain.calls_for("quiz")) == 3
