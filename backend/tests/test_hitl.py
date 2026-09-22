"""Human-in-the-loop: the viva scoring node is gated by interrupt_before."""
import pytest

from backend.app.graph.service import AgentRequestError
from backend.app.models import AgentRequest
from backend.tests.conftest import final, run, types
from backend.tests.fakes import server_error


async def _partial(service, sid="s"):
    await run(service, sid, action="viva_start")
    return await run(service, sid, message="ANSWER: a partial answer", agent="viva")


async def test_partial_answer_pauses_before_score_is_finalised(service, brain, make_session):
    make_session("s")
    events = await _partial(service)
    f = final(events)
    assert f["interrupted"] is True
    assert "interrupt" in types(events)
    assert f["interrupt"]["prompt"] == "Can you say more about X?"
    assert f["interrupt"]["evaluation"]["verdict"] == "partial"
    assert f["content"] == "Can you say more about X?"
    # nothing has been scored yet, and the graph is genuinely parked before the scoring node
    assert f["state"]["viva_score"]["asked"] == 0
    snap = await service.graph.aget_state({"configurable": {"thread_id": "s"}})
    assert snap.next == ("viva_score",)


async def test_resume_with_elaboration_rescore_and_asks_next_question(service, brain, make_session):
    make_session("s")
    await _partial(service)
    events = await run(service, "s", resume={"elaboration": "here is more detail"})
    f = final(events)
    assert f["interrupted"] is False
    assert f["state"]["viva_score"]["asked"] == 1
    assert f["state"]["viva_score"]["correct"] == 1                  # elaboration re-evaluated as correct (0.9)
    assert "Next Question" in f["content"] and "Question 2" in f["content"]
    assert "elaborated" in brain.calls_for("viva_eval")[-1].text


async def test_resume_without_elaboration_finalises_as_is(service, brain, make_session):
    make_session("s")
    await _partial(service)
    events = await run(service, "s", resume={"elaboration": ""})
    v = final(events)["state"]["viva_score"]
    assert v["asked"] == 1 and v["correct"] == 0 and v["points"] == 0.5   # partial credit, not "correct"
    assert len(brain.calls_for("viva_eval")) == 1                        # no second evaluation


async def test_plain_message_while_paused_is_treated_as_elaboration(service, brain, make_session):
    make_session("s")
    await _partial(service)
    events = await run(service, "s", message="Let me clarify what I meant", agent="viva")
    assert final(events)["state"]["viva_score"]["asked"] == 1
    assert "clarify what I meant" in brain.calls_for("viva_eval")[-1].text


async def test_unrelated_action_while_paused_abandons_the_pending_evaluation(service, brain, make_session):
    make_session("s")
    await _partial(service)
    events = await run(service, "s", action="generate_test", params={"num_questions": 1})
    f = final(events)
    assert f["interrupted"] is False and f["state"]["quiz"]
    snap = await service.graph.aget_state({"configurable": {"thread_id": "s"}})
    assert snap.next == () and snap.values["viva_pending_eval"] is None
    assert f["state"]["viva_score"]["asked"] == 0                       # abandoned, not scored


async def test_resume_with_nothing_paused_is_rejected(service, brain, make_session):
    make_session("s")
    with pytest.raises(AgentRequestError) as exc:
        await service.prepare("s", AgentRequest(resume={"elaboration": "hi"}))
    assert exc.value.status == 409


async def test_clear_answers_do_not_interrupt(service, brain, make_session):
    make_session("s")
    await run(service, "s", action="viva_start")
    events = await run(service, "s", message="ANSWER: great answer", agent="viva")
    assert "interrupt" not in types(events)
    assert final(events)["state"]["viva_score"]["asked"] == 1


async def test_hitl_can_be_disabled_per_request(service, brain, make_session):
    make_session("s")
    await run(service, "s", action="viva_start")
    events = await run(service, "s", message="ANSWER: a partial answer", agent="viva", params={"hitl": False})
    f = final(events)
    assert f["interrupted"] is False and f["state"]["viva_score"]["asked"] == 1


async def test_retry_after_resume_does_not_ask_for_confirmation_twice(service, brain, make_session):
    make_session("s")
    await _partial(service)
    brain.fail("viva_eval", server_error())        # the re-evaluation after the human's elaboration fails once
    events = await run(service, "s", resume={"elaboration": "more"})
    f = final(events)
    assert "retry" in types(events)
    assert f["interrupted"] is False               # retried through viva_score_auto: no second interrupt
    assert f["state"]["viva_score"]["asked"] == 1
