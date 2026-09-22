"""Supervisor routing and mid-conversation handoffs that preserve viva state."""
from backend.tests.conftest import final, run


def route_of(events):
    return next(e for e in events if e["type"] == "route")


async def test_question_during_viva_hands_off_to_doubt_and_keeps_viva_score(service, brain, make_session):
    make_session("s")
    await run(service, "s", action="viva_start")
    await run(service, "s", message="ANSWER: great answer", agent="viva")
    before = (await service.snapshot("s"))["viva_score"]
    assert before["asked"] == 1 and before["active"]

    # student is on the viva tab but asks a question of their own
    events = await run(service, "s", message="What does a gradient mean?", agent="viva")
    assert route_of(events)["agent"] == "doubt"
    assert route_of(events)["source"] == "llm"
    f = final(events)
    assert "explanation" in f["content"].lower()
    assert "viva is paused" in f["content"].lower()

    after = (await service.snapshot("s"))["viva_score"]
    assert after == before                                            # untouched by the handoff

    # ...and the viva resumes: the next answer is scored against the SAME pending question
    events = await run(service, "s", message="ANSWER: great again", agent="viva")
    assert route_of(events)["agent"] == "viva"
    resumed = final(events)["state"]["viva_score"]
    assert resumed["asked"] == 2 and resumed["correct"] == 2
    assert brain.calls_for("viva_eval")[-1].text.count(before["pending_question"]) >= 1


async def test_answer_on_viva_tab_skips_the_classifier_llm(service, brain, make_session):
    make_session("s")
    await run(service, "s", action="viva_start")
    brain.calls.clear()
    events = await run(service, "s", message="ANSWER: great answer", agent="viva")
    assert route_of(events)["source"] == "heuristic"
    assert brain.calls_for("supervisor") == []                        # no LLM routing call was needed


async def test_explicit_actions_route_deterministically(service, brain, make_session):
    make_session("s")
    ev = await run(service, "s", action="generate_test", params={"num_questions": 2, "difficulty": "easy"})
    r = route_of(ev)
    assert (r["agent"], r["target"], r["source"]) == ("test", "test_generate", "action")
    assert brain.calls_for("supervisor") == []
    quiz = final(ev)["state"]["quiz"]
    assert len(quiz["questions"]) == 2 and quiz["difficulty"] == "easy"


async def test_free_text_intent_classification(service, brain, make_session):
    make_session("s")
    ev = await run(service, "s", message="Quiz me on this lecture")
    r = route_of(ev)
    assert (r["agent"], r["source"]) == ("test", "llm")
    # parameters extracted by the classifier drive the test node (num_questions=3, hard)
    assert final(ev)["state"]["quiz"]["difficulty"] == "hard"
    assert len(final(ev)["state"]["quiz"]["questions"]) == 3

    ev = await run(service, "s", message="Please make notes", stream=True)
    assert route_of(ev)["agent"] == "notes"


async def test_messages_are_tagged_with_the_handling_agent(service, brain, make_session):
    make_session("s")
    await run(service, "s", message="What is a gradient?")
    await run(service, "s", action="viva_start")
    doubt = await service.history("s", "doubt")
    viva = await service.history("s", "viva")
    assert [m["role"] for m in doubt] == ["user", "assistant"]
    assert [m["role"] for m in viva] == ["user", "assistant"]
    assert doubt[0]["content"] == "What is a gradient?"


async def test_ending_viva_resets_score_and_summarises(service, brain, make_session):
    make_session("s")
    await run(service, "s", action="viva_start")
    await run(service, "s", message="ANSWER: great answer", agent="viva")
    ev = await run(service, "s", action="viva_end")
    f = final(ev)
    assert "Viva Session Complete" in f["content"] and "100%" in f["content"]
    assert f["state"]["viva_score"]["active"] is False and f["state"]["viva_score"]["asked"] == 0
