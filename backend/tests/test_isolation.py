"""Multi-tenant isolation and native persistence: thread_id == session_id."""
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.graph import build_graph
from backend.app.graph.service import AgentService
from backend.tests.conftest import final, run

ALPHA = "ALPHA-LECTURE " * 80
BETA = "BETA-LECTURE " * 80


async def test_sessions_do_not_share_state(service, brain, make_session):
    make_session("A", transcript=ALPHA)
    make_session("B", transcript=BETA)

    await run(service, "A", action="viva_start")
    await run(service, "A", message="ANSWER: great answer", agent="viva")     # scores 1/1 in A
    await run(service, "B", message="What is beta?", agent="doubt")

    a, b = await service.snapshot("A"), await service.snapshot("B")
    assert a["viva_score"]["asked"] == 1 and a["viva_score"]["correct"] == 1
    assert b["viva_score"] == {}                      # B never started a viva

    hist_a = await service.history("A")
    hist_b = await service.history("B")
    assert not any("beta" in m["content"].lower() for m in hist_a)
    assert [m["content"] for m in hist_b if m["role"] == "user"] == ["What is beta?"]


async def test_each_thread_only_sees_its_own_transcript(service, brain, make_session):
    make_session("A", transcript=ALPHA)
    make_session("B", transcript=BETA)
    await run(service, "A", message="Explain?", agent="doubt")
    await run(service, "B", message="Explain?", agent="doubt")
    await run(service, "A", message="Explain again?", agent="doubt")

    doubt_calls = brain.calls_for("doubt")
    assert len(doubt_calls) == 3
    assert "ALPHA-LECTURE" in doubt_calls[0].text and "BETA-LECTURE" not in doubt_calls[0].text
    assert "BETA-LECTURE" in doubt_calls[1].text and "ALPHA-LECTURE" not in doubt_calls[1].text
    assert "ALPHA-LECTURE" in doubt_calls[2].text and "BETA-LECTURE" not in doubt_calls[2].text


async def test_concurrent_sessions_stay_isolated(service, brain, make_session):
    ids = [f"S{i}" for i in range(6)]
    for i, sid in enumerate(ids):
        make_session(sid, transcript=f"UNIQUE-{i} " * 100)
    brain.delays["doubt"] = 0.05

    results = await asyncio.gather(*(run(service, sid, message=f"question from {sid}?", agent="doubt") for sid in ids))
    for sid, events in zip(ids, results):
        assert final(events)["thread_id"] == sid
        users = [m["content"] for m in await service.history(sid) if m["role"] == "user"]
        assert users == [f"question from {sid}?"]


async def test_same_session_requests_are_serialised(service, brain, make_session):
    make_session("X")
    brain.delays["doubt"] = 0.1
    await asyncio.gather(
        run(service, "X", message="first?", agent="doubt"),
        run(service, "X", message="second?", agent="doubt"),
    )
    roles = [m["role"] for m in await service.history("X")]
    assert roles == ["user", "assistant", "user", "assistant"]     # never user,user,assistant,assistant


async def test_state_survives_restart_with_sqlite_checkpointer(brain, make_session, tmp_path):
    make_session("P", transcript=ALPHA)
    db = str(tmp_path / "cp.sqlite")

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        svc = AgentService(build_graph(saver))
        await run(svc, "P", action="viva_start")
        await run(svc, "P", message="ANSWER: great answer", agent="viva")

    # "restart": brand-new graph + saver over the same file
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        svc = AgentService(build_graph(saver))
        snap = await svc.snapshot("P")
        assert snap["viva_score"]["asked"] == 1
        assert snap["viva_score"]["pending_question"].startswith("Question 2")
        # and the conversation continues where it left off
        events = await run(svc, "P", message="ANSWER: great again", agent="viva")
        assert final(events)["state"]["viva_score"]["asked"] == 2


async def test_abandoned_stream_releases_the_session_lock(service, brain, make_session):
    """A client that disconnects mid-stream must not wedge the session for everyone after it."""
    from backend.app.models import AgentRequest
    make_session("s")
    brain.delays["doubt"] = 0.05
    stream = service.stream("s", AgentRequest(message="first?", agent="doubt"))
    await stream.__anext__()                       # 'start' event: the run holds the lock now
    await stream.aclose()                          # ...and the client goes away

    events = await asyncio.wait_for(run(service, "s", message="second?", agent="doubt"), timeout=5)
    assert final(events)["error"] is None
    users = [m["content"] for m in await service.history("s") if m["role"] == "user"]
    assert users[-1] == "second?"
