"""HTTP layer: the unified endpoint, SSE streaming, and removal of the legacy routes."""
import json

import pytest

from backend.tests.fakes import rate_limit_error


def parse_sse(response) -> list[tuple[str, dict]]:
    events, name, data = [], None, []
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data.append(line.split(":", 1)[1].strip())
        elif line == "" and name:
            events.append((name, json.loads("".join(data) or "{}")))
            name, data = None, []
    return events


def stream(client, sid, body):
    with client.stream("POST", f"/api/sessions/{sid}/agent", json=body) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("text/event-stream")
        return parse_sse(r)


def test_streaming_delivers_word_by_word_tokens_then_final(client, make_session):
    make_session("s1")
    events = stream(client, "s1", {"message": "What is a gradient?", "agent": "doubt"})
    names = [n for n, _ in events]
    assert names[0] == "start" and names[1] == "route" and names[-2:] == ["final", "done"]

    tokens = [d["text"] for n, d in events if n == "token"]
    assert len(tokens) >= 5                                            # incremental, not one blob
    assert all(len(t.split()) <= 1 for t in tokens)                    # a word (plus whitespace) at a time
    final = dict(events)["final"]
    assert "".join(tokens) == final["content"] == "Here is a clear explanation for: What is a gradient?"
    assert final["agent"] == "doubt" and final["thread_id"] == "s1"


def test_tokens_arrive_before_the_run_finishes(client, make_session, brain):
    make_session("s1")
    brain.delays["doubt"] = 0.2
    events = stream(client, "s1", {"message": "Explain slowly?", "agent": "doubt"})
    names = [n for n, _ in events]
    assert names.index("token") < names.index("final")


def test_non_streaming_returns_one_json_document(client, make_session):
    make_session("s1")
    r = client.post("/api/sessions/s1/agent", json={"message": "What is a gradient?", "agent": "doubt", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["agent"] == "doubt" and body["interrupted"] is False and body["error"] is None
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["content"] == "Here is a clear explanation for: What is a gradient?"


def test_viva_interrupt_and_resume_over_http(client, make_session):
    make_session("s1")
    client.post("/api/sessions/s1/agent", json={"action": "viva_start", "stream": False})

    events = stream(client, "s1", {"message": "ANSWER: partial thing", "agent": "viva"})
    names = [n for n, _ in events]
    assert "interrupt" in names and names[-2:] == ["final", "done"]
    intr = dict(events)["interrupt"]
    assert intr["prompt"] == "Can you say more about X?" and intr["evaluation"]["verdict"] == "partial"
    assert dict(events)["final"]["interrupted"] is True

    state = client.get("/api/sessions/s1/agent/state").json()
    assert state["interrupt"]["node"] == "viva_score" and state["viva_score"]["asked"] == 0

    r = client.post("/api/sessions/s1/agent", json={"resume": {"elaboration": "more detail"}, "stream": False})
    body = r.json()
    assert body["interrupted"] is False and body["state"]["viva_score"]["asked"] == 1
    assert "Next Question" in body["content"]


def test_resume_with_nothing_paused_returns_409(client, make_session):
    make_session("s1")
    r = client.post("/api/sessions/s1/agent", json={"resume": {"elaboration": "x"}})
    assert r.status_code == 409 and "Nothing to resume" in r.json()["detail"]


@pytest.mark.parametrize("body", [{}, {"message": "   "}, {"agent": "doubt"}, {"message": "hi", "action": "nope"},
                                  {"params": {"num_questions": 99}, "action": "generate_test"}])
def test_invalid_requests_are_rejected(client, make_session, body):
    make_session("s1")
    assert client.post("/api/sessions/s1/agent", json=body).status_code == 422


def test_unknown_session_is_404(client):
    assert client.post("/api/sessions/nope/agent", json={"message": "hi"}).status_code == 404
    assert client.get("/api/sessions/nope/agent/state").status_code == 404
    assert client.get("/api/sessions/nope/agent/history").status_code == 404


@pytest.mark.parametrize("method,path", [
    ("post", "/api/sessions/s1/notes"), ("get", "/api/sessions/s1/notes"),
    ("post", "/api/sessions/s1/chat"), ("get", "/api/sessions/s1/chat/history"),
    ("post", "/api/sessions/s1/viva"), ("post", "/api/sessions/s1/test/generate"),
    ("post", "/api/sessions/s1/test/submit"), ("get", "/api/sessions/s1/test/results"),
])
def test_legacy_routes_are_gone(client, make_session, method, path):
    make_session("s1")
    r = getattr(client, method)(path, **({"json": {}} if method == "post" else {}))
    assert r.status_code in (404, 405)


def test_quiz_flow_never_leaks_the_answer_key(client, make_session):
    make_session("s1")
    r = client.post("/api/sessions/s1/agent", json={"action": "generate_test", "params": {"num_questions": 3}, "stream": False})
    body = r.json()
    qs = body["state"]["quiz"]["questions"]
    assert len(qs) == 3
    dumped = json.dumps(body)
    assert "answer_key" not in dumped and "correct_answer" not in json.dumps(qs) and "explanation" not in json.dumps(qs)
    assert "answer_key" not in json.dumps(client.get("/api/sessions/s1/agent/state").json())

    # answer the first question right (fake generator makes question i correct at index i % 4), others wrong
    answers = {q["id"]: "0" for q in qs}
    r = client.post("/api/sessions/s1/agent", json={"action": "submit_test", "params": {"answers": answers}, "stream": False})
    result = r.json()["state"]["test_result"]
    assert result["total"] == 3 and 0 <= result["score"] <= 3
    assert all("correct_answer" in item and "explanation" in item for item in result["results"])   # revealed only after submit


def test_submit_without_a_generated_quiz(client, make_session):
    make_session("s1")
    r = client.post("/api/sessions/s1/agent", json={"action": "submit_test", "params": {"answers": {"q": "0"}}, "stream": False})
    assert "no active test" in r.json()["content"].lower()


def test_history_and_state_come_from_the_checkpoint(client, make_session):
    make_session("s1")
    client.post("/api/sessions/s1/agent", json={"message": "What is a gradient?", "agent": "doubt", "stream": False})
    client.post("/api/sessions/s1/agent", json={"action": "viva_start", "stream": False})
    doubt = client.get("/api/sessions/s1/agent/history", params={"agent": "doubt"}).json()["messages"]
    assert [m["role"] for m in doubt] == ["user", "assistant"] and doubt[0]["content"] == "What is a gradient?"
    everything = client.get("/api/sessions/s1/agent/history").json()["messages"]
    assert len(everything) == 4
    assert client.get("/api/sessions/s1/agent/state").json()["viva_score"]["active"] is True


def test_notes_state_falls_back_to_notes_saved_by_another_session(client, make_session):
    make_session("s1", notes_markdown="# Saved earlier")
    assert client.get("/api/sessions/s1/agent/state").json()["notes_draft"] == "# Saved earlier"


def test_errors_surface_in_the_stream_as_a_final_event(client, make_session, brain):
    make_session("s1")
    brain.fail_always("doubt", rate_limit_error)
    events = stream(client, "s1", {"message": "Explain?", "agent": "doubt"})
    names = [n for n, _ in events]
    assert names.count("retry") == 2 and names[-2:] == ["final", "done"]
    assert dict(events)["final"]["error"] == "rate_limit"
    assert "rate-limited" in "".join(d["text"] for n, d in events if n == "token")


def test_concurrent_requests_to_different_sessions_over_http(client, make_session):
    from concurrent.futures import ThreadPoolExecutor
    ids = [f"c{i}" for i in range(4)]
    for sid in ids:
        make_session(sid)

    def call(sid):
        return client.post(f"/api/sessions/{sid}/agent", json={"message": f"hello from {sid}?", "agent": "doubt", "stream": False}).json()

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(call, ids))
    for sid, body in zip(ids, results):
        assert body["thread_id"] == sid and f"hello from {sid}?" in body["content"]
