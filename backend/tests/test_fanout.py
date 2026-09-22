"""Notes: parallel fan-out (Summarizer / LaTeX / DiagramMapper) and fan-in (Merger)."""
import json

from backend.app.graph.notes import build_segments
from backend.tests.conftest import final, run, types
from backend.tests.fakes import Call

MATH_LECTURE = "\n".join(f"Line {i}: the equation for the loss uses a derivative and a matrix, x = {i} + 2. " * 3 for i in range(60))


def write_timeline(tmp_path, video_id, lines):
    folder = tmp_path / video_id
    folder.mkdir(parents=True, exist_ok=True)
    snippets = [{"text": ln, "start": i * 20.0, "duration": 20.0} for i, ln in enumerate(lines)]
    (folder / "en.json").write_text(json.dumps(snippets))
    (folder / "cues.json").write_text(json.dumps([
        {"start": 25.0, "cue_type": "slide", "trigger_text": "look at this slide"},
        {"start": 1100.0, "cue_type": "diagram", "trigger_text": "as shown in this diagram"},
    ]))
    return snippets


def overlap(a: Call, b: Call) -> bool:
    return a.started < b.ended and b.started < a.ended


async def test_workers_run_in_parallel_and_merger_waits_for_all(service, brain, make_session, tmp_path):
    lines = MATH_LECTURE.splitlines()
    write_timeline(tmp_path, "vid", lines)
    make_session("s", transcript="\n".join(lines), video_id="vid")
    brain.delays["notes_summary"] = 0.3
    brain.delays["notes_latex"] = 0.3

    events = await run(service, "s", action="generate_notes")
    f = final(events)
    assert f["error"] is None and f["state"]["notes_draft"]

    summary, latex = brain.calls_for("notes_summary")[0], brain.calls_for("notes_latex")[0]
    assert overlap(summary, latex), "the LLM workers must execute concurrently, not one after the other"

    # fan-in: all three workers reported before the merger's message was produced
    progress = [e for e in events if e["type"] == "notes_progress"]
    assert {p["worker"] for p in progress} == {"summarizer", "latex", "diagram_mapper"}
    assert all(p["status"] == "done" for p in progress)
    last_progress = max(i for i, e in enumerate(events) if e["type"] == "notes_progress")
    first_token = min(i for i, e in enumerate(events) if e["type"] == "token")
    assert last_progress < first_token


async def test_merged_notes_place_equations_and_frames_under_the_right_headers(service, brain, make_session, tmp_path):
    lines = MATH_LECTURE.splitlines()
    snippets = write_timeline(tmp_path, "vid", lines)
    total = snippets[-1]["start"] + 20
    frames = ["/storage/vid/unique_frames/frame_01_slide_25s.png",
              f"/storage/vid/unique_frames/frame_02_diagram_{int(total) - 30}s.png"]
    make_session("s", transcript="\n".join(lines), frame_urls=frames, video_id="vid")

    f = final(await run(service, "s", action="generate_notes"))
    notes = f["state"]["notes_draft"]
    assert notes.startswith("# Test Lecture")
    assert "## Heading 1" in notes and "## Summary" in notes and "**Key terms:** term1" in notes

    sections = notes.split("## ")
    first = next(s for s in sections if s.startswith("Heading 1"))
    last_heading = max((s for s in sections if s.startswith("Heading ")), key=lambda s: int(s.split()[1].split("\n")[0]))
    assert "$$\nE = mc^2\n$$" in first                                  # equation attached to section 1
    assert "![look at this slide](/storage/vid/unique_frames/frame_01_slide_25s.png)" in first   # early frame -> first header
    assert "as shown in this diagram" in last_heading                 # late frame -> last header
    # checkpoint stays small: intermediates are dropped after the merge
    snap = await service.graph.aget_state({"configurable": {"thread_id": "s"}})
    assert not snap.values["notes_segments"] and snap.values["notes_summary"] is None


async def test_segments_cover_the_whole_transcript_in_order():
    lines = [f"sentence {i}" for i in range(300)]
    timeline = [{"text": t, "start": i * 5.0, "duration": 5.0} for i, t in enumerate(lines)]
    segs = build_segments("\n".join(lines), timeline)
    assert 3 <= len(segs) <= 8
    assert segs[0]["start"] == 0 and segs[-1]["end"] == 300 * 5.0
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(segs, segs[1:]))
    assert " ".join(s["text"] for s in segs).split() == " ".join(lines).split()


async def test_frames_spread_by_order_when_no_timestamps_exist(service, brain, make_session):
    frames = [f"/storage/v/frame_{i}.png" for i in range(6)]      # no _<n>s suffix, no timeline on disk
    make_session("s", transcript=MATH_LECTURE, frame_urls=frames)
    notes = final(await run(service, "s", action="generate_notes"))["state"]["notes_draft"]
    assert all(u in notes for u in frames)                         # every frame lands somewhere


async def test_non_critical_worker_failure_still_delivers_notes(service, brain, make_session):
    make_session("s", transcript=MATH_LECTURE)
    brain.override("notes_latex", lambda t: "this is not json at all")
    events = await run(service, "s", action="generate_notes")
    f = final(events)
    assert f["state"]["notes_draft"] and "$$" not in f["state"]["notes_draft"]
    assert "latex" in f["content"].lower() and "could not be generated" in f["content"]
    assert f["error"] is None


async def test_rate_limited_worker_recovers_locally_on_the_fallback_model(service, brain, make_session):
    from backend.tests.fakes import rate_limit_error
    make_session("s", transcript=MATH_LECTURE)
    brain.fail("notes_summary", rate_limit_error(), on_fallback=False)
    events = await run(service, "s", action="generate_notes")
    f = final(events)
    assert f["state"]["notes_draft"] and f["error"] is None
    assert "retry" not in types(events)                               # handled inside the worker, no graph-level retry
    assert [c.fallback for c in brain.calls_for("notes_summary")] == [False, True]
    assert len(brain.calls_for("notes_latex")) == 1                   # the other workers were not re-run


async def test_critical_worker_failure_retries_the_whole_fanout(service, brain, make_session):
    from backend.tests.fakes import rate_limit_error
    make_session("s", transcript=MATH_LECTURE)
    brain.fail("notes_summary", rate_limit_error(), rate_limit_error())   # local retry fails too
    events = await run(service, "s", action="generate_notes")
    f = final(events)
    retry = next(e for e in events if e["type"] == "retry")
    assert retry["node"] == "notes_prepare" and retry["kind"] == "rate_limit"
    assert f["state"]["notes_draft"] and f["error"] is None
    assert len(brain.calls_for("notes_summary")) == 3
    assert len(brain.calls_for("notes_latex")) == 2                   # the whole fan-out re-ran


async def test_summarizer_output_is_sized_to_the_token_budget(service, brain, make_session, monkeypatch):
    from backend.config import get_settings
    make_session("s", transcript=MATH_LECTURE)
    await run(service, "s", action="generate_notes")
    roomy = brain.calls_for("notes_summary")[-1]
    monkeypatch.setattr(get_settings(), "LLM_MAX_OUTPUT_TOKENS", 600)
    await run(service, "s", action="generate_notes", params={"refresh": True})
    tight = brain.calls_for("notes_summary")[-1]
    words = lambda c: int(__import__("re").search(r"about (\d+) words", c.text).group(1))
    assert tight.max_tokens == 600 and roomy.max_tokens == 3500
    assert words(tight) < words(roomy)


async def test_saved_notes_are_served_without_any_llm_call_until_refresh(service, brain, make_session):
    make_session("s", transcript=MATH_LECTURE)
    await run(service, "s", action="generate_notes")
    n = len(brain.calls)
    f = final(await run(service, "s", action="generate_notes"))
    assert len(brain.calls) == n and "saved lecture notes" in f["content"] and f["state"]["notes_draft"]
    await run(service, "s", action="generate_notes", params={"refresh": True})
    assert len(brain.calls) > n


async def test_no_transcript_short_circuits(service, brain, make_session):
    make_session("s", transcript="")
    f = final(await run(service, "s", action="generate_notes"))
    assert "No transcript" in f["content"] and brain.calls == []


async def test_equations_already_written_inline_are_not_repeated(service, brain, make_session):
    brain.override("notes_summary", lambda t: "TITLE: T\nSUMMARY: S\n=== SECTION 1 ===\nHEADING: One\nCONTENT:\n- Energy: $$E = mc^2$$")
    make_session("s", transcript=MATH_LECTURE)
    notes = final(await run(service, "s", action="generate_notes"))["state"]["notes_draft"]
    assert notes.count("mc^2") == 1                                   # the extracted duplicate was skipped
