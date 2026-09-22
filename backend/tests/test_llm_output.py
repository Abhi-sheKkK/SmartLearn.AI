"""Robustness of model-output handling: LaTeX backslashes, truncation, malformed structure."""
import json

import pytest

from backend.app.graph.llm import MalformedOutputError, extract_json, _has_mangled_latex
from backend.app.graph.notes import parse_equations, parse_summary
from backend.tests.conftest import final, run, types

LATEX_SUMMARY = r"""TITLE: Backprop
SUMMARY: Gradients flow backwards.
=== SECTION 1 ===
HEADING: The rule
KEY_TERMS: gradient
CONTENT:
- Update $w \leftarrow w - \eta \nabla L$ and $\frac{\partial L}{\partial \theta}$
- Also $\beta$, $\rho$, $\times$, $\text{loss}$
"""
LATEX_EQUATIONS = r"""@@ EQUATION section=1
LATEX: \nabla_{\mathbf{w}} L = \frac{\partial L}{\partial \mathbf{w}}
EXPLAIN: Gradient of the loss.
"""


async def test_latex_survives_the_whole_notes_pipeline_unmangled(service, brain, make_session):
    brain.override("notes_summary", lambda t: LATEX_SUMMARY)
    brain.override("notes_latex", lambda t: LATEX_EQUATIONS)
    make_session("s", transcript="the equation uses a derivative and a matrix, x = 1 + 2. " * 40)
    notes = final(await run(service, "s", action="generate_notes"))["state"]["notes_draft"]
    for snippet in (r"\nabla L", r"\frac{\partial L}{\partial \theta}", r"\beta", r"\rho", r"\times", r"\text{loss}",
                    r"\nabla_{\mathbf{w}} L = \frac{\partial L}{\partial \mathbf{w}}"):
        assert snippet in notes, snippet
    assert not any(ch in notes for ch in ("\x0c", "\x08", "\t"))          # no form-feeds / tabs from mangled escapes


async def test_truncated_summary_still_produces_notes_and_keeps_frames(service, brain, make_session):
    # only sections 1-2 came back (token limit); frames for later sections must be re-homed, not dropped
    cut = "TITLE: T\nSUMMARY: S\n=== SECTION 1 ===\nHEADING: One\nCONTENT:\n- a\n=== SECTION 2 ===\nHEADING: Two\nCONTENT:\n- b\n=== SECTION 3 ===\nHEADING: Thr"
    brain.override("notes_summary", lambda t: cut)
    frames = [f"/storage/v/f_{i}.png" for i in range(6)]
    make_session("s", transcript="the equation is x = 1 + 2 and a matrix. " * 400, frame_urls=frames)
    f = final(await run(service, "s", action="generate_notes"))
    notes = f["state"]["notes_draft"]
    assert "## One" in notes and "## Two" in notes and "## Thr" not in notes and f["error"] is None
    assert all(u in notes for u in frames)


def test_summary_parser_variants():
    out = parse_summary("SUMMARY: s\n## SECTION 1 ##\nHEADING: A\nCONTENT: inline start\nmore\n== SECTION 2 ==\ncontent without markers")
    assert [s["heading"] for s in out["sections"]] == ["A", "Section 2"]
    assert out["sections"][0]["content"] == "inline start\nmore"
    assert out["sections"][1]["content"] == "content without markers"


def test_summary_parser_rejects_output_without_sections():
    with pytest.raises(MalformedOutputError):
        parse_summary("Sure! Here are your notes: ...")


def test_equation_parser():
    eq = parse_equations("@@ EQUATION section=3\nLATEX: $$a^2 + b^2 = c^2$$\nEXPLAIN: Pythagoras\n@@ EQUATION\nLATEX: \\[ E = mc^2 \\]")
    assert eq["equations"] == [
        {"section_id": 3, "latex": "a^2 + b^2 = c^2", "explanation": "Pythagoras"},
        {"section_id": 1, "latex": "E = mc^2", "explanation": ""},
    ]
    assert parse_equations("NONE") == {"equations": []}
    with pytest.raises(MalformedOutputError):
        parse_equations("I could not find any equations in this lecture.")


# ------------------------------------------------------------------ JSON handling for the short calls

def test_json_with_unescaped_latex_is_repaired():
    assert extract_json(r'{"q": "What is $\eta$ in \mathbf{w}?"}')["q"] == r"What is $\eta$ in \mathbf{w}?"


def test_legal_escapes_are_left_alone():
    assert extract_json('{"a": "x\\ny \\"q\\" \\\\"}')["a"] == 'x\ny "q" \\'


def test_mangled_latex_is_detected():
    assert _has_mangled_latex(json.loads(r'{"a": "\frac{1}{2}"}'))          # \f decoded to form-feed
    assert _has_mangled_latex(json.loads(r'{"a": ["ok", {"b": "\beta"}]}'))  # \b decoded to backspace
    assert not _has_mangled_latex({"a": "fine \\frac"})


async def test_quiz_json_with_mangled_latex_is_retried(service, brain, make_session):
    good = json.dumps({"questions": [{"question": "What is $\\eta$?", "options": ["a", "b", "c", "d"], "correct_answer": 0}]})
    bad = r'{"questions": [{"question": "Compute $\frac{1}{2}$", "options": ["a","b","c","d"], "correct_answer": 0}]}'
    replies = iter([bad, good])
    brain.override("quiz", lambda t: next(replies))
    make_session("s")
    events = await run(service, "s", action="generate_test", params={"num_questions": 1})
    f = final(events)
    assert types(events).count("retry") == 1 and f["error"] is None
    assert f["state"]["quiz"]["questions"][0]["question"] == "What is $\\eta$?"
