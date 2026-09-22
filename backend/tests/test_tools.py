"""Tools bound to the sub-agents: python_repl / web_search (DoubtAgent), text_to_speech (VivaAgent)."""
import sys
import types as pytypes

import pytest

from backend.app.graph import tools
from backend.tests.conftest import final, run, types


# ------------------------------------------------------------------ python_repl sandbox

def test_python_repl_computes():
    assert tools.python_repl.invoke({"code": "import math\nprint(math.factorial(6))"}) == "720"
    assert tools.python_repl.invoke({"code": "import numpy as np\nprint(int(np.dot([1,2,3],[4,5,6])))"}) == "32"


@pytest.mark.parametrize("code", [
    "import os\nprint(os.listdir('.'))",
    "import subprocess",
    "from pathlib import Path",
    "print(open('/etc/hosts').read())",
    "print(().__class__.__mro__)",
    "eval('1+1')",
    "getattr(int, 'real')",
    "import numpy as np\nnp.fromfile('/etc/hosts')",
])
def test_python_repl_blocks_escape_hatches(code):
    assert tools.python_repl.invoke({"code": code}).startswith("Error:")


def test_python_repl_child_gets_an_empty_environment_and_temp_cwd(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value")
    seen = {}
    real_run = tools.subprocess.run

    def spy(*args, **kwargs):
        seen.update(env=kwargs["env"], cwd=kwargs["cwd"], argv=args[0])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", spy)
    assert tools.run_python_sandboxed("print('hi')") == "hi"
    assert "GROQ_API_KEY" not in seen["env"] and "super-secret-value" not in str(seen["env"])
    assert "-I" in seen["argv"]                                       # isolated interpreter mode
    assert "smartlearn-repl-" in seen["cwd"]


def test_python_repl_enforces_timeout(monkeypatch):
    monkeypatch.setattr(tools, "TIMEOUT_SECONDS", 1)
    out = tools.run_python_sandboxed("while True:\n    pass")
    assert "exceeded" in out


def test_python_repl_reports_runtime_errors():
    assert "ZeroDivisionError" in tools.python_repl.invoke({"code": "print(1/0)"})


# ------------------------------------------------------------------ web_search

def test_web_search_formats_results(monkeypatch):
    class FakeDDGS:
        def __init__(self, **kw): pass
        def text(self, query, max_results=5):
            return [{"title": "NumPy docs", "href": "https://numpy.org", "body": "Array library."}][:max_results]
    monkeypatch.setitem(sys.modules, "ddgs", pytypes.SimpleNamespace(DDGS=FakeDDGS))
    out = tools.web_search.invoke({"query": "numpy dot"})
    assert "NumPy docs" in out and "https://numpy.org" in out


def test_web_search_failure_is_returned_not_raised(monkeypatch):
    class Boom:
        def __init__(self, **kw): pass
        def text(self, *a, **k): raise RuntimeError("network down")
    monkeypatch.setitem(sys.modules, "ddgs", pytypes.SimpleNamespace(DDGS=Boom))
    assert "unavailable" in tools.web_search.invoke({"query": "x"})


# ------------------------------------------------------------------ text_to_speech

def test_tts_strips_markdown_and_caches(monkeypatch, tmp_path):
    spoken = []

    class FakeGTTS:
        def __init__(self, text, lang="en"): spoken.append(text)
        def save(self, path): open(path, "wb").write(b"mp3")
    monkeypatch.setitem(sys.modules, "gtts", pytypes.SimpleNamespace(gTTS=FakeGTTS))

    url = tools.text_to_speech.invoke({"text": "**What** is `$x^2$`? See [docs](http://a.b)"})
    assert url.startswith("/storage/tts/") and url.endswith(".mp3")
    assert "*" not in spoken[0] and "`" not in spoken[0] and "http" not in spoken[0]
    assert (tmp_path / "tts" / url.rsplit("/", 1)[1]).exists()
    assert tools.text_to_speech.invoke({"text": "**What** is `$x^2$`? See [docs](http://a.b)"}) == url
    assert len(spoken) == 1                                            # second call served from cache


def test_tts_failure_returns_empty_string(monkeypatch):
    class Boom:
        def __init__(self, *a, **k): raise RuntimeError("offline")
    monkeypatch.setitem(sys.modules, "gtts", pytypes.SimpleNamespace(gTTS=Boom))
    assert tools.text_to_speech.invoke({"text": "hello there"}) == ""


# ------------------------------------------------------------------ wired into the agents

async def test_doubt_agent_calls_python_tool_and_streams_tool_events(service, brain, make_session):
    make_session("s")
    events = await run(service, "s", message="Please calculate 6 times 7", agent="doubt")
    f = final(events)
    assert [e["name"] for e in events if e["type"] == "tool_start"] == ["python_repl"]
    assert [e["output"] for e in events if e["type"] == "tool_end"] == ["42"]
    assert "result is 42" in f["content"]
    assert [c.tools for c in brain.calls_for("doubt")] == [True, True]   # tool-bound both rounds
    # tool plumbing messages are not part of the visible history
    assert all("42" not in m["content"] or m["role"] == "assistant" for m in await service.history("s", "doubt"))


async def test_tools_can_be_disabled_by_configuration(service, brain, make_session, monkeypatch):
    from backend.config import get_settings
    monkeypatch.setattr(get_settings(), "ENABLE_PYTHON_TOOL", False)
    monkeypatch.setattr(get_settings(), "ENABLE_WEB_SEARCH", False)
    import backend.app.graph as g
    from backend.app.graph.service import AgentService
    from backend.app.graph import open_checkpointer
    make_session("s")
    async with open_checkpointer(get_settings()) as saver:
        svc = AgentService(g.build_graph(saver))
        events = await run(svc, "s", message="Please calculate 6 times 7", agent="doubt")
    assert [c.tools for c in brain.calls_for("doubt")] == [False]
    assert "tool_start" not in types(events)


async def test_viva_question_is_voiced_via_the_tts_tool(service, brain, make_session, monkeypatch):
    from backend.config import get_settings
    monkeypatch.setattr(get_settings(), "ENABLE_TTS", True)

    class FakeGTTS:
        def __init__(self, text, lang="en"): self.text = text
        def save(self, path): open(path, "wb").write(b"mp3")
    monkeypatch.setitem(sys.modules, "gtts", pytypes.SimpleNamespace(gTTS=FakeGTTS))

    make_session("s")
    events = await run(service, "s", action="viva_start")
    f = final(events)
    assert f["state"]["audio_url"].startswith("/storage/tts/")
    assert any(e["type"] == "audio" and e["url"] == f["state"]["audio_url"] for e in events)

    events = await run(service, "s", action="viva_start", params={"tts": False})
    assert final(events)["state"]["audio_url"] == ""


async def test_tts_outage_never_breaks_the_viva(service, brain, make_session, monkeypatch):
    from backend.config import get_settings
    monkeypatch.setattr(get_settings(), "ENABLE_TTS", True)

    class Boom:
        def __init__(self, *a, **k): raise RuntimeError("offline")
    monkeypatch.setitem(sys.modules, "gtts", pytypes.SimpleNamespace(gTTS=Boom))
    make_session("s")
    f = final(await run(service, "s", action="viva_start"))
    assert f["error"] is None and f["state"]["viva_score"]["pending_question"] and f["state"]["audio_url"] == ""
