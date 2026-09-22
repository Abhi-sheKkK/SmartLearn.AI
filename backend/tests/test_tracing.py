"""LangSmith tracing is wired from settings into the process env and into every run's config."""
import os
from types import SimpleNamespace

import pytest

from backend.app.graph.tracing import configure_tracing, run_config

KEYS = ["LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
        "LANGCHAIN_PROJECT", "LANGSMITH_PROJECT", "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def cfg(**kw):
    base = dict(LANGCHAIN_TRACING_V2="true", LANGCHAIN_API_KEY="ls-key", LANGCHAIN_PROJECT="proj", LANGCHAIN_ENDPOINT="")
    return SimpleNamespace(**{**base, **kw})


def test_enabled_exports_all_langsmith_variables(monkeypatch):
    status = configure_tracing(cfg(LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"))
    assert status["enabled"] is True
    for name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        assert os.environ[name] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == os.environ["LANGSMITH_API_KEY"] == "ls-key"
    assert os.environ["LANGCHAIN_PROJECT"] == os.environ["LANGSMITH_PROJECT"] == "proj"
    assert os.environ["LANGCHAIN_ENDPOINT"] == "https://eu.api.smith.langchain.com"


def test_tracing_requested_without_a_key_is_disabled_not_broken():
    status = configure_tracing(cfg(LANGCHAIN_API_KEY=""))
    assert status["enabled"] is False and "no LANGCHAIN_API_KEY" in status["reason"]
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_explicitly_disabled():
    assert configure_tracing(cfg(LANGCHAIN_TRACING_V2="false"))["enabled"] is False
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_run_config_carries_thread_and_trace_metadata():
    c = run_config("sess-9", video_id="abc")
    assert c["configurable"]["thread_id"] == "sess-9"
    assert c["metadata"] == {"session_id": "sess-9", "thread_id": "sess-9", "video_id": "abc"}
    assert "session:sess-9" in c["tags"] and c["run_name"] == "smartlearn-agent"


def test_app_startup_applies_tracing_settings(monkeypatch, brain):
    from fastapi.testclient import TestClient
    from backend.config import get_settings
    from backend.main import app

    monkeypatch.setattr(get_settings(), "LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setattr(get_settings(), "LANGCHAIN_API_KEY", "ls-from-dotenv")
    monkeypatch.setattr(get_settings(), "LANGCHAIN_PROJECT", "smartlearn-test")
    assert "LANGCHAIN_API_KEY" not in os.environ
    with TestClient(app):                                            # runs the FastAPI lifespan
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "ls-from-dotenv"
        assert os.environ["LANGCHAIN_PROJECT"] == "smartlearn-test"
