import pytest
from fastapi.testclient import TestClient

from backend import database
from backend.app.graph import build_graph, open_checkpointer
from backend.app.graph.llm import reset_learned_limits, set_llm_factory
from backend.app.graph.service import AgentService
from backend.app.models import AgentRequest
from backend.config import get_settings
from backend.tests.fakes import Brain

TRANSCRIPT = "Neural networks learn by gradient descent. The derivative of the loss gives the gradient. " * 6


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    settings = get_settings()
    monkeypatch.setattr(settings, "CHECKPOINT_BACKEND", "memory")
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(settings, "LLM_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(settings, "ENABLE_TTS", False)
    monkeypatch.setattr(settings, "GROQ_MODEL", "primary-model")
    monkeypatch.setattr(settings, "GROQ_FALLBACK_MODEL", "fallback-model")
    monkeypatch.setattr(settings, "LLM_MAX_OUTPUT_TOKENS", 0)
    reset_learned_limits()
    monkeypatch.setattr(settings, "LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(database, "client", None)  # never touch a real Supabase from tests
    for store in (database._local_sessions, database._local_messages, database._local_test_results, database._local_video_materials):
        store.clear()
    yield
    set_llm_factory(None)
    reset_learned_limits()


@pytest.fixture
def brain():
    b = Brain()
    set_llm_factory(b.factory)
    return b


@pytest.fixture
def make_session():
    def _make(session_id: str, transcript: str = TRANSCRIPT, frame_urls=None, video_id: str = None, **extra):
        database.insert_session({
            "id": session_id, "youtube_url": f"https://youtu.be/{session_id}", "video_id": video_id or session_id,
            "status": "completed", "transcript_text": transcript, "frame_urls": frame_urls or [], **extra,
        })
        return session_id
    return _make


@pytest.fixture
async def service(brain):
    async with open_checkpointer(get_settings()) as saver:
        yield AgentService(build_graph(saver))


@pytest.fixture
def client(brain):
    from backend.main import app
    with TestClient(app) as c:
        yield c


async def run(service: AgentService, session_id: str, **kwargs) -> list[dict]:
    """Execute one turn and return all events."""
    return [e async for e in service.stream(session_id, AgentRequest(**kwargs))]


def final(events: list[dict]) -> dict:
    assert events[-1]["type"] in ("final", "error"), events[-1]
    return events[-1]


def types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]
