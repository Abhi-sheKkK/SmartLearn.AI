"""Entry node: hydrate lecture context into the thread's state and reset per-turn flags.

The API no longer assembles agent state by hand. It hands the graph one new user input; this node
fetches the lecture transcript / frames from the session record the first time a thread needs them
(later turns reuse the checkpointed copy) so every other node just reads state.
"""
import asyncio

from langchain_core.runnables import RunnableConfig
from backend.database import get_session


async def load_context(state: dict, config: RunnableConfig) -> dict:
    update: dict = {
        # per-turn resets
        "error": None,
        "retry_count": 0,
        "use_fallback": False,
        "tool_rounds": 0,
        "viva_continue": False,
        "audio_url": "",
        "route": None,
    }

    if not state.get("transcript"):
        session_id = state.get("session_id") or config["configurable"]["thread_id"]
        session = await asyncio.to_thread(get_session, session_id) or {}
        transcript = session.get("transcript_text") or ""
        frame_urls = session.get("frame_urls") or []
        if not isinstance(frame_urls, list):
            frame_urls = []
        update.update(
            transcript=transcript,
            frame_urls=frame_urls,
            meta={
                "video_id": session.get("video_id", ""),
                "title": session.get("title") or "",
                "transcript_chars": len(transcript),
                "frame_count": len(frame_urls),
            },
        )
    return update
