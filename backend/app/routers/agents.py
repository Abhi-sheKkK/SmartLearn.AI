"""Unified agent API.

    POST /api/sessions/{id}/agent            run one turn (SSE stream by default, JSON with stream=false)
    GET  /api/sessions/{id}/agent/state      checkpointed state projection (viva score, quiz, notes, ...)
    GET  /api/sessions/{id}/agent/history    conversation history, optionally filtered by agent

This replaces the separate /notes, /chat, /test/generate and /test/submit routes: the supervisor node
inside the graph decides which specialist handles the turn, and LangGraph's checkpointer (keyed by
thread_id == session_id) owns all conversation state.
"""
import asyncio
import json
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.app.graph.service import AgentRequestError, AgentService
from backend.app.models import AgentRequest, AgentResponse
from backend.database import get_session

router = APIRouter(tags=["agents"])


def _service(request: Request) -> AgentService:
    service = getattr(request.app.state, "agent_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Agent graph is not initialised")
    return service


async def _require_session(session_id: str) -> dict:
    session = await asyncio.to_thread(get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"


async def _sse_stream(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for event in events:
        yield _sse(event)
    yield "event: done\ndata: {}\n\n"


@router.post("/sessions/{session_id}/agent")
async def run_agent(session_id: str, body: AgentRequest, request: Request):
    """Run one turn of the multi-agent graph for this session."""
    await _require_session(session_id)
    service = _service(request)

    try:
        await service.prepare(session_id, body)  # reject impossible requests (e.g. resume with nothing paused) with a real HTTP status
    except AgentRequestError as e:
        raise HTTPException(status_code=e.status, detail=e.detail)

    if body.stream:
        return StreamingResponse(
            _sse_stream(service.stream(session_id, body)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    final = await service.run_once(session_id, body)
    if final.get("type") == "error":
        raise HTTPException(status_code=final.get("status", 500), detail=final.get("message", "Agent run failed"))
    return AgentResponse.model_validate({k: v for k, v in final.items() if k != "type"})


@router.get("/sessions/{session_id}/agent/state")
async def get_agent_state(session_id: str, request: Request):
    session = await _require_session(session_id)
    view = await _service(request).snapshot(session_id)
    if not view["notes_draft"]:
        # Notes saved by an earlier session of the same video (or generated before checkpointing).
        view["notes_draft"] = session.get("notes_markdown") or ""
    return view


@router.get("/sessions/{session_id}/agent/history")
async def get_agent_history(session_id: str, request: Request, agent: Optional[str] = None):
    await _require_session(session_id)
    return {"messages": await _service(request).history(session_id, agent)}
