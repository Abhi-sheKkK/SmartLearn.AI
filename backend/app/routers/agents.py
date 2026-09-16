import asyncio
from fastapi import APIRouter, HTTPException
from backend.app.models import (
    ChatRequest, ChatResponse, NotesResponse,
    GenerateTestRequest, SubmitTestRequest,
    TestResponse, TestQuestion, TestResultResponse
)
from backend.database import (
    get_session, update_session,
    insert_message, get_messages,
    insert_test_result, get_test_results,
    save_video_material, get_video_material
)
from backend.app.agents.supervisor import get_graph
from backend.app.agents.notes_agent import notes_node
from backend.app.agents.test_agent import generate_test_questions, evaluate_test, get_or_build_mcq_pool
from langchain_core.messages import HumanMessage

router = APIRouter(tags=["agents"])

# In-memory store for test data (answer keys) keyed by session_id
_test_store = {}


@router.post("/sessions/{session_id}/notes", response_model=NotesResponse)
async def generate_notes(session_id: str):
    """Generate or retrieve cached AI notes for the session's lecture."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    transcript = session.get("transcript_text", "")
    frame_urls = session.get("frame_urls", [])
    video_id = session.get("video_id", "")
    
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript available. Process the video first.")
    
    # 1. Check if notes are already saved in session or video material cache
    existing_notes = session.get("notes_markdown") or get_video_material(video_id, "notes_markdown")
    if existing_notes:
        print(f"[NOTES MATERIAL CACHE HIT] Returning saved notes for video {video_id} (Session {session_id})")
        update_session(session_id, {"notes_markdown": existing_notes})
        return NotesResponse(
            markdown=existing_notes,
            frame_urls=frame_urls if isinstance(frame_urls, list) else []
        )
    
    # 2. Run the notes agent directly to generate new notes
    print(f"[AGENT: NOTES] Generating lecture notes for video {video_id}...")
    state = {
        "messages": [HumanMessage(content="Generate comprehensive lecture notes")],
        "session_id": session_id,
        "transcript": transcript,
        "frame_urls": frame_urls if isinstance(frame_urls, list) else [],
        "current_agent": "notes",
        "notes": "",
        "test_data": {},
        "viva_score": {},
    }
    
    result = await asyncio.to_thread(notes_node, state)
    notes_md = result.get("notes", "")
    
    # Save notes to session and global video material cache
    update_session(session_id, {"notes_markdown": notes_md})
    save_video_material(video_id, "notes_markdown", notes_md)
    
    return NotesResponse(
        markdown=notes_md,
        frame_urls=frame_urls if isinstance(frame_urls, list) else []
    )


@router.get("/sessions/{session_id}/notes", response_model=NotesResponse)
async def get_notes(session_id: str):
    """Get previously generated notes."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    notes_md = session.get("notes_markdown", "")
    frame_urls = session.get("frame_urls", [])
    
    return NotesResponse(
        markdown=notes_md or "",
        frame_urls=frame_urls if isinstance(frame_urls, list) else []
    )


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat_with_agent(session_id: str, request: ChatRequest):
    """Send a message to the doubt/viva agent."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    transcript = session.get("transcript_text", "")
    frame_urls = session.get("frame_urls", [])
    
    # Save user message
    insert_message({
        "session_id": session_id,
        "role": "user",
        "content": request.message,
        "agent_type": request.agent_type
    })
    
    # Get conversation history
    history = get_messages(session_id, request.agent_type)
    
    # Build message list from history
    from langchain_core.messages import AIMessage
    lang_messages = []
    for msg in history:
        if msg.get("role") == "user":
            lang_messages.append(HumanMessage(content=msg["content"]))
        else:
            lang_messages.append(AIMessage(content=msg["content"]))
    
    # Add the current user message
    lang_messages.append(HumanMessage(content=request.message))
    
    # Prepare state
    state = {
        "messages": lang_messages,
        "session_id": session_id,
        "transcript": transcript,
        "frame_urls": frame_urls if isinstance(frame_urls, list) else [],
        "current_agent": request.agent_type,
        "notes": "",
        "test_data": {},
        "viva_score": session.get("viva_score", {"asked": 0, "correct": 0}),
    }
    
    # Route to the right agent
    if request.agent_type == "viva":
        from backend.app.agents.viva_agent import viva_node
        result = await asyncio.to_thread(viva_node, state)
    else:
        from backend.app.agents.doubt_agent import doubt_node
        result = await asyncio.to_thread(doubt_node, state)
    
    # Extract response
    response_messages = result.get("messages", [])
    response_content = response_messages[-1].content if response_messages else "I couldn't generate a response."
    
    # Save assistant message
    insert_message({
        "session_id": session_id,
        "role": "assistant",
        "content": response_content,
        "agent_type": request.agent_type
    })
    
    # Update viva score if applicable
    if request.agent_type == "viva" and "viva_score" in result:
        update_session(session_id, {"viva_score": result["viva_score"]})
    
    return ChatResponse(
        role="assistant",
        content=response_content,
        agent_type=request.agent_type
    )


@router.get("/sessions/{session_id}/chat/history")
async def get_chat_history(session_id: str, agent_type: str = "doubt"):
    """Get chat history for a session."""
    messages = get_messages(session_id, agent_type)
    return {"messages": messages}


@router.post("/sessions/{session_id}/test/generate")
async def generate_test(session_id: str, request: GenerateTestRequest):
    """Generate or sample from a large pre-saved MCQ question bank (N=30) for a video."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    transcript = session.get("transcript_text", "")
    video_id = session.get("video_id", "")
    
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript available.")
    
    test_data = await asyncio.to_thread(
        get_or_build_mcq_pool, video_id, transcript, request.num_questions, request.difficulty
    )
    
    if not test_data["questions"]:
        raise HTTPException(status_code=500, detail="Failed to generate test questions.")
    
    _test_store[session_id] = test_data
    
    return {
        "questions": test_data["questions"]
    }


@router.post("/sessions/{session_id}/test/submit")
async def submit_test(session_id: str, request: SubmitTestRequest):
    """Submit test answers and get results."""
    test_data = _test_store.get(session_id)
    if not test_data:
        raise HTTPException(status_code=404, detail="No test found. Generate a test first.")
    
    results = evaluate_test(test_data, request.answers)
    
    # Save results to database
    insert_test_result({
        "session_id": session_id,
        "questions": test_data["questions"],
        "user_answers": request.answers,
        "score": results["score"],
        "total": results["total"]
    })
    
    return results


@router.get("/sessions/{session_id}/test/results")
async def get_test_results_endpoint(session_id: str):
    """Get test results for a session."""
    results = get_test_results(session_id)
    return {"results": results}
