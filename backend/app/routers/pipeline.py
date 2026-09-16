import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.database import get_session
from backend.app.pipeline.processor import process_video

router = APIRouter(tags=["pipeline"])


@router.post("/sessions/{session_id}/process")
async def trigger_processing(session_id: str, background_tasks: BackgroundTasks):
    """Trigger video processing pipeline for a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    youtube_url = session.get("youtube_url")
    if not youtube_url:
        raise HTTPException(status_code=400, detail="No YouTube URL in session")
    
    # Run processing in background
    background_tasks.add_task(process_video, youtube_url, session_id)
    
    return {"status": "processing", "message": "Video processing started"}


@router.get("/sessions/{session_id}/frames")
async def get_frames(session_id: str):
    """Get extracted frame URLs for a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    frame_urls = session.get("frame_urls", [])
    return {"frames": frame_urls}
