import uuid
from fastapi import APIRouter, HTTPException
from backend.app.models import CreateSessionRequest, SessionResponse, StatusResponse
from backend.database import insert_session, get_session, update_session, get_completed_session_by_video_id
from backend.app.pipeline.youtube_data import extract_id

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new learning session from a YouTube URL."""
    video_id = extract_id(request.youtube_url)
    if video_id == "None":
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Check if this video has already been processed in a previous session
    existing = get_completed_session_by_video_id(video_id)
    
    session_id = str(uuid.uuid4())
    session_data = {
        "id": session_id,
        "youtube_url": request.youtube_url,
        "video_id": video_id,
        "status": "completed" if existing else "pending",
    }
    
    if existing:
        session_data["transcript_text"] = existing.get("transcript_text", "")
        session_data["frame_urls"] = existing.get("frame_urls", [])
        session_data["notes_markdown"] = existing.get("notes_markdown", "")
    
    result = insert_session(session_data)
    
    return SessionResponse(
        id=session_id,
        youtube_url=request.youtube_url,
        video_id=video_id,
        status=session_data["status"],
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_details(session_id: str):
    """Get session details."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        id=session.get("id", session_id),
        youtube_url=session.get("youtube_url", ""),
        video_id=session.get("video_id", ""),
        title=session.get("title"),
        status=session.get("status", "unknown"),
        created_at=session.get("created_at"),
    )


@router.get("/{session_id}/status", response_model=StatusResponse)
async def get_session_status(session_id: str):
    """Poll session processing status."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = session.get("status", "unknown")
    messages_map = {
        "pending": "Session created, waiting to start processing...",
        "extracting_id": "Extracting video info...",
        "downloading_transcript": "Downloading transcript...",
        "chunking_transcript": "Analyzing transcript structure...",
        "processing_cues": "Analyzing visual cues with AI...",
        "extracting_frames": "Extracting key visual frames...",
        "deduplicating_frames": "Filtering deduplicated frames...",
        "uploading_frames": "Saving visual frames...",
        "ready": "Processing complete! Your dashboard is ready.",
        "completed": "Processing complete! Your dashboard is ready.",
        "error": session.get("error_message") or "An error occurred during processing.",
    }
    
    return StatusResponse(
        status=status,
        message=messages_map.get(status, "Processing...")
    )
