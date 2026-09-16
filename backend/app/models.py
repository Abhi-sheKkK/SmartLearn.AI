from pydantic import BaseModel
from typing import Optional

# Request models
class CreateSessionRequest(BaseModel):
    youtube_url: str

class ChatRequest(BaseModel):
    message: str
    agent_type: str = "doubt"

class GenerateTestRequest(BaseModel):
    num_questions: int = 10
    difficulty: str = "medium"

class SubmitTestRequest(BaseModel):
    answers: dict[str, str]

# Response models
class SessionResponse(BaseModel):
    id: str
    youtube_url: str
    video_id: str
    title: Optional[str] = None
    status: str
    created_at: Optional[str] = None

class ChatResponse(BaseModel):
    role: str
    content: str
    agent_type: str

class NotesResponse(BaseModel):
    markdown: str
    latex: Optional[str] = None
    frame_urls: list[str] = []

class TestQuestion(BaseModel):
    id: str
    question: str
    options: list[str]

class TestResponse(BaseModel):
    questions: list[TestQuestion]

class TestResultResponse(BaseModel):
    score: int
    total: int
    results: list[dict]

class StatusResponse(BaseModel):
    status: str
    message: str = ""
