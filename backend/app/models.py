from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal, Optional


# Request models
class CreateSessionRequest(BaseModel):
    youtube_url: str


AgentName = Literal["doubt", "viva", "test", "notes"]
AgentAction = Literal["generate_notes", "generate_test", "submit_test", "viva_start", "viva_end"]


class AgentParams(BaseModel):
    """Optional knobs for a turn. Which ones apply depends on the action / routed agent."""
    num_questions: Optional[int] = Field(default=None, ge=1, le=30)
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    answers: Optional[dict[str, str]] = None        # submit_test: {question_id: option_index}
    hitl: bool = True                               # viva: pause for elaboration on partial answers
    tts: Optional[bool] = None                      # viva: synthesize the question as audio
    refresh: bool = False                           # notes: regenerate instead of returning saved notes


class ResumePayload(BaseModel):
    """Answer to a human-in-the-loop interrupt (the viva scoring gate)."""
    elaboration: str = ""                           # empty / omitted = finalize the score as it stands


class AgentRequest(BaseModel):
    """Single request shape for every agent interaction: POST /api/sessions/{id}/agent."""
    message: Optional[str] = Field(default=None, max_length=8000)
    agent: Optional[AgentName] = None               # hint (which tab the user is on); the supervisor decides
    action: Optional[AgentAction] = None            # explicit command; bypasses intent classification
    params: AgentParams = Field(default_factory=AgentParams)
    resume: Optional[ResumePayload] = None
    stream: bool = True                             # SSE (text/event-stream) vs. a single JSON response

    @model_validator(mode="after")
    def _needs_something(self):
        if self.message is not None and not self.message.strip():
            self.message = None
        if self.message is None and self.action is None and self.resume is None:
            raise ValueError("Provide a message, an action, or a resume payload")
        return self


# Response models
class SessionResponse(BaseModel):
    id: str
    youtube_url: str
    video_id: str
    title: Optional[str] = None
    status: str
    created_at: Optional[str] = None


class AgentStateView(BaseModel):
    """Client-safe projection of the checkpointed graph state."""
    viva_score: dict = {}
    quiz: Optional[dict] = None                     # questions only -- never the answer key
    test_result: Optional[dict] = None
    notes_draft: str = ""
    audio_url: str = ""
    active_agent: Optional[str] = None
    interrupt: Optional[dict[str, Any]] = None      # set while the run is paused for human input


class AgentMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    agent: Optional[str] = None


class AgentResponse(BaseModel):
    thread_id: str
    agent: Optional[str] = None
    content: str = ""
    messages: list[AgentMessage] = []
    interrupted: bool = False
    interrupt: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    state: AgentStateView = AgentStateView()


class StatusResponse(BaseModel):
    status: str
    message: str = ""
