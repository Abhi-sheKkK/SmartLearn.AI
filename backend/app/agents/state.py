from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class SmartLearnState(TypedDict):
    """Shared state for the SmartLearn.AI multi-agent system."""
    messages: Annotated[list, add_messages]
    session_id: str
    transcript: str
    frame_urls: list[str]
    current_agent: str  # 'supervisor', 'notes', 'doubt', 'viva', 'test'
    notes: str  # Generated notes markdown
    test_data: dict  # Generated test questions and answers
    viva_score: dict  # Viva tracking data
