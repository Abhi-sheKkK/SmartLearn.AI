from langchain_core.messages import AIMessage, HumanMessage
from backend.config import get_settings
from backend.app.agents.state import SmartLearnState
from langchain_groq import ChatGroq


def create_llm():
    settings = get_settings()
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.4,
        max_tokens=800,
    )


def doubt_node(state: SmartLearnState) -> dict:
    """
    Conversational chatbot that answers doubts about the lecture.
    Uses transcript and frames as context.
    """
    llm = create_llm()
    transcript = state.get("transcript", "")
    frame_urls = state.get("frame_urls", [])
    messages = state.get("messages", [])
    
    # Get the latest user message
    last_message = messages[-1] if messages else None
    user_question = last_message.content if last_message and hasattr(last_message, 'content') else "Hello"
    
    # Truncate transcript for context
    max_chars = 15000
    context_transcript = transcript[:max_chars] if transcript else "No transcript available."
    
    # Build conversation history (last 10 messages for context)
    chat_history = ""
    recent_messages = messages[-10:] if len(messages) > 10 else messages
    for msg in recent_messages[:-1]:  # Exclude the latest message
        role = "Student" if isinstance(msg, HumanMessage) else "Tutor"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        chat_history += f"{role}: {content}\n"
    
    prompt = f"""You are an expert AI tutor for SmartLearn.AI. A student is watching a YouTube lecture and has questions.

Your role:
- Answer the student's question clearly and thoroughly
- Reference specific parts of the lecture transcript when relevant
- If the question is about a visual concept, describe what the relevant frame/diagram shows
- Use examples and analogies to explain complex concepts
- Be encouraging and supportive
- Format your response in Markdown for readability
- Use LaTeX for any mathematical notation ($...$ for inline, $$...$$ for display)
- Keep answers focused and concise but comprehensive

LECTURE TRANSCRIPT (for reference):
{context_transcript}

{f'The lecture has {len(frame_urls)} visual frames/diagrams available.' if frame_urls else ''}

PREVIOUS CONVERSATION:
{chat_history}

STUDENT'S QUESTION: {user_question}

Provide a helpful, well-formatted answer:"""
    
    response = llm.invoke(prompt)
    
    return {
        "messages": [AIMessage(content=response.content)],
    }
