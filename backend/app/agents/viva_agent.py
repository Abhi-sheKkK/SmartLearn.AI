import json
from langchain_core.messages import AIMessage, HumanMessage
from backend.config import get_settings
from backend.app.agents.state import SmartLearnState
from langchain_groq import ChatGroq


def create_llm():
    settings = get_settings()
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.5,
        max_tokens=600,
    )


def viva_node(state: SmartLearnState) -> dict:
    """
    Conducts an interactive viva/interview session.
    Asks questions, evaluates answers, provides feedback.
    """
    llm = create_llm()
    transcript = state.get("transcript", "")
    messages = state.get("messages", [])
    viva_score = state.get("viva_score", {"asked": 0, "correct": 0})
    
    last_message = messages[-1] if messages else None
    user_input = last_message.content if last_message and hasattr(last_message, 'content') else ""
    
    # Truncate transcript for context
    max_chars = 15000
    context_transcript = transcript[:max_chars] if transcript else "No transcript available."
    
    # Build conversation history
    chat_history = ""
    recent_messages = messages[-12:] if len(messages) > 12 else messages
    for msg in recent_messages[:-1]:
        role = "Student" if isinstance(msg, HumanMessage) else "Examiner"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        chat_history += f"{role}: {content}\n"
    
    is_start = "start" in user_input.lower() or "begin" in user_input.lower() or viva_score.get("asked", 0) == 0
    is_end = "end" in user_input.lower() or "stop" in user_input.lower() or "quit" in user_input.lower()
    
    if is_end:
        asked = viva_score.get("asked", 0)
        correct = viva_score.get("correct", 0)
        summary = f"""## 🎓 Viva Session Complete!

**Questions Asked:** {asked}
**Good Answers:** {correct}
**Score:** {correct}/{asked} ({(correct/asked*100) if asked > 0 else 0:.0f}%)

{'Excellent performance! 🌟' if asked > 0 and correct/asked >= 0.7 else 'Keep practicing! You\'ll improve. 💪'}

Feel free to start another viva session whenever you\'re ready."""
        return {
            "messages": [AIMessage(content=summary)],
            "viva_score": {"asked": 0, "correct": 0}
        }
    
    if is_start and not chat_history.strip():
        prompt = f"""You are a friendly but thorough viva examiner for SmartLearn.AI.

Based on this lecture transcript, ask the student a conceptual question to test their understanding.
Start with a medium-difficulty question.

LECTURE CONTENT:
{context_transcript}

INSTRUCTIONS:
- Ask ONE clear, focused question
- The question should test understanding, not just memorization
- Start with: "Welcome to your viva session! Let's begin.\n\n" then the question
- Format the question clearly in Markdown
- After the question, add: "\n\n*Take your time to think and respond.*"
"""
    else:
        prompt = f"""You are a viva examiner for SmartLearn.AI. You're in an ongoing viva session.

LECTURE CONTENT (for reference):
{context_transcript}

PREVIOUS CONVERSATION:
{chat_history}

STUDENT'S LATEST ANSWER: {user_input}

INSTRUCTIONS:
1. Evaluate the student's answer:
   - Is it correct? Partially correct? Incorrect?
   - Provide brief, constructive feedback
2. Give the correct/complete answer if the student was wrong or incomplete
3. Then ask the NEXT question (different topic from previous questions)
4. Format your response as:

**Evaluation:** [Your feedback on their answer]

**Correct Answer:** [Brief correct answer if needed]

---

**Next Question:** [Your next question]

*Take your time to think and respond.*

Adjust difficulty: If the student answered well, make the next question harder. If they struggled, make it slightly easier."""
    
    response = llm.invoke(prompt)
    
    # Update viva score tracking
    new_score = viva_score.copy() if isinstance(viva_score, dict) else {"asked": 0, "correct": 0}
    if not is_start or chat_history.strip():
        new_score["asked"] = new_score.get("asked", 0) + 1
        # Simple heuristic: if feedback contains positive indicators, count as correct
        response_lower = response.content.lower()
        if any(word in response_lower for word in ["correct", "excellent", "great", "well done", "right", "good answer", "exactly"]):
            new_score["correct"] = new_score.get("correct", 0) + 1
    
    return {
        "messages": [AIMessage(content=response.content)],
        "viva_score": new_score
    }
