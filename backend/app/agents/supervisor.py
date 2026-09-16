import json
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from backend.config import get_settings
from backend.app.agents.state import SmartLearnState
from backend.app.agents.notes_agent import notes_node
from backend.app.agents.doubt_agent import doubt_node
from backend.app.agents.viva_agent import viva_node
from backend.app.agents.test_agent import test_node


def create_llm():
    settings = get_settings()
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.1,
    )


def supervisor_node(state: SmartLearnState):
    """
    Supervisor decides which agent to route to based on the latest user message.
    """
    llm = create_llm()
    
    last_message = state["messages"][-1] if state["messages"] else None
    if not last_message:
        return {"current_agent": "supervisor"}
    
    user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    routing_prompt = f"""You are a routing agent for SmartLearn.AI, an educational platform.
Based on the user's message, decide which specialized agent should handle it.

Available agents:
- "notes": Generate or retrieve lecture notes. Triggers: generate notes, create notes, make notes, summarize lecture, show notes
- "doubt": Answer questions about the lecture content. Triggers: questions about the topic, explain something, what is, how does, why
- "viva": Conduct viva/interview practice. Triggers: start viva, interview me, viva practice, ask me questions, oral exam
- "test": Generate or handle MCQ tests. Triggers: generate test, quiz me, create mcq, take test, test my knowledge
- "end": The conversation is finished or it's a greeting/thanks. Triggers: bye, thanks, done

User message: "{user_input}"

Respond with ONLY a JSON object: {{"agent": "<agent_name>", "reason": "<brief reason>"}}"""
    
    response = llm.invoke(routing_prompt)
    
    try:
        # Try to parse the response as JSON
        content = response.content.strip()
        # Handle cases where LLM wraps in code blocks
        if content.startswith('```'):
            content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        routing = json.loads(content)
        target = routing.get("agent", "doubt")
    except (json.JSONDecodeError, AttributeError):
        # Default to doubt agent if parsing fails
        target = "doubt"
    
    # Validate target
    valid_agents = ["notes", "doubt", "viva", "test", "end"]
    if target not in valid_agents:
        target = "doubt"
    
    if target == "end":
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content="Thank you for using SmartLearn.AI! Feel free to come back anytime. 🎓")],
            "current_agent": "end"
        }
    
    return Command(
        goto=target,
        update={"current_agent": target}
    )


def build_graph():
    """Build and compile the multi-agent graph."""
    graph = StateGraph(SmartLearnState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("notes", notes_node)
    graph.add_node("doubt", doubt_node)
    graph.add_node("viva", viva_node)
    graph.add_node("test", test_node)
    
    # Entry point
    graph.add_edge(START, "supervisor")
    
    # All agents return to supervisor... but since we're using single-turn,
    # each agent call goes: supervisor -> agent -> END
    graph.add_edge("notes", END)
    graph.add_edge("doubt", END)
    graph.add_edge("viva", END)
    graph.add_edge("test", END)
    
    return graph.compile()


# Compiled graph singleton
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
