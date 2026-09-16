import json
from langchain_core.messages import AIMessage
from backend.config import get_settings
from backend.app.agents.state import SmartLearnState
from langchain_groq import ChatGroq


def create_llm():
    settings = get_settings()
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0.3,
        max_tokens=950,
    )


def notes_node(state: SmartLearnState) -> dict:
    """
    Generates comprehensive Markdown notes from the lecture transcript.
    Includes references to key frames/diagrams.
    """
    llm = create_llm()
    transcript = state.get("transcript", "")
    frame_urls = state.get("frame_urls", [])
    
    if not transcript:
        return {
            "messages": [AIMessage(content="No transcript available. Please process a video first.")],
            "notes": ""
        }
    
    # Truncate very long transcripts for the context window
    # Groq models typically have 8k-32k context
    max_chars = 25000
    truncated_transcript = transcript[:max_chars]
    if len(transcript) > max_chars:
        truncated_transcript += "\n\n[... transcript truncated for processing ...]"
    
    # Build frame references for inline embedding
    frame_refs = ""
    if frame_urls:
        frame_refs = "\nAVAILABLE VISUAL DIAGRAMS/SLIDES (Embed these inline in the notes):\n"
        for i, url in enumerate(frame_urls, 1):
            frame_refs += f"- Image {i}: ![Lecture Diagram {i}]({url})\n"
    
    prompt = f"""You are an expert note-taker and educator. Create comprehensive, well-structured lecture notes from the following transcript.

IMPORTANT GUIDELINES:
1. Create detailed, organized notes in Markdown format
2. Use proper heading hierarchy (# for main topic, ## for subtopics, ### for sub-subtopics)
3. Include key definitions, concepts, and explanations
4. Use bullet points and numbered lists for clarity
5. Include any mathematical formulas using LaTeX notation (wrap in $...$ for inline or $$...$$ for block)
6. Add blockquotes for important takeaways
7. Include code blocks where code examples are mentioned
8. CRITICAL INSTRUCTION FOR IMAGES:
   - Do NOT create a "Visual References" or "Lecture Frames" section at the end of the notes.
   - Embed each provided lecture frame image INLINE directly inside the specific section or subtopic where that diagram, formula, slide, or code example is explained.
   - Image embed syntax: ![Descriptive Caption](image_url)
9. Create a brief summary at the end.

{frame_refs}

TRANSCRIPT:
{truncated_transcript}

Generate the comprehensive lecture notes now:"""
    
    response = llm.invoke(prompt)
    notes_content = response.content
    
    return {
        "messages": [AIMessage(content="I've generated comprehensive notes from the lecture. You can view them in the Notes tab.")],
        "notes": notes_content
    }
