import json
import uuid
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
        max_tokens=600,
    )

def generate_test_questions(transcript: str, num_questions: int = 10, difficulty: str = "medium") -> dict:
    """
    Generate MCQ test questions from the transcript.
    Returns dict with questions list and answer key.
    """
    llm = create_llm()
    
    max_chars = 20000
    context = transcript[:max_chars] if transcript else "No transcript available."
    
    prompt = f"""You are an expert test creator for educational content. Generate {num_questions} multiple-choice questions (MCQs) based on the following lecture transcript.

Difficulty level: {difficulty}
- easy: Basic recall and definition questions
- medium: Understanding and application questions
- hard: Analysis, evaluation, and tricky conceptual questions

LECTURE TRANSCRIPT:
{context}

INSTRUCTIONS:
- Generate exactly {num_questions} questions
- Each question must have exactly 4 options (A, B, C, D)
- Only one option should be correct
- Questions should cover different topics from the lecture
- Include a brief explanation for the correct answer

Respond with ONLY a valid JSON object in this exact format:
{{
  "questions": [
    {{
      "question": "What is ...?",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "correct_answer": 0,
      "explanation": "Brief explanation of why this is correct"
    }}
  ]
}}

The correct_answer field is the 0-based index of the correct option."""
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # Clean up response - remove code blocks if present
    if content.startswith('```'):
        content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    
    try:
        data = json.loads(content)
        questions = data.get("questions", [])
        
        # Add IDs to each question
        processed_questions = []
        answer_key = {}
        
        for i, q in enumerate(questions):
            q_id = f"q_{uuid.uuid4().hex[:8]}"
            processed_questions.append({
                "id": q_id,
                "question": q["question"],
                "options": q["options"],
            })
            answer_key[q_id] = {
                "correct_answer": q["correct_answer"],
                "explanation": q.get("explanation", "")
            }
        
        return {
            "questions": processed_questions,
            "answer_key": answer_key
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing test questions: {e}")
        return {"questions": [], "answer_key": {}}


import random
from backend.database import get_video_material, save_video_material

TARGET_POOL_SIZE = 30

def get_or_build_mcq_pool(video_id: str, transcript: str, num_questions: int = 10, difficulty: str = "medium") -> dict:
    """
    Builds and maintains a large pool of unique MCQs (target N=30) per video.
    Returns a random sample of `num_questions` from the pool.
    """
    cache_key = f"mcq_pool_{difficulty}"
    
    # Retrieve existing question pool from cache
    existing_pool = get_video_material(video_id, cache_key) or {"questions": [], "answer_key": {}}
    pool_questions = list(existing_pool.get("questions", []))
    pool_answer_key = dict(existing_pool.get("answer_key", {}))
    
    # If pool already has reached TARGET_POOL_SIZE (or enough questions), sample directly!
    if len(pool_questions) >= TARGET_POOL_SIZE or (len(pool_questions) >= num_questions and len(pool_questions) >= 20):
        print(f"[MCQ POOL HIT] Sampling {num_questions} questions from pre-saved pool of {len(pool_questions)} MCQs for video {video_id}...")
        sample_size = min(num_questions, len(pool_questions))
        sampled_questions = random.sample(pool_questions, sample_size)
        sampled_keys = {q["id"]: pool_answer_key[q["id"]] for q in sampled_questions if q["id"] in pool_answer_key}
        return {
            "questions": sampled_questions,
            "answer_key": sampled_keys
        }
    
    # Otherwise, generate a new batch of MCQs to grow the pool towards TARGET_POOL_SIZE
    needed = max(num_questions, TARGET_POOL_SIZE - len(pool_questions))
    gen_count = min(needed, 5)
    print(f"[MCQ POOL BUILDING] Pool currently has {len(pool_questions)}/{TARGET_POOL_SIZE} MCQs. Generating {gen_count} new unique MCQs...")
    
    new_batch = generate_test_questions(transcript, gen_count, difficulty)
    new_qs = new_batch.get("questions", [])
    new_key = new_batch.get("answer_key", {})
    
    # Deduplicate against existing pool questions by lowercased question stem
    existing_stems = {q["question"].strip().lower() for q in pool_questions}
    added_count = 0
    
    for q in new_qs:
        stem = q["question"].strip().lower()
        if stem not in existing_stems:
            existing_stems.add(stem)
            pool_questions.append(q)
            if q["id"] in new_key:
                pool_answer_key[q["id"]] = new_key[q["id"]]
            added_count += 1
            
    print(f"[MCQ POOL UPDATE] Added {added_count} unique MCQs. Total pool size: {len(pool_questions)}.")
    
    updated_pool = {
        "questions": pool_questions,
        "answer_key": pool_answer_key
    }
    save_video_material(video_id, cache_key, updated_pool)
    
    # Sample num_questions for this user's test
    sample_size = min(num_questions, len(pool_questions))
    sampled_questions = random.sample(pool_questions, sample_size)
    sampled_keys = {q["id"]: pool_answer_key[q["id"]] for q in sampled_questions if q["id"] in pool_answer_key}
    
    return {
        "questions": sampled_questions,
        "answer_key": sampled_keys
    }


def evaluate_test(test_data: dict, user_answers: dict) -> dict:
    """
    Evaluate user's test answers against the answer key.
    Returns score and detailed results.
    """
    answer_key = test_data.get("answer_key", {})
    questions = test_data.get("questions", [])
    
    results = []
    score = 0
    total = len(questions)
    
    for q in questions:
        q_id = q["id"]
        user_answer = user_answers.get(q_id)
        correct_info = answer_key.get(q_id, {})
        correct_idx = correct_info.get("correct_answer", -1)
        explanation = correct_info.get("explanation", "")
        
        is_correct = False
        if user_answer is not None:
            try:
                is_correct = int(user_answer) == correct_idx
            except (ValueError, TypeError):
                is_correct = False
        
        if is_correct:
            score += 1
        
        results.append({
            "question_id": q_id,
            "question": q["question"],
            "options": q["options"],
            "user_answer": int(user_answer) if user_answer is not None else None,
            "correct_answer": correct_idx,
            "is_correct": is_correct,
            "explanation": explanation
        })
    
    return {
        "score": score,
        "total": total,
        "percentage": round(score / total * 100, 1) if total > 0 else 0,
        "results": results
    }


def test_node(state: SmartLearnState) -> dict:
    """
    LangGraph node for test generation.
    Generates MCQ questions from the transcript.
    """
    transcript = state.get("transcript", "")
    messages = state.get("messages", [])
    
    last_message = messages[-1] if messages else None
    user_input = last_message.content if last_message and hasattr(last_message, 'content') else ""
    
    if not transcript:
        return {
            "messages": [AIMessage(content="No transcript available. Please process a video first.")],
        }
    
    # Parse number of questions and difficulty from user message
    num_q = 10
    difficulty = "medium"
    
    user_lower = user_input.lower()
    if "easy" in user_lower:
        difficulty = "easy"
    elif "hard" in user_lower:
        difficulty = "hard"
    
    # Try to extract number
    import re
    numbers = re.findall(r'\d+', user_input)
    if numbers:
        num = int(numbers[0])
        if 1 <= num <= 20:
            num_q = num
    
    test_data = generate_test_questions(transcript, num_q, difficulty)
    
    if not test_data["questions"]:
        return {
            "messages": [AIMessage(content="Sorry, I couldn't generate test questions. Please try again.")],
        }
    
    return {
        "messages": [AIMessage(content=f"I've generated {len(test_data['questions'])} {difficulty} MCQ questions. Go to the Test tab to take the test!")],
        "test_data": test_data
    }
