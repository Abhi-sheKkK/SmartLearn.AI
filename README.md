# SmartLearn.AI 🎓✨

Transform YouTube lectures into interactive learning experiences using AI.

## Features

- **📝 Smart LaTeX Notes**: AI-generated structured notes with embedded visual diagrams, slides, and flowcharts extracted directly from YouTube video lectures.
- **💬 Doubt Assistant Chatbot**: Multi-turn chat agent that answers student questions by cross-referencing video transcripts, timestamp visual cues, and LLM knowledge.
- **🎤 Viva / Interview Practice**: Interactive 1-on-1 viva session where an AI examiner asks conceptual questions, scores your verbal responses, and provides targeted feedback.
- **📋 MCQ Quiz Engine**: Auto-generates customizable quizzes with instant grading and detailed explanations per question.
- **⚡ Visual Frame Extraction**: Automatically detects moments in the lecture where instructors point to slides, diagrams, or code on screen using LLM cue analysis and SSIM image deduplication.

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic v2
- **Multi-Agent Framework**: LangGraph StateGraph (Supervisor & Specialized Agent nodes)
- **LLM Engine**: Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`)
- **Database & Storage**: Supabase PostgreSQL + Supabase Storage (with local JSON fallback)
- **Video Processing**: `yt-dlp`, `youtube-transcript-api`, OpenCV, `scikit-image` (SSIM), FFmpeg
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphic design system), Vanilla JS, KaTeX, Highlight.js, Marked.js

## Project Structure

```
smartlearn_ai/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph Supervisor and Agent Nodes
│   │   │   ├── supervisor.py
│   │   │   ├── notes_agent.py
│   │   │   ├── doubt_agent.py
│   │   │   ├── viva_agent.py
│   │   │   └── test_agent.py
│   │   ├── pipeline/        # Video Transcript & Frame Extraction
│   │   │   ├── youtube_data.py
│   │   │   └── processor.py
│   │   ├── routers/         # FastAPI REST API Endpoints
│   │   │   ├── sessions.py
│   │   │   ├── pipeline.py
│   │   │   └── agents.py
│   │   └── models.py        # Pydantic Models
│   ├── config.py            # Environment Configuration
│   ├── database.py          # Supabase & Local Database Handler
│   └── main.py              # FastAPI Application Entrypoint
├── frontend/
│   ├── css/                 # Glassmorphism Design System & Styles
│   ├── js/                  # Modules (api, app, dashboard, notes, chat, viva, test)
│   ├── index.html           # Landing Page
│   └── dashboard.html       # Workspace Dashboard
├── schema.sql               # PostgreSQL Database Schema
├── render.yaml              # Render Deployment Blueprint
└── requirements.txt         # Python Dependencies
```

## Setup & Running Locally

1. **Clone & Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   SUPABASE_URL=your_supabase_url (optional, defaults to local storage if empty)
   SUPABASE_KEY=your_supabase_anon_key
   SUPABASE_STORAGE_BUCKET=smartlearn-frames
   GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
   ```

3. **Database Setup (Optional - Supabase)**:
   Run the queries in `schema.sql` inside your Supabase SQL Editor.

4. **Start the Application**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

5. **Access the Application**:
   Open `http://localhost:8000` in your web browser.

## Deployment on Render

1. Connect your repository to Render.
2. Render will automatically detect `render.yaml` blueprint.
3. Set the `GROQ_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY` environment variables in the Render Dashboard.

---
© 2025 SmartLearn.AI
