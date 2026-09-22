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
- **Multi-Agent Framework**: LangGraph `StateGraph` with a supervisor, parallel workers, human-in-the-loop interrupts and SQLite checkpointing (see [Agent architecture](#agent-architecture))
- **LLM Engine**: Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`)
- **Database & Storage**: Supabase PostgreSQL + Supabase Storage (with local JSON fallback)
- **Video Processing**: `yt-dlp`, `youtube-transcript-api`, OpenCV, `scikit-image` (SSIM), FFmpeg
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphic design system), Vanilla JS, KaTeX, Highlight.js, Marked.js

## Project Structure

```
smartlearn_ai/
├── backend/
│   ├── app/
│   │   ├── graph/           # LangGraph multi-agent system (state, nodes, builder, service)
│   │   │   ├── state.py         # AgentState schema
│   │   │   ├── builder.py       # graph wiring + compile (checkpointer, interrupts)
│   │   │   ├── supervisor.py    # intent routing / handoffs
│   │   │   ├── doubt.py  viva.py  quiz.py  notes.py   # specialist agents
│   │   │   ├── resilience.py    # retry / fallback edges
│   │   │   ├── tools.py         # python_repl, web_search, text_to_speech
│   │   │   ├── llm.py           # model factory, error classification
│   │   │   ├── service.py       # per-request execution + SSE event mapping
│   │   │   └── tracing.py       # LangSmith wiring
│   │   └── tests/ (backend/tests) # pytest suite (fake LLM, no network)
│   │   ├── pipeline/        # Video Transcript & Frame Extraction
│   │   │   ├── youtube_data.py
│   │   │   └── processor.py
│   │   ├── routers/         # FastAPI REST API Endpoints
│   │   │   ├── sessions.py
│   │   │   ├── pipeline.py
│   │   │   └── agents.py    # POST /api/sessions/{id}/agent
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

   # Optional -- multi-agent graph
   GROQ_FALLBACK_MODEL=openai/gpt-oss-20b   # used on rate limits / malformed output; empty disables
   LLM_MAX_OUTPUT_TOKENS=0                  # 0 = learn each model's limit automatically
   CHECKPOINT_BACKEND=sqlite                # or "memory"
   ENABLE_PYTHON_TOOL=true                  # sandboxed python_repl for the doubt agent
   ENABLE_WEB_SEARCH=true                   # DuckDuckGo lookups for the doubt agent
   ENABLE_TTS=true                          # spoken viva questions (gTTS)

   # Optional -- LangSmith tracing (switched off automatically if no key is set)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_key
   LANGCHAIN_PROJECT=smartlearn-ai
   ```

3. **Database Setup (Optional - Supabase)**:
   Run the queries in `schema.sql` inside your Supabase SQL Editor.

4. **Start the Application**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

5. **Access the Application**:
   Open `http://localhost:8000` in your web browser.

## Agent architecture

Every agent interaction goes through **one endpoint**, `POST /api/sessions/{id}/agent`. A LangGraph
`StateGraph` decides what to do; the checkpointer (`thread_id == session_id`) owns all conversation state.

```
START -> load_context -> supervisor -+-> doubt_agent <-> doubt_tools (python_repl, web_search)
                                     +-> viva_ask / viva_evaluate -> viva_score* -> viva_ask ...   (* human gate)
                                     +-> test_generate / test_grade
                                     +-> notes_prepare => [summarizer | latex | diagram_mapper] => notes_merge
   any failed node -> retry_gate -> (same model | fallback model) -> node,   or -> error_handler
```

- **Supervisor**: explicit `action`s route deterministically; otherwise an LLM intent classifier returns a
  schema-validated decision. Routing is per turn, so a question asked mid-viva is answered by the doubt agent
  while `viva_score` stays untouched and the viva resumes afterwards.
- **Human-in-the-loop**: the graph is compiled with `interrupt_before=["viva_score"]`. A partial answer pauses
  the run before scoring and asks the student to elaborate; resume with `{"resume": {"elaboration": "..."}}`
  (empty = finalize as is).
- **Fan-out / fan-in**: notes are built by three parallel workers (summary, LaTeX equations, diagram mapping)
  merged into one document. Workers reply in a delimited text format, not JSON, so LaTeX backslashes survive.
- **Resilience**: failures are classified (`rate_limit`, `too_large`, `malformed`, `transient`, `fatal`) and routed
  by conditional edges. Rate limits switch to the fallback model without waiting; a per-model output-token limit
  is learned from the API's own error and applied to later calls; a fallback model that doesn't exist is dropped.
- **Streaming**: `stream: true` (default) returns Server-Sent Events -- `start`, `route`, `token` (word by word),
  `tool_start`/`tool_end`, `retry`, `audio`, `notes_progress`, `interrupt`, `final`. `stream: false` returns one JSON
  document from the same code path.

```jsonc
POST /api/sessions/{id}/agent
{ "message": "What is a gradient?", "agent": "doubt" }                  // free text (supervisor routes it)
{ "action": "generate_test", "params": { "num_questions": 5, "difficulty": "easy" } }
{ "action": "submit_test",   "params": { "answers": { "<question_id>": "2" } } }
{ "action": "generate_notes", "params": { "refresh": true } }            // also: viva_start, viva_end
{ "resume": { "elaboration": "I meant that ..." } }                      // answer a human-in-the-loop interrupt

GET /api/sessions/{id}/agent/state      // viva score, quiz (never the answer key), notes, pending interrupt
GET /api/sessions/{id}/agent/history?agent=doubt
```

**Security note -- `python_repl`.** It deliberately does *not* use LangChain's in-process `PythonREPLTool`.
Snippets run in a separate interpreter with an empty environment, a temp working directory, CPU/file limits, a
timeout and an AST allowlist. That is defence in depth, not a security boundary: for untrusted multi-tenant
production use, run it in a container/microVM or set `ENABLE_PYTHON_TOOL=false`.

## Tests

```bash
python -m pytest        # ~110 tests, ~5s, no network or API keys needed (scripted fake LLM)
```

Covers multi-tenant isolation and persistence across restarts, supervisor handoffs, the HITL gate, parallel
fan-out, streaming over HTTP, and the retry / fallback edges.

## Deployment on Render

1. Connect your repository to Render.
2. Render will automatically detect `render.yaml` blueprint.
3. Set the `GROQ_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY` environment variables in the Render Dashboard (and `LANGCHAIN_API_KEY` to enable LangSmith tracing).
4. The SQLite checkpoint file lives under `backend/storage/`; on Render's ephemeral disk attach a persistent disk there (or set `CHECKPOINT_DB_PATH`) if conversations must survive redeploys.

---
© 2025 SmartLearn.AI
