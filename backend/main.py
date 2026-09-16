import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title='SmartLearn.AI API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/api/health')
def health_check():
    return {"status": "ok"}

try:
    from backend.app.routers import sessions, pipeline, agents
    app.include_router(sessions.router, prefix="/api")
    app.include_router(pipeline.router, prefix="/api")
    app.include_router(agents.router, prefix="/api")
except ImportError as e:
    print(f"Error importing routers: {e}")

storage_dir = Path(__file__).parent / 'storage'
os.makedirs(storage_dir, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_dir)), name="storage")

frontend_dir = Path(__file__).parent.parent / 'frontend'
if not frontend_dir.exists():
    try:
        frontend_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating frontend directory: {e}")

try:
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
except Exception as e:
    print(f"Error mounting frontend: {e}")
