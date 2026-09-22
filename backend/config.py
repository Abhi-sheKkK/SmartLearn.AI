import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_STORAGE_BUCKET: str = "smartlearn-frames"
    GROQ_MODEL: str = "qwen/qwen3.8-27b"

    # Optional proxy for youtube_transcript_api, to work around YouTube rate-limiting
    # or blocking requests from cloud-hosted server IPs. Only used if configured.
    WEBSHARE_PROXY_USERNAME: str = ""
    WEBSHARE_PROXY_PASSWORD: str = ""
    YT_HTTP_PROXY: str = ""
    YT_HTTPS_PROXY: str = ""

    # --- Multi-agent graph -------------------------------------------------
    # Secondary model used when the primary model is rate-limited or returns malformed output.
    GROQ_FALLBACK_MODEL: str = "openai/gpt-oss-20b"   # empty string disables the fallback
    LLM_MAX_RETRIES: int = 2          # graph-level retries per turn (attempt 1 = same model, 2+ = fallback model)
    LLM_RETRY_BACKOFF: float = 1.0    # seconds; doubled on each retry, or Retry-After if the API supplied one
    # Hard ceiling on output tokens per call. 0 = automatic: the graph learns each model's limit from
    # "Request too large ... Limit N" errors (e.g. Groq's on-demand output-tokens-per-minute cap).
    LLM_MAX_OUTPUT_TOKENS: int = 0

    # Checkpointing: "sqlite" (durable, default) or "memory" (ephemeral, used by tests).
    CHECKPOINT_BACKEND: str = "sqlite"
    CHECKPOINT_DB_PATH: str = ""      # defaults to <STORAGE_DIR>/checkpoints.sqlite

    # Agent tools
    ENABLE_PYTHON_TOOL: bool = True
    ENABLE_WEB_SEARCH: bool = True
    ENABLE_TTS: bool = True

    # LangSmith tracing. Exported into os.environ at startup (LangChain reads the process env, not .env).
    # Tracing is switched off automatically (with a warning) if no API key is configured.
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "smartlearn-ai"
    LANGCHAIN_ENDPOINT: str = ""

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = Path(__file__).resolve().parent / "storage"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
