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

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = Path(__file__).resolve().parent / "storage"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
