"""Checkpointer lifecycle.

`AsyncSqliteSaver` makes conversation state durable across requests *and* restarts (thread_id ==
session_id); `MemorySaver` is the ephemeral option for tests / throwaway runs.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.memory import MemorySaver


@asynccontextmanager
async def open_checkpointer(settings) -> AsyncIterator[object]:
    if str(settings.CHECKPOINT_BACKEND).lower() == "memory":
        yield MemorySaver()
        return

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = Path(settings.CHECKPOINT_DB_PATH) if settings.CHECKPOINT_DB_PATH else Path(settings.STORAGE_DIR) / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
