"""Environment-backed configuration with no implicit secret-file loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path
    sec_user_agent: str
    openai_api_key: str | None
    openai_model: str

    @classmethod
    def from_environment(cls, database_path: Path | None = None) -> "Settings":
        return cls(
            database_path=database_path or Path(os.getenv("RESEARCH_AGENT_DB", "data/research-agent.sqlite3")),
            sec_user_agent=os.getenv(
                "SEC_USER_AGENT",
                "resilient-financial-research-agent/1.0 contact@example.com",
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        )
