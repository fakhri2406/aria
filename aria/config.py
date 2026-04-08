"""Centralised settings loaded from environment / .env file."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-6"
    langsmith_api_key: str = ""
    langsmith_project: str = "aria"
    tavily_api_key: str = ""
    hf_token: str = ""


settings = Settings()
