"""MCP server configuration models."""

from __future__ import annotations

import sys
from typing import Any

from pydantic import BaseModel, model_validator

from aria.config import settings
from aria.mcp.client import MCPClientManager


class ServerConfig(BaseModel):
    """Base configuration for an MCP server."""

    name: str
    command: str
    args: list[str] = []
    env: dict[str, str] | None = None

    def build_client(self) -> MCPClientManager:
        return MCPClientManager(
            name=self.name,
            command=self.command,
            args=self.args,
            env=self.env,
        )


class TavilyServerConfig(ServerConfig):
    """Tavily web-search MCP server (npm package)."""

    name: str = "tavily"
    command: str = "npx"
    args: list[str] = ["-y", "@tavily/mcp-server"]

    @model_validator(mode="before")
    @classmethod
    def _inject_api_key(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.setdefault("env", {"TAVILY_API_KEY": settings.tavily_api_key})
        return data


class ArxivServerConfig(ServerConfig):
    """ArXiv paper search MCP server (local Python module)."""

    name: str = "arxiv"
    command: str = sys.executable
    args: list[str] = ["-m", "aria.mcp.arxiv_server"]


def get_default_servers() -> list[ServerConfig]:
    """Return configs for all built-in MCP servers."""
    configs: list[ServerConfig] = [ArxivServerConfig()]
    if settings.tavily_api_key:
        configs.append(TavilyServerConfig())
    return configs
