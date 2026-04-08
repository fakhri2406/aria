"""Async MCP client manager for connecting to tool servers over stdio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Self

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, Tool

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Async context manager wrapping a single MCP server connection."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.name = name
        self._server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
            cwd=cwd,
        )
        self._stdio_cm: Any = None
        self._session_cm: Any = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> Self:
        try:
            self._stdio_cm = stdio_client(self._server_params)
            read_stream, write_stream = await self._stdio_cm.__aenter__()

            self._session_cm = ClientSession(read_stream, write_stream)
            session = await self._session_cm.__aenter__()
            self._session = session

            await session.initialize()
        except Exception as exc:
            await self._cleanup()
            raise RuntimeError(f"Failed to connect to MCP server '{self.name}': {exc}") from exc
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._cleanup()

    async def _cleanup(self) -> None:
        for cm_attr in ("_session_cm", "_stdio_cm"):
            cm = getattr(self, cm_attr, None)
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    logger.warning("Error closing %s for server '%s'", cm_attr, self.name)
                finally:
                    setattr(self, cm_attr, None)
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(
                f"MCP server '{self.name}' is not connected. "
                "Use 'async with' to establish a connection first."
            )
        return self._session

    async def list_tools(self) -> list[Tool]:
        session = self._require_session()
        result = await session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        session = self._require_session()
        return await session.call_tool(name, arguments)
