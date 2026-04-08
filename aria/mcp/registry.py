"""MCP tool registry — discovers tools from servers and wraps them as LangChain tools."""

from __future__ import annotations

import logging
from typing import Any, Self

from langchain_core.tools import StructuredTool
from mcp.types import TextContent, Tool
from pydantic import Field, create_model

from aria.mcp.client import MCPClientManager
from aria.mcp.servers import ServerConfig, get_default_servers

logger = logging.getLogger(__name__)

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _schema_to_model(name: str, schema: dict[str, Any]) -> type:
    """Convert a JSON Schema (from MCP tool inputSchema) to a Pydantic model."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    field_defs: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        python_type = _JSON_TYPE_MAP.get(prop_schema.get("type", ""), Any)
        description = prop_schema.get("description", "")

        if prop_name in required:
            field_defs[prop_name] = (python_type, Field(description=description))
        else:
            default = prop_schema.get("default")
            field_defs[prop_name] = (
                python_type,
                Field(default=default, description=description),
            )

    return create_model(name, **field_defs)


class MCPToolRegistry:
    """Connects to MCP servers, discovers tools, and exposes them as LangChain tools."""

    def __init__(self, configs: list[ServerConfig] | None = None) -> None:
        self._configs = configs if configs is not None else get_default_servers()
        self._clients: list[MCPClientManager] = []
        self._tools: dict[str, tuple[MCPClientManager, Tool]] = {}

    async def __aenter__(self) -> Self:
        for config in self._configs:
            client = config.build_client()
            try:
                await client.__aenter__()
            except RuntimeError:
                logger.warning("Skipping MCP server '%s' — failed to connect", config.name)
                continue
            self._clients.append(client)

        await self._discover_tools()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for client in reversed(self._clients):
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                logger.warning("Error closing MCP server '%s'", client.name)
        self._clients.clear()
        self._tools.clear()

    async def _discover_tools(self) -> None:
        for client in self._clients:
            try:
                tools = await client.list_tools()
            except Exception:
                logger.warning("Failed to list tools for server '%s'", client.name)
                continue

            for tool in tools:
                key = tool.name
                if key in self._tools:
                    key = f"{client.name}_{tool.name}"
                    logger.warning("Tool name collision: '%s' renamed to '%s'", tool.name, key)
                self._tools[key] = (client, tool)

    def get_tool_names(self) -> list[str]:
        return list(self._tools)

    def get_langchain_tools(self) -> list[StructuredTool]:
        lc_tools: list[StructuredTool] = []

        for tool_name, (client, tool) in self._tools.items():
            args_model = _schema_to_model(tool_name, tool.inputSchema)

            async def _invoke(
                _client: MCPClientManager = client,
                _tool_name: str = tool.name,
                **kwargs: Any,
            ) -> str:
                result = await _client.call_tool(_tool_name, kwargs)
                parts = [item.text for item in result.content if isinstance(item, TextContent)]
                return "\n".join(parts)

            lc_tools.append(
                StructuredTool.from_function(
                    coroutine=_invoke,
                    name=tool_name,
                    description=tool.description or "",
                    args_schema=args_model,
                )
            )

        return lc_tools
