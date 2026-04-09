"""MCP integration layer — client, server configs, and tool registry."""

from aria.mcp.client import MCPClientManager
from aria.mcp.registry import MCPToolRegistry

__all__ = [
    "MCPClientManager",
    "MCPToolRegistry"
]
