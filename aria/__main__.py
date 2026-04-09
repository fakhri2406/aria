"""ARIA CLI — entry point for the ``aria`` command."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel

from aria import __version__
from aria.config import settings

app = typer.Typer(
    name="aria",
    help="ARIA — Agentic Research & Intelligence Assistant",
    add_completion=False,
)
console = Console()


def _run_async(coro):
    """Bridge from sync Typer command to an async coroutine."""
    return asyncio.run(coro)


@app.command()
def run(
    question: str = typer.Argument(..., help="Research question to investigate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full traceback on error"),
) -> None:
    """Run ARIA on a research question and generate a report."""
    try:
        report = _run_async(_run_question(question))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as exc:
        if verbose:
            console.print_exception()
        else:
            console.print(Panel(str(exc), title="Error", border_style="red"))
        raise typer.Exit(code=1) from None

    console.print()
    console.print(
        Panel(report, title="[bold green]✔ ARIA Report[/bold green]", border_style="green")
    )


async def _run_question(question: str) -> str:
    from aria.orchestrator import ARIAOrchestrator

    with console.status("[bold cyan]Researching…[/bold cyan]", spinner="dots"):
        async with ARIAOrchestrator(settings) as orchestrator:
            return await orchestrator.run(question)


@app.command()
def version() -> None:
    """Print ARIA version information."""
    console.print(f"ARIA v{__version__}")


@app.command()
def config() -> None:
    """Show current settings (API keys masked)."""
    from rich.table import Table

    table = Table(title="ARIA Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for field_name, field_value in settings.model_dump().items():
        value_str = str(field_value)
        if "key" in field_name or "token" in field_name:
            value_str = _mask(value_str)
        table.add_row(field_name, value_str)

    console.print(table)


def _mask(value: str) -> str:
    """Mask a secret value, showing only the last 4 characters."""
    if len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


@app.command()
def tools() -> None:
    """List available MCP tools."""
    _run_async(_list_tools())


async def _list_tools() -> None:
    from rich.table import Table

    from aria.mcp.registry import MCPToolRegistry

    with console.status("[bold cyan]Connecting to MCP servers…[/bold cyan]", spinner="dots"):
        async with MCPToolRegistry() as registry:
            tool_names = registry.get_tool_names()

    if not tool_names:
        console.print("[yellow]No tools discovered.[/yellow]")
        return

    table = Table(title="Available MCP Tools")
    table.add_column("#", style="dim")
    table.add_column("Tool Name", style="cyan")

    for i, name in enumerate(tool_names, 1):
        table.add_row(str(i), name)

    console.print(table)


if __name__ == "__main__":
    app()
