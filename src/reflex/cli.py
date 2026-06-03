"""``reflex`` command-line interface.

Commands
--------
* ``reflex run``     — interactive REPL against a persistent agent.
* ``reflex chat``    — one-shot prompt (scriptable).
* ``reflex inspect`` — show memory-tier statistics for a database.
* ``reflex config``  — print the fully-resolved configuration.
* ``reflex version`` — print the package version.

All commands accept ``-c/--config PATH`` to load a YAML config; environment variables
(``REFLEX_*``) and code defaults fill in the rest.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ReflexConfig
from .runtime.agent import Agent
from .version import __version__

app = typer.Typer(
    name="reflex",
    help="ReFlex.AI — persistent cognitive architecture for long-running agents.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to a YAML config file.", exists=False),
]


def _load(config: Path | None) -> ReflexConfig:
    return ReflexConfig.load(config)


@app.command()
def version() -> None:
    """Print the ReFlex version."""
    console.print(f"ReFlex.AI {__version__}")


@app.command(name="config")
def show_config(config: ConfigOpt = None) -> None:
    """Print the fully-resolved configuration as YAML."""
    cfg = _load(config)
    console.print(Panel(cfg.to_yaml(), title="Resolved configuration", expand=False))


@app.command()
def chat(
    message: Annotated[str, typer.Argument(help="The message to send.")],
    config: ConfigOpt = None,
    show_memory: Annotated[bool, typer.Option(help="Print retrieved memory hits.")] = False,
) -> None:
    """Send a single message and print the response."""

    async def _run() -> None:
        cfg = _load(config)
        async with Agent.from_config(cfg) as agent:
            turn = await agent.turn(message)
            if show_memory and turn.retrieval.hits:
                console.print(Panel(turn.retrieval.render(), title="Retrieved memory"))
            console.print(turn.response)
            console.print(
                f"[dim]integrity={turn.integrity.score:.2f} "
                f"revised={turn.revised} latency={turn.latency_s:.2f}s[/dim]"
            )

    asyncio.run(_run())


@app.command()
def run(config: ConfigOpt = None) -> None:
    """Start an interactive REPL. Type ':stats', ':recall <q>', or ':quit'."""

    async def _run() -> None:
        cfg = _load(config)
        async with Agent.from_config(cfg) as agent:
            console.print(
                Panel(
                    f"ReFlex.AI {__version__} — agent '{cfg.agent.name}'\n"
                    f"model={agent.llm.model}  session={agent.session_id}\n"
                    "Commands: :stats  :recall <query>  :quit",
                    title="ReFlex REPL",
                )
            )
            while True:
                try:
                    line = console.input("[bold cyan]you[/bold cyan] › ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]bye[/dim]")
                    break
                if not line:
                    continue
                if line in (":quit", ":q", ":exit"):
                    break
                if line == ":stats":
                    _print_stats(agent.stats())
                    continue
                if line.startswith(":recall "):
                    bundle = agent.recall(line[len(":recall ") :])
                    console.print(Panel(bundle.render() or "(no hits)", title="recall"))
                    continue
                turn = await agent.turn(line)
                console.print(f"[bold green]{cfg.agent.name}[/bold green] › {turn.response}")
                if turn.integrity.flags:
                    console.print(
                        f"[yellow]⚠ {len(turn.integrity.flags)} integrity flag(s); "
                        f"revised={turn.revised}[/yellow]"
                    )

    asyncio.run(_run())


@app.command()
def eval(
    config: ConfigOpt = None,
    distractors: Annotated[int, typer.Option(help="Unrelated turns to grow memory.")] = 100,
    top_k: Annotated[int, typer.Option(help="Retrieval depth for the recall probe.")] = 10,
    seed: Annotated[int, typer.Option(help="Random seed (reproducibility).")] = 1234,
) -> None:
    """Run the memory-retention benchmark and print reproducible metrics."""
    from .eval import run_retention

    async def _run() -> None:
        cfg = _load(config)
        result = await run_retention(cfg, distractors=distractors, top_k=top_k, seed=seed)
        console.print(Panel(result.render(), title="Retention benchmark"))

    asyncio.run(_run())


@app.command()
def inspect(config: ConfigOpt = None) -> None:
    """Show memory-tier statistics for the configured database."""

    async def _run() -> None:
        cfg = _load(config)
        async with Agent.from_config(cfg) as agent:
            _print_stats(agent.stats())

    asyncio.run(_run())


def _print_stats(stats: dict[str, int]) -> None:
    table = Table(title="Memory tiers")
    table.add_column("tier", style="cyan")
    table.add_column("size", justify="right", style="magenta")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
