"""`langvoice` command line. Thin: every command is one call into a module."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import config

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def make(
    source: Path = typer.Argument(..., help=".txt (one sentence per line) or .jsonl"),
    engine: str = typer.Option("chatterbox", "--engine", "-e"),
    voice: Optional[str] = typer.Option(None, help="default voice for lines that name none"),
    out: Path = typer.Option(config.OUT_ROOT, "--out", "-o", help="directory of clip pairs"),
    limit: Optional[int] = typer.Option(None, help="only the first N lines (a quick look)"),
    device: Optional[str] = typer.Option(None, help="cuda or cpu (default: cuda if present)"),
    private: bool = typer.Option(False, "--private",
                                 help="the voice is someone who did not consent: a local test only"),
) -> None:
    """Render every line into a clip pair, skipping lines already made."""
    from . import engines, lines as lines_mod
    from .make import make as run_make

    try:
        lines = lines_mod.read(source)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if voice:
        lines = [line if line.voice else line.model_copy(update={"voice": voice}) for line in lines]
    if private and out.resolve().is_relative_to((Path.home() / "Downloads").resolve()):
        # ~/Downloads is where langclips-push looks, and restic backs it up.
        raise typer.BadParameter("--private clips must not go under ~/Downloads", param_hint="--out")
    written = run_make(engines.load(engine, device=device), lines, out, limit=limit,
                       private=private, echo=typer.echo)
    typer.echo(f"{len(written)} new clip(s) in {out}")


@app.command()
def voices(engine: str = typer.Option("azure", "--engine", "-e"),
           locale: str = typer.Option("sv-SE", help="e.g. sv-SE")) -> None:
    """List an engine's voices for a locale (Azure asks its API)."""
    from . import engines

    loaded = engines.load(engine)
    if not hasattr(loaded, "voices"):
        raise typer.BadParameter(f"{engine} has no voice listing", param_hint="--engine")
    for voice in loaded.voices(locale):
        styles = ", ".join(voice.get("StyleList") or []) or "-"
        typer.echo(f"{voice['ShortName']:32s} {voice.get('Gender', ''):7s} "
                   f"{voice.get('VoiceType', ''):8s} styles: {styles}")


@app.command()
def check() -> None:
    """Verify this machine can render, without downloading anything."""
    from . import check as checks

    if not checks.run(echo=typer.echo):
        raise typer.Exit(1)


def main() -> None:  # pragma: no cover -- console-script entry point
    app()
