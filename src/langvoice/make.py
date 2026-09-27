"""`langvoice make`: a file of lines in, a directory of clip pairs out.

The pairs are lang-clips schema 1, written through langprep's own sidecar
model, so ``langprep verify``/``review`` and ``langclips-push`` read them like
any other clip. Two optional fields mark them synthetic: ``tags`` gets ``tts``,
and a ``synth`` block records engine, voice, settings, what the expression
asked for and what was ignored. The schema's rule is that optional additions do
not bump the version, and importers ignore what they don't know.

Re-running is cheap: every clip carries a key over everything that shaped its
audio, and a line whose key is already in the directory is skipped before the
engine is even asked.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from langprep import sidecar

from .engines.base import Engine
from .lines import Line
from .render import render


def key_for(engine: Engine, line: Line) -> str:
    identity = {
        "engine": engine.name, "model": engine.model, "voice": engine.voice_name(line.voice),
        "lang": line.lang, "text": line.text,
        "expression": line.expression.model_dump(exclude_none=True), "params": line.params,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def existing_keys(directory: Path) -> set[str]:
    keys = set()
    for path in sidecar.clips_in(directory):
        synth = json.loads(path.read_text(encoding="utf-8")).get("synth") or {}
        if synth.get("key"):
            keys.add(synth["key"])
    return keys


def build_sidecar(engine: Engine, line: Line, *, clip_id: str, created_at: str, key: str,
                  audio_file: str, rendering, private: bool = False) -> sidecar.Sidecar:
    return sidecar.Sidecar(
        id=clip_id,
        created_at=created_at,
        language=line.lang,
        audio=sidecar.AudioInfo(file=audio_file, duration_ms=rendering.duration_ms),
        text=line.shown,
        # The text was given, not heard: "manual" is the schema's word for that.
        text_source="manual",
        focus=[sidecar.Focus(term=term) for term in line.focus],
        translation=line.translation,
        # "private-voice": cloned from someone who never agreed to it. Such a
        # clip is for judging an engine locally and must never reach Hub.
        tags=["tts", engine.name, *(["private-voice"] if private else []), *line.tags],
        source=sidecar.Source(title=line.source),
        synth={
            "engine": engine.name,
            "model": engine.model,
            "voice": engine.voice_name(line.voice),
            # Some engines license per voice (Piper); the voice's licence is the one that binds.
            "license": (engine.license_for(line.voice) if hasattr(engine, "license_for")
                        else engine.license),
            "watermark": engine.watermark,
            "params": rendering.params,
            "expression": line.expression.model_dump(exclude_none=True),
            "applied": rendering.applied,
            "ignored": rendering.ignored,
            "consent": not private,
            "key": key,
        },
    )


def make(engine: Engine, lines: list[Line], out: Path, *,
         limit: int | None = None, private: bool = False, echo=print) -> list[Path]:
    """Render every line not already in `out`. Returns the sidecars written.

    Engines load their model on the first say(), so a re-run with nothing new
    never loads one.
    """
    out.mkdir(parents=True, exist_ok=True)
    have = existing_keys(out)
    written: list[Path] = []
    todo = lines[:limit] if limit else lines

    with tempfile.TemporaryDirectory(prefix="langvoice-") as scratch:
        for number, line in enumerate(todo, start=1):
            key = key_for(engine, line)
            if key in have:
                echo(f"[{number}/{len(todo)}] skip (already made): {line.shown[:60]}")
                continue
            now = datetime.now().astimezone()
            clip_id = sidecar.clip_id(now, key)
            audio_file = f"{clip_id}.webm"
            try:
                rendering = render(engine, line, out / audio_file, scratch=Path(scratch))
            except (RuntimeError, ValueError, FileNotFoundError) as error:
                # One bad sentence must not cost the other 499.
                echo(f"[{number}/{len(todo)}] FAILED: {line.shown[:60]} -- {error}")
                continue
            card = build_sidecar(engine, line, clip_id=clip_id, key=key, audio_file=audio_file,
                                 created_at=now.isoformat(timespec="seconds"),
                                 rendering=rendering, private=private)
            # Audio first, sidecar last: the .json is the commit marker.
            written.append(card.write(out))
            have.add(key)
            extra = f"  ignored: {', '.join(rendering.ignored)}" if rendering.ignored else ""
            echo(f"[{number}/{len(todo)}] {clip_id}  {rendering.duration_ms / 1000:.1f}s  "
                 f"{line.shown[:60]}{extra}")
    return written
