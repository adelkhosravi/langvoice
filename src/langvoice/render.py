"""One line in, one finished WebM out.

The renderer is where an expression the engine cannot honour gets honoured
anyway: pauses by synthesising each phrase apart and joining them with silence,
speed by ffmpeg's ``atempo``, which changes tempo without changing pitch. What
neither can do lands in ``ignored``, and the sidecar says so.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import config
from .engines.base import Engine
from .lines import Line

# Opus in WebM at the extension's bitrate, so Hub stores and plays synthetic and
# captured clips through one code path. -16 LUFS matches langprep's clean takes.
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
BITRATE = "48k"


@dataclass
class Rendering:
    file: Path
    duration_ms: int
    params: dict[str, Any]
    applied: dict[str, str] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)


def seed_for(text: str, voice: str) -> int:
    """A stable seed per sentence and voice, so a re-run reproduces the take."""
    return int.from_bytes(hashlib.sha256(f"{voice}\x1f{text}".encode()).digest()[:4], "big")


def render(engine: Engine, line: Line, out: Path, *, scratch: Path) -> Rendering:
    expression = line.expression
    voice = engine.voice_name(line.voice)
    seed = seed_for(line.shown, voice)
    applied: dict[str, str] = {}
    ignored: list[str] = []

    native_rate = expression.rate if engine.capabilities.rate else None
    phrases = line.phrases()
    pause_ms = expression.pause_ms if expression.pause_ms is not None else config.DEFAULT_PAUSE_MS
    if len(phrases) > 1 and not engine.capabilities.pause:
        # Separate synthesis per phrase: a comma gives a pause of whatever
        # length the model fancies, and a learner asked for a real one.
        pieces = [engine.say(p, lang=line.lang, voice=line.voice, emotion=expression.emotion,
                             rate=native_rate, params=line.params, seed=seed) for p in phrases]
        rate = pieces[0].rate
        samples = join(pieces, rate, pause_ms)
        params = pieces[0].params
        applied["pause"] = f"joined:{pause_ms}ms"
    else:
        # A capable engine gets the marks and turns them into its own breaks.
        text = line.text if engine.capabilities.pause else line.shown
        audio = engine.say(text, lang=line.lang, voice=line.voice, emotion=expression.emotion,
                           rate=native_rate, params=line.params, seed=seed,
                           pause_ms=pause_ms if engine.capabilities.pause else None)
        samples, rate, params = audio.samples, audio.rate, audio.params
        if len(phrases) > 1:
            applied["pause"] = "native"

    if expression.emotion is not None and not engine.capabilities.emotion:
        ignored.append("emotion")

    tempo = None
    if expression.rate is not None and expression.rate != 1.0:
        if engine.capabilities.rate:
            applied["rate"] = "native"
        else:
            tempo = expression.rate
            applied["rate"] = f"atempo:{tempo}"

    wav = scratch / "say.wav"
    sf.write(str(wav), samples, rate, subtype="PCM_16")
    encode(wav, out, tempo=tempo)
    duration_ms = int(len(samples) / rate / (tempo or 1.0) * 1000)
    return Rendering(file=out, duration_ms=duration_ms, params=params,
                     applied=applied, ignored=ignored)


def join(pieces, rate: int, pause_ms: int) -> np.ndarray:
    gap = np.zeros(int(rate * pause_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, piece in enumerate(pieces):
        if piece.rate != rate:
            raise ValueError("an engine returned phrases at different sample rates")
        if index:
            parts.append(gap)
        parts.append(piece.samples.astype(np.float32))
    return np.concatenate(parts)


def encode(wav: Path, out: Path, *, tempo: float | None = None) -> Path:
    filters = ([f"atempo={tempo}"] if tempo else []) + [LOUDNORM]
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(wav), "-af", ",".join(filters),
           "-c:a", "libopus", "-b:a", BITRATE, "-vn", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed encoding {out.name}: {done.stderr.strip()[-400:]}")
    return out
