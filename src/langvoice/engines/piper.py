"""Piper (rhasspy), small CPU voices trained on one Swedish speaker each.

Robotic next to a large model, but the listener's verdict on 2026-09-26 was
that Piper at least *sounds Swedish*, where Chatterbox v3 sounded Russian or
Norwegian. For a learner, correct sounds beat natural melody.

Licences differ per voice, so the sidecar records the voice's, not Piper's.
Piper's own code is GPL-3.0; that binds the program, not audio it renders.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import config
from .base import Audio, Capabilities

VOICES_REPO = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# lisa is left out on purpose: its card states no licence, and it was
# fine-tuned from a Norwegian voice.
VOICES = {
    "sv_SE-alma-medium": ("sv/sv_SE/alma/medium", "CC-BY-4.0"),
    "sv_SE-nst-medium": ("sv/sv_SE/nst/medium", "CC0-1.0 (NST data)"),
}
DEFAULT_VOICE = "sv_SE-alma-medium"


class Piper:
    name = "piper"
    model = "piper"
    license = "GPL-3.0 (software)"
    watermark = None
    # length_scale is a real speed control; pauses and emotion are not there.
    capabilities = Capabilities(rate=True, pause=False, emotion=False)

    def __init__(self, device: str | None = None) -> None:
        self._loaded: dict[str, Any] = {}

    def voice_name(self, voice: str | None) -> str:
        return voice or DEFAULT_VOICE

    def license_for(self, voice: str | None) -> str:
        return VOICES[self.voice_name(voice)][1]

    def _voice(self, name: str):
        if name not in VOICES:
            raise ValueError(f"unknown Piper voice {name!r}; known: {', '.join(VOICES)}")
        if name not in self._loaded:
            from piper import PiperVoice

            self._loaded[name] = PiperVoice.load(str(self._fetch(name)))
        return self._loaded[name]

    def _fetch(self, name: str):
        import urllib.request

        directory = config.MODELS / "piper"
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in (".onnx", ".onnx.json"):
            target = directory / f"{name}{suffix}"
            if not target.exists() or target.stat().st_size == 0:
                partial = target.with_suffix(target.suffix + ".partial")
                urllib.request.urlretrieve(f"{VOICES_REPO}/{VOICES[name][0]}/{name}{suffix}", partial)
                partial.replace(target)
        return directory / f"{name}.onnx"

    def say(self, text: str, *, lang: str, voice: str | None, emotion: float | None,
            rate: float | None, params: dict[str, Any], seed: int,
            pause_ms: int | None = None) -> Audio:
        from piper import SynthesisConfig

        name = self.voice_name(voice)
        piper_voice = self._voice(name)
        # Piper's length_scale is the inverse of a speed: larger is slower.
        settings = {"length_scale": 1.0 / (rate or 1.0), **params}
        chunks = list(piper_voice.synthesize(text, syn_config=SynthesisConfig(**settings)))
        if not chunks:
            raise RuntimeError(f"Piper produced nothing for {text[:40]!r}")
        samples = np.concatenate([
            np.frombuffer(c.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            for c in chunks
        ])
        return Audio(samples=samples, rate=chunks[0].sample_rate,
                     params={"voice": name, **settings})
