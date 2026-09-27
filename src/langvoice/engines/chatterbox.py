"""Chatterbox Multilingual v3 (Resemble AI, MIT weights), on the laptop GPU.

What it can and cannot do, read from ``chatterbox/mtl_tts.py`` rather than the
marketing page: ``generate()`` takes exaggeration, cfg_weight, temperature and
three sampling knobs. There is no speed, no pause and no SSML, so those come
from the renderer. Every output carries a PerTh watermark.

The voice is copied from a reference recording. With none given, the model's
own built-in voice (``conds.pt``) is used. Only give it recordings of someone
who agreed to it -- never a Common Voice volunteer, whose terms forbid
identifying them and who never agreed to be cloned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import config
from .base import Audio, Capabilities

# from_pretrained defaults to v2. v3 is the one with the better Swedish, and
# passing it explicitly keeps a library upgrade from changing our voice.
T3_MODEL = "v3"
DEFAULTS = {"cfg_weight": 0.5, "temperature": 0.8}
# Below this the delivery goes flat enough to sound broken, whatever "flat" asked for.
MIN_EXAGGERATION = 0.25


class Chatterbox:
    name = "chatterbox"
    model = f"chatterbox-multilingual-{T3_MODEL}"
    license = "MIT"
    watermark = "perth"
    capabilities = Capabilities(rate=False, pause=False, emotion=True)

    def __init__(self, device: str | None = None) -> None:
        # The model loads on the first say(), not here: a re-run where every
        # line is already made should not spend 20 s and 4 GB of VRAM on nothing.
        self.device = device
        self.tts = None
        self._voices: dict[str | None, Any] = {}

    def _load(self) -> None:
        config.use_cache_dirs()
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = ChatterboxMultilingualTTS.from_pretrained(self.device, t3_model=T3_MODEL)
        # prepare_conditionals overwrites tts.conds, so the built-in voice is
        # kept aside and each reference voice is prepared once, not per line.
        self._voices = {None: self.tts.conds}

    def voice_name(self, voice: str | None) -> str:
        return Path(voice).stem if voice else "builtin"

    def _use_voice(self, voice: str | None, exaggeration: float) -> None:
        if voice not in self._voices:
            path = Path(voice).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"reference voice not found: {path}")
            self.tts.prepare_conditionals(str(path), exaggeration=exaggeration)
            self._voices[voice] = self.tts.conds
        self.tts.conds = self._voices[voice]

    def say(self, text: str, *, lang: str, voice: str | None, emotion: float | None,
            rate: float | None, params: dict[str, Any], seed: int,
            pause_ms: int | None = None) -> Audio:
        if self.tts is None:
            self._load()
        exaggeration = max(MIN_EXAGGERATION, 0.5 if emotion is None else emotion)
        settings = {**DEFAULTS, **params}
        self._use_voice(voice, exaggeration)
        # Sampling is random. Seeding from the text makes a re-run reproduce the
        # same take, so a clip that was judged good stays the clip that was judged.
        self._torch.manual_seed(seed)
        wav = self.tts.generate(text, language_id=lang, exaggeration=exaggeration, **settings)
        samples = wav.squeeze(0).numpy().astype("float32")
        return Audio(samples=samples, rate=self.tts.sr,
                     params={"exaggeration": exaggeration, **settings, "seed": seed})
