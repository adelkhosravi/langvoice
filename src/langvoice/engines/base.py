"""The engine contract.

An engine turns one phrase into samples and says what it can do natively.
Everything else -- splitting at pauses, slowing down, encoding, the sidecar --
belongs to the renderer, so a new engine is one small class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Capabilities:
    # Each flag means "the engine does this itself, better than post-processing
    # would". False is not a refusal: the renderer fills in rate and pauses.
    rate: bool = False
    pause: bool = False
    emotion: bool = False


@dataclass
class Audio:
    samples: np.ndarray  # mono float32 in [-1, 1]
    rate: int
    # What the engine actually used, for the sidecar.
    params: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    name: str
    model: str
    license: str
    # Set when the engine marks its output (Chatterbox embeds PerTh). Recorded
    # so nobody later mistakes a watermark detector's hit for a problem.
    watermark: str | None
    capabilities: Capabilities

    def voice_name(self, voice: str | None) -> str:
        """The name the sidecar records for `voice` (None = the default voice)."""
        ...

    def say(self, text: str, *, lang: str, voice: str | None, emotion: float | None,
            rate: float | None, params: dict[str, Any], seed: int,
            pause_ms: int | None = None) -> Audio:
        """Synthesise one phrase. `rate` is passed only when capabilities.rate.

        With capabilities.pause, `text` still holds its "|" marks and
        `pause_ms` says how long each one is.
        """
        ...
