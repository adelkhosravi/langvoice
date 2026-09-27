"""What goes in: one sentence per line, with an optional expression.

Two formats. A ``.txt`` file is one sentence per line and nothing else, which is
what a transcript or a word-list export usually is. A ``.jsonl`` file carries
the rest::

    {"text": "Jag heter Anna. | Vad heter du?", "expression": {"rate": 0.85}}

A ``|`` in the text is a pause. It never reaches the sidecar: the learner reads
the sentence, not our markup.

The expression is a *request*. Each engine does what it can natively, the
renderer fills in what post-processing can (speed, pauses), and whatever is
left is recorded as ignored in the sidecar rather than dropped silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Emotion is an intensity, not a mood: Chatterbox's only lever is how animated
# the delivery is. The labels are names for points on that scale, so a line can
# say "calm" without knowing what an engine calls it.
EMOTION_LABELS = {"flat": 0.25, "calm": 0.35, "neutral": 0.5, "lively": 0.75, "intense": 1.0}


class Expression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 0 (flat) to 1 (intense); a label from EMOTION_LABELS is turned into its number.
    emotion: float | None = None
    # 1.0 is the voice's own pace; 0.85 is "a bit slower", the useful setting
    # for a beginner. Outside 0.5-2.0 even atempo stops sounding like speech.
    rate: float | None = Field(default=None, ge=0.5, le=2.0)
    # Length of each "|" pause.
    pause_ms: int | None = Field(default=None, ge=0, le=5000)

    @field_validator("emotion", mode="before")
    @classmethod
    def _label(cls, value: Any) -> Any:
        if isinstance(value, str):
            if value not in EMOTION_LABELS:
                raise ValueError(f"unknown emotion {value!r}; known: {', '.join(EMOTION_LABELS)}")
            return EMOTION_LABELS[value]
        return value

    @field_validator("emotion")
    @classmethod
    def _range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("emotion is 0 (flat) to 1 (intense)")
        return value


class Line(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    lang: str = "sv"
    # An engine voice name, or a path to a reference wav for engines that clone.
    voice: str | None = None
    expression: Expression = Field(default_factory=Expression)
    # Engine-specific overrides, passed through untouched (e.g. {"cfg_weight": 0.3}).
    params: dict[str, Any] = Field(default_factory=dict)
    # Carried into the sidecar as-is.
    focus: list[str] = Field(default_factory=list)
    translation: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not display_text(value):
            raise ValueError("text is empty")
        return value

    def phrases(self) -> list[str]:
        """The text split at its pauses."""
        return [p.strip() for p in self.text.split("|") if p.strip()]

    @property
    def shown(self) -> str:
        return display_text(self.text)


def display_text(text: str) -> str:
    """The sentence as the learner sees it: pause marks gone, spaces tidied."""
    return re.sub(r"\s+", " ", text.replace("|", " ")).strip()


def read(path: Path) -> list[Line]:
    """Read a .txt or .jsonl file. Blank lines and ``#`` comments are skipped."""
    path = Path(path)
    lines: list[Line] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            if path.suffix == ".jsonl":
                lines.append(Line.model_validate(json.loads(raw)))
            else:
                lines.append(Line(text=raw))
        except (ValueError, json.JSONDecodeError) as error:
            # Name the line: a 500-line file with one bad row is otherwise a hunt.
            raise ValueError(f"{path.name}:{number}: {error}") from error
    return lines
