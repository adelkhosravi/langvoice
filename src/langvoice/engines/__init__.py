"""TTS engines. Each is imported only when asked for: they pull in torch or SDKs."""

from __future__ import annotations

import importlib.util

from .base import Engine

NAMES = ("chatterbox", "piper", "azure")


def load(name: str, **options) -> Engine:
    if name == "chatterbox":
        # find_spec, not import: chatterbox's __init__ drags in torch, which
        # costs seconds a run of nothing-but-skips should not pay.
        if importlib.util.find_spec("chatterbox") is None:
            raise SystemExit("chatterbox is not installed; run: uv sync --extra chatterbox")
        from .chatterbox import Chatterbox

        return Chatterbox(**options)
    if name == "piper":
        if importlib.util.find_spec("piper") is None:
            raise SystemExit("piper is not installed; run: uv sync --extra piper")
        from .piper import Piper

        return Piper(**options)
    if name == "azure":
        # Plain HTTPS through httpx: nothing to install beyond the base.
        from .azure import Azure

        return Azure(**options)
    raise SystemExit(f"unknown engine {name!r}; known: {', '.join(NAMES)}")
