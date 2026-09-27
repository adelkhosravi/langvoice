"""Paths and settings.

Model weights live under ``~/.cache/langvoice``, which restic's ``.cache``
exclude already covers: they are re-downloadable, several gigabytes, and
nothing to back up.
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE_ROOT = Path(os.environ.get("LANGVOICE_CACHE", Path.home() / ".cache" / "langvoice"))
MODELS = CACHE_ROOT / "models"

# Beside the extension's lang-clips folder, so langclips-push takes it with
# --dir and nothing synthetic lands among hand-captured clips by accident.
OUT_ROOT = Path(os.environ.get("LANGVOICE_OUT", Path.home() / "Downloads" / "lang-clips-tts"))

# A download is refused below this. The Chatterbox weights alone are 3.2 GB,
# and / was at 88% when this tool was written.
MIN_FREE_GB = 5.0

# A "|" in a line's text is a pause of this length, unless the line says otherwise.
DEFAULT_PAUSE_MS = 400


def use_cache_dirs() -> None:
    """Point HuggingFace and torch at CACHE_ROOT. Call before any model import."""
    MODELS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODELS / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
