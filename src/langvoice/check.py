"""`langvoice check`: can this machine render, and is there room to try?

Nothing here downloads. The disk check is first because it is the one most
likely to fail: the Chatterbox weights are 3.2 GB and / had 12 GB free when
this was written.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass

from . import config

CHATTERBOX_REPO = "models--ResembleAI--chatterbox"
CHATTERBOX_FILES = ("t3_mtl23ls_v3.safetensors", "s3gen.pt", "ve.pt", "conds.pt")


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    fix: str = ""

    def line(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        text = f"{mark}  {self.name:18s} {self.detail}"
        return text if self.ok or not self.fix else f"{text}\n          → {self.fix}"


def _disk() -> Result:
    config.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(config.CACHE_ROOT).free / 1e9
    return Result("disk", free >= config.MIN_FREE_GB, f"{free:.1f} GB free under {config.CACHE_ROOT}",
                  f"need {config.MIN_FREE_GB:.0f} GB: `uv cache prune`, or clear ~/.cache/langprep/work")


def _ffmpeg() -> Result:
    if not shutil.which("ffmpeg"):
        return Result("ffmpeg", False, "missing", "sudo pacman -S ffmpeg")
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout
    ok = "libopus" in encoders
    return Result("ffmpeg", ok, "libopus available" if ok else "built without libopus",
                  "install an ffmpeg with libopus: every clip is Opus in WebM")


def _gpu() -> Result:
    if not shutil.which("nvidia-smi"):
        return Result("gpu", False, "no nvidia-smi: rendering will run on the CPU (slow)")
    done = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if done.returncode != 0 or not done.stdout.strip():
        return Result("gpu", False, "nvidia-smi found no GPU", "check the driver: nvidia-smi")
    name, total, used = (part.strip() for part in done.stdout.splitlines()[0].split(","))
    free_gb = (int(total) - int(used)) / 1024
    # Chatterbox at fp32 wants about 4-5 GB; a browser can hold a gigabyte of VRAM.
    return Result("gpu", free_gb >= 4.5, f"{name}, {free_gb:.1f} of {int(total) / 1024:.1f} GB free",
                  "close GPU-heavy apps (browsers, ollama) before a run")


def _chatterbox() -> list[Result]:
    if importlib.util.find_spec("chatterbox") is None:
        return [Result("chatterbox", False, "not installed", "uv sync --extra chatterbox")]
    out = [Result("chatterbox", True, "installed")]
    import torch

    cuda = torch.cuda.is_available()
    out.append(Result("torch cuda", cuda, f"torch {torch.__version__}, "
                      + (f"CUDA {torch.version.cuda}" if cuda else "CPU only"),
                      "reinstall torch with CUDA wheels"))
    snapshots = config.MODELS / "huggingface" / "hub" / CHATTERBOX_REPO / "snapshots"
    present = {p.name for p in snapshots.glob("*/*")} if snapshots.exists() else set()
    missing = [f for f in CHATTERBOX_FILES if f not in present]
    out.append(Result("weights", not missing,
                      "cached" if not missing else f"not yet downloaded ({len(missing)} files, ~3.2 GB)",
                      "the first `langvoice make` downloads them"))
    return out


def run(echo=print) -> bool:
    results = [_disk(), _ffmpeg(), _gpu(), *_chatterbox()]
    for result in results:
        echo(result.line())
    # The weights not being there yet is expected before a first run, not a failure.
    return all(r.ok for r in results if r.name != "weights")
