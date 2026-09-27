# langvoice

Turns Swedish sentences into synthetic learning clips for
[Hub](https://hub.adelkh.com). It is the text-to-speech sibling of `langprep`,
which cuts *recorded* audio into clips. Both write lang-clips schema 1
(`~/projects/lang-clips/SCHEMA.md`) through langprep's sidecar model, so
`langprep verify`, `langprep review` and `langclips-push` read the output
unchanged.

Why TTS at all: recorded audio from the radio or TV can't be redistributed,
and a TTS engine with a commercial licence gives you audio you're allowed to ship.

## Install

```bash
uv sync --group dev                        # light: sidecars + ffmpeg, no torch
uv sync --extra chatterbox --group dev     # laptop: Chatterbox v3 (torch 2.6, ~4 GB)
uv run langvoice check
uv run pytest
```

This is its own venv, not langprep's, on purpose: chatterbox-tts pins
`torch==2.6` and `numpy<2`, while langprep runs torch 2.14 and numpy 2.5.
Weights (3.2 GB) download on the first `make` into `~/.cache/langvoice`, which
restic excludes.

## Use

```bash
uv run langvoice make sentences.txt                  # one sentence per line
uv run langvoice make lines.jsonl -o ~/Downloads/lang-clips-tts
```

A `.jsonl` line can carry an expression:

```json
{"text": "Jag heter Anna. | Vad heter du?", "expression": {"emotion": "calm", "rate": 0.85, "pause_ms": 500},
 "focus": ["heter"], "translation": "My name is Anna. What is your name?"}
```

| Field | Meaning | Chatterbox |
|---|---|---|
| `emotion` | 0 (flat) to 1 (intense), or `flat`/`calm`/`neutral`/`lively`/`intense` | native (`exaggeration`) |
| `rate` | 1.0 is the voice's own pace, 0.85 a bit slower | ffmpeg `atempo` (no native speed) |
| `\|` in text + `pause_ms` | a pause, default 400 ms | each phrase is synthesised separately and joined with silence |
| `params` | engine-specific overrides, e.g. `{"cfg_weight": 0.3}` | passed through |
| `voice` | a reference wav to copy the voice from | cloned; default is the built-in voice |

Whatever an engine can't do, and post-processing can't fill in, is listed
under `synth.ignored` in the sidecar. Nothing is dropped silently.

Only use a reference voice from someone who agreed to it. Never use a Common
Voice volunteer.

## Checking the output

```bash
cd ~/projects/langprep
uv run langprep verify ~/Downloads/lang-clips-tts    # kb-whisper round trip
uv run langprep review ~/Downloads/lang-clips-tts --serve
```

Re-running `make` skips every line already in the directory (keyed on text,
voice, expression and settings), and the seed is derived from the sentence, so
a re-run reproduces the same take.
