import json
import shutil
import subprocess

import numpy as np
import pytest

from langprep import sidecar
from langvoice import lines
from langvoice.engines.base import Audio, Capabilities
from langvoice.make import make

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
RATE = 24_000


class FakeEngine:
    name = "fake"
    model = "fake-1"
    license = "MIT"
    watermark = None

    def __init__(self, capabilities=Capabilities()):
        self.capabilities = capabilities
        self.calls = []

    def voice_name(self, voice):
        return voice or "builtin"

    def say(self, text, *, lang, voice, emotion, rate, params, seed, pause_ms=None):
        self.calls.append({"text": text, "emotion": emotion, "rate": rate, "seed": seed})
        # 0.1 s of tone per word: long enough for ffmpeg, cheap to make.
        n = int(RATE * 0.1 * len(text.split()))
        tone = 0.1 * np.sin(np.arange(n) / RATE * 2 * np.pi * 220).astype(np.float32)
        return Audio(samples=tone, rate=RATE, params={"seed": seed})


def duration_s(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return float(out)


def test_txt_and_jsonl_parse(tmp_path):
    (tmp_path / "a.txt").write_text("Hej där.\n\n# a comment\nJag heter Anna.\n")
    assert [line.text for line in lines.read(tmp_path / "a.txt")] == ["Hej där.", "Jag heter Anna."]
    (tmp_path / "b.jsonl").write_text(json.dumps(
        {"text": "Hej. | Hur mår du?", "expression": {"emotion": "calm", "rate": 0.85}}) + "\n")
    line = lines.read(tmp_path / "b.jsonl")[0]
    assert line.expression.emotion == lines.EMOTION_LABELS["calm"]
    assert line.phrases() == ["Hej.", "Hur mår du?"]
    assert line.shown == "Hej. Hur mår du?"


def test_bad_line_is_named(tmp_path):
    (tmp_path / "c.jsonl").write_text('{"text": "ok"}\n{"text": "x", "expression": {"rate": 9}}\n')
    with pytest.raises(ValueError, match="c.jsonl:2"):
        lines.read(tmp_path / "c.jsonl")


@needs_ffmpeg
def test_sidecar_is_schema_1_and_marks_synthetic(tmp_path):
    engine = FakeEngine()
    [path] = make(engine, [lines.Line(text="Jag heter Anna.", focus=["heter"])], tmp_path,
                  echo=lambda *_: None)
    card = sidecar.load(path)  # langprep's own parser accepts it
    assert card.text_source == "manual"
    assert (tmp_path / card.audio.file).exists()
    assert "tts" in card.tags and card.focus[0].term == "heter"
    synth = json.loads(path.read_text())["synth"]
    assert synth["engine"] == "fake" and synth["voice"] == "builtin" and synth["key"]


@needs_ffmpeg
def test_pauses_are_joined_and_marks_never_reach_the_text(tmp_path):
    engine = FakeEngine()
    line = lines.Line(text="Hej. | Hur mår du?", expression=lines.Expression(pause_ms=1000))
    [path] = make(engine, [line], tmp_path, echo=lambda *_: None)
    card = sidecar.load(path)
    assert card.text == "Hej. Hur mår du?"
    assert [c["text"] for c in engine.calls] == ["Hej.", "Hur mår du?"]
    # 1 word + 3 words of tone, plus the 1 s gap.
    assert duration_s(tmp_path / card.audio.file) == pytest.approx(1.4, abs=0.1)
    assert json.loads(path.read_text())["synth"]["applied"]["pause"] == "joined:1000ms"


@needs_ffmpeg
def test_rate_falls_back_to_atempo_and_unsupported_emotion_is_recorded(tmp_path):
    engine = FakeEngine()
    line = lines.Line(text="ett två tre fyra fem",
                      expression=lines.Expression(rate=0.5, emotion=0.8))
    [path] = make(engine, [line], tmp_path, echo=lambda *_: None)
    synth = json.loads(path.read_text())["synth"]
    assert synth["applied"]["rate"] == "atempo:0.5"
    assert synth["ignored"] == ["emotion"]
    assert engine.calls[0]["rate"] is None  # the engine was not asked to do it
    assert duration_s(tmp_path / sidecar.load(path).audio.file) == pytest.approx(1.0, abs=0.1)


@needs_ffmpeg
def test_native_capabilities_are_passed_through(tmp_path):
    engine = FakeEngine(Capabilities(rate=True, pause=True, emotion=True))
    line = lines.Line(text="Hej. | Då.", expression=lines.Expression(rate=0.85, emotion=0.8))
    [path] = make(engine, [line], tmp_path, echo=lambda *_: None)
    assert engine.calls == [{"text": "Hej. | Då.", "emotion": 0.8, "rate": 0.85,
                             "seed": engine.calls[0]["seed"]}]
    synth = json.loads(path.read_text())["synth"]
    assert synth["applied"] == {"pause": "native", "rate": "native"} and synth["ignored"] == []


@needs_ffmpeg
def test_rerun_skips_made_lines_and_seed_is_stable(tmp_path):
    engine = FakeEngine()
    batch = [lines.Line(text="Hej."), lines.Line(text="Hej.", expression=lines.Expression(rate=0.85))]
    assert len(make(engine, batch, tmp_path, echo=lambda *_: None)) == 2
    assert make(engine, batch, tmp_path, echo=lambda *_: None) == []
    assert len(engine.calls) == 2
    # Same sentence and voice, so the same take -- only the tempo differs.
    assert engine.calls[0]["seed"] == engine.calls[1]["seed"]


@needs_ffmpeg
def test_private_voice_is_marked(tmp_path):
    [path] = make(FakeEngine(), [lines.Line(text="Hej.")], tmp_path, private=True,
                  echo=lambda *_: None)
    data = json.loads(path.read_text())
    assert "private-voice" in data["tags"] and data["synth"]["consent"] is False


def test_azure_ssml_has_native_rate_and_breaks():
    from langvoice.engines.azure import ssml

    body = ssml("Hej & välkommen. | Hur mår du?", lang="sv", voice="sv-SE-SofieNeural",
                rate=0.85, pause_ms=500)
    assert 'xml:lang="sv-SE"' in body and 'name="sv-SE-SofieNeural"' in body
    assert '<prosody rate="-15%">' in body
    assert 'välkommen.<break time="500ms"/>Hur mår du?' in body
    assert "&amp;" in body  # text is escaped, not injected as markup
    assert "prosody" not in ssml("Hej.", lang="sv", voice="v", rate=None, pause_ms=None)
