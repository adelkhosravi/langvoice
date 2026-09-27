"""Azure AI Speech neural voices, through the REST API.

Chosen after listening: Piper sounds Swedish but robotic, and Chatterbox v3
sounded Russian or Norwegian. Azure's sv-SE voices are trained on native
Swedish studio recordings, so they should get the sounds *and* the melody right.
Its terms let you use the output commercially.

Speed and pauses are native (SSML ``<prosody rate>`` and ``<break>``), so the
renderer sends the text with its ``|`` marks and nothing is post-processed. The
Swedish voices have no speaking styles, so emotion is recorded as ignored.

Credentials come from ``AZURE_SPEECH_KEY`` and ``AZURE_SPEECH_REGION``, in the
environment or ``~/.secret``. Every response is cached by a hash of its exact
request, so re-rendering a directory, or the same sentence in a bench, costs
nothing. The free F0 tier is 500k characters a month.
"""

from __future__ import annotations

import hashlib
import io
import time
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import soundfile as sf

from .. import config
from .base import Audio, Capabilities

DEFAULT_VOICE = "sv-SE-SofieNeural"
# 24 kHz 16-bit PCM in a RIFF header: lossless, so the only encode is ours.
OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"
LOCALES = {"sv": "sv-SE"}
RETRIES = 4


def credentials() -> tuple[str, str]:
    from langprep.enrich.llm import secret

    key, region = secret("AZURE_SPEECH_KEY"), secret("AZURE_SPEECH_REGION")
    if not key or not region:
        raise SystemExit(
            "Azure needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (e.g. swedencentral), "
            "in the environment or ~/.secret"
        )
    return key, region


def ssml(text: str, *, lang: str, voice: str, rate: float | None, pause_ms: int | None) -> str:
    """The request body. `|` in the text becomes a break of `pause_ms`."""
    locale = LOCALES.get(lang, lang)
    parts = [escape(p.strip()) for p in text.split("|") if p.strip()]
    body = f'<break time="{pause_ms or config.DEFAULT_PAUSE_MS}ms"/>'.join(parts)
    if rate and rate != 1.0:
        # Relative percent, which Azure applies to the voice's own pace:
        # 0.85 becomes "-15%".
        body = f'<prosody rate="{round((rate - 1.0) * 100):+d}%">{body}</prosody>'
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang={quoteattr(locale)}><voice name={quoteattr(voice)}>{body}</voice></speak>')


class Azure:
    name = "azure"
    model = "azure-neural"
    license = "Azure terms (commercial use allowed)"
    watermark = None
    capabilities = Capabilities(rate=True, pause=True, emotion=False)

    def __init__(self, device: str | None = None) -> None:
        self._client = None
        self.cache = config.CACHE_ROOT / "azure"

    def voice_name(self, voice: str | None) -> str:
        return voice or DEFAULT_VOICE

    def _http(self):
        if self._client is None:
            import httpx

            key, region = credentials()
            self._client = httpx.Client(
                base_url=f"https://{region}.tts.speech.microsoft.com",
                headers={"Ocp-Apim-Subscription-Key": key, "User-Agent": "langvoice"},
                timeout=60.0,
            )
        return self._client

    def voices(self, locale: str = "sv-SE") -> list[dict]:
        response = self._http().get("/cognitiveservices/voices/list")
        response.raise_for_status()
        return [v for v in response.json() if v.get("Locale") == locale]

    def _fetch(self, body: str) -> bytes:
        self.cache.mkdir(parents=True, exist_ok=True)
        cached = self.cache / f"{hashlib.sha256((OUTPUT_FORMAT + body).encode()).hexdigest()}.wav"
        if cached.exists() and cached.stat().st_size > 0:
            return cached.read_bytes()
        for attempt in range(RETRIES):
            response = self._http().post(
                "/cognitiveservices/v1", content=body.encode(),
                headers={"Content-Type": "application/ssml+xml",
                         "X-Microsoft-OutputFormat": OUTPUT_FORMAT},
            )
            # F0 allows few requests at once; 429 means wait, not fail.
            if response.status_code == 429 and attempt < RETRIES - 1:
                time.sleep(float(response.headers.get("Retry-After", 2 ** attempt)))
                continue
            if response.status_code in (401, 403):
                raise SystemExit(f"Azure refused the key ({response.status_code}): "
                                 "check AZURE_SPEECH_KEY and that the region matches the resource")
            if response.status_code != 200:
                raise RuntimeError(f"Azure {response.status_code}: {response.text[:200]}")
            if not response.content:
                raise RuntimeError("Azure returned no audio (a voice name it does not know?)")
            partial = cached.with_suffix(".partial")
            partial.write_bytes(response.content)
            partial.replace(cached)
            return response.content
        raise RuntimeError("Azure kept rate-limiting; try again in a minute")

    def say(self, text: str, *, lang: str, voice: str | None, emotion: float | None,
            rate: float | None, params: dict[str, Any], seed: int,
            pause_ms: int | None = None) -> Audio:
        name = self.voice_name(voice)
        body = ssml(text, lang=lang, voice=name, rate=rate, pause_ms=pause_ms)
        samples, sample_rate = sf.read(io.BytesIO(self._fetch(body)), dtype="float32")
        return Audio(samples=samples, rate=sample_rate,
                     params={"voice": name, "format": OUTPUT_FORMAT,
                             **({"prosody_rate": rate} if rate else {})})
