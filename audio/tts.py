"""ElevenLabs voice alerts (Plan.md Task 15).

Pre-generates a short spoken phrase per alert `reason` (Task 7's enum) at
startup and caches the audio bytes in memory, base64-encoded and ready to
broadcast -- a live alert must never wait on the ElevenLabs API.

Playback is server-generated, browser-played: this module only produces and
caches the base64 audio; server/app.py broadcasts it as a
{"type": "audio", "reason": str, "data": "<base64>"} message over the same
/ws/stream fan-out frames and events already use, and Task 16's HUD plays it
via new Audio(). No server-side playback (simpleaudio/playsound) -- that
would need CoreAudio passthrough this Mac's Docker-free setup doesn't
require, and the wire shape for browser playback was already reserved by
Task 10's docstring.

Degrades the same way db/pool.py does: a missing or placeholder
ELEVENLABS_API_KEY never crashes the server -- warm_up() just leaves the
cache empty and get_audio_b64() returns None for everything, so the alert
path in server/app.py silently skips the audio broadcast.
"""

import asyncio
import base64
import logging

from elevenlabs.client import ElevenLabs

import config

logger = logging.getLogger("drivesense.audio")

# Task 7's reason enum -> a short spoken phrase, trimmed to what reads
# naturally out loud rather than a full sentence.
PHRASES = {
    "light_green": "Light's green, go!",
    "lead_accelerating": "Car ahead is moving!",
}

_PLACEHOLDER_MARKERS = ("your_api_key_here",)


def _key_looks_configured(key: str | None) -> bool:
    if not key:
        return False
    return not any(marker in key for marker in _PLACEHOLDER_MARKERS)


class VoiceCache:
    def __init__(self) -> None:
        self._audio_b64: dict[str, str] = {}
        self.enabled = False

    async def warm_up(self) -> None:
        """Pre-generate every phrase in PHRASES. Runs once at FastAPI
        startup so the first real alert never waits on the ElevenLabs API.
        Never raises -- a failure here just leaves the cache (partially)
        empty and voice alerts silently don't fire."""
        if not _key_looks_configured(config.ELEVENLABS_API_KEY):
            logger.warning(
                "ELEVENLABS_API_KEY not configured (still a placeholder or unset) -- "
                "voice alerts disabled. See DEMO.md and .env.example."
            )
            return

        client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        for reason, text in PHRASES.items():
            try:
                audio_bytes = await asyncio.to_thread(self._synthesize, client, text)
                self._audio_b64[reason] = base64.b64encode(audio_bytes).decode("ascii")
                logger.info(
                    "cached voice line for reason=%s (%d bytes)", reason, len(audio_bytes)
                )
            except Exception:
                logger.exception("failed to generate voice line for reason=%s", reason)

        self.enabled = bool(self._audio_b64)

    def _synthesize(self, client: ElevenLabs, text: str) -> bytes:
        """Runs on a worker thread via asyncio.to_thread -- convert() is a
        synchronous, blocking network call and must not run directly on the
        event loop, even though this only happens at startup."""
        chunks = client.text_to_speech.convert(
            voice_id=config.ELEVENLABS_VOICE_ID,
            text=text,
            model_id=config.ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",
        )
        return b"".join(chunks)

    def get_audio_b64(self, reason: str | None) -> str | None:
        if reason is None:
            return None
        return self._audio_b64.get(reason)

    def stats(self) -> dict:
        return {"enabled": self.enabled, "cached_reasons": sorted(self._audio_b64.keys())}
