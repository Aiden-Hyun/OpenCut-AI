"""Audio / TTS models (used by app.routes.tts).

TTSRequest mirrors the tts-service `TTSGenerateRequest` schema
(services/tts-service/app.py) so `request.model_dump()` forwards every
field the microservice expects. camelCase aliases match the payload the
web client sends (apps/web/src/types/ai.ts TTSRequest).
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field


class TTSRequest(BaseModel):
    """Text-to-speech generation request, proxied to the tts-service."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to speak")
    language: str = Field(default="en", description="Language code")
    speaker_wav: str | None = Field(
        default=None,
        validation_alias=AliasChoices("speaker_wav", "speakerWav"),
        description="Path to a reference speaker WAV for voice cloning",
    )
    speaker: str | None = Field(
        default=None, description="Built-in speaker name (e.g. 'male', 'female')"
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
