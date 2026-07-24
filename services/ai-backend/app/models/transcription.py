"""Transcription models (used by app.services.whisper_service).

Field names and types match how WhisperService constructs these objects
from faster-whisper output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionWord(BaseModel):
    """A single word with timing and confidence."""

    word: str
    start: float
    end: float
    probability: float = 0.0


class TranscriptionSegment(BaseModel):
    """A transcript segment with optional word-level timestamps."""

    id: int = 0
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    words: list[TranscriptionWord] = Field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionResult(BaseModel):
    """Full transcription result for a media file."""

    text: str = ""
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    language: str = ""
    duration: float = 0.0
