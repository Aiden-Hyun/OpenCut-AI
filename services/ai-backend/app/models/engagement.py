"""Engagement scoring, clip detection, and YouTube job models.

Consumed by:
  - app.services.engagement.* (scorer, hook_analyzer, face_presence)
  - app.services.clip_detector, app.services.youtube_service
  - app.routes.engagement, app.routes.youtube

Every field name matches a keyword used at a construction site, and every
attribute read elsewhere (e.g. score.hook.composite, clip.duration) is
provided as a field or computed property.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# -- Per-signal sub-scores --------------------------------------------------

class HookScore(BaseModel):
    """First-3-seconds hook strength."""

    visual_novelty: float = 0.0
    audio_energy_spike: float = 0.0
    early_face_present: bool = False
    hook_type: str = ""
    hook_type_confidence: float = 0.0
    speech_rate: float = 0.0
    composite: float = 0.0


class EnergyScore(BaseModel):
    """Audio energy / dynamic range."""

    mean_energy: float = 0.0
    peak_energy: float = 0.0
    energy_variance: float = 0.0
    has_dynamic_range: bool = False
    composite: float = 0.0


class CuriosityScore(BaseModel):
    """Curiosity-gap triggers in the transcript."""

    has_question: bool = False
    has_bold_claim: bool = False
    has_open_loop: bool = False
    gap_count: int = 0
    composite: float = 0.0


class FacePresenceScore(BaseModel):
    """Face presence ratio vs. optimal target."""

    face_ratio: float = 0.0
    is_optimal: bool = False
    early_face_present: bool = False
    composite: float = 0.0


class EmotionalArcScore(BaseModel):
    """Emotional arc / pacing structure."""

    has_strong_open: bool = False
    has_buildup: bool = False
    has_peak: bool = False
    peak_timestamp: float = 0.0
    dominant_emotion: str = ""
    composite: float = 0.0


class AudioSyncScore(BaseModel):
    """Beat detection and caption/beat alignment."""

    bpm: float | None = None
    beat_count: int = 0
    caption_beat_alignment: float = 0.0
    composite: float = 0.0


class ViralityScore(BaseModel):
    """LLM-estimated viral potential across 4 dimensions."""

    hook_strength: float = 0.0
    shareability: float = 0.0
    emotional_impact: float = 0.0
    standalone_value: float = 0.0
    reason: str = ""
    suggested_title: str = ""
    composite: float = 0.0


class EnhancementSuggestion(BaseModel):
    """An actionable suggestion tied to a weak signal."""

    signal: str
    current_score: float = 0.0
    suggestion: str = ""
    action_type: str = "manual"
    expected_impact: str = "medium"


# -- Aggregate engagement score ---------------------------------------------

# Relative weight of each signal in the composite. Hook matters most for
# short-form retention; audio-sync is a minor polish signal.
_SIGNAL_WEIGHTS: dict[str, float] = {
    "hook": 0.25,
    "curiosity": 0.15,
    "energy": 0.15,
    "emotional_arc": 0.15,
    "virality": 0.15,
    "face_presence": 0.10,
    "audio_sync": 0.05,
}


class EngagementScore(BaseModel):
    """Full engagement breakdown for a clip."""

    hook: HookScore = Field(default_factory=HookScore)
    energy: EnergyScore = Field(default_factory=EnergyScore)
    curiosity: CuriosityScore = Field(default_factory=CuriosityScore)
    audio_sync: AudioSyncScore = Field(default_factory=AudioSyncScore)
    face_presence: FacePresenceScore = Field(default_factory=FacePresenceScore)
    emotional_arc: EmotionalArcScore = Field(default_factory=EmotionalArcScore)
    virality: ViralityScore = Field(default_factory=ViralityScore)
    suggestions: list[EnhancementSuggestion] = Field(default_factory=list)

    @property
    def composite(self) -> float:
        """Weighted average of the seven signal composites (0-100)."""
        total = (
            self.hook.composite * _SIGNAL_WEIGHTS["hook"]
            + self.curiosity.composite * _SIGNAL_WEIGHTS["curiosity"]
            + self.energy.composite * _SIGNAL_WEIGHTS["energy"]
            + self.emotional_arc.composite * _SIGNAL_WEIGHTS["emotional_arc"]
            + self.virality.composite * _SIGNAL_WEIGHTS["virality"]
            + self.face_presence.composite * _SIGNAL_WEIGHTS["face_presence"]
            + self.audio_sync.composite * _SIGNAL_WEIGHTS["audio_sync"]
        )
        return round(total, 1)

    @property
    def grade(self) -> str:
        c = self.composite
        if c >= 80:
            return "A"
        if c >= 65:
            return "B"
        if c >= 50:
            return "C"
        if c >= 35:
            return "D"
        return "F"

    def to_response(self) -> dict[str, Any]:
        """Serialize with the computed composite + grade included."""
        data = self.model_dump()
        data["composite"] = self.composite
        data["grade"] = self.grade
        return data


# -- Clip scoring requests / results ----------------------------------------

class ScoreClipRequest(BaseModel):
    """Inputs needed to score a single clip."""

    audio_path: str | None = None
    video_path: str | None = None
    transcript_text: str | None = None
    transcript_segments: list[dict[str, Any]] | None = None
    start: float = 0.0
    end: float = 0.0
    title: str | None = ""


class ScoreBatchRequest(BaseModel):
    """A batch of clips to score in parallel."""

    clips: list[ScoreClipRequest] = Field(default_factory=list)


class ScoredClip(BaseModel):
    """A detected clip with its engagement score."""

    index: int = 0
    title: str = ""
    start: float = 0.0
    end: float = 0.0
    transcript_preview: str = ""
    tags: list[str] = Field(default_factory=list)
    engagement: EngagementScore = Field(default_factory=EngagementScore)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


# -- YouTube / job models ---------------------------------------------------

class YouTubeVideoMeta(BaseModel):
    """Metadata for a source YouTube video."""

    video_id: str = ""
    title: str = "Untitled"
    channel_name: str = "Unknown"
    channel_id: str = ""
    duration_seconds: float = 0.0
    thumbnail_url: str = ""
    upload_date: str = ""
    view_count: int | None = None
    is_live: bool = False
    is_private: bool = False
    warning: str | None = None


class JobStatus(BaseModel):
    """Status of a background job (YouTube-to-Reels pipeline)."""

    job_id: str
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    result: Any | None = None
    error: str | None = None
