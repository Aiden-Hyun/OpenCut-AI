"""Image generation / prompt enhancement / infographic models
(used by app.routes.generate).

ImageGenParams mirrors the image-service `ImageGenParams` schema
(services/image-service/app.py) so `params.model_dump()` forwards every
field the microservice expects. camelCase aliases match the web client
payload (apps/web/src/types/ai.ts ImageGenParams).
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class ImageGenParams(BaseModel):
    """Text-to-image parameters forwarded to the image-service."""

    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = Field(
        default="",
        validation_alias=AliasChoices("negative_prompt", "negativePrompt"),
        description="Negative prompt",
    )
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=30, ge=1, le=100)
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=30.0,
        validation_alias=AliasChoices("guidance_scale", "guidanceScale"),
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class EnhancePromptRequest(BaseModel):
    """Request to expand a short prompt into a detailed image prompt."""

    prompt: str = Field(..., description="Original short prompt")
    style: str = Field(default="", description="Desired visual style")


class InfographicRequest(BaseModel):
    """Request to render a data infographic overlay (PNG)."""

    topic: str = Field(..., description="Infographic title / topic")
    data_points: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of {label|key, value} items to render",
    )
    style: str = Field(default="modern", description="Visual style hint")
    width: int = Field(default=1920, ge=64, le=4096)
    height: int = Field(default=1080, ge=64, le=4096)
    background_color: tuple[int, int, int, int] = Field(
        default=(0, 0, 0, 0),
        description="RGBA fill; default transparent for video overlay",
    )
