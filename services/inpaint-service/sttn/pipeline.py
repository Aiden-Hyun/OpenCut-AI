"""STTN sliding-window video inpainting pipeline.

Adapted from the video-subtitle-remover project (Apache-2.0),
backend/inpaint/sttn_auto_inpaint.py and backend/tools/inpaint_tools.py:
https://github.com/YaoFANGUK/video-subtitle-remover

Simplified for rectangular subtitle regions supplied by the caller: no OCR
subtitle detection, no GUI hooks, no VRAM heuristics. Frames are cropped to
full-width horizontal bands around each region (VSR's speed trick: the
model only ever sees a 640x120 strip, not the whole frame), inpainted with a
sliding temporal window, then composited back into the original frames.

The caller supplies a SCHEDULE rather than one region, so a compilation
whose burned-in captions move (and sometimes double up) is handled in a
single pass: each entry names a time range and the regions active in it.
Entries with no regions — and any gap between entries — are copied through
untouched, and the whole job still produces one continuous encode, so there
are no re-encode seams. Multiple regions in one entry are inpainted as
sequential passes over the same frame buffer, so a later pass sees the
earlier pass's output and adjacent boxes cannot undo each other.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import torch

from .auto_sttn import InpaintGenerator

logger = logging.getLogger(__name__)

# Defaults matching video-subtitle-remover's config
NEIGHBOR_STRIDE = 5     # sttnNeighborStride
REF_LENGTH = 10         # sttnReferenceLength
MAX_LOAD_NUM = 50       # sttnMaxLoadNum (frames per chunk)

# STTN model input resolution (width x height); bands keep this 16:3 aspect
MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 120

# Progress weight of copying one frame through relative to inpainting one
# region-frame. Pass-through stretches are ~50x cheaper than an STTN pass,
# but they still cost something, so counting them keeps progress monotonic
# and lets it reach 1.0 exactly at the last written frame.
PASSTHROUGH_COST = 0.02


class InpaintCancelled(Exception):
    """Raised to abort inpaint_video when the caller requests cancellation."""


@dataclass
class InpaintProgress:
    """Progress snapshot handed to inpaint_video's on_progress callback.

    ``fraction`` is weighted by region-frames — the actual unit of work —
    so it advances evenly whether the current stretch has zero, one or two
    active regions. ``frames_written`` / ``total_frames`` stay honest frame
    counts for the user-facing message.
    """

    fraction: float
    frames_written: int
    total_frames: int


@dataclass
class _Region:
    """One rectangle to erase, in pixels, with its STTN band layout."""

    x0: int
    y0: int
    x1: int
    y1: int
    bands: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class _Span:
    """A contiguous half-open frame range and the regions active in it."""

    first: int
    last: int
    regions: list[_Region]

    @property
    def length(self) -> int:
        return self.last - self.first


class STTNInpaint:
    """Wraps the pretrained STTN generator for band-wise video inpainting."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = InpaintGenerator().to(self.device)
        state = _load_checkpoint(model_path)
        self.model.load_state_dict(state["netG"])
        self.model.eval()
        self.neighbor_stride = NEIGHBOR_STRIDE
        self.ref_length = REF_LENGTH

    def get_ref_index(self, neighbor_ids: list[int], length: int) -> list[int]:
        """Sample reference frames across the whole chunk."""
        ref_index = []
        for i in range(0, length, self.ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index

    @torch.no_grad()
    def inpaint(
        self,
        frames: list[np.ndarray],
        window_cb: Callable[[int, int], None] | None = None,
    ) -> list[np.ndarray]:
        """Inpaint a list of BGR uint8 frames at MODEL_INPUT_WIDTH x HEIGHT.

        Returns RGB float frames (numpy) in the same order. The caller
        converts back to BGR when compositing (BGR2RGB is symmetric).
        window_cb(windows_done, windows_total) reports sliding-window progress.
        """
        frame_length = len(frames)
        windows = list(range(0, frame_length, self.neighbor_stride))
        # BGR -> RGB, [0,255] -> [-1,1], (T,H,W,C) -> (T,C,H,W)
        arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
        feats = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255)
        feats = (feats * 2 - 1).to(self.device)
        comp_frames: list[np.ndarray | None] = [None] * frame_length

        feats = self.model.encoder(
            feats.view(frame_length, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
        )
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, frame_length, c, feat_h, feat_w)
        # Sliding temporal window over the chunk
        for window_index, f in enumerate(windows):
            neighbor_ids = [
                i for i in range(
                    max(0, f - self.neighbor_stride),
                    min(frame_length, f + self.neighbor_stride + 1),
                )
            ]
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
            pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :]))
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = pred_img[i].astype(np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    # Blend overlapping windows 50/50 like the original
                    comp_frames[idx] = (
                        comp_frames[idx].astype(np.float32) * 0.5
                        + img.astype(np.float32) * 0.5
                    )
            if window_cb:
                window_cb(window_index + 1, len(windows))
        return comp_frames


def _load_checkpoint(model_path: str) -> dict:
    """Load the pretrained checkpoint, preferring safe weights-only mode."""
    try:
        return torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        logger.warning("weights_only load failed for %s; falling back", model_path)
        return torch.load(model_path, map_location="cpu")


def compute_bands(width: int, height: int, y0: int, y1: int) -> tuple[list[tuple[int, int]], int]:
    """Full-width horizontal band(s) covering mask rows [y0, y1).

    Bands are split_h = width * 3 / 16 pixels tall (the 16:3 aspect of the
    model input), centred on the region like VSR's get_inpaint_area_by_mask
    and clamped to the frame. Regions taller than one band get several
    stacked bands so the whole rectangle is covered.
    """
    split_h = min(int(width * 3 / 16), height)
    bands: list[tuple[int, int]] = []
    region_h = max(1, y1 - y0)
    n = max(1, math.ceil(region_h / split_h))
    for i in range(n):
        center = y0 + (i + 0.5) * region_h / n
        ymin = int(round(center - split_h / 2))
        ymin = max(0, min(ymin, height - split_h))
        band = (ymin, ymin + split_h)
        if band not in bands:
            bands.append(band)
    return bands, split_h


def _plan_spans(
    schedule: list[dict],
    width: int,
    height: int,
    total: int,
    fps: float,
) -> list[_Span]:
    """Turn a seconds/fractions schedule into frame-indexed pixel spans.

    The returned spans are sorted, non-overlapping and cover the whole
    video: gaps between entries (and entries without regions) come back as
    spans with an empty region list, i.e. pure pass-through.
    """
    spans: list[_Span] = []
    cursor = 0
    for entry in sorted(schedule, key=lambda e: float(e["start"])):
        end = float(entry["end"])
        first = max(cursor, int(math.floor(float(entry["start"]) * fps)))
        last = total if math.isinf(end) else min(total, int(math.ceil(end * fps)))
        if last <= first:
            logger.info(
                "Skipping schedule entry %.2f-%.2fs: empty after clamping to %d frames",
                float(entry["start"]), end, total,
            )
            continue
        if first > cursor:
            spans.append(_Span(first=cursor, last=first, regions=[]))

        regions: list[_Region] = []
        for rect in entry.get("regions") or []:
            fx1, fy1, fx2, fy2 = rect
            x0 = max(0, min(width - 1, int(round(fx1 * width))))
            x1 = max(x0 + 1, min(width, int(round(fx2 * width))))
            y0 = max(0, min(height - 1, int(round(fy1 * height))))
            y1 = max(y0 + 1, min(height, int(round(fy2 * height))))
            bands, _ = compute_bands(width, height, y0, y1)
            regions.append(_Region(x0=x0, y0=y0, x1=x1, y1=y1, bands=bands))

        spans.append(_Span(first=first, last=last, regions=regions))
        cursor = last
    if cursor < total:
        spans.append(_Span(first=cursor, last=total, regions=[]))
    return spans


def inpaint_video(
    inpainter: STTNInpaint,
    input_path: str,
    output_path: str,
    schedule: list[dict],
    on_progress: Callable[[InpaintProgress], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    """Remove a time-varying set of regions from a video by STTN inpainting.

    :param schedule: entries ``{"start": seconds, "end": seconds, "regions":
        [(x1, y1, x2, y2), ...]}`` with coordinates as fractions (0-1) of
        the frame. ``end`` may be ``float("inf")`` for "to the end". Entries
        are sorted and clamped so they never overlap; frames covered by no
        entry, or by an entry with no regions, are copied through untouched.
        A single whole-video entry reproduces the old single-region job.
    :param on_progress: callback(InpaintProgress)
    :param should_cancel: polled between chunks, between regions and between
        sliding windows; return True to abort promptly via InpaintCancelled
    :returns: number of frames written
    :raises InpaintCancelled: when should_cancel() turns True
    """
    reader = cv2.VideoCapture(input_path)
    if not reader.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fps = reader.get(cv2.CAP_PROP_FPS)
    total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
    if not fps or fps != fps or fps <= 0:  # NaN or invalid
        fps = 30.0
    if width <= 0 or height <= 0:
        reader.release()
        raise ValueError("Video has invalid dimensions.")
    if total <= 0:
        reader.release()
        raise ValueError("Video reports no frames; cannot schedule inpainting.")

    spans = _plan_spans(schedule, width, height, total, fps)
    # Unit of work = one region-frame; copied frames count for a fraction.
    total_units = sum(
        span.length * (len(span.regions) if span.regions else PASSTHROUGH_COST)
        for span in spans
    )
    logger.info(
        "Inpainting %s: %dx%d @ %.2ffps, %d frames, %d span(s), %.0f region-frames",
        input_path, width, height, fps, total, len(spans), total_units,
    )
    for span in spans:
        logger.info(
            "  frames %d-%d (%.2f-%.2fs): %s",
            span.first, span.last, span.first / fps, span.last / fps,
            [
                f"px ({r.x0},{r.y0})-({r.x1},{r.y1}) bands={r.bands}"
                for r in span.regions
            ]
            or "pass-through",
        )

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        reader.release()
        raise ValueError(f"Cannot open video writer: {output_path}")

    frames_done = 0
    units_done = 0.0

    def report(units: float) -> None:
        if not on_progress:
            return
        on_progress(
            InpaintProgress(
                fraction=min(1.0, units / total_units) if total_units > 0 else 1.0,
                frames_written=frames_done,
                total_frames=max(total, frames_done),
            )
        )

    try:
        for span in spans:
            if should_cancel and should_cancel():
                raise InpaintCancelled("Cancelled between schedule entries.")
            if not span.regions:
                # Nothing to erase here: copy the frames straight through.
                for _ in range(span.length):
                    success, image = reader.read()
                    if not success:
                        break
                    writer.write(image)
                    frames_done += 1
                    units_done += PASSTHROUGH_COST
                    report(units_done)
                continue

            remaining = span.length
            while remaining > 0:
                if should_cancel and should_cancel():
                    raise InpaintCancelled("Cancelled between chunks.")
                # Read the next chunk of frames from this span
                frames_hr: list[np.ndarray] = []
                while len(frames_hr) < min(MAX_LOAD_NUM, remaining):
                    success, image = reader.read()
                    if not success:
                        break
                    frames_hr.append(image)
                if not frames_hr:
                    break
                remaining -= len(frames_hr)

                # One STTN pass per region, cumulative: each pass crops from
                # the frames the previous pass already wrote into, so
                # overlapping or adjacent boxes cannot undo each other.
                for region_index, region in enumerate(span.regions):
                    if should_cancel and should_cancel():
                        raise InpaintCancelled("Cancelled between regions.")
                    bands = region.bands
                    comps: dict[int, list[np.ndarray]] = {}
                    for k, (ymin, ymax) in enumerate(bands):
                        band_frames = [
                            cv2.resize(
                                f[ymin:ymax, :, :],
                                (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
                            )
                            for f in frames_hr
                        ]

                        def window_cb(
                            done_windows: int,
                            total_windows: int,
                            band_index: int = k,
                            done_regions: int = region_index,
                        ) -> None:
                            """Report fine-grained progress while a chunk is inpainted.

                            Also the cancellation point: STTNInpaint.inpaint calls
                            this after every sliding window, so a cancel takes
                            effect within seconds instead of waiting out the whole
                            chunk.
                            """
                            if should_cancel and should_cancel():
                                raise InpaintCancelled(
                                    "Cancelled between sliding windows."
                                )
                            band_fraction = (
                                band_index + done_windows / total_windows
                            ) / len(bands)
                            report(
                                units_done
                                + (done_regions + band_fraction) * len(frames_hr)
                            )

                        comps[k] = inpainter.inpaint(band_frames, window_cb=window_cb)

                    # Composite this region's bands back into the frames. The
                    # mask is a hard rectangle, so copying the band's pixels
                    # inside the region's x range is the whole compositing step.
                    for j, frame in enumerate(frames_hr):
                        for k, (ymin, ymax) in enumerate(bands):
                            comp = cv2.resize(comps[k][j], (width, ymax - ymin))
                            # inpaint() returned RGB; swap back to BGR for cv2
                            comp = cv2.cvtColor(
                                comp.astype(np.uint8), cv2.COLOR_RGB2BGR
                            )
                            top = max(region.y0, ymin)
                            bottom = min(region.y1, ymax)
                            if bottom <= top:
                                continue
                            frame[top:bottom, region.x0:region.x1, :] = comp[
                                top - ymin:bottom - ymin, region.x0:region.x1, :
                            ]

                units_done += len(frames_hr) * len(span.regions)
                for frame in frames_hr:
                    writer.write(frame)
                    frames_done += 1
                report(units_done)

        # Metadata frame counts can under-report; never drop trailing frames.
        while True:
            success, image = reader.read()
            if not success:
                break
            writer.write(image)
            frames_done += 1
        report(total_units)
    finally:
        reader.release()
        writer.release()

    if frames_done == 0:
        raise ValueError("No frames could be read from the video.")
    return frames_done
