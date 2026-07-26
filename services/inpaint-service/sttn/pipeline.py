"""STTN sliding-window video inpainting pipeline.

Adapted from the video-subtitle-remover project (Apache-2.0),
backend/inpaint/sttn_auto_inpaint.py and backend/tools/inpaint_tools.py:
https://github.com/YaoFANGUK/video-subtitle-remover

Simplified for a fixed rectangular subtitle region supplied by the caller:
no OCR subtitle detection, no GUI hooks, no VRAM heuristics. Frames are
cropped to full-width horizontal bands around the region (VSR's speed
trick: the model only ever sees a 640x120 strip, not the whole frame),
inpainted with a sliding temporal window, then composited back into the
original frames using the region mask.
"""

import logging
import math
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


def inpaint_video(
    inpainter: STTNInpaint,
    input_path: str,
    output_path: str,
    region: tuple[float, float, float, float],
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Remove the given region from a video by STTN inpainting.

    :param region: (x1, y1, x2, y2) as fractions (0-1) of the frame
    :param on_progress: callback(frames_done, total_frames)
    :returns: number of frames written
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

    fx1, fy1, fx2, fy2 = region
    x0 = max(0, min(width - 1, int(round(fx1 * width))))
    x1 = max(x0 + 1, min(width, int(round(fx2 * width))))
    y0 = max(0, min(height - 1, int(round(fy1 * height))))
    y1 = max(y0 + 1, min(height, int(round(fy2 * height))))

    # Binary mask of the subtitle region, float for compositing
    mask = np.zeros((height, width, 1), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0

    bands, split_h = compute_bands(width, height, y0, y1)
    logger.info(
        "Inpainting %s: %dx%d @ %.2ffps, %d frames, region px (%d,%d)-(%d,%d), bands=%s",
        input_path, width, height, fps, total, x0, y0, x1, y1, bands,
    )

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        reader.release()
        raise ValueError(f"Cannot open video writer: {output_path}")

    frames_done = 0
    try:
        while True:
            # Read the next chunk of frames
            frames_hr: list[np.ndarray] = []
            while len(frames_hr) < MAX_LOAD_NUM:
                success, image = reader.read()
                if not success:
                    break
                frames_hr.append(image)
            if not frames_hr:
                break

            # Crop + resize each band, run STTN per band
            comps: dict[int, list[np.ndarray]] = {}
            for k, (ymin, ymax) in enumerate(bands):
                band_frames = [
                    cv2.resize(
                        f[ymin:ymax, :, :],
                        (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
                    )
                    for f in frames_hr
                ]

                def window_cb(done_windows: int, total_windows: int, band_index: int = k) -> None:
                    """Report fine-grained progress while a chunk is inpainted."""
                    if not on_progress:
                        return
                    chunk_fraction = (band_index + done_windows / total_windows) / len(bands)
                    virtual_done = frames_done + chunk_fraction * len(frames_hr)
                    on_progress(int(virtual_done), max(total, frames_done + len(frames_hr)))

                comps[k] = inpainter.inpaint(band_frames, window_cb=window_cb)

            # Composite inpainted bands back into the original frames
            for j, frame in enumerate(frames_hr):
                for k, (ymin, ymax) in enumerate(bands):
                    comp = cv2.resize(comps[k][j], (width, ymax - ymin))
                    # inpaint() returned RGB; swap back to BGR for cv2
                    comp = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    mask_area = mask[ymin:ymax, :]
                    frame[ymin:ymax, :, :] = (
                        mask_area * comp + (1 - mask_area) * frame[ymin:ymax, :, :]
                    )
                writer.write(frame)
                frames_done += 1
                if on_progress:
                    on_progress(frames_done, max(total, frames_done))
    finally:
        reader.release()
        writer.release()

    if frames_done == 0:
        raise ValueError("No frames could be read from the video.")
    return frames_done
