"""Heuristic burned-in subtitle region detection.

Pure OpenCV/numpy — no OCR, no ML models, no weight downloads. The goal is
not to read the text but to find WHERE recurring subtitle text lives so the
caller can pre-fill the /inpaint region.

How it works:

1.  Sample ~24 frames uniformly across the video, skipping the first and
    last 3% (intros/outros often carry title cards, not subtitles).
2.  Per frame, build a "text-likeness" mask: grayscale -> morphological
    gradient (text is dense fine edges) -> Otsu threshold -> horizontal
    dilation (merges letters into word/line blobs) -> connected components
    filtered to text-like geometry (height ~1-8% of the frame, wide
    aspect, sane fill ratio).
3.  Accumulate the per-frame masks into a recurrence heatmap. Real
    subtitles sit at a fixed place across many frames; transient scene
    edges do not.
4.  Down-weight CONSTANT overlays (watermarks/logos): subtitle content
    CHANGES between samples, so the masked grayscale signal of a subtitle
    row varies across frames, while a watermark row's signal is
    essentially constant. Rows that recur in nearly every frame but never
    vary are excluded from the band search.
5.  Row-profile the heatmap to find the strongest contiguous horizontal
    band (rows above a fraction of the peak, merged across small gaps),
    then column-profile within the band for the x extent. Pad the box
    slightly and clamp to [0, 1].
6.  Gate the result on peak DOMINANCE: busy footage (traffic, crowds,
    foliage) yields moderate text-like recurrence across the whole frame,
    so the strongest band must clearly stand out against the frame's
    median row activity before it is trusted as subtitles.

Coordinates in the result are FRACTIONS (0-1) of the frame, matching the
region format expected by /inpaint.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
FRAME_SAMPLES = 24        # frames sampled uniformly across the video
EDGE_SKIP_FRAC = 0.03     # skip the first/last 3% of the video
WORK_WIDTH = 960          # downscale frames to at most this width

# Text-like connected-component geometry (fractions of frame height/ratios)
COMP_MIN_HEIGHT_FRAC = 0.010   # smaller than ~1% of the frame: noise
COMP_MAX_HEIGHT_FRAC = 0.085   # taller than ~8%: not a subtitle line
COMP_MIN_ASPECT = 1.2          # width/height — dilated words are wide blobs
COMP_MIN_FILL = 0.15           # area / bbox area; filters sparse lattices
COMP_MAX_FILL = 0.95           # ... and solid rectangles (UI panels)

# Watermark exclusion: rows whose text mask recurs in >= this fraction of
# frames but whose masked-content signal varies by less than this std
# (gray levels) across frames are constant overlays, not subtitles.
WATERMARK_MIN_RECURRENCE = 0.8
WATERMARK_MAX_STD = 2.5

# Band extraction from the recurrence heatmap
BAND_ROW_FRAC = 0.35       # keep rows above this fraction of the peak profile
BAND_GAP_FRAC = 0.02       # merge row runs separated by <= 2% of the height
BAND_MIN_HEIGHT_FRAC = 0.008
COL_FRAC = 0.45            # keep columns above this fraction of the col peak
PAD_FRAC = 0.015           # padding added to every side of the final box

MIN_HIT_RATIO = 0.25       # found=false when the band recurs less than this
# found=false unless the band's peak row profile exceeds the frame's median
# row profile by this factor. Rendered text is far denser in edges than any
# background row of the same video; without this, uniformly busy footage
# (e.g. night traffic full of high-contrast blobs) fakes a subtitle band.
MIN_PEAK_DOMINANCE = 5.0

_GRAD_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# ksize is (width, height): dilate 9 px horizontally, 3 vertically so the
# letters of a word fuse into one component without bridging text rows.
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))


def _prepare_gray(frame: np.ndarray) -> np.ndarray:
    """Downscale a BGR frame to the working width and convert to grayscale."""
    height, width = frame.shape[:2]
    if width > WORK_WIDTH:
        scale = WORK_WIDTH / width
        frame = cv2.resize(
            frame,
            (WORK_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _text_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean text-likeness mask for one grayscale frame.

    Morphological gradient highlights the dense letter edges of rendered
    text; Otsu binarizes it adaptively; horizontal dilation fuses letters
    into word/line blobs; then connected components are kept only if their
    geometry looks like a text line.
    """
    height, _ = gray.shape
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, _GRAD_KERNEL)
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dilated = cv2.dilate(binary, _DILATE_KERNEL)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if num <= 1:
        return np.zeros(gray.shape, dtype=bool)

    w = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    h = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    bbox_area = w * h
    keep = np.zeros(num, dtype=bool)
    keep[1:] = (
        (h >= max(2.0, height * COMP_MIN_HEIGHT_FRAC))
        & (h <= height * COMP_MAX_HEIGHT_FRAC)
        & (w > COMP_MIN_ASPECT * h)
        & (area >= COMP_MIN_FILL * bbox_area)
        & (area <= COMP_MAX_FILL * bbox_area)
    )
    return keep[labels]


def _merge_runs(active: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Contiguous True runs in a 1-D bool array as half-open (start, end).

    Runs separated by at most `gap` False entries are merged — subtitle
    blocks have small inter-line gaps that should not split the band.
    """
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 <= gap:
            prev = i
        else:
            runs.append((start, prev + 1))
            start = prev = i
    runs.append((start, prev + 1))
    return runs


def _sample_frames(video_path: str) -> list[np.ndarray]:
    """Read ~FRAME_SAMPLES grayscale frames spread uniformly over the video."""
    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        if total <= 0:
            raise ValueError("Video reports no frames; cannot sample.")
        first = int(total * EDGE_SKIP_FRAC)
        last = max(first, total - 1 - int(total * EDGE_SKIP_FRAC))
        count = min(FRAME_SAMPLES, last - first + 1)
        indices = np.unique(np.linspace(first, last, count).round().astype(int))

        grays: list[np.ndarray] = []
        for idx in indices:
            reader.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = reader.read()
            if not ok or frame is None:
                continue
            grays.append(_prepare_gray(frame))
        if not grays:
            raise ValueError("No frames could be read from the video.")
        return grays
    finally:
        reader.release()


def detect_subtitle_region(video_path: str) -> dict:
    """Locate the recurring burned-in subtitle band in a video.

    :returns: ``{found, region: {x1, y1, x2, y2} | None, hit_ratio,
        frames_sampled}`` with region coordinates as fractions (0-1) of
        the frame. ``hit_ratio`` is the fraction of sampled frames with
        text-like content inside the detected band.
    """
    grays = _sample_frames(video_path)
    n = len(grays)
    stack = np.stack([_text_mask(g) for g in grays])  # (n, H, W) bool
    _, height, width = stack.shape

    no_band = {
        "found": False,
        "region": None,
        "hit_ratio": 0.0,
        "frames_sampled": n,
    }

    # Per-pixel recurrence across samples (0-1)
    heat = stack.mean(axis=0, dtype=np.float32)

    # Cross-frame content variance per row, for the watermark check. The
    # signal is the mean masked gray value of the row: constant overlays
    # keep it flat across frames, changing subtitle text does not.
    gray_stack = np.stack(grays).astype(np.float32)
    row_signal = (gray_stack * stack).sum(axis=2) / width  # (n, H)
    row_std = row_signal.std(axis=0)

    # Row recurrence: fraction of frames where the row has any real text
    row_counts = stack.sum(axis=2)  # (n, H)
    row_hit = row_counts >= max(3, int(round(0.005 * width)))
    row_recurrence = row_hit.mean(axis=0)

    row_profile = heat.sum(axis=1)
    watermark_rows = (row_recurrence >= WATERMARK_MIN_RECURRENCE) & (
        row_std < WATERMARK_MAX_STD
    )
    if watermark_rows.any():
        logger.info(
            "Excluding %d watermark-like rows (recurrence >= %.2f, std < %.1f)",
            int(watermark_rows.sum()), WATERMARK_MIN_RECURRENCE, WATERMARK_MAX_STD,
        )
        row_profile[watermark_rows] = 0.0

    # Light smoothing so single noisy rows don't fragment the band
    row_profile = np.convolve(row_profile, np.ones(5, np.float32) / 5, mode="same")

    peak = float(row_profile.max())
    if peak <= 0.0:
        return no_band

    # Peak dominance: how far the strongest band rises above the frame's
    # baseline edge activity (median row). Uniformly busy footage stays
    # near 1; burned-in text spikes well past MIN_PEAK_DOMINANCE.
    median_profile = float(np.median(row_profile))
    dominant = peak >= MIN_PEAK_DOMINANCE * median_profile

    bands = _merge_runs(
        row_profile >= BAND_ROW_FRAC * peak,
        gap=max(2, int(round(BAND_GAP_FRAC * height))),
    )
    if not bands:
        return no_band
    # Strongest band = largest integrated row profile
    y0, y1 = max(bands, key=lambda b: float(row_profile[b[0]:b[1]].sum()))
    if y1 - y0 < max(3, int(round(BAND_MIN_HEIGHT_FRAC * height))):
        return no_band

    # Horizontal extent from the column profile inside the band
    col_profile = heat[y0:y1].sum(axis=0)
    cols = np.flatnonzero(col_profile >= COL_FRAC * float(col_profile.max()))
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    # How many sampled frames actually show text inside the band?
    band_area = (y1 - y0) * width
    min_pixels = max(20, int(0.001 * band_area))
    band_counts = stack[:, y0:y1, :].sum(axis=(1, 2))
    hit_ratio = float((band_counts >= min_pixels).mean())

    found = dominant and hit_ratio >= MIN_HIT_RATIO
    region = None
    if found:
        region = {
            "x1": round(max(0.0, x0 / width - PAD_FRAC), 4),
            "y1": round(max(0.0, y0 / height - PAD_FRAC), 4),
            "x2": round(min(1.0, x1 / width + PAD_FRAC), 4),
            "y2": round(min(1.0, y1 / height + PAD_FRAC), 4),
        }
    logger.info(
        "Detection on %s: found=%s region=%s hit_ratio=%.3f dominance=%.2f (%d frames)",
        video_path, found, region, hit_ratio,
        peak / max(median_profile, 1e-6), n,
    )
    return {
        "found": found,
        "region": region,
        "hit_ratio": round(hit_ratio, 4),
        "frames_sampled": n,
    }
