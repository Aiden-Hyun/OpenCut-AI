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

``detect_subtitle_timeline`` applies the same machinery to a SLIDING
TIMELINE instead of the whole file: compilations mix source clips whose
burned-in captions sit at different places, and some scenes carry two at
once (an editor caption mid-frame plus dialogue subtitles at the bottom).
It walks the video in short windows, keeps EVERY qualifying band per
window (not just the strongest), and groups adjacent windows with the same
region set into contiguous segments that /inpaint can erase in one pass.
Two deliberate differences from the whole-video mode:

*   Watermark rows are computed ONCE from a whole-video sample and reused
    for every window. Within a 4 s window a real subtitle line often shows
    the same text in every sampled frame, so a per-window variance test
    would mistake it for a constant overlay; across the whole file a
    subtitle row always varies somewhere, a watermark row never does.
*   Band candidates use a lower row-profile threshold
    (``TIMELINE_BAND_ROW_FRAC``), because a secondary one-line caption
    sits far below the peak of a two-line subtitle block and would never
    become a candidate at the whole-video threshold. Every candidate still
    has to clear the min-height, hit-ratio and peak-dominance gates on its
    own.
"""

import logging
import math
from dataclasses import dataclass

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

# --- Time-windowed multi-region mode (detect_subtitle_timeline) ------------
WINDOW_SECONDS = 4.0          # default window length in seconds
WINDOW_FRAME_SAMPLES = 7      # frames sampled per window
# Hard cap on frames decoded for the walk. A 13-minute video at 4 s windows
# would want ~1400 frames; the window is widened instead so detection stays
# in the tens of seconds rather than minutes.
MAX_TIMELINE_FRAMES = 400
# Candidate threshold for the windowed mode, deliberately below
# BAND_ROW_FRAC: next to a two-line subtitle block a single mid-frame
# caption peaks at roughly a quarter of the block's row profile, so 0.35
# would drop it before the per-band gates ever see it.
TIMELINE_BAND_ROW_FRAC = 0.20
SEGMENT_MERGE_IOU = 0.6       # per-region IoU for "these windows match"
MIN_SEGMENT_SECONDS = 1.5     # shorter segments fold into a neighbour

# Extra per-band gates for the windowed mode. Seven frames of a 4 s window
# carry far less evidence than 24 frames spread over the whole file, so busy
# footage throws up blobs that clear the whole-video gates. Measured over the
# night-driving footage (real burned-in captions vs background blobs that
# cleared MIN_PEAK_DOMINANCE):
#   * caption bands: height 0.04-0.12, aspect 2.5-8.1, coverage 0.14-0.83
#   * background blobs: height up to 0.38, aspect down to 0.4,
#     coverage 0.016-0.108
# Coverage — the share of the band box actually filled with recurring
# text-like pixels — is the discriminator that survives both directions:
# dominance ratios move with the footage's baseline edge density (blobs in
# soft footage reached 12x while real captions in busy footage sat at 3x),
# coverage does not. Aspect stays low enough for short CJK caption lines.
TIMELINE_BAND_MAX_HEIGHT_FRAC = 0.22   # taller than ~2 subtitle lines
TIMELINE_BAND_MIN_ASPECT = 2.0         # width/height of the band box
TIMELINE_MIN_COVERAGE = 0.12           # recurring text pixels / band area

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


@dataclass
class _Analysis:
    """Recurrence analysis of one set of sampled frames.

    Shared by the whole-video and time-windowed detectors: ``heat`` is the
    per-pixel recurrence map, ``row_profile`` its smoothed row sums with
    watermark rows zeroed, and ``stack`` the per-frame text masks used for
    hit ratios.
    """

    stack: np.ndarray        # (n, H, W) bool text masks
    heat: np.ndarray         # (H, W) float32 recurrence 0-1
    row_profile: np.ndarray  # (H,) float32, watermark rows zeroed
    peak: float
    median_profile: float

    @property
    def frames(self) -> int:
        return int(self.stack.shape[0])

    @property
    def height(self) -> int:
        return int(self.stack.shape[1])

    @property
    def width(self) -> int:
        return int(self.stack.shape[2])


def _watermark_rows(grays: list[np.ndarray], stack: np.ndarray) -> np.ndarray:
    """Rows holding a CONSTANT overlay (watermark/logo) rather than text.

    The signal is the mean masked gray value of each row: it stays flat
    across frames for a fixed overlay and moves as subtitle text changes.
    """
    width = stack.shape[2]
    gray_stack = np.stack(grays).astype(np.float32)
    row_signal = (gray_stack * stack).sum(axis=2) / width  # (n, H)
    row_std = row_signal.std(axis=0)

    # Row recurrence: fraction of frames where the row has any real text
    row_counts = stack.sum(axis=2)  # (n, H)
    row_hit = row_counts >= max(3, int(round(0.005 * width)))
    row_recurrence = row_hit.mean(axis=0)
    return (row_recurrence >= WATERMARK_MIN_RECURRENCE) & (row_std < WATERMARK_MAX_STD)


def _analyze(
    grays: list[np.ndarray],
    exclude_rows: np.ndarray | None = None,
) -> _Analysis | None:
    """Build the recurrence analysis for a set of sampled frames.

    :param exclude_rows: precomputed watermark row mask; when omitted it is
        derived from these same frames (whole-video mode). The windowed
        detector passes a whole-video mask instead, because a 4 s window is
        too short to tell a static subtitle apart from a watermark.
    :returns: the analysis, or None when no text-like pixel recurs at all.
    """
    stack = np.stack([_text_mask(g) for g in grays])  # (n, H, W) bool
    heat = stack.mean(axis=0, dtype=np.float32)
    row_profile = heat.sum(axis=1)

    watermark_rows = (
        _watermark_rows(grays, stack) if exclude_rows is None else exclude_rows
    )
    if watermark_rows.any():
        if exclude_rows is None:
            logger.info(
                "Excluding %d watermark-like rows (recurrence >= %.2f, std < %.1f)",
                int(watermark_rows.sum()),
                WATERMARK_MIN_RECURRENCE,
                WATERMARK_MAX_STD,
            )
        row_profile = row_profile.copy()
        row_profile[watermark_rows] = 0.0

    # Light smoothing so single noisy rows don't fragment the band
    row_profile = np.convolve(row_profile, np.ones(5, np.float32) / 5, mode="same")

    peak = float(row_profile.max())
    if peak <= 0.0:
        return None
    return _Analysis(
        stack=stack,
        heat=heat,
        row_profile=row_profile,
        peak=peak,
        median_profile=float(np.median(row_profile)),
    )


def _band_candidates(
    analysis: _Analysis, row_frac: float = BAND_ROW_FRAC
) -> list[tuple[int, int]]:
    """Row runs of the profile above ``row_frac`` of its peak."""
    return _merge_runs(
        analysis.row_profile >= row_frac * analysis.peak,
        gap=max(2, int(round(BAND_GAP_FRAC * analysis.height))),
    )


def _band_is_tall_enough(analysis: _Analysis, y0: int, y1: int) -> bool:
    return y1 - y0 >= max(3, int(round(BAND_MIN_HEIGHT_FRAC * analysis.height)))


def _band_dominance(analysis: _Analysis, y0: int, y1: int) -> float:
    """How far the band's peak row rises above the frame's median row.

    Uniformly busy footage (traffic, crowds, foliage) stays near 1; rendered
    text spikes well past MIN_PEAK_DOMINANCE.
    """
    band_peak = float(analysis.row_profile[y0:y1].max())
    return band_peak / max(analysis.median_profile, 1e-6)


def _band_coverage(analysis: _Analysis, y0: int, y1: int, x0: int, x1: int) -> float:
    """Share of the band box filled with recurring text-like pixels.

    Rendered text fills its box densely and repeatedly; a bright structural
    edge (a car pillar, a light streak) only grazes the box it spans.
    """
    return float(analysis.heat[y0:y1, x0:x1].mean())


def _band_hit_ratio(analysis: _Analysis, y0: int, y1: int) -> float:
    """Fraction of sampled frames showing text-like content in the band."""
    band_area = (y1 - y0) * analysis.width
    min_pixels = max(20, int(0.001 * band_area))
    band_counts = analysis.stack[:, y0:y1, :].sum(axis=(1, 2))
    return float((band_counts >= min_pixels).mean())


def _band_columns(analysis: _Analysis, y0: int, y1: int) -> tuple[int, int]:
    """Half-open x extent of a band from the column profile inside it."""
    col_profile = analysis.heat[y0:y1].sum(axis=0)
    cols = np.flatnonzero(col_profile >= COL_FRAC * float(col_profile.max()))
    return int(cols[0]), int(cols[-1]) + 1


def _band_region(analysis: _Analysis, y0: int, y1: int) -> dict:
    """Padded, clamped fractional box for a band, x extent from its columns."""
    x0, x1 = _band_columns(analysis, y0, y1)
    width, height = analysis.width, analysis.height
    return {
        "x1": round(max(0.0, x0 / width - PAD_FRAC), 4),
        "y1": round(max(0.0, y0 / height - PAD_FRAC), 4),
        "x2": round(min(1.0, x1 / width + PAD_FRAC), 4),
        "y2": round(min(1.0, y1 / height + PAD_FRAC), 4),
    }


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

    no_band = {
        "found": False,
        "region": None,
        "hit_ratio": 0.0,
        "frames_sampled": n,
    }

    analysis = _analyze(grays)
    if analysis is None:
        return no_band

    # Peak dominance: how far the strongest band rises above the frame's
    # baseline edge activity (median row). Uniformly busy footage stays
    # near 1; burned-in text spikes well past MIN_PEAK_DOMINANCE.
    dominant = analysis.peak >= MIN_PEAK_DOMINANCE * analysis.median_profile

    bands = _band_candidates(analysis)
    if not bands:
        return no_band
    # Strongest band = largest integrated row profile
    y0, y1 = max(
        bands, key=lambda b: float(analysis.row_profile[b[0]:b[1]].sum())
    )
    if not _band_is_tall_enough(analysis, y0, y1):
        return no_band

    hit_ratio = _band_hit_ratio(analysis, y0, y1)
    found = dominant and hit_ratio >= MIN_HIT_RATIO
    region = _band_region(analysis, y0, y1) if found else None
    logger.info(
        "Detection on %s: found=%s region=%s hit_ratio=%.3f dominance=%.2f (%d frames)",
        video_path, found, region, hit_ratio,
        analysis.peak / max(analysis.median_profile, 1e-6), n,
    )
    return {
        "found": found,
        "region": region,
        "hit_ratio": round(hit_ratio, 4),
        "frames_sampled": n,
    }


# ---------------------------------------------------------------------------
# Time-windowed multi-region detection
# ---------------------------------------------------------------------------


def _region_iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two fractional boxes."""
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _pair_regions(a: list[dict], b: list[dict]) -> list[tuple[int, int, float]]:
    """Greedily pair equal-length region sets by descending IoU.

    Index order would usually do (bands come out top-to-bottom), but greedy
    pairing keeps the comparison stable when a window picks the regions up
    in a different order.
    """
    scores = sorted(
        ((_region_iou(ra, rb), i, j) for i, ra in enumerate(a) for j, rb in enumerate(b)),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for iou, i, j in scores:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, iou))
    return pairs


def _region_set_similarity(a: list[dict], b: list[dict]) -> float | None:
    """Mean IoU of the best pairing, or None when the counts differ."""
    if len(a) != len(b):
        return None
    if not a:
        return 1.0
    pairs = _pair_regions(a, b)
    return sum(iou for _, _, iou in pairs) / len(pairs)


def _same_region_set(a: list[dict], b: list[dict]) -> bool:
    """True when both sets hold the same count of overlapping regions."""
    if len(a) != len(b):
        return False
    if not a:
        return True
    return all(iou >= SEGMENT_MERGE_IOU for _, _, iou in _pair_regions(a, b))


def _union_region_sets(a: list[dict], b: list[dict]) -> list[dict]:
    """Union each matched pair of near-identical regions into one box."""
    if not a or len(a) != len(b):
        return a
    merged = [dict(region) for region in a]
    for i, j, _ in _pair_regions(a, b):
        merged[i] = {
            "x1": round(min(a[i]["x1"], b[j]["x1"]), 4),
            "y1": round(min(a[i]["y1"], b[j]["y1"]), 4),
            "x2": round(max(a[i]["x2"], b[j]["x2"]), 4),
            "y2": round(max(a[i]["y2"], b[j]["y2"]), 4),
        }
    return merged


def _window_regions(
    analysis: _Analysis, min_hit_ratio: float = MIN_HIT_RATIO
) -> tuple[list[dict], list[float]]:
    """Every band in one window that independently looks like subtitles.

    A band qualifies when it clears TIMELINE_BAND_ROW_FRAC of the window's
    peak row profile, the min-height check, ``min_hit_ratio`` within the
    window and MIN_PEAK_DOMINANCE against the window's median row — plus the
    windowed-mode geometry and mean-dominance gates that keep busy footage
    from passing a bright blob off as a caption.
    """
    regions: list[dict] = []
    ratios: list[float] = []
    for y0, y1 in _band_candidates(analysis, TIMELINE_BAND_ROW_FRAC):
        if not _band_is_tall_enough(analysis, y0, y1):
            continue
        band_height = (y1 - y0) / analysis.height
        if band_height > TIMELINE_BAND_MAX_HEIGHT_FRAC:
            continue
        x0, x1 = _band_columns(analysis, y0, y1)
        band_width = (x1 - x0) / analysis.width
        if band_width < TIMELINE_BAND_MIN_ASPECT * band_height:
            continue
        if _band_dominance(analysis, y0, y1) < MIN_PEAK_DOMINANCE:
            continue
        if _band_coverage(analysis, y0, y1, x0, x1) < TIMELINE_MIN_COVERAGE:
            continue
        hit_ratio = _band_hit_ratio(analysis, y0, y1)
        if hit_ratio < min_hit_ratio:
            continue
        regions.append(_band_region(analysis, y0, y1))
        ratios.append(hit_ratio)
    return regions, ratios


def _merge_windows(windows: list[dict]) -> list[dict]:
    """Fold adjacent windows with matching region sets into segments."""
    segments: list[dict] = []
    for window in windows:
        previous = segments[-1] if segments else None
        if previous and _same_region_set(previous["regions"], window["regions"]):
            previous["end"] = window["end"]
            previous["regions"] = _union_region_sets(
                previous["regions"], window["regions"]
            )
            previous["ratios"].extend(window["ratios"])
            continue
        segments.append(
            {
                "start": window["start"],
                "end": window["end"],
                "regions": window["regions"],
                "ratios": list(window["ratios"]),
            }
        )
    return segments


def _absorb_short_segments(segments: list[dict], min_seconds: float) -> list[dict]:
    """Fold sub-``min_seconds`` segments into their best-matching neighbour.

    Window-level noise otherwise produces a flickery timeline (a single
    window where one of two captions dropped out becomes its own segment).
    The host neighbour keeps its own region set and swallows the time range.
    """

    def preference(short: dict, neighbour: dict | None) -> tuple[float, float]:
        """(region-set similarity, neighbour duration); higher hosts better."""
        if neighbour is None:
            return (-1.0, -1.0)
        similarity = _region_set_similarity(short["regions"], neighbour["regions"])
        return (
            similarity if similarity is not None else -0.5,
            neighbour["end"] - neighbour["start"],
        )

    while len(segments) > 1:
        index = next(
            (
                i
                for i, s in enumerate(segments)
                if s["end"] - s["start"] < min_seconds
            ),
            None,
        )
        if index is None:
            break
        short = segments[index]
        before = segments[index - 1] if index > 0 else None
        after = segments[index + 1] if index + 1 < len(segments) else None
        if preference(short, before) >= preference(short, after):
            before["end"] = short["end"]  # type: ignore[index]
            before["ratios"].extend(short["ratios"])  # type: ignore[index]
        else:
            after["start"] = short["start"]  # type: ignore[index]
            after["ratios"].extend(short["ratios"])  # type: ignore[index]
        segments.pop(index)
    return segments


def _relink_segments(segments: list[dict]) -> list[dict]:
    """Second merge pass: neighbours can match once shorts were absorbed."""
    merged: list[dict] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        if previous and _same_region_set(previous["regions"], segment["regions"]):
            previous["end"] = segment["end"]
            previous["regions"] = _union_region_sets(
                previous["regions"], segment["regions"]
            )
            previous["ratios"].extend(segment["ratios"])
            continue
        merged.append(segment)
    return merged


def detect_subtitle_timeline(
    video_path: str, window_seconds: float = WINDOW_SECONDS
) -> dict:
    """Locate burned-in subtitle regions over TIME, not just over the file.

    Walks the video in ~``window_seconds`` windows of ~WINDOW_FRAME_SAMPLES
    frames each (the window widens when the total would exceed
    MAX_TIMELINE_FRAMES), collects every qualifying band per window, then
    groups adjacent windows with the same region set into segments. A
    segment with ``regions: []`` has nothing to erase.

    :returns: ``{segments: [{start, end, regions: [{x1, y1, x2, y2}],
        hit_ratio}], duration, frames_sampled, windows}`` — seconds for
        times, fractions (0-1) of the frame for coordinates.
    """
    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        fps = reader.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:  # NaN or invalid
            fps = 30.0
        if total <= 0:
            raise ValueError("Video reports no frames; cannot sample.")
        duration = total / fps

        # Watermark rows come from a whole-video sample: within one window a
        # static subtitle is indistinguishable from a fixed overlay.
        global_grays = _sample_frames(video_path)
        frames_sampled = len(global_grays)
        exclude_rows = _watermark_rows(
            global_grays, np.stack([_text_mask(g) for g in global_grays])
        )
        if exclude_rows.any():
            logger.info(
                "Timeline detection excluding %d watermark-like rows",
                int(exclude_rows.sum()),
            )

        window = max(0.5, float(window_seconds))
        count = max(1, math.ceil(duration / window))
        if count * WINDOW_FRAME_SAMPLES > MAX_TIMELINE_FRAMES:
            widened = duration * WINDOW_FRAME_SAMPLES / MAX_TIMELINE_FRAMES
            logger.info(
                "Widening detection window %.1fs -> %.1fs: %.0fs of video at "
                "%d frames/window would need %d frames (cap %d)",
                window, widened, duration, WINDOW_FRAME_SAMPLES,
                count * WINDOW_FRAME_SAMPLES, MAX_TIMELINE_FRAMES,
            )
            window = widened
            count = max(1, math.ceil(duration / window))

        windows: list[dict] = []
        for index in range(count):
            first = int(round(index * window * fps))
            last = min(total, int(round((index + 1) * window * fps)))
            if last - first < 2:  # too short to carry a recurrence signal
                continue
            samples = min(WINDOW_FRAME_SAMPLES, last - first)
            indices = np.unique(
                np.linspace(first, last - 1, samples).round().astype(int)
            )
            grays: list[np.ndarray] = []
            for frame_index in indices:
                reader.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = reader.read()
                if not ok or frame is None:
                    continue
                grays.append(_prepare_gray(frame))
            frames_sampled += len(grays)
            if len(grays) < 2:
                continue

            analysis = _analyze(grays, exclude_rows=exclude_rows)
            regions, ratios = ([], []) if analysis is None else _window_regions(analysis)
            windows.append(
                {
                    "start": first / fps,
                    "end": last / fps,
                    "regions": regions,
                    "ratios": ratios,
                }
            )
    finally:
        reader.release()

    segments = _relink_segments(
        _absorb_short_segments(_merge_windows(windows), MIN_SEGMENT_SECONDS)
    )

    result = {
        "segments": [
            {
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "regions": segment["regions"],
                "hit_ratio": round(
                    sum(segment["ratios"]) / len(segment["ratios"]), 4
                )
                if segment["ratios"]
                else 0.0,
            }
            for segment in segments
        ],
        "duration": round(duration, 3),
        "frames_sampled": frames_sampled,
        "windows": len(windows),
    }
    logger.info(
        "Timeline detection on %s: %.1fs, %d windows of %.1fs, %d segments",
        video_path, duration, len(windows), window, len(result["segments"]),
    )
    for segment in result["segments"]:
        logger.info(
            "  %6.2f-%6.2fs: %d region(s) %s hit_ratio=%.2f",
            segment["start"], segment["end"], len(segment["regions"]),
            [
                f"({r['x1']:.2f},{r['y1']:.2f})-({r['x2']:.2f},{r['y2']:.2f})"
                for r in segment["regions"]
            ],
            segment["hit_ratio"],
        )
    return result
