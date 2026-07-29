# inpaint-service

STTN video inpainting microservice for removing burned-in (hardcoded)
subtitles. Port **8427**.

The STTN implementation is vendored from
[video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
(Apache-2.0), which builds on the original
[STTN](https://github.com/researchmm/STTN) research code (ECCV 2020).

## API

- `GET /health` — `{status, model: {installed, loaded}}`
- `POST /detect-region` — multipart `file` →
  `{found, region: {x1, y1, x2, y2} | null, hit_ratio, frames_sampled}`.
  Synchronous OpenCV heuristic (no OCR, no weights) that locates the
  recurring burned-in subtitle band; constant overlays (watermarks/logos)
  are excluded by their lack of cross-frame content variance, and
  uniformly busy footage is rejected by a peak-dominance gate.
- `POST /detect-timeline` — multipart `file` + optional form field
  `window_seconds` (default 4) →
  `{segments: [{start, end, regions: [{x1, y1, x2, y2}], hit_ratio}], duration, frames_sampled, windows}`.
  Time-varying multi-region variant of `/detect-region` for compilations
  whose captions move between clips (and sometimes double up): the video is
  walked in short windows, every qualifying band per window is kept, and
  adjacent windows with the same region set are grouped into segments.
  Segments with `regions: []` have nothing to erase. Windows are widened
  automatically so a long video stays under a ~400 frame sampling budget.
- `POST /inpaint` — multipart `file` plus **either**
  - form fields `x1, y1, x2, y2` (fractions 0-1 of the frame; one box for
    the whole video), **or**
  - form field `schedule`, JSON
    `[{"start": seconds, "end": seconds, "regions": [{"x1": …, "y1": …, "x2": …, "y2": …}]}]`
    as returned by `/detect-timeline` (it takes precedence over `x1..y2`).
    Entries are sorted and clamped so they never overlap; frames covered by
    no entry — or by an entry with an empty `regions` list — are copied
    through untouched, and the whole job is still one continuous encode.
    Malformed schedules are rejected with 422.

  → `{job_id}`
- `GET /jobs/{job_id}` — `{status: queued|processing|done|error|cancelled, progress, message}`
- `POST /jobs/{job_id}/cancel` — abort a queued/running job (the worker
  polls the flag between STTN sliding windows and cleans up its temp
  files); no-op on finished jobs. Returns the job status.
- `GET /result/{job_id}` — processed mp4 (H.264, original audio remuxed)

## Model weights

Weights (`infer_model.pth`, ~66 MB) are lazy-downloaded on first use from

```
https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/backend/models/sttn-auto/infer_model.pth
```

into `/root/.cache/opencutai/sttn/` (docker volume `inpaint_models`).
Override the source with `STTN_MODEL_URL`.

### Pre-seeding weights on flaky networks

Downloads from inside the Docker VM can hang. Download on the HOST and
copy into the container volume instead:

```bash
curl -L -o /tmp/infer_model.pth \
  https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/backend/models/sttn-auto/infer_model.pth
docker compose up -d inpaint-service
docker exec opencut-ai-inpaint-service-1 mkdir -p /root/.cache/opencutai/sttn
docker cp /tmp/infer_model.pth \
  opencut-ai-inpaint-service-1:/root/.cache/opencutai/sttn/infer_model.pth
```

## Notes

- CPU-only torch inside Docker; processing is slow — expect several
  minutes per minute of video.
- Frames are cropped to a full-width horizontal band around the region
  (16:3 aspect, resized to the model's 640x120 input) so the model never
  sees the whole frame, then composited back — VSR's speed trick.
- Two regions active at the same timestamp cost two STTN passes over those
  frames, run cumulatively (the second pass sees the first pass's output),
  so job progress is weighted by region-frames rather than frames.
