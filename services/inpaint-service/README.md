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
- `POST /inpaint` — multipart `file` + form fields `x1, y1, x2, y2`
  (fractions 0-1 of the frame; box containing the subtitles) → `{job_id}`
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
