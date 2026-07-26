"""STTN video inpainting microservice.

Standalone FastAPI service that removes burned-in (hardcoded) subtitles
from video by inpainting a caller-supplied rectangular region with STTN
(Spatial-Temporal Transformer Networks). Jobs run in a background thread,
are polled by id and can be cancelled mid-run. /detect-region suggests the
subtitle region via a pure-OpenCV heuristic. Runs on port 8427.

Model weights come from the video-subtitle-remover project (Apache-2.0)
and are lazy-downloaded on first use into ~/.cache.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from detect import detect_subtitle_region

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = os.getenv("RESULT_DIR", "results")
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "opencutai", "sttn"),
)
MODEL_PATH = os.path.join(MODEL_DIR, "infer_model.pth")
# STTN weights shipped inside the video-subtitle-remover repo (Apache-2.0).
# See README.md for pre-seeding the volume on flaky networks.
MODEL_URL = os.getenv(
    "STTN_MODEL_URL",
    "https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/backend/models/sttn-auto/infer_model.pth",
)
MODEL_MIN_BYTES = 60_000_000  # sanity check: real checkpoint is ~66 MB
DOWNLOAD_ATTEMPTS = int(os.getenv("STTN_DOWNLOAD_ATTEMPTS", "5"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}

# ---------------------------------------------------------------------------
# Job store (in-memory; jobs are ephemeral like the container filesystem)
# ---------------------------------------------------------------------------


@dataclass
class Job:
    job_id: str
    status: str = "queued"  # queued | processing | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result_path: str | None = None
    # Set by /jobs/{id}/cancel; polled by the worker between sliding windows.
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
# STTN on CPU is heavy; run one job at a time so concurrent uploads queue up.
_work_lock = threading.Lock()


def _update_job(job_id: str, **updates) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)


def _cancel_requested(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return bool(job and job.cancel_requested)


def _job_snapshot(job: Job) -> dict:
    """Public status shape shared by /jobs/{id} and /jobs/{id}/cancel."""
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": round(job.progress, 4),
        "message": job.message,
        "error": job.error,
    }


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


class InpaintService:
    """Singleton wrapping the STTN inpainter with lazy weight download."""

    _instance: "InpaintService | None" = None
    _inpainter = None
    _load_lock = threading.Lock()

    def __new__(cls) -> "InpaintService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_installed(self) -> bool:
        try:
            return os.path.getsize(MODEL_PATH) >= MODEL_MIN_BYTES
        except OSError:
            return False

    @property
    def is_loaded(self) -> bool:
        return self._inpainter is not None

    def download_weights(self, on_status=None) -> None:
        """Download the STTN checkpoint with retries (DNS can be flaky in the VM)."""
        if self.is_installed:
            return
        os.makedirs(MODEL_DIR, exist_ok=True)
        part_path = MODEL_PATH + ".part"
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                if on_status:
                    on_status(f"Downloading STTN weights (attempt {attempt}/{DOWNLOAD_ATTEMPTS})...")
                logger.info("Downloading STTN weights from %s (attempt %d)", MODEL_URL, attempt)
                request = urllib.request.Request(
                    MODEL_URL, headers={"User-Agent": "opencutai-inpaint-service"}
                )
                with urllib.request.urlopen(request, timeout=120) as resp, open(part_path, "wb") as f:
                    shutil.copyfileobj(resp, f, length=1024 * 1024)
                if os.path.getsize(part_path) < MODEL_MIN_BYTES:
                    raise ValueError(
                        f"Downloaded file too small ({os.path.getsize(part_path)} bytes); "
                        "expected a ~66 MB checkpoint."
                    )
                os.replace(part_path, MODEL_PATH)
                logger.info("STTN weights saved to %s", MODEL_PATH)
                return
            except Exception as e:  # noqa: BLE001 - retry any network hiccup
                last_error = e
                logger.warning("Weight download attempt %d failed: %s", attempt, e)
                if os.path.exists(part_path):
                    os.remove(part_path)
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Failed to download STTN weights: {last_error}")

    def get_inpainter(self, on_status=None):
        """Return the loaded STTN inpainter, downloading/loading on first use."""
        if self._inpainter is not None:
            return self._inpainter
        with self._load_lock:
            if self._inpainter is not None:
                return self._inpainter
            self.download_weights(on_status=on_status)
            if on_status:
                on_status("Loading STTN model...")
            from sttn.pipeline import STTNInpaint

            logger.info("Loading STTN model from %s...", MODEL_PATH)
            self._inpainter = STTNInpaint(MODEL_PATH, device="cpu")
            logger.info("STTN model loaded.")
            return self._inpainter


inpaint_service = InpaintService()

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def _mux_output(raw_path: str, source_path: str, final_path: str) -> None:
    """Encode the inpainted frames to H.264 and remux the original audio."""
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_path,
        "-i", source_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        final_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr[-500:]}")


def _process_job(
    job_id: str,
    input_path: str,
    region: tuple[float, float, float, float],
) -> None:
    """Worker thread: download weights, inpaint frames, mux audio."""
    from sttn.pipeline import InpaintCancelled, inpaint_video

    raw_path = os.path.join(RESULT_DIR, f"{job_id}_raw.mp4")
    final_path = os.path.join(RESULT_DIR, f"{job_id}.mp4")
    with _work_lock:
        try:
            if _cancel_requested(job_id):
                raise InpaintCancelled("Cancelled while queued.")
            _update_job(job_id, status="processing", progress=0.01, message="Preparing model...")
            inpainter = inpaint_service.get_inpainter(
                on_status=lambda msg: _update_job(job_id, message=msg)
            )
            if _cancel_requested(job_id):
                raise InpaintCancelled("Cancelled while preparing model.")
            _update_job(job_id, progress=0.05, message="Inpainting frames...")

            def on_progress(done: int, total: int) -> None:
                _update_job(
                    job_id,
                    progress=0.05 + 0.85 * (done / total),
                    message=f"Inpainting frames ({done}/{total})...",
                )

            start = time.time()
            frames = inpaint_video(
                inpainter, input_path, raw_path, region, on_progress,
                should_cancel=lambda: _cancel_requested(job_id),
            )

            if _cancel_requested(job_id):
                raise InpaintCancelled("Cancelled before encoding.")
            _update_job(job_id, progress=0.92, message="Encoding output video...")
            _mux_output(raw_path, input_path, final_path)

            elapsed = time.time() - start
            logger.info("Job %s done: %d frames in %.1fs", job_id, frames, elapsed)
            _update_job(
                job_id,
                status="done",
                progress=1.0,
                message=f"Done: {frames} frames in {elapsed:.0f}s",
                result_path=final_path,
            )
        except InpaintCancelled:
            logger.info("Inpaint job %s cancelled", job_id)
            _update_job(job_id, status="cancelled", message="Cancelled")
        except Exception as e:
            logger.exception("Inpaint job %s failed", job_id)
            _update_job(job_id, status="error", error=str(e)[:500], message="Failed")
        finally:
            # Drop the input and any partial output so cancelled/failed jobs
            # leave nothing behind; only a completed final_path survives.
            for path in (raw_path, input_path):
                if os.path.exists(path):
                    os.remove(path)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenCutAI Inpaint Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3100",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Return service health and model status."""
    return {
        "status": "ok",
        "service": "inpaint",
        "model": {
            "installed": inpaint_service.is_installed,
            "loaded": inpaint_service.is_loaded,
            "model_name": "sttn",
        },
    }


@app.post("/detect-region")
async def detect_region(file: UploadFile = File(...)):
    """Detect the burned-in subtitle region of the uploaded video.

    Synchronous (no job): samples ~24 frames and returns the strongest
    recurring text band as fractions (0-1) of the frame, ready to feed
    into /inpaint. Pure OpenCV heuristic — no OCR, no model weights.
    Response: {found, region: {x1, y1, x2, y2} | null, hit_ratio,
    frames_sampled}.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    detect_id = uuid.uuid4().hex[:12]
    input_path = os.path.join(UPLOAD_DIR, f"detect_{detect_id}{ext}")
    try:
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)
    except Exception:
        logger.exception("Failed to store upload for detection %s", detect_id)
        raise HTTPException(status_code=500, detail="Failed to store upload.")

    try:
        # CPU-bound OpenCV work; keep the event loop free.
        result = await run_in_threadpool(detect_subtitle_region, input_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Region detection %s failed", detect_id)
        raise HTTPException(status_code=500, detail="Region detection failed.")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    logger.info(
        "Region detection %s (%s): found=%s hit_ratio=%.3f",
        detect_id, file.filename, result["found"], result["hit_ratio"],
    )
    return result


@app.post("/inpaint")
async def inpaint(
    file: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
):
    """Start a subtitle-removal job for the uploaded video.

    Region coordinates are FRACTIONS (0-1) of the frame; (x1, y1) is the
    top-left and (x2, y2) the bottom-right of the box containing the
    burned-in subtitles. Returns a job_id to poll via /jobs/{job_id}.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="Region must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 (fractions of the frame).",
        )

    job_id = uuid.uuid4().hex[:12]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    try:
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)
    except Exception:
        logger.exception("Failed to store upload for job %s", job_id)
        raise HTTPException(status_code=500, detail="Failed to store upload.")

    with _jobs_lock:
        _jobs[job_id] = Job(job_id=job_id, message="Waiting for worker...")

    thread = threading.Thread(
        target=_process_job,
        args=(job_id, input_path, (x1, y1, x2, y2)),
        daemon=True,
    )
    thread.start()

    logger.info(
        "Inpaint job %s queued: file=%s region=(%.3f,%.3f)-(%.3f,%.3f)",
        job_id, file.filename, x1, y1, x2, y2,
    )
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Poll job status: queued | processing | done | error | cancelled, progress 0-1."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return _job_snapshot(job)


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Request cancellation of a queued or running inpaint job.

    Sets a flag that the worker polls between STTN sliding windows, so a
    processing job aborts within a few seconds, frees the worker lock and
    removes its temp files. A still-queued job is cancelled immediately.
    Cancelling a done/errored/cancelled job is a no-op that returns the
    current status.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.status in ("queued", "processing"):
            job.cancel_requested = True
            if job.status == "queued":
                # The worker has not picked it up yet; it will observe the
                # flag when it does and clean up the stored upload.
                job.status = "cancelled"
                job.message = "Cancelled"
            else:
                job.message = "Cancelling..."
            logger.info("Cancel requested for inpaint job %s", job_id)
        return _job_snapshot(job)


@app.get("/result/{job_id}")
async def job_result(job_id: str):
    """Download the processed video for a completed job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "done" or not job.result_path:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not finished (status: {job.status}).",
        )
    if not os.path.exists(job.result_path):
        raise HTTPException(status_code=410, detail="Result file no longer exists.")
    return FileResponse(
        job.result_path,
        media_type="video/mp4",
        filename=f"{job_id}-clean.mp4",
    )
