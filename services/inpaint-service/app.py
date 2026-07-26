"""STTN video inpainting microservice.

Standalone FastAPI service that removes burned-in (hardcoded) subtitles
from video by inpainting a caller-supplied rectangular region with STTN
(Spatial-Temporal Transformer Networks). Jobs run in a background thread
and are polled by id. Runs on port 8427.

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
    status: str = "queued"  # queued | processing | done | error
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result_path: str | None = None
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
    raw_path = os.path.join(RESULT_DIR, f"{job_id}_raw.mp4")
    final_path = os.path.join(RESULT_DIR, f"{job_id}.mp4")
    with _work_lock:
        try:
            _update_job(job_id, status="processing", progress=0.01, message="Preparing model...")
            inpainter = inpaint_service.get_inpainter(
                on_status=lambda msg: _update_job(job_id, message=msg)
            )
            _update_job(job_id, progress=0.05, message="Inpainting frames...")

            from sttn.pipeline import inpaint_video

            def on_progress(done: int, total: int) -> None:
                _update_job(
                    job_id,
                    progress=0.05 + 0.85 * (done / total),
                    message=f"Inpainting frames ({done}/{total})...",
                )

            start = time.time()
            frames = inpaint_video(inpainter, input_path, raw_path, region, on_progress)

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
        except Exception as e:
            logger.exception("Inpaint job %s failed", job_id)
            _update_job(job_id, status="error", error=str(e)[:500], message="Failed")
        finally:
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
    """Poll job status: queued | processing | done | error, progress 0-1."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": round(job.progress, 4),
        "message": job.message,
        "error": job.error,
    }


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
