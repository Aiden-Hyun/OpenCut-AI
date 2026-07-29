"""REST routes for burned-in subtitle removal (STTN video inpainting).

Proxies requests to the inpaint-service microservice. Jobs are long-running
background tasks on the service side; the frontend polls /jobs/{job_id},
can abort via /jobs/{job_id}/cancel, and downloads the processed video from
/result/{job_id}. /detect-region synchronously suggests one subtitle box for
the whole video; /detect-timeline suggests a per-segment schedule for
compilations whose captions move between clips.
"""

import logging

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inpaint", tags=["inpaint"])

# Uploads can be large and CPU inpainting is slow; be generous.
UPLOAD_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
RESULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
# Region detection is synchronous on the service side (~seconds).
DETECT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
# Timeline detection walks the whole file window by window (up to ~400
# decoded frames), so it needs more headroom than single-region detection.
DETECT_TIMELINE_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


def _service_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Inpaint service is not available. Ensure inpaint-service is running on "
        f"{settings.INPAINT_SERVICE_URL}",
    )


@router.post("/detect-region")
async def detect_region(file: UploadFile = File(...)):
    """Detect the burned-in subtitle region in a video.

    Synchronous passthrough to the inpaint service's OpenCV heuristic.
    Returns {found, region: {x1, y1, x2, y2} | null, hit_ratio,
    frames_sampled} with coordinates as fractions (0-1) of the frame.
    """
    try:
        async with httpx.AsyncClient(timeout=DETECT_TIMEOUT) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            resp = await client.post(
                f"{settings.INPAINT_SERVICE_URL}/detect-region", files=files
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.ConnectError:
        raise _service_unavailable()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Region detection proxy failed")
        raise HTTPException(status_code=500, detail="Region detection failed.")


@router.post("/detect-timeline")
async def detect_timeline(
    file: UploadFile = File(...),
    window_seconds: float | None = Form(None),
):
    """Detect burned-in subtitle regions over time in a video.

    Synchronous passthrough to the inpaint service's windowed heuristic.
    Returns {segments: [{start, end, regions: [{x1, y1, x2, y2}],
    hit_ratio}], duration, frames_sampled, windows} — seconds for times,
    fractions (0-1) of the frame for coordinates. Feed the segments back as
    the `schedule` field of /remove-subtitles.
    """
    try:
        async with httpx.AsyncClient(timeout=DETECT_TIMELINE_TIMEOUT) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            data = (
                {"window_seconds": str(window_seconds)}
                if window_seconds is not None
                else None
            )
            resp = await client.post(
                f"{settings.INPAINT_SERVICE_URL}/detect-timeline",
                files=files,
                data=data,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.ConnectError:
        raise _service_unavailable()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Timeline detection proxy failed")
        raise HTTPException(status_code=500, detail="Timeline detection failed.")


@router.post("/remove-subtitles")
async def remove_subtitles(
    file: UploadFile = File(...),
    x1: float | None = Form(None),
    y1: float | None = Form(None),
    x2: float | None = Form(None),
    y2: float | None = Form(None),
    schedule: str | None = Form(None),
):
    """Start a burned-in subtitle removal job.

    Either `schedule` — JSON [{start, end, regions: [{x1, y1, x2, y2}]}] from
    /detect-timeline, which takes precedence — or the four region fields for
    a single box covering the whole video. Coordinates are fractions (0-1) of
    the frame. Returns {job_id} for polling.
    """
    try:
        async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            data: dict[str, str] = {}
            if schedule:
                data["schedule"] = schedule
            for key, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
                if value is not None:
                    data[key] = str(value)
            resp = await client.post(
                f"{settings.INPAINT_SERVICE_URL}/inpaint", files=files, data=data
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.ConnectError:
        raise _service_unavailable()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inpaint proxy failed")
        raise HTTPException(status_code=500, detail="Subtitle removal failed to start.")


@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Poll the status of an inpaint job."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{settings.INPAINT_SERVICE_URL}/jobs/{job_id}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.ConnectError:
        raise _service_unavailable()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inpaint status proxy failed")
        raise HTTPException(status_code=500, detail="Failed to fetch job status.")


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Request cancellation of an inpaint job (no-op if already finished)."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.INPAINT_SERVICE_URL}/jobs/{job_id}/cancel"
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text if e.response else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.ConnectError:
        raise _service_unavailable()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inpaint cancel proxy failed")
        raise HTTPException(status_code=500, detail="Failed to cancel job.")


@router.get("/result/{job_id}")
async def job_result(job_id: str):
    """Stream the processed video for a completed inpaint job."""
    client = httpx.AsyncClient(timeout=RESULT_TIMEOUT)
    try:
        req = client.build_request(
            "GET", f"{settings.INPAINT_SERVICE_URL}/result/{job_id}"
        )
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            detail = (await resp.aread()).decode(errors="replace")
            await resp.aclose()
            await client.aclose()
            raise HTTPException(status_code=resp.status_code, detail=detail)

        async def stream():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        headers = {
            "Content-Disposition": resp.headers.get(
                "content-disposition", f'attachment; filename="{job_id}-clean.mp4"'
            ),
        }
        if resp.headers.get("content-length"):
            headers["Content-Length"] = resp.headers["content-length"]
        return StreamingResponse(stream(), media_type="video/mp4", headers=headers)
    except httpx.ConnectError:
        await client.aclose()
        raise _service_unavailable()
    except HTTPException:
        raise
    except Exception:
        await client.aclose()
        logger.exception("Inpaint result proxy failed")
        raise HTTPException(status_code=500, detail="Failed to fetch job result.")
