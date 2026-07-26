"""REST routes for burned-in subtitle removal (STTN video inpainting).

Proxies requests to the inpaint-service microservice. Jobs are long-running
background tasks on the service side; the frontend polls /jobs/{job_id}
and downloads the processed video from /result/{job_id}.
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


def _service_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Inpaint service is not available. Ensure inpaint-service is running on "
        f"{settings.INPAINT_SERVICE_URL}",
    )


@router.post("/remove-subtitles")
async def remove_subtitles(
    file: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
):
    """Start a burned-in subtitle removal job.

    Region coordinates are fractions (0-1) of the frame describing the box
    that contains the subtitles. Returns {job_id} for polling.
    """
    try:
        async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            data = {"x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2)}
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
