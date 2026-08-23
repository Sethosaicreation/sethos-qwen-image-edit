"""Authenticated HTTPS worker for Qwen Image Edit 2511 on Vast.ai."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from inference import ENGINE, InferenceError
from media import MediaError, download_image, encode_webp
from schema import PhotoEditRequest, _signed_image_url


TOKEN = os.getenv("VAST_WORKER_TOKEN", "").strip()
SPOOL = Path(os.getenv("AI_WORKER_SPOOL", "/workspace/spool")).resolve()
SPOOL.mkdir(parents=True, exist_ok=True)
CONCURRENCY = max(1, min(2, int(os.getenv("AI_WORKER_CONCURRENCY", "1"))))
RESULT_TTL = max(300, int(os.getenv("AI_WORKER_RESULT_TTL_SECONDS", "3600")))
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=CONCURRENCY, thread_name_prefix="qwen-vast")

app = FastAPI(title="Sethos Qwen Vast Worker", version="1.0.0", docs_url=None, redoc_url=None)


class VastJobRequest(BaseModel):
    job_id: str = Field(pattern=r"^[a-f0-9-]{36}$")
    provider: str = Field(pattern=r"^qwen-2511-edit$")
    model: str = Field(min_length=3, max_length=160)
    task: str = Field(pattern=r"^(image_edit|benchmark)$")
    input: dict[str, Any]
    character_id: str | None = Field(default=None, max_length=80)
    premium: bool = False
    shadow_mode: bool = False


def authorize(authorization: str | None = Header(default=None)) -> None:
    if len(TOKEN) < 32:
        raise HTTPException(status_code=503, detail="worker token is not configured")
    expected = f"Bearer {TOKEN}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="invalid authorization")


def public_base_url() -> str:
    configured = os.getenv("AI_WORKER_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    mapped_port = os.getenv("VAST_TCP_PORT_8000", "").strip()
    if not mapped_port.isdigit():
        raise RuntimeError("VAST_TCP_PORT_8000 is unavailable")
    return f"https://sethos-vast-worker:{mapped_port}"


def bounded_text(value: Any, minimum: int, maximum: int, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    text = value.strip()
    if not minimum <= len(text) <= maximum:
        raise ValueError(f"{field} has an invalid length")
    return text


def first_reference(data: dict[str, Any]) -> str:
    direct = data.get("reference_image_url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    references = data.get("reference_images")
    if isinstance(references, list):
        for value in references:
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("reference image is required")


def second_reference(data: dict[str, Any], source: str) -> str:
    direct = data.get("style_image_url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    references = data.get("reference_images")
    if isinstance(references, list):
        values = [value.strip() for value in references if isinstance(value, str) and value.strip()]
        if len(values) > 1:
            return values[1]
    return ""


def photo_request(request: VastJobRequest) -> PhotoEditRequest:
    data = request.input
    source = first_reference(data)
    style = second_reference(data, source)
    prompt = bounded_text(data.get("prompt"), 3, 2500, "prompt")
    negative = data.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative.strip()) > 1000:
        raise ValueError("negative_prompt has an invalid length")
    mode = data.get("edit_mode", "free")
    fidelity = data.get("fidelity", "identity")
    ratio = data.get("aspect_ratio", "source")
    quality = data.get("quality", "standard")
    if mode not in {"outfit", "background", "hair", "relight", "free"}:
        mode = "free"
    if fidelity not in {"identity", "balanced", "creative"}:
        fidelity = "identity"
    if ratio not in {"source", "1:1", "4:5", "3:4", "9:16", "16:9"}:
        ratio = "source"
    if quality not in {"preview", "standard", "quality"}:
        quality = "standard"
    seed = data.get("seed", -1)
    if isinstance(seed, bool) or not isinstance(seed, int) or not -1 <= seed <= 2_147_483_647:
        seed = -1
    return PhotoEditRequest(
        contract_version="sethos.qwen.image-edit-2511.v1",
        source_image_url=_signed_image_url(source, "source_image", True),
        style_image_url=_signed_image_url(style, "style_image", False),
        prompt=prompt,
        negative_prompt=negative.strip(),
        edit_mode=mode,
        fidelity=fidelity,
        aspect_ratio=ratio,
        quality=quality,
        seed=seed,
    )


def prune() -> None:
    cutoff = time.time() - RESULT_TTL
    with LOCK:
        expired = [key for key, value in JOBS.items() if float(value.get("completed_at", time.time())) < cutoff]
        for key in expired:
            result_path = JOBS[key].get("result_path")
            if result_path:
                Path(str(result_path)).unlink(missing_ok=True)
            JOBS.pop(key, None)


def run_job(worker_job_id: str, request: VastJobRequest) -> None:
    started = time.monotonic()
    try:
        parsed = photo_request(request)
        with LOCK:
            JOBS[worker_job_id]["status"] = "generating"
        with tempfile.TemporaryDirectory(prefix="sethos-qwen-vast-") as temporary:
            root = Path(temporary)
            source = download_image(parsed.source_image_url, root / "source-image")
            style = download_image(parsed.style_image_url, root / "style-image") if parsed.style_image_url else None
            image, metadata = ENGINE.generate(parsed, source, style)
        encoded, output_bytes = encode_webp(image)
        output = SPOOL / f"{request.job_id}.webp"
        output.write_bytes(base64.b64decode(encoded, validate=True))
        sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        signature = hmac.new(TOKEN.encode(), f"{worker_job_id}:{sha256}".encode(), hashlib.sha256).hexdigest()
        elapsed = round(time.monotonic() - started, 3)
        with LOCK:
            JOBS[worker_job_id].update(
                status="completed",
                result_path=str(output),
                result_url=f"{public_base_url()}/v1/results/{worker_job_id}?signature={signature}",
                extension="webp",
                sha256=sha256,
                metrics={
                    "generation_seconds": elapsed,
                    "gpu_seconds": elapsed,
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "seed": metadata["seed"],
                    "bytes": output_bytes,
                },
                completed_at=time.time(),
            )
    except (InferenceError, MediaError, OSError, ValueError, RuntimeError) as error:
        with LOCK:
            JOBS[worker_job_id].update(status="failed", error=str(error)[:1500], completed_at=time.time())


@app.get("/health")
def health(_: None = Depends(authorize)) -> dict[str, Any]:
    with LOCK:
        active = sum(1 for value in JOBS.values() if value.get("status") in {"queued", "generating"})
    return {
        "status": "ready",
        "profile": "DAILY_WORKER",
        "configured_providers": ["qwen-2511-edit"],
        "active_jobs": active,
        "concurrency": CONCURRENCY,
    }


@app.post("/v1/jobs", status_code=202)
def create_job(request: VastJobRequest, _: None = Depends(authorize)) -> dict[str, str]:
    prune()
    worker_job_id = uuid.uuid4().hex
    with LOCK:
        JOBS[worker_job_id] = {"status": "queued", "request_job_id": request.job_id, "created_at": time.time()}
    EXECUTOR.submit(run_job, worker_job_id, request)
    return {"job_id": worker_job_id, "status": "queued"}


@app.get("/v1/jobs/{worker_job_id}")
def job_status(worker_job_id: str, _: None = Depends(authorize)) -> dict[str, Any]:
    if len(worker_job_id) != 32 or any(character not in "0123456789abcdef" for character in worker_job_id):
        raise HTTPException(status_code=404, detail="job not found")
    with LOCK:
        result = dict(JOBS.get(worker_job_id) or {})
    if not result:
        raise HTTPException(status_code=404, detail="job not found")
    result.pop("result_path", None)
    return result


@app.get("/v1/results/{worker_job_id}")
def result(worker_job_id: str, signature: str) -> FileResponse:
    with LOCK:
        job = dict(JOBS.get(worker_job_id) or {})
    output = Path(str(job.get("result_path", ""))).resolve()
    sha256 = str(job.get("sha256", ""))
    expected = hmac.new(TOKEN.encode(), f"{worker_job_id}:{sha256}".encode(), hashlib.sha256).hexdigest()
    if not sha256 or not hmac.compare_digest(signature, expected) or not output.is_file() or SPOOL not in output.parents:
        raise HTTPException(status_code=404, detail="result not found")
    return FileResponse(output, filename=output.name, media_type="image/webp")


def tls_file(environment_name: str, destination: Path, expected_header: bytes) -> Path:
    encoded = os.getenv(environment_name, "").strip()
    if not encoded:
        raise RuntimeError(f"{environment_name} is required")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise RuntimeError(f"{environment_name} is invalid") from error
    if expected_header not in payload or len(payload) > 16_384:
        raise RuntimeError(f"{environment_name} is invalid")
    destination.write_bytes(payload)
    destination.chmod(0o600)
    return destination


if __name__ == "__main__":
    if len(TOKEN) < 32:
        raise SystemExit("VAST_WORKER_TOKEN must contain at least 32 characters")
    certificate = tls_file("VAST_WORKER_TLS_CERT_B64", Path("/tmp/sethos-worker.crt"), b"BEGIN CERTIFICATE")
    private_key = tls_file("VAST_WORKER_TLS_KEY_B64", Path("/tmp/sethos-worker.key"), b"BEGIN PRIVATE KEY")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("AI_WORKER_PORT", "8000")),
        access_log=False,
        ssl_certfile=str(certificate),
        ssl_keyfile=str(private_key),
    )
