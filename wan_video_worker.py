"""Authenticated single-GPU Wan2.2 TI2V-5B worker for Vast.ai."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import random
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
import uvicorn

from media import MediaError, download_image
from schema import _signed_image_url


TOKEN = os.getenv("VAST_WORKER_TOKEN", "").strip()
SPOOL = Path(os.getenv("AI_WORKER_SPOOL", "/workspace/spool-video")).resolve()
MODEL_CACHE = Path(os.getenv("WAN_MODEL_CACHE", "/workspace/huggingface-cache")).resolve()
WAN_ROOT = Path(os.getenv("WAN_CODE_ROOT", "/opt/Wan2.2")).resolve()
MODEL_ID = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B").strip()
MODEL_REVISION = os.getenv(
    "WAN_MODEL_REVISION", "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
).strip()
GENERATION_TIMEOUT = max(900, min(10_800, int(os.getenv("WAN_GENERATION_TIMEOUT_SECONDS", "7200"))))
RESULT_TTL = max(300, int(os.getenv("AI_WORKER_RESULT_TTL_SECONDS", "7200")))
SPOOL.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wan22-ti2v")

app = FastAPI(title="Sethos Wan2.2 TI2V Vast Worker", version="1.0.0", docs_url=None, redoc_url=None)


class VastVideoJobRequest(BaseModel):
    job_id: str = Field(pattern=r"^[a-f0-9-]{36}$")
    provider: str = Field(pattern=r"^wan-ti2v-5b$")
    model: str = Field(min_length=3, max_length=160)
    task: str = Field(pattern=r"^(video|benchmark)$")
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


def video_parameters(request: VastVideoJobRequest) -> dict[str, Any]:
    data = request.input
    source = _signed_image_url(data.get("reference_image_url"), "source_image", True)
    prompt = bounded_text(data.get("prompt"), 3, 2500, "prompt")
    negative = data.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative.strip()) > 1000:
        raise ValueError("negative_prompt has an invalid length")
    duration = data.get("duration", 15)
    if isinstance(duration, bool) or not isinstance(duration, int) or duration not in {5, 10, 15}:
        raise ValueError("duration must be 5, 10 or 15 seconds")
    if data.get("aspect_ratio", "9:16") != "9:16":
        raise ValueError("only the 9:16 aspect ratio is supported")
    if data.get("resolution", "720p") != "720p":
        raise ValueError("only 720p is supported")
    seed = data.get("seed", -1)
    if isinstance(seed, bool) or not isinstance(seed, int) or not -1 <= seed <= 2_147_483_647:
        raise ValueError("seed is invalid")
    if seed < 0:
        seed = random.SystemRandom().randint(0, 2_147_483_647)
    return {
        "reference_image_url": source,
        "prompt": prompt,
        "negative_prompt": negative.strip(),
        "duration": duration,
        "frames": duration * 24 + 1,
        "seed": seed,
        "size": "704*1280",
    }


def model_path() -> Path:
    with MODEL_LOCK:
        resolved = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(MODEL_CACHE),
        )
    path = Path(resolved).resolve()
    if not path.is_dir():
        raise RuntimeError("Wan2.2 model snapshot is unavailable")
    return path


def effective_prompt(prompt: str, negative_prompt: str) -> str:
    identity_guard = (
        "The supplied image is immutable frame zero. Preserve the exact adult person, face, "
        "hair, body proportions, clothing, lighting, framing and background. Use restrained "
        "natural movement, stable anatomy, realistic physics, no speech and no lip-sync."
    )
    result = f"{prompt.strip()} {identity_guard}"
    if negative_prompt:
        result += f" Avoid these defects: {negative_prompt.strip()}"
    return result[:3900]


def generate_video(parameters: dict[str, Any], reference: Path, output: Path) -> dict[str, Any]:
    if not (WAN_ROOT / "generate.py").is_file():
        raise RuntimeError("Wan2.2 runtime is missing")
    checkpoint = model_path()
    command = [
        "python", "-u", str(WAN_ROOT / "generate.py"),
        "--task", "ti2v-5B",
        "--size", str(parameters["size"]),
        "--ckpt_dir", str(checkpoint),
        "--image", str(reference),
        "--prompt", effective_prompt(parameters["prompt"], parameters["negative_prompt"]),
        "--frame_num", str(parameters["frames"]),
        "--base_seed", str(parameters["seed"]),
        "--save_file", str(output),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    completed = subprocess.run(
        command,
        cwd=str(WAN_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=GENERATION_TIMEOUT,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "Wan2.2 generation failed")[-4000:]
        raise RuntimeError(detail.strip())
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError("Wan2.2 did not produce a readable MP4")
    return probe_video(output)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets:format=duration",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        raise RuntimeError("ffprobe rejected the generated MP4")
    data = json.loads(completed.stdout)
    stream = (data.get("streams") or [{}])[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = float((data.get("format") or {}).get("duration") or 0)
    frames = int(stream.get("nb_read_packets") or 0)
    rate = str(stream.get("avg_frame_rate") or "0/1").split("/", 1)
    fps = float(rate[0]) / max(1.0, float(rate[1])) if len(rate) == 2 else 0.0
    if width < 640 or height < 1080 or height <= width or duration < 4.5 or fps < 20:
        raise RuntimeError("generated MP4 does not satisfy the vertical 720p contract")
    return {"width": width, "height": height, "duration": duration, "frames": frames, "fps": fps}


def prune() -> None:
    cutoff = time.time() - RESULT_TTL
    with LOCK:
        expired = [key for key, value in JOBS.items() if float(value.get("completed_at", time.time())) < cutoff]
        for key in expired:
            result_path = JOBS[key].get("result_path")
            if result_path:
                Path(str(result_path)).unlink(missing_ok=True)
            JOBS.pop(key, None)


def run_job(worker_job_id: str, request: VastVideoJobRequest) -> None:
    started = time.monotonic()
    try:
        parameters = video_parameters(request)
        with LOCK:
            JOBS[worker_job_id]["status"] = "loading"
        with tempfile.TemporaryDirectory(prefix="sethos-wan22-") as temporary:
            reference = download_image(parameters["reference_image_url"], Path(temporary) / "reference-image")
            output = SPOOL / f"{request.job_id}.mp4"
            output.unlink(missing_ok=True)
            with LOCK:
                JOBS[worker_job_id]["status"] = "generating"
            metrics = generate_video(parameters, reference, output)
        sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        signature = hmac.new(TOKEN.encode(), f"{worker_job_id}:{sha256}".encode(), hashlib.sha256).hexdigest()
        elapsed = round(time.monotonic() - started, 3)
        metrics.update(generation_seconds=elapsed, gpu_seconds=elapsed, seed=parameters["seed"], bytes=output.stat().st_size)
        with LOCK:
            JOBS[worker_job_id].update(
                status="completed",
                result_path=str(output),
                result_url=f"{public_base_url()}/v1/results/{worker_job_id}?signature={signature}",
                extension="mp4",
                sha256=sha256,
                metrics=metrics,
                completed_at=time.time(),
            )
    except (MediaError, OSError, ValueError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        with LOCK:
            JOBS[worker_job_id].update(status="failed", error=str(error)[:4000], completed_at=time.time())


@app.get("/health")
def health(_: None = Depends(authorize)) -> dict[str, Any]:
    with LOCK:
        active = sum(1 for value in JOBS.values() if value.get("status") in {"queued", "loading", "generating"})
    return {
        "status": "ready",
        "profile": "VIDEO_WORKER",
        "configured_providers": ["wan-ti2v-5b"],
        "model": "Wan2.2-TI2V-5B",
        "model_revision": MODEL_REVISION,
        "active_jobs": active,
        "concurrency": 1,
    }


@app.post("/v1/jobs", status_code=202)
def create_job(request: VastVideoJobRequest, _: None = Depends(authorize)) -> dict[str, str]:
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
    return FileResponse(output, filename=output.name, media_type="video/mp4")


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
