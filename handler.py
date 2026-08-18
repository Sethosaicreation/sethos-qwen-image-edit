"""RunPod Serverless entry point for Sethos Qwen Image Edit."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from inference import ENGINE, InferenceError
from media import MediaError, download_image, encode_webp
from schema import CONTRACT_VERSION, InputError, parse_request


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("sethos.qwen-image-edit")


def handler(event: Any) -> dict[str, Any]:
    started = time.monotonic()
    job_id = str(event.get("id", "unknown"))[:190] if isinstance(event, dict) else "unknown"
    try:
        request = parse_request(event)
        prompt_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        LOGGER.info("job=%s contract=%s prompt_sha256=%s style=%s quality=%s", job_id, request.contract_version, prompt_hash, bool(request.style_image_url), request.quality)
        with tempfile.TemporaryDirectory(prefix="sethos-qwen-edit-") as temporary:
            root = Path(temporary)
            source_path = download_image(request.source_image_url, root / "source-image")
            style_path = download_image(request.style_image_url, root / "style-image") if request.style_image_url else None
            image, metadata = ENGINE.generate(request, source_path, style_path)
            encoded, output_bytes = encode_webp(image)
        result = {
            "contract_version": CONTRACT_VERSION,
            "prompt_sha256": prompt_hash,
            "applied_parameters": request.public_parameters(metadata["seed"], metadata["width"], metadata["height"]),
            "image_base64": encoded,
            "mime_type": "image/webp",
            "bytes": output_bytes,
            "worker_elapsed_ms": round((time.monotonic() - started) * 1000),
        }
        LOGGER.info("job=%s completed bytes=%d elapsed_ms=%d", job_id, output_bytes, result["worker_elapsed_ms"])
        return result
    except (InputError, MediaError, InferenceError) as error:
        code = error.code if isinstance(error, InputError) else error.__class__.__name__.upper()
        LOGGER.error("job=%s failed code=%s message=%s", job_id, code, error)
        raise RuntimeError(json.dumps({"code": code, "message": str(error)}, ensure_ascii=False)) from error
    except Exception as error:  # pragma: no cover
        LOGGER.exception("job=%s unexpected failure", job_id)
        raise RuntimeError(json.dumps({"code": "UNEXPECTED_WORKER_ERROR", "message": "Erreur interne inattendue du worker photo."}, ensure_ascii=False)) from error


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
