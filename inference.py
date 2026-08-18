"""Qwen Image Edit inference with RunPod cached-model discovery."""

from __future__ import annotations

import inspect
import math
import os
import secrets
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from schema import PhotoEditRequest


MODEL_ID = os.getenv("QWEN_IMAGE_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
MODEL_REVISION = os.getenv("QWEN_IMAGE_MODEL_REVISION", "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9")
MODEL_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")


class InferenceError(RuntimeError):
    pass


def resolve_model_dir() -> Path:
    configured = os.getenv("QWEN_IMAGE_MODEL_DIR", "").strip()
    if configured:
        path = Path(configured)
        if path.is_dir():
            return path
    root = MODEL_CACHE_ROOT / "models--Qwen--Qwen-Image-Edit-2511"
    pinned = root / "snapshots" / MODEL_REVISION
    if pinned.is_dir():
        return pinned
    reference = root / "refs" / "main"
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        candidate = root / "snapshots" / revision
        if revision and candidate.is_dir():
            return candidate
    snapshots = sorted((root / "snapshots").glob("*"), key=lambda path: path.stat().st_mtime, reverse=True) if (root / "snapshots").is_dir() else []
    if snapshots:
        return snapshots[0]
    raise InferenceError("Poids Qwen-Image-Edit-2511 introuvables. Ajoutez Qwen/Qwen-Image-Edit-2511 dans le champ Model de l’endpoint RunPod.")


def directed_prompt(request: PhotoEditRequest) -> str:
    mode = {
        "outfit": "Edit only the clothing or outfit requested by the user.",
        "background": "Edit only the background or environment requested by the user.",
        "hair": "Edit only the hairstyle or makeup requested by the user.",
        "relight": "Edit only the lighting, color grade, and atmosphere requested by the user.",
        "free": "Apply the requested image edit precisely.",
    }[request.edit_mode]
    fidelity = {
        "identity": "Preserve exactly the same adult person's facial identity, body proportions, pose, camera angle, expression, and every area not explicitly requested to change.",
        "balanced": "Preserve the person's identity, pose, and main composition while applying the requested edit naturally.",
        "creative": "Preserve recognizable subject identity while allowing composition changes required by the edit.",
    }[request.fidelity]
    references = "Use Figure 1 as the source photo."
    if request.style_image_url:
        references += " Use Figure 2 only as the requested clothing, material, style, or environment reference. Never copy Figure 2's identity."
    ratio = "" if request.aspect_ratio == "source" else f" Compose the result in {request.aspect_ratio} aspect ratio."
    return f"{references} {mode} User instruction: {request.prompt} {fidelity}{ratio}".strip()


def output_dimensions(source: Image.Image, request: PhotoEditRequest) -> tuple[int, int]:
    if request.aspect_ratio == "source":
        ratio = source.width / source.height
    else:
        left, right = request.aspect_ratio.split(":", 1)
        ratio = int(left) / int(right)
    pixels = {"preview": 786_432, "standard": 1_048_576, "quality": 1_474_560}[request.quality]
    height = math.sqrt(pixels / ratio)
    width = height * ratio
    width = max(256, int(round(width / 32)) * 32)
    height = max(256, int(round(height / 32)) * 32)
    while width * height > pixels:
        if width >= height:
            width -= 32
        else:
            height -= 32
    return width, height


class QwenEngine:
    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import torch
                from diffusers import QwenImageEditPlusPipeline

                if not torch.cuda.is_available():
                    raise InferenceError("GPU CUDA indisponible sur le worker RunPod.")
                pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    str(resolve_model_dir()),
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                pipeline.to("cuda")
                pipeline.set_progress_bar_config(disable=True)
                self._pipeline = pipeline
                return pipeline
            except InferenceError:
                raise
            except Exception as error:
                raise InferenceError(f"Chargement de Qwen Image Edit impossible : {error}") from error

    def generate(self, request: PhotoEditRequest, source_path: Path, style_path: Path | None) -> tuple[Image.Image, dict[str, int]]:
        try:
            import torch

            pipeline = self._load()
            source = Image.open(source_path).convert("RGB")
            images = [source]
            if style_path is not None:
                images.append(Image.open(style_path).convert("RGB"))
            seed = request.seed if request.seed >= 0 else secrets.randbelow(2_147_483_648)
            width, height = output_dimensions(source, request)
            arguments: dict[str, Any] = {
                "image": images,
                "prompt": directed_prompt(request),
                "negative_prompt": request.negative_prompt or " ",
                "generator": torch.Generator(device="cpu").manual_seed(seed),
                "true_cfg_scale": 4.0,
                "guidance_scale": 1.0,
                "num_inference_steps": request.steps,
                "num_images_per_prompt": 1,
            }
            signature = inspect.signature(pipeline.__call__).parameters
            if "width" in signature:
                arguments["width"] = width
            if "height" in signature:
                arguments["height"] = height
            with self._inference_lock, torch.inference_mode():
                result = pipeline(**arguments)
            output = result.images[0]
            return output, {"seed": seed, "width": output.width, "height": output.height}
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError(f"Modification Qwen interrompue : {error}") from error


ENGINE = QwenEngine()
