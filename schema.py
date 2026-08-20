"""Strict input contract for the Sethos Qwen Image Edit RunPod worker."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse


CONTRACT_VERSION = "sethos.qwen.image-edit-2511.v1"
ALLOWED_MODES = {"outfit", "background", "hair", "relight", "free"}
ALLOWED_FIDELITY = {"identity", "balanced", "creative"}
ALLOWED_RATIOS = {"source", "1:1", "4:5", "3:4", "9:16", "16:9"}
QUALITY_STEPS = {"preview": 20, "standard": 30, "quality": 40}
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InputError(ValueError):
    def __init__(self, message: str, code: str = "INVALID_INPUT") -> None:
        super().__init__(message)
        self.code = code


def _text(value: Any, field: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str):
        raise InputError(f"Le champ {field} doit être du texte.")
    cleaned = CONTROL_CHARACTERS.sub("", value).strip()
    if not minimum <= len(cleaned) <= maximum:
        raise InputError(f"Le champ {field} doit contenir entre {minimum} et {maximum} caractères.")
    return cleaned


def _choice(value: Any, field: str, allowed: set[str], default: str) -> str:
    candidate = value if isinstance(value, str) else default
    if candidate not in allowed:
        raise InputError(f"Valeur invalide pour {field}.")
    return candidate


def _signed_image_url(value: Any, field: str, required: bool) -> str:
    if value in (None, "") and not required:
        return ""
    if not isinstance(value, str) or len(value) > 1600:
        raise InputError(f"URL invalide pour {field}.")
    parsed = urlparse(value)
    allowed_host = os.getenv("SETHOS_INPUT_HOST", "sethosaicreation.fr").lower()
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != allowed_host:
        raise InputError(f"Hôte non autorisé pour {field}.")
    if parsed.username or parsed.password or parsed.fragment:
        raise InputError(f"Chemin non autorisé pour {field}.")
    query = parse_qs(parsed.query, strict_parsing=True)
    editor_url = parsed.path == "/admin/api/photo-editor-runpod.php" \
        and set(query) == {"action", "id", "slot", "token"} \
        and query.get("action") == ["input"] and query.get("slot") in (["source"], ["style"]) \
        and re.fullmatch(r"pe_[a-f0-9]{24}", query.get("id", [""])[0]) is not None \
        and len(query.get("token", [])) == 1 and re.fullmatch(r"[a-f0-9]{64}", query["token"][0]) is not None
    influencer_url = field == "source_image" and parsed.path == "/admin/api/influencer-studio.php" \
        and set(query) == {"action", "id", "token"} and query.get("action") == ["input"] \
        and re.fullmatch(r"inf_[a-f0-9]{24}", query.get("id", [""])[0]) is not None \
        and len(query.get("token", [])) == 1 and re.fullmatch(r"[a-f0-9]{64}", query["token"][0]) is not None
    if not editor_url and not influencer_url:
        raise InputError(f"Signature invalide pour {field}.")
    return value


@dataclass(frozen=True)
class PhotoEditRequest:
    contract_version: str
    source_image_url: str
    style_image_url: str
    prompt: str
    negative_prompt: str
    edit_mode: str
    fidelity: str
    aspect_ratio: str
    quality: str
    seed: int

    @property
    def steps(self) -> int:
        return QUALITY_STEPS[self.quality]

    def public_parameters(self, effective_seed: int, width: int, height: int) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "edit_mode": self.edit_mode,
            "fidelity": self.fidelity,
            "aspect_ratio": self.aspect_ratio,
            "quality": self.quality,
            "steps": self.steps,
            "seed": effective_seed,
            "width": width,
            "height": height,
            "style_reference": bool(self.style_image_url),
        }


def parse_request(event: Any) -> PhotoEditRequest:
    if not isinstance(event, dict) or not isinstance(event.get("input"), dict):
        raise InputError("La requête RunPod doit contenir un objet input.")
    data = event["input"]
    contract = data.get("contract_version")
    if contract != CONTRACT_VERSION:
        raise InputError("Version de contrat worker incompatible.", "CONTRACT_MISMATCH")
    prompt = _text(data.get("prompt"), "prompt", 3, 2500)
    negative = data.get("negative_prompt", "")
    if not isinstance(negative, str):
        raise InputError("Le prompt négatif doit être du texte.")
    negative = CONTROL_CHARACTERS.sub("", negative).strip()
    if len(negative) > 1000:
        raise InputError("Le prompt négatif dépasse 1 000 caractères.")
    seed = data.get("seed", -1)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < -1 or seed > 2_147_483_647:
        raise InputError("Seed invalide.")
    return PhotoEditRequest(
        contract_version=contract,
        source_image_url=_signed_image_url(data.get("source_image"), "source_image", True),
        style_image_url=_signed_image_url(data.get("style_image"), "style_image", False),
        prompt=prompt,
        negative_prompt=negative,
        edit_mode=_choice(data.get("edit_mode"), "edit_mode", ALLOWED_MODES, "free"),
        fidelity=_choice(data.get("fidelity"), "fidelity", ALLOWED_FIDELITY, "identity"),
        aspect_ratio=_choice(data.get("aspect_ratio"), "aspect_ratio", ALLOWED_RATIOS, "source"),
        quality=_choice(data.get("quality"), "quality", set(QUALITY_STEPS), "standard"),
        seed=seed,
    )
