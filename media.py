"""Bounded image download and encoding helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 15 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class MediaError(RuntimeError):
    pass


def download_image(url: str, destination: Path) -> Path:
    try:
        with requests.get(url, stream=True, timeout=(15, 90), allow_redirects=False) as response:
            if response.status_code != 200:
                raise MediaError(f"La référence image est inaccessible (HTTP {response.status_code}).")
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type not in ALLOWED_CONTENT_TYPES:
                raise MediaError("La référence distante n’est pas une image JPG, PNG ou WebP.")
            length = int(response.headers.get("content-length", "0") or 0)
            if length > MAX_IMAGE_BYTES:
                raise MediaError("La référence distante dépasse 15 Mio.")
            received = 0
            with destination.open("xb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > MAX_IMAGE_BYTES:
                        raise MediaError("La référence distante dépasse 15 Mio.")
                    output.write(chunk)
    except requests.RequestException as error:
        raise MediaError("Impossible de télécharger la référence image.") from error
    if destination.stat().st_size < 1:
        raise MediaError("La référence image est vide.")
    try:
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as error:
        raise MediaError("La référence image est illisible.") from error
    if width < 64 or height < 64 or width > 16384 or height > 16384 or width * height > 100_000_000:
        raise MediaError("Les dimensions de la référence image sont invalides.")
    return destination


def encode_webp(image: Image.Image) -> tuple[str, int]:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="WEBP", quality=95, method=6)
    payload = buffer.getvalue()
    if len(payload) < 1024 or len(payload) > 7_000_000:
        raise MediaError("L’image produite dépasse la taille de résultat autorisée.")
    return base64.b64encode(payload).decode("ascii"), len(payload)
