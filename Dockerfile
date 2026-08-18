FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    QWEN_IMAGE_MODEL_ID=Qwen/Qwen-Image-Edit-2511 \
    QWEN_IMAGE_MODEL_REVISION=6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY requirements-worker.txt /worker/requirements-worker.txt
RUN python -m pip install -r /worker/requirements-worker.txt

COPY schema.py media.py inference.py handler.py /worker/
RUN python -m py_compile /worker/schema.py /worker/media.py /worker/inference.py /worker/handler.py \
    && python -c "from diffusers import QwenImageEditPlusPipeline; assert QwenImageEditPlusPipeline"

ENTRYPOINT []
CMD ["python", "-u", "/worker/handler.py"]
