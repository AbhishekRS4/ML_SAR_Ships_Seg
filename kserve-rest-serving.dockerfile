# Build from repo root:
#   docker build \
#     -f kserve_scripts/kserve-rest-serving.dockerfile \
#     -t <YOUR_REGISTRY>/sar-ships-seg-kserve:v1 .
#
# Run locally for testing (checkpoint must be bind-mounted):
#   docker run --rm --gpus all \
#     -p 8080:8080 \
#     -v /path/to/checkpoint.pth:/mnt/models/model.pth:ro \
#     <YOUR_REGISTRY>/sar-ships-seg-kserve:v1 \
#     --model_name=sar-ships-seg \
#     --file-model-ckpt=/mnt/models/model.pth \
#     --http_port=8080

FROM python:3.10-slim

WORKDIR /app

# install linux package dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# copy pyproject.toml and uv.lock from repo root for reproducible dependency installation
COPY pyproject.toml uv.lock ./

# install kserve serving dependencies using uv (into the system Python, no venv)
RUN uv sync --only-group dev-inference-kserve --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

# put src/ on PYTHONPATH so the package-relative imports in the runtime
# (models.*, data_handler.*, inference.*) resolve correctly
ENV PYTHONPATH="/app/src"

# copy application source
COPY src/ ./src/

# model checkpoint is injected at runtime by the KServe storage initialiser
# (STORAGE_URI in deploy_rest.yaml); create the expected mount point
RUN mkdir -p /mnt/models

EXPOSE 8080

# kserve.model_server.parser provides standard KServe flags; custom flags
# (--file-model-ckpt, --model-compile-mode, --which-gpu) are defined in
# src/inference/infer_kserve_rest.py
ENTRYPOINT ["python", "-m", "inference.infer_kserve_rest"]
