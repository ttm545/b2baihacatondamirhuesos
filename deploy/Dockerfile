ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} AS runtime
USER root
WORKDIR /release
COPY requirements-api.txt requirements-baseline.txt ./
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    'torch==2.6.0+cpu' 'numpy==2.2.6' 'transformers==4.57.6' 'navec==0.10.0' \
    -r requirements-api.txt -r requirements-baseline.txt

FROM runtime
COPY vk_mail_search ./vk_mail_search
COPY solutions/tiny_encoder ./solutions/tiny_encoder
COPY solutions/semantic_ensemble ./solutions/semantic_ensemble
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_HOME=/tmp/huggingface \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    VK_DATA_DIR=/data VK_SEARCHER=period VK_CONTEXT_MODE=ensemble VK_TINY_THREADS=1 \
    VK_TINY_MODEL=/models/rubert-tiny-retriever VK_CONTEXT_INDEX=/models/tiny-fast.npz \
    VK_WORD_VECTORS=/models/navec.tar VK_SEMANTIC_WEIGHT=0.085 \
    VK_TINY_PROJECTION=/release/solutions/tiny_encoder/assets/projection-80.npz \
    VK_TINY_ADAPTER=/release/solutions/tiny_encoder/assets/adapter-80.npz \
    VK_CONTEXT_ADAPTER=/release/solutions/tiny_encoder/assets/qwen-context.npz
USER 65534:65534
CMD ["python", "-m", "uvicorn", "solutions.semantic_ensemble.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
