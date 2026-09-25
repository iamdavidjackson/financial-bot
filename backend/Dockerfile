FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Use the CPU-only PyTorch wheels so the image doesn't pull in CUDA.
COPY api/requirements.txt api/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r api/requirements.txt

COPY core core
COPY api api
COPY trained_models trained_models

WORKDIR /app/api
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
