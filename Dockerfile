# ---- Stage 1 : build du frontend React ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2 : API Python (sans CUDA) ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
# Installe torch/torchvision en version CPU (index officiel) puis le reste
RUN grep -v -E '^torch' requirements.txt > /tmp/req-no-torch.txt \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/req-no-torch.txt

COPY src/ src/
COPY api/ api/
COPY models/checkpoints/best.pt models/checkpoints/best.pt
COPY --from=frontend /app/frontend/dist frontend/dist

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
