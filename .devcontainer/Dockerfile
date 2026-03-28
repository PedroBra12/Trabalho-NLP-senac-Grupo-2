# ── Base: CUDA 12.4 + cuDNN 9 on Ubuntu 22.04 ────────────────────────────────
# Matches the vGPU driver stack; change cuda version tag if needed.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── System deps ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        curl \
        git \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip directly into python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ── Install Ollama (CPU+GPU binary, manages its own CUDA libs) ────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Persistent volumes ────────────────────────────────────────────────────────
# /app/.chromadb  → ChromaDB index (mount to preserve across restarts)
# /app/docs       → drop PDFs / TXTs here to index them
# /root/.ollama   → Ollama model weights (mount to avoid re-downloading)
VOLUME ["/app/.chromadb", "/app/docs", "/root/.ollama"]

# ── Expose Ollama port ────────────────────────────────────────────────────────
EXPOSE 11434
