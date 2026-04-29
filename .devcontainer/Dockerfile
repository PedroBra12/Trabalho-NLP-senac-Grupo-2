FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        curl \
        git \
        zstd \
        ffmpeg \
        libsm6 \
        libxext6 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip into python3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Ollama
RUN curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.20.3 sh

WORKDIR /app
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/app/.qdrant", "/app/docs", "/root/.ollama"]

# Start Ollama automatically whenever a shell opens in the container
RUN echo 'pgrep -x ollama > /dev/null || ollama serve &' >> ~/.bashrc

EXPOSE 11434
