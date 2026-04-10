# localfit RunPod Template
# Pre-built with Ollama, cloudflared, and localfit
# Pod starts ready to serve in ~30s (just pull model)
#
# Build:  docker build -t localfit/runpod:latest .
# Push:   docker push localfit/runpod:latest

FROM runpod/base:1.0.3-cuda1281-ubuntu2404

LABEL maintainer="localfit"
LABEL description="GPU toolkit for AI models — pre-built with Ollama, Cloudflare tunnel, and localfit"
LABEL org.opencontainers.image.source="https://github.com/AnassKartit/localfit"

# Install system deps (pciutils for GPU detection, lsof for port checks)
RUN apt-get update -qq && \
    apt-get install -y -qq pciutils lsof curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install cloudflared
RUN curl -fsSL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && \
    chmod +x /usr/local/bin/cloudflared

# Install localfit + huggingface_hub
RUN pip install --no-cache-dir localfit huggingface_hub hf_transfer

# Startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 11434 22

CMD ["/app/start.sh"]
