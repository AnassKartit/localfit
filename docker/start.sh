#!/bin/bash
# localfit RunPod startup script
# Starts Ollama, pulls model, creates Cloudflare tunnel

set -e

echo "╭──────────────────────────────────────────╮"
echo "│  localfit — try any AI model, one command │"
echo "╰──────────────────────────────────────────╯"

# Get model from env vars (set by localfit --cloud or RunPod template)
MODEL_TAG="${LOCALFIT_MODEL_TAG:-}"
MODEL_REPO="${LOCALFIT_MODEL_REPO:-}"
MODEL_FILE="${LOCALFIT_MODEL_FILE:-}"

# Start Ollama
echo "[localfit] Starting Ollama..."
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_FLASH_ATTENTION=1 ollama serve > /tmp/ollama.log 2>&1 &

# Wait for Ollama
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "[localfit] Ollama ready"
        break
    fi
    sleep 1
done

# Pull model
if [ -n "$MODEL_TAG" ]; then
    # Ollama registry model (e.g. gemma3:4b, qwen3:8b)
    echo "[localfit] Pulling $MODEL_TAG..."
    ollama pull "$MODEL_TAG"
    SERVE_MODEL="$MODEL_TAG"

elif [ -n "$MODEL_REPO" ] && [ -n "$MODEL_FILE" ]; then
    # HuggingFace GGUF — download and import into Ollama
    echo "[localfit] Downloading $MODEL_REPO / $MODEL_FILE..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python3 -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download('$MODEL_REPO', '$MODEL_FILE')
os.symlink(path, '/tmp/model.gguf')
print(f'Downloaded: {path}')
"
    echo "[localfit] Importing into Ollama..."
    echo "FROM /tmp/model.gguf" > /tmp/Modelfile
    ollama create localmodel -f /tmp/Modelfile
    SERVE_MODEL="localmodel"

else
    echo "[localfit] No model specified."
    echo "[localfit] Set LOCALFIT_MODEL_TAG (e.g. gemma3:4b)"
    echo "[localfit]  or LOCALFIT_MODEL_REPO + LOCALFIT_MODEL_FILE"
    echo "[localfit] Starting SSH only — connect and run manually."
    sleep infinity
fi

# Show GPU info
echo "[localfit] GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "  (no nvidia-smi)"

# Start Cloudflare tunnel
echo "[localfit] Starting Cloudflare tunnel..."
nohup cloudflared tunnel --url http://localhost:11434 > /tmp/cloudflared.log 2>&1 &
sleep 8

# Extract tunnel URL
TUNNEL_URL=$(grep -oP 'https://[\w-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -1)

# Write status file (localfit polls this via SSH)
if [ -n "$TUNNEL_URL" ]; then
    echo "LOCALFIT_TUNNEL=$TUNNEL_URL" > /tmp/localfit_status
    echo "LOCALFIT_MODEL=$SERVE_MODEL" >> /tmp/localfit_status
else
    echo "LOCALFIT_TUNNEL=FAILED" > /tmp/localfit_status
fi
echo "LOCALFIT_READY" >> /tmp/localfit_status

# Show connection info
echo ""
echo "╭──────────────────────────────────────────╮"
echo "│  READY                                   │"
echo "│                                          │"
if [ -n "$TUNNEL_URL" ]; then
echo "│  Tunnel: $TUNNEL_URL"
echo "│  API:    $TUNNEL_URL/v1"
fi
echo "│  Model:  $SERVE_MODEL"
echo "│                                          │"
echo "│  Test:                                   │"
echo "│  curl $TUNNEL_URL/v1/chat/completions \\"
echo "│    -d '{\"model\":\"$SERVE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
echo "╰──────────────────────────────────────────╯"

# Keep alive — restart tunnel if it dies
while true; do
    if ! pgrep -f cloudflared > /dev/null; then
        echo "[localfit] Restarting tunnel..."
        nohup cloudflared tunnel --url http://localhost:11434 > /tmp/cloudflared.log 2>&1 &
        sleep 8
        NEW_URL=$(grep -oP 'https://[\w-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -1)
        if [ -n "$NEW_URL" ]; then
            echo "LOCALFIT_TUNNEL=$NEW_URL" > /tmp/localfit_status
            echo "LOCALFIT_MODEL=$SERVE_MODEL" >> /tmp/localfit_status
            echo "LOCALFIT_READY" >> /tmp/localfit_status
            echo "[localfit] New tunnel: $NEW_URL"
        fi
    fi
    sleep 30
done
