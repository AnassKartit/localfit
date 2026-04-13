"""localfit cloud — provision GPU pods on RunPod, serve models remotely.

Usage:
    localfit serve gemma4:31b --cloud              # auto-provision + tunnel
    localfit serve gemma4:31b --cloud --budget 2h  # auto-stop after 2 hours
    localfit stop                                   # kill pod immediately
    localfit login runpod                           # save API key
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import urllib.parse
import threading

from pathlib import Path
from rich.console import Console

console = Console()
CONFIG_DIR = Path.home() / ".localfit"
RUNPOD_KEY_FILE = CONFIG_DIR / "runpod_key"
MODAL_KEY_FILE = CONFIG_DIR / "modal_key"

# Modal endpoints for popular models
MODAL_MODELS = {
    "glm-5.1": {
        "url": "https://api.us-west-2.modal.direct/v1",
        "model_id": "zai-org/GLM-5.1-FP8",
        "name": "GLM-5.1 (754B MoE)",
    },
    "glm-5": {
        "url": "https://api.us-west-2.modal.direct/v1",
        "model_id": "zai-org/GLM-5-FP8",
        "name": "GLM-5 (744B MoE)",
    },
}


# ── Modal ──


def save_modal_key(key):
    """Save Modal API token."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODAL_KEY_FILE.write_text(key.strip())
    MODAL_KEY_FILE.chmod(0o600)
    console.print(f"  [green]✓[/] Modal token saved")


def get_modal_key():
    """Get Modal API token."""
    key = os.environ.get("MODAL_TOKEN") or os.environ.get("MODAL_API_KEY")
    if key:
        return key
    if MODAL_KEY_FILE.exists():
        return MODAL_KEY_FILE.read_text().strip()
    return None


def modal_serve(model_query, tool=None):
    """Connect to a model via Modal's API.

    No pod management needed -- just token + URL.
    Works with any OpenAI-compatible client.
    """
    token = get_modal_key()
    if not token:
        console.print(f"\n  [bold]Modal — serverless cloud GPUs[/]\n")
        console.print(f"  1. Get a token at [cyan]https://modal.com/settings/tokens[/]")
        console.print(f"  2. Paste it below:\n")
        try:
            token = input("  Modal token: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not token:
            return None
        save_modal_key(token)

    # Find model endpoint
    modal_info = None
    query_lower = model_query.lower().replace(":", "-").replace(" ", "-")
    for mid, info in MODAL_MODELS.items():
        if mid in query_lower or query_lower in mid:
            modal_info = info
            break

    if not modal_info:
        # Custom model -- user provides their own Modal endpoint
        console.print(f"\n  [dim]Model '{model_query}' not in Modal presets.[/]")
        console.print(f"  [dim]Enter your Modal endpoint URL (OpenAI-compatible):[/]")
        try:
            url = input("  URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not url:
            return None
        modal_info = {"url": url, "model_id": model_query, "name": model_query}

    api_base = modal_info["url"]
    model_id = modal_info["model_id"]

    console.print(f"\n  [green]✓[/] Modal: {modal_info['name']}")
    console.print(f"  [dim]API: {api_base}[/]")
    console.print(f"  [dim]Model: {model_id}[/]")

    # Test connection
    try:
        req = urllib.request.Request(
            f"{api_base}/models",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            models = [m.get("id", "") for m in data.get("data", [])]
            console.print(f"  [green]✓[/] Connected — {len(models)} models available")
    except Exception as e:
        console.print(f"  [yellow]Warning: Could not verify connection ({e})[/]")
        console.print(f"  [dim]Continuing anyway — model may need a cold start...[/]")

    # Launch tool if requested
    if tool:
        from localfit.cli import _launch_tool_with_endpoint

        # Set auth header for tools that need it
        os.environ["OPENAI_API_KEY"] = token
        os.environ["OPENAI_API_BASE"] = api_base
        _launch_tool_with_endpoint(tool, api_base, model_id)

    return {"api_base": api_base, "model_id": model_id, "token": token}


# GPU options with pricing (approximate, RunPod varies by availability)
GPU_OPTIONS = []  # populated at runtime from API

# Rough tok/s estimates by GPU family (used when no benchmark data available)
_TOK_S_ESTIMATES = {
    "3090": 35,
    "4090": 45,
    "5090": 55,
    "A5000": 25,
    "A6000": 40,
    "A100": 90,
    "H100": 120,
    "H200": 130,
    "L4": 20,
    "L40": 45,
    "L40S": 50,
    "V100": 15,
    "MI300": 80,
    "RTX PRO 6000": 60,
    "RTX PRO 4500": 35,
    "RTX 5000 Ada": 40,
    "RTX 6000 Ada": 45,
}


def _estimate_tok_s(gpu_name):
    """Estimate tok/s from GPU name."""
    for key, tps in _TOK_S_ESTIMATES.items():
        if key.lower() in gpu_name.lower():
            return tps
    return 30  # default


def fetch_gpu_options(api_key):
    """Fetch real-time GPU pricing from RunPod API."""
    result = _runpod_api(
        "{ gpuTypes { id displayName memoryInGb communityPrice securePrice secureCloud communityCloud } }",
        api_key,
    )
    gpus = []
    for g in result.get("data", {}).get("gpuTypes", []):
        vram = g.get("memoryInGb", 0)
        name = g.get("displayName", "")
        if vram < 16:
            continue
        # Prefer community price (cheaper), fall back to secure
        community = g.get("communityPrice") or 0
        secure = g.get("securePrice") or 0
        has_community = g.get("communityCloud", False) and community > 0
        has_secure = g.get("secureCloud", False) and secure > 0
        if has_community:
            price = community
        elif has_secure:
            price = secure
        else:
            continue  # not available at all
        gpus.append(
            {
                "id": g["id"],
                "name": name,
                "vram": vram,
                "price": round(price, 2),
                "tok_s": _estimate_tok_s(name),
                "community": has_community,
                "secure": has_secure,
            }
        )
    # Sort by price
    gpus.sort(key=lambda x: x["price"])
    return gpus


# Docker image: lightweight RunPod base with CUDA + SSH built in (~6GB, boots in ~60s)
RUNPOD_IMAGE = "runpod/base:1.0.3-cuda1281-ubuntu2404"

# Ollama-based startup script (replaces old llama-server compile approach)
# Installs Ollama (~21s) + cloudflared (<1s), pulls model via Ollama, starts tunnel
STARTUP_SCRIPT_OLLAMA = """#!/bin/bash
set -e

echo "[localfit] Starting setup..."

# Install system deps for GPU detection
apt-get update -qq && apt-get install -y -qq pciutils lsof > /dev/null 2>&1

# Install Ollama
echo "[localfit] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -3

# Install cloudflared
echo "[localfit] Installing cloudflared..."
curl -fsSL -o /usr/local/bin/cloudflared \\
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \\
  && chmod +x /usr/local/bin/cloudflared

# Start Ollama server
echo "[localfit] Starting Ollama..."
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_FLASH_ATTENTION=1 ollama serve > /tmp/ollama.log 2>&1 &
sleep 3

# Wait for Ollama to be ready
for i in $(seq 1 30); do
  if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "[localfit] Ollama ready!"
    break
  fi
  sleep 1
done

# Pull model (either Ollama tag or GGUF from HuggingFace)
{pull_cmd}

# Start Cloudflare tunnel
echo "[localfit] Starting Cloudflare tunnel..."
nohup cloudflared tunnel --url http://localhost:11434 > /tmp/cloudflared.log 2>&1 &
sleep 8

# Extract tunnel URL
TUNNEL_URL=$(grep -oP 'https://[\\w-]+\\.trycloudflare\\.com' /tmp/cloudflared.log | head -1)
if [ -n "$TUNNEL_URL" ]; then
  echo "LOCALFIT_TUNNEL=$TUNNEL_URL" > /tmp/localfit_status
  echo "[localfit] Tunnel ready: $TUNNEL_URL"
else
  echo "LOCALFIT_TUNNEL=FAILED" > /tmp/localfit_status
  echo "[localfit] Tunnel failed!"
fi

echo "LOCALFIT_READY" >> /tmp/localfit_status
echo "[localfit] All ready!"

# Keep alive
while true; do
  # Restart tunnel if it dies
  if ! pgrep -f cloudflared > /dev/null; then
    echo "[localfit] Restarting tunnel..."
    nohup cloudflared tunnel --url http://localhost:11434 > /tmp/cloudflared.log 2>&1 &
    sleep 8
    NEW_URL=$(grep -oP 'https://[\\w-]+\\.trycloudflare\\.com' /tmp/cloudflared.log | head -1)
    if [ -n "$NEW_URL" ]; then
      echo "LOCALFIT_TUNNEL=$NEW_URL" > /tmp/localfit_status
      echo "LOCALFIT_READY" >> /tmp/localfit_status
    fi
  fi
  sleep 30
done
"""


def _build_pull_cmd(model_tag=None, hf_repo=None, hf_filename=None):
    """Build the model pull command for the startup script.

    Two modes:
      1. Ollama tag (e.g. "gemma3:4b") — uses `ollama pull`
      2. HuggingFace GGUF — downloads file, imports into Ollama via Modelfile
    """
    if model_tag:
        return f'''echo "[localfit] Pulling model: {model_tag}"
ollama pull {model_tag}
echo "[localfit] Model pulled!"'''

    if hf_repo and hf_filename:
        return f'''echo "[localfit] Downloading GGUF from HuggingFace..."
pip install -q huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('{hf_repo}', '{hf_filename}')
import os; os.symlink(path, '/tmp/model.gguf')
print(f'Downloaded: {{path}}')
"

echo "[localfit] Importing into Ollama..."
cat > /tmp/Modelfile << 'EOF'
FROM /tmp/model.gguf
EOF
ollama create localmodel -f /tmp/Modelfile
echo "[localfit] Model imported as 'localmodel'!"'''

    return 'echo "[localfit] ERROR: No model specified!"'


def save_runpod_key(key):
    """Save RunPod API key securely."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RUNPOD_KEY_FILE.write_text(key.strip())
    RUNPOD_KEY_FILE.chmod(0o600)
    console.print(f"  [green]✓[/] API key saved to {RUNPOD_KEY_FILE}")


def get_runpod_key():
    """Get RunPod API key."""
    # Check env first
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    # Check file
    if RUNPOD_KEY_FILE.exists():
        return RUNPOD_KEY_FILE.read_text().strip()
    return None


def _runpod_api(query, api_key):
    """Execute a RunPod GraphQL query."""
    import subprocess

    result = subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            f"https://api.runpod.io/graphql?api_key={api_key}",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({"query": query}),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def _runpod_rest(method, path, api_key, payload=None):
    """Execute a RunPod REST API request."""
    url = f"https://rest.runpod.io/v1{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "localfit",
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body
        return {"error": {"status": e.code, "body": parsed}}


def create_pod_rest(api_key, payload):
    """Create a pod via the REST API."""
    return _runpod_rest("POST", "/pods", api_key, payload)


def get_pod_rest(api_key, pod_id):
    """Get a pod via the REST API."""
    return _runpod_rest("GET", f"/pods/{pod_id}", api_key)


def list_gpus(api_key):
    """List available GPU types with pricing."""
    query = "{ gpuTypes { id displayName memoryInGb } }"
    return _runpod_api(query, api_key)


def create_pod(
    api_key,
    gpu_id,
    name,
    model_tag=None,
    hf_repo=None,
    hf_filename=None,
    container_disk=30,
    volume_disk=50,
    network_volume_id=None,
):
    """Create a RunPod GPU pod with Ollama + Cloudflare tunnel.

    Two model modes:
      - model_tag: Ollama registry tag (e.g. "gemma3:4b", "qwen3:8b")
      - hf_repo + hf_filename: HuggingFace GGUF file to download and import
    """

    # Build the model pull command
    pull_cmd = _build_pull_cmd(
        model_tag=model_tag, hf_repo=hf_repo, hf_filename=hf_filename
    )

    # Build startup script
    script = STARTUP_SCRIPT_OLLAMA.replace("{pull_cmd}", pull_cmd)

    # Escape for GraphQL
    script_escaped = (
        script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )

    volume_part = ""
    if network_volume_id:
        volume_part = f'networkVolumeId: "{network_volume_id}"'
    elif volume_disk > 0:
        volume_part = f"volumeInGb: {volume_disk}"

    model_label = model_tag or f"{hf_repo}/{hf_filename}"

    query = f'''
    mutation {{
      podFindAndDeployOnDemand(
        input: {{
          cloudType: ALL
          gpuCount: 1
          containerDiskInGb: {container_disk}
          {volume_part}
          gpuTypeId: "{gpu_id}"
          name: "{name}"
          imageName: "{RUNPOD_IMAGE}"
          ports: "11434/http,22/tcp"
          volumeMountPath: "/workspace"
          startSsh: true
          dockerArgs: "bash -c '{script_escaped}'"
          env: [
            {{ key: "LOCALFIT_MODEL", value: "{model_label}" }}
          ]
        }}
      ) {{
        id
        machineId
        machine {{
          podHostId
        }}
      }}
    }}
    '''

    return _runpod_api(query, api_key)


STARTUP_SCRIPT_IMAGE = """#!/bin/bash
set -e
echo "[localfit] Installing diffusers..."
pip install -q -U 'git+https://github.com/huggingface/diffusers.git' transformers accelerate sentencepiece 2>&1 | tail -2

echo "[localfit] Installing cloudflared..."
curl -fsSL -o /usr/local/bin/cloudflared \\
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \\
  && chmod +x /usr/local/bin/cloudflared

echo "[localfit] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[localfit] Loading {model_repo}..."
python3 << 'PYEOF'
import torch,json,base64,io,time,os,re,threading
from http.server import HTTPServer,BaseHTTPRequestHandler
from diffusers import DiffusionPipeline
print(f"GPU: {{torch.cuda.get_device_name(0)}}")
p=DiffusionPipeline.from_pretrained("{model_repo}",torch_dtype=torch.bfloat16).to("cuda")
print("MODEL_LOADED")
class H(BaseHTTPRequestHandler):
 def do_POST(self):
  if self.path=="/v1/images/generations":
   l=int(self.headers.get("Content-Length",0));b=json.loads(self.rfile.read(l)) if l else {{}}
   s=b.get("size","512x512");w,h=(int(x) for x in s.split("x"))
   r=p(prompt=b.get("prompt","cat"),width=w,height=h,num_inference_steps=b.get("steps",4))
   buf=io.BytesIO();r.images[0].save(buf,format="PNG")
   d=json.dumps({{"created":int(time.time()),"data":[{{"b64_json":base64.b64encode(buf.getvalue()).decode()}}]}}).encode()
   self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",len(d));self.end_headers();self.wfile.write(d)
 def do_GET(self):
  d=json.dumps({{"status":"ok","model":"{model_repo}"}}).encode()
  self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",len(d));self.end_headers();self.wfile.write(d)
 def log_message(self,*a):pass
srv=HTTPServer(("0.0.0.0",8189),H);threading.Thread(target=srv.serve_forever,daemon=True).start()
print("API on :8189")
os.system("nohup cloudflared tunnel --url http://localhost:8189 >/tmp/cf.log 2>&1 &")
time.sleep(10)
with open("/tmp/cf.log") as f:log=f.read()
m=re.search(r"https://[\\w-]+\\.trycloudflare\\.com",log)
if m:
 url=m.group(0);print(f"LOCALFIT_TUNNEL={{url}}")
 import urllib.request as ur;ur.urlopen("https://ntfy.sh/localfit-img-{ntfy_topic}",data=url.encode(),timeout=5)
 print(f"LOCALFIT_READY")
else:
 print("LOCALFIT_TUNNEL=FAILED")
while True:time.sleep(30)
PYEOF
"""


def create_pod_image(api_key, gpu_id, name, model_repo, container_disk=40):
    """Create a RunPod GPU pod for image generation with diffusers + Cloudflare tunnel."""
    import re as _re

    ntfy_topic = _re.sub(r"[^a-z0-9]+", "-", model_repo.lower())[:40]
    script = STARTUP_SCRIPT_IMAGE.replace("{model_repo}", model_repo).replace(
        "{ntfy_topic}", ntfy_topic
    )

    script_escaped = (
        script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )

    query = f'''
    mutation {{
      podFindAndDeployOnDemand(
        input: {{
          cloudType: ALL
          gpuCount: 1
          containerDiskInGb: {container_disk}
          gpuTypeId: "{gpu_id}"
          name: "{name}"
          imageName: "runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204"
          ports: "8189/http"
          dockerArgs: "bash -c '{script_escaped}'"
          env: [
            {{ key: "LOCALFIT_MODEL", value: "{model_repo}" }}
          ]
        }}
      ) {{
        id
        machine {{
          gpuDisplayName
        }}
      }}
    }}
    '''

    return _runpod_api(query, api_key)


def get_pod(api_key, pod_id):
    """Get pod status."""
    query = f'''
    {{ pod(input: {{ podId: "{pod_id}" }}) {{
        id
        name
        desiredStatus
        runtime {{
          uptimeInSeconds
          ports {{
            ip
            isIpPublic
            privatePort
            publicPort
          }}
        }}
    }} }}
    '''
    return _runpod_api(query, api_key)


def stop_pod(api_key, pod_id):
    """Stop a pod."""
    query = (
        f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }} }}'
    )
    return _runpod_api(query, api_key)


def terminate_pod(api_key, pod_id):
    """Terminate a pod (delete)."""
    query = f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}'
    return _runpod_api(query, api_key)


def cloud_serve(model_query, budget_hours=2, gpu_preference=None):
    """Full flow: pick GPU → create pod → wait ready → tunnel → configure tools."""
    from localfit.backends import fetch_hf_model, get_machine_specs

    api_key = get_runpod_key()
    if not api_key:
        console.print(f"\n  [red]RunPod API key not set.[/]")
        console.print(
            f"  [dim]Get one at: https://www.runpod.io/console/user/settings[/]"
        )
        console.print(f"  [cyan]localfit login runpod[/]\n")
        return None

    specs = get_machine_specs()
    gpu_total = specs["gpu_total_mb"]

    # Fetch model info
    console.print(f"\n  [bold]Fetching model info...[/]")
    data = fetch_hf_model(model_query, silent=True)
    if not data or not data["gguf_files"]:
        console.print(f"  [red]Model not found: {model_query}[/]")
        return None

    # Check if it fits locally first
    fits_local = [f for f in data["gguf_files"] if f["size_gb"] * 1024 < gpu_total]
    if fits_local:
        best_local = fits_local[-1]
        console.print(f"  [green]This model fits your local GPU![/]")
        console.print(
            f"  [dim]Best local quant: {best_local['quant']} ({best_local['size_gb']}GB)[/]"
        )
        console.print(f"  [dim]Use: localfit --serve {model_query}[/]")
        try:
            ans = input("  Still use cloud? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if ans != "y":
            return None

    # Ask budget first
    console.print(f"\n  [bold]How much do you want to spend?[/]\n")
    console.print(f"  [bold cyan]1[/]  $1    [dim](~2h on RTX 3090)[/]")
    console.print(
        f"  [bold cyan]2[/]  $2    [dim](~4h on RTX 3090 or ~3h on RTX 4090)[/]"
    )
    console.print(f"  [bold cyan]3[/]  $5    [dim](~10h on RTX 3090 or ~2h on A100)[/]")
    console.print(f"  [bold cyan]4[/]  $10   [dim](~20h on RTX 3090)[/]")
    console.print(f"  [bold cyan]5[/]  Custom amount")
    console.print()
    try:
        budget_choice = input(f"  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    budget_map = {"1": 1.0, "2": 2.0, "3": 5.0, "4": 10.0}
    if budget_choice in budget_map:
        max_spend = budget_map[budget_choice]
    elif budget_choice == "5":
        try:
            max_spend = float(input("  Max spend ($): ").strip().replace("$", ""))
        except (EOFError, KeyboardInterrupt, ValueError):
            return None
    else:
        try:
            max_spend = float(budget_choice.replace("$", ""))
        except ValueError:
            return None

    # Fetch real-time GPU pricing from RunPod
    console.print(f"  [dim]Fetching live GPU pricing...[/]")
    live_gpus = fetch_gpu_options(api_key)
    if not live_gpus:
        console.print(f"  [red]Could not fetch GPU pricing from RunPod.[/]")
        return None

    # Smart matching — find best GPU + quant for budget using live prices
    from localfit.matcher import (
        find_best_match,
        find_recommended,
        get_quality_label,
        GPUS,
    )

    # Override matcher's GPU list with live data
    GPUS.clear()
    GPUS.extend(live_gpus)

    options = find_best_match(data["gguf_files"], max_spend)
    if not options:
        console.print(f"  [red]Budget too low for any GPU with this model.[/]")
        console.print(f"  [dim]Minimum: ~$0.18 (30 min on cheapest GPU)[/]")
        return None

    recommended = find_recommended(data["gguf_files"], max_spend)

    console.print(f"\n  [bold]Best options for ${max_spend:.2f}:[/]\n")

    available_gpus = []
    for i, opt in enumerate(options[:6], 1):  # show top 6 options max
        gpu = opt["gpu"]
        quant = opt["quant"]
        h = int(opt["hours"])
        m = int((opt["hours"] - h) * 60)
        time_str = f"{h}h{m:02d}m" if m else f"{h}h"
        label, color = get_quality_label(opt["quality_score"])

        is_rec = (
            recommended
            and opt["gpu"]["name"] == recommended["gpu"]["name"]
            and opt["quant"]["quant"] == recommended["quant"]["quant"]
        )
        rec_tag = " [bold green]← RECOMMENDED[/]" if is_rec else ""

        console.print(
            f"  [bold cyan]{i}[/]  {gpu['name']:<12} {gpu['vram']}GB  "
            f"~{opt['tok_s']} tok/s  "
            f"[bold]{time_str}[/]  "
            f"[{color}]{quant['quant']} ({label})[/{color}]  "
            f"[dim]{quant['size_gb']}GB[/]"
            f"{rec_tag}"
        )
        available_gpus.append(opt)

    # Show upgrade hint if quality is poor
    if recommended and recommended.get("upgrade_hint"):
        hint = recommended["upgrade_hint"]
        console.print(
            f"\n  [yellow]Tip: ${hint['min_cost']:.2f} more gets you {hint['gpu']} with {hint['quant']} (much better quality)[/]"
        )

    console.print()
    try:
        ans = input(f"  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    try:
        idx = int(ans) - 1
        chosen = available_gpus[idx]
        gpu = chosen["gpu"]
        best_quant = chosen["quant"]
        budget_hours = chosen["hours"]
    except (ValueError, IndexError):
        return None

    cost = gpu["price"] * budget_hours
    storage_cost = 0.006 * budget_hours  # ~$0.006/hr for 60GB disk

    # Hard approval with full cost breakdown
    console.print(f"\n  [bold]─── Cost Estimate ───[/]")
    console.print(f"  GPU:       {gpu['name']} ({gpu['vram']}GB VRAM)")
    console.print(f"  Model:     {best_quant['quant']} ({best_quant['size_gb']}GB)")
    console.print(
        f"  Speed:     ~{gpu.get('tok_s', gpu.get('tok_s_est', 0))} tok/s estimated"
    )
    console.print(f"  Duration:  {budget_hours}h (auto-stop)")
    console.print()
    console.print(f"  [bold]Costs:[/]")
    console.print(
        f"    GPU compute:  ${gpu['price']:.2f}/hr × {budget_hours}h = [bold]${cost:.2f}[/]"
    )
    console.print(f"    Storage:      ${storage_cost:.3f}")
    console.print(f"    [bold]Total:        ${cost + storage_cost:.2f}[/]")
    console.print()
    console.print(
        f"  [dim]vs Claude Sonnet API for {budget_hours}h: ~${budget_hours * 13.75:.2f}[/]"
    )
    console.print(
        f"  [dim]vs Opus API for {budget_hours}h: ~${budget_hours * 22.84:.2f}[/]"
    )
    console.print()

    try:
        confirm = input(f"  Proceed? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if confirm not in ("y", "yes"):
        console.print(f"  [dim]Cancelled.[/]")
        return None

    console.print(f"\n  [bold]Provisioning {gpu['name']}...[/]")

    model_name = data.get("name", "")

    # Create pod with Ollama
    try:
        result = create_pod(
            api_key=api_key,
            gpu_id=gpu["id"],
            name=f"localfit-{model_name[:20]}",
            hf_repo=data["repo_id"],
            hf_filename=best_quant["filename"],
            container_disk=30,
            volume_disk=50,
        )

        if "errors" in result:
            console.print(
                f"  [red]Failed: {result['errors'][0].get('message', 'Unknown error')}[/]"
            )
            return None

        pod_data = result.get("data", {}).get("podFindAndDeployOnDemand", {})
        pod_id = pod_data.get("id")
        console.print(f"  [green]✓[/] Pod created: {pod_id}")

        # Save pod info for stop/status
        pod_info = {
            "pod_id": pod_id,
            "gpu": gpu["name"],
            "model": model_name,
            "quant": best_quant["quant"],
            "budget_hours": budget_hours,
            "started_at": time.time(),
            "price_per_hour": gpu["price"],
        }
        pod_file = CONFIG_DIR / "active_pod.json"
        pod_file.write_text(json.dumps(pod_info, indent=2))

        # Wait for pod to be ready (SSH available)
        console.print(f"  [dim]Waiting for pod to start (~60s)...[/]")
        ssh_port = None
        ssh_ip = None
        for i in range(120):  # 10 min max
            try:
                pod_status = get_pod(api_key, pod_id)
                pod = pod_status.get("data", {}).get("pod", {})
                runtime = pod.get("runtime")
                if runtime and runtime.get("ports"):
                    for p in runtime["ports"]:
                        if p.get("privatePort") == 22:
                            ssh_port = p.get("publicPort")
                            ssh_ip = p.get("ip")
                    if ssh_port and ssh_ip:
                        console.print(f"  [green]✓[/] Pod running: {ssh_ip}:{ssh_port}")
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            console.print(f"  [red]Pod failed to start in 10 minutes.[/]")
            return pod_id

        # Wait for Ollama + model + tunnel to be ready via SSH
        console.print(f"  [dim]Waiting for Ollama + model + tunnel...[/]")
        tunnel_url = None
        ssh_key = os.path.expanduser("~/.ssh/id_ed25519")
        ssh_cmd_base = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            f"root@{ssh_ip}",
            "-p",
            str(ssh_port),
            "-i",
            ssh_key,
        ]

        for j in range(180):  # 15 min for install + model pull
            try:
                check = subprocess.run(
                    ssh_cmd_base + ["cat /tmp/localfit_status 2>/dev/null"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                output = check.stdout.strip()
                if "LOCALFIT_READY" in output:
                    # Extract tunnel URL
                    for line in output.split("\n"):
                        if line.startswith("LOCALFIT_TUNNEL="):
                            tunnel_url = line.split("=", 1)[1].strip()
                    if tunnel_url and tunnel_url != "FAILED":
                        console.print(f"  [green]✓[/] Tunnel ready: {tunnel_url}")
                        break
                    else:
                        console.print(f"  [yellow]Tunnel failed, retrying...[/]")
                        # Trigger tunnel restart
                        subprocess.run(
                            ssh_cmd_base
                            + [
                                "pkill -f cloudflared; sleep 1; nohup cloudflared tunnel --url http://localhost:11434 > /tmp/cloudflared.log 2>&1 &"
                            ],
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        time.sleep(10)
                        # Re-read tunnel URL
                        check2 = subprocess.run(
                            ssh_cmd_base
                            + [
                                "grep -oP 'https://[\\w-]+\\.trycloudflare\\.com' /tmp/cloudflared.log | head -1"
                            ],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        tunnel_url = check2.stdout.strip()
                        if tunnel_url:
                            console.print(f"  [green]✓[/] Tunnel ready: {tunnel_url}")
                            break
                elif (
                    "Installing Ollama" in output
                    or "Pulling" in output
                    or "Downloading" in output
                ):
                    # Still setting up, show progress
                    pass
            except Exception:
                pass
            time.sleep(5)
        else:
            console.print(f"  [red]Setup failed after 15 minutes.[/]")
            return pod_id

        if not tunnel_url:
            console.print(f"  [red]Could not get tunnel URL.[/]")
            return pod_id

        # Verify the endpoint works
        console.print(f"  [dim]Verifying endpoint...[/]")
        model_tag = "localmodel"  # GGUF imported as 'localmodel'
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    f"{tunnel_url}/v1/chat/completions",
                    data=json.dumps(
                        {
                            "model": model_tag,
                            "messages": [{"role": "user", "content": "Say OK"}],
                            "max_tokens": 5,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                resp = urllib.request.urlopen(req, timeout=90)
                resp_data = json.loads(resp.read())
                if resp_data.get("choices"):
                    console.print(f"  [green]✓[/] Endpoint verified!")
                    break
            except Exception as e:
                if attempt < 5:
                    time.sleep(10)
        else:
            console.print(
                f"  [yellow]Warning: Could not verify endpoint, but tunnel is up.[/]"
            )

        # Update pod info with tunnel URL
        pod_info["tunnel_url"] = tunnel_url
        pod_info["model_tag"] = model_tag
        pod_info["ssh_ip"] = ssh_ip
        pod_info["ssh_port"] = ssh_port
        pod_file.write_text(json.dumps(pod_info, indent=2))

        # Auto-stop timer
        def _auto_stop():
            time.sleep(budget_hours * 3600)
            console.print(
                f"\n  [yellow]Budget expired ({budget_hours}h). Stopping pod...[/]"
            )
            try:
                terminate_pod(api_key, pod_id)
            except Exception:
                pass

        timer = threading.Thread(target=_auto_stop, daemon=True)
        timer.start()

        # Done!
        from localfit.remote import _launch_remote_tool_menu, _print_ready

        console.print(
            f"  [dim]Estimated speed: ~{gpu.get('tok_s', gpu.get('tok_s_est', 0))} tok/s[/]"
        )
        _print_ready(
            tunnel_url,
            f"{model_name} {best_quant['quant']}",
            model_query,
            {"name": gpu["name"], "vram_gb": gpu["vram"]},
            data["repo_id"],
            provider_name="RunPod",
            cost_text=f"${gpu['price']:.2f}/hr (auto-stop in {budget_hours}h = ${cost:.2f})",
            status_command="localfit --cloud-status",
            stop_command="localfit stop",
        )
        _launch_remote_tool_menu(tunnel_url, model_tag)

        return pod_id

    except Exception as e:
        console.print(f"  [red]Error: {e}[/]")
        import traceback

        traceback.print_exc()
        return None


def cloud_stop():
    """Stop the active cloud pod."""
    api_key = get_runpod_key()
    pod_file = CONFIG_DIR / "active_pod.json"

    if not pod_file.exists():
        console.print(f"  [dim]No active cloud pod.[/]")
        return

    info = json.loads(pod_file.read_text())
    pod_id = info["pod_id"]
    elapsed = (time.time() - info["started_at"]) / 3600
    cost = elapsed * info["price_per_hour"]

    console.print(f"\n  [bold]Stopping pod {pod_id}...[/]")
    console.print(f"  Runtime: {elapsed:.1f}h · Cost: ${cost:.2f}")

    try:
        terminate_pod(api_key, pod_id)
        console.print(f"  [green]✓[/] Pod terminated. Billing stopped.")
        pod_file.unlink()
    except Exception as e:
        console.print(f"  [red]Error: {e}[/]")


def cloud_status():
    """Show active cloud pod status."""
    pod_file = CONFIG_DIR / "active_pod.json"
    if not pod_file.exists():
        console.print(f"  [dim]No active cloud pod.[/]")
        return

    info = json.loads(pod_file.read_text())
    elapsed = (time.time() - info["started_at"]) / 3600
    cost = elapsed * info["price_per_hour"]
    remaining = info["budget_hours"] - elapsed

    console.print(f"\n  [bold]Active pod:[/] {info['pod_id']}")
    console.print(f"  Model: {info['model']} {info['quant']}")
    console.print(f"  GPU: {info['gpu']}")
    if info.get("tunnel_url"):
        console.print(f"  API: {info['tunnel_url']}/v1")
    console.print(f"  Runtime: {elapsed:.1f}h · Cost so far: ${cost:.2f}")
    if remaining > 0:
        console.print(f"  [green]Remaining: {remaining:.1f}h[/]")
    else:
        console.print(f"  [red]Budget exceeded by {-remaining:.1f}h![/]")
    console.print()
