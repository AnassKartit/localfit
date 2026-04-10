"""localfit makeitfit — quantize any model to fit your GPU, remotely with Unsloth.

Flow:
    1. Check your GPU + fetch model metadata
    2. Show what fits as-is, what needs quantization, what's impossible
    3. User picks: existing quant / remote quantize (Kaggle free / RunPod paid)
    4. Unsloth quantizes on the cloud GPU
    5. Result uploaded to user's HuggingFace repo
    6. localfit runs the new quant immediately

Usage:
    localfit --makeitfit llama-4-scout-17b
    localfit --makeitfit mistral-7b-instruct
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import base64

from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
CONFIG_DIR = Path.home() / ".localfit"
RUNPOD_QUANT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


# ── VRAM heuristics for Unsloth 4-bit loading ──
# To quantize a model you need to load it in 4-bit first.
# Rule of thumb: model_params_B * 0.6 GB needed on GPU.
# T4 = 15GB usable, A100-40 = 38GB usable, A100-80 = 78GB usable
_QUANT_GPU_TIERS = [
    # (max_model_gb_bf16, gpu_name, vram_gb, approx_cost_per_hr, tier)
    (14, "T4 16GB (Kaggle free)", 15, 0.0, "kaggle"),
    (28, "RTX 3090 24GB (RunPod)", 24, 0.30, "runpod_3090"),
    (55, "A100 40GB (RunPod)", 40, 1.50, "runpod_a100_40"),
    (110, "A100 80GB (RunPod)", 80, 2.50, "runpod_a100_80"),
]

# Best quant for each GPU tier (quality vs size tradeoff)
_TIER_DEFAULT_QUANT = {
    "kaggle": "q4_k_m",
    "runpod_3090": "q5_k_m",
    "runpod_a100_40": "q6_k",
    "runpod_a100_80": "q8_0",
}

# Unsloth quant method → human label + quality score
QUANT_OPTIONS = [
    ("q2_k",   "Q2_K   — 2-bit, extreme compression, lowest quality",  2),
    ("q3_k_m", "Q3_K_M — 3-bit, very small, usable",                    3),
    ("q4_k_m", "Q4_K_M — 4-bit, best size/quality balance ★ recommended", 4),
    ("q5_k_m", "Q5_K_M — 5-bit, near-lossless for most tasks",          5),
    ("q6_k",   "Q6_K   — 6-bit, excellent quality",                     6),
    ("q8_0",   "Q8_0   — 8-bit, almost identical to full precision",    7),
]


def _estimate_remote_quant_disk_gb(bf16_gb):
    """Rough scratch space for model download + f16 GGUF + final quant."""
    if not bf16_gb:
        return 40
    return max(40, min(200, int(bf16_gb * 3.2 + 12)))


def _build_runpod_quant_startup():
    """Container start command for RunPod quantization pods."""
    return (
        "set -uo pipefail; "
        "mkdir -p /tmp/localfit-quant; "
        "printf '%s' \"$LOCALFIT_SCRIPT\" | base64 -d > /tmp/localfit-quant/quant.py; "
        "python3 /tmp/localfit-quant/quant.py 2>&1 | tee /tmp/localfit-quant/quant.log; "
        "status=${PIPESTATUS[0]}; "
        "echo \"$status\" > /tmp/localfit-quant/exit_code; "
        "sleep infinity"
    )


def _build_runpod_quant_pod_request(
    gpu_id, cloud_type, model_slug, script_b64, hf_token, container_disk_gb
):
    """Build the REST payload for a RunPod quantization pod."""
    return {
        "name": f"localfit-quant-{model_slug}",
        "imageName": "unsloth/unsloth:latest",
        "gpuTypeIds": [gpu_id],
        "gpuCount": 1,
        "cloudType": cloud_type,
        "computeType": "GPU",
        "containerDiskInGb": container_disk_gb,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": {
            "LOCALFIT_SCRIPT": script_b64,
            "HF_TOKEN": hf_token,
        },
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [_build_runpod_quant_startup()],
    }


def _get_hf_token():
    """Read saved HuggingFace token."""
    # Check env first
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def _get_hf_username(token):
    """Get HuggingFace username from token via API."""
    for endpoint in ["https://huggingface.co/api/whoami-v2", "https://huggingface.co/api/whoami"]:
        try:
            req = urllib.request.Request(
                endpoint,
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
                return data.get("name")
        except Exception:
            continue
    return None


def _bf16_size_gb(gguf_files):
    """Estimate full BF16 size from GGUF files (reverse-engineer from Q8_0 or largest)."""
    if not gguf_files:
        return None
    # Q8_0 ≈ 1 byte/param → BF16 ≈ 2x Q8_0
    q8 = [f for f in gguf_files if "Q8" in f.get("quant", "").upper()]
    if q8:
        return round(q8[0]["size_gb"] * 2.0, 1)
    # Largest quant is closest to BF16
    return round(max(f["size_gb"] for f in gguf_files) * 1.3, 1)


def _can_quantize(bf16_gb):
    """
    Check if quantization is technically possible and on which GPU tier.
    Returns list of feasible tiers, or empty list if impossible.

    Unsloth loads model in 4-bit first. Rule: needs ~bf16_gb * 0.45 VRAM.
    """
    needed_vram = bf16_gb * 0.45  # 4-bit loading overhead
    feasible = []
    for max_gb, gpu_name, vram_gb, cost, tier in _QUANT_GPU_TIERS:
        if bf16_gb <= max_gb and needed_vram <= vram_gb - 2:  # 2GB buffer
            feasible.append({
                "tier": tier,
                "gpu_name": gpu_name,
                "vram_gb": vram_gb,
                "cost_per_hr": cost,
                "est_minutes": max(10, int(bf16_gb * 1.5)),  # ~1.5 min/GB
            })
    return feasible


def _generate_quant_script(model_repo, quant_method, hf_repo_id, hf_token):
    """
    Generate a Python script that quantizes a HF model to GGUF and uploads to HF.
    Uses llama.cpp native tools (not Unsloth's broken converter).

    Flow: download model → convert_hf_to_gguf.py → llama-quantize → upload
    """
    model_repo_literal = json.dumps(model_repo)
    quant_method_literal = json.dumps(quant_method)
    hf_repo_id_literal = json.dumps(hf_repo_id)
    hf_token_literal = json.dumps(hf_token)
    return f'''\
import os
import shutil
import subprocess
import sys
import time
import traceback

MODEL_REPO = {model_repo_literal}
QUANT_METHOD = {quant_method_literal}
HF_REPO_ID = {hf_repo_id_literal}
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or {hf_token_literal}
WORKDIR = "/tmp/localfit-quant"
HF_MODEL_DIR = os.path.join(WORKDIR, "hf_model")

def run(cmd, check=False, **kw):
    print(f">> {{cmd}}", flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    if check and r.returncode != 0:
        print(f"LOCALFIT_ERROR=command_failed:{{cmd[:80]}}", flush=True)
        sys.exit(1)
    return r

try:
    # ── 1. Install deps ──
    print("LOCALFIT_STATUS=installing_deps", flush=True)
    os.makedirs(WORKDIR, exist_ok=True)
    run("apt-get update -qq && apt-get install -y -qq git cmake build-essential > /dev/null 2>&1", check=True)
    run("pip install huggingface_hub hf_transfer gguf transformers sentencepiece protobuf safetensors numpy -q", check=True)

    # ── 2. Get llama.cpp (just the converter script + build quantizer) ──
    print("LOCALFIT_STATUS=setting_up_llama_cpp", flush=True)
    if not os.path.exists("/tmp/llama.cpp"):
        run("git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp", check=True)

    # Build llama-quantize (needed for Q4_K_M etc — F16/Q8_0 can skip this)
    quantizer = "/tmp/llama.cpp/build/bin/llama-quantize"
    needs_quantizer = QUANT_METHOD not in ("f16", "f32", "bf16")
    if needs_quantizer and not os.path.exists(quantizer):
        print("LOCALFIT_STATUS=building_llama_quantize", flush=True)
        run("cmake /tmp/llama.cpp -B /tmp/llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF > /dev/null 2>&1", check=True)
        run("cmake --build /tmp/llama.cpp/build --config Release --target llama-quantize -j$(nproc) > /dev/null 2>&1", check=True)

    # ── 3. Download model from HuggingFace ──
    print("LOCALFIT_STATUS=downloading_model", flush=True)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=HF_MODEL_DIR,
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
    )

    # ── 4. Convert to F16 GGUF ──
    print("LOCALFIT_STATUS=converting_to_gguf", flush=True)
    model_name = MODEL_REPO.split("/")[-1].lower().replace(" ", "-")
    f16_path = os.path.join(WORKDIR, f"{{model_name}}-f16.gguf")
    run(f"python /tmp/llama.cpp/convert_hf_to_gguf.py {{HF_MODEL_DIR}} --outtype f16 --outfile {{f16_path}}", check=True)

    # Clean up safetensors to free disk
    shutil.rmtree(HF_MODEL_DIR, ignore_errors=True)

    # ── 5. Quantize (skip for f16/bf16) ──
    if needs_quantizer:
        print(f"LOCALFIT_STATUS=quantizing_{{QUANT_METHOD}}", flush=True)
        quant_path = os.path.join(WORKDIR, f"{{model_name}}-{{QUANT_METHOD}}.gguf")
        run(f"{{quantizer}} {{f16_path}} {{quant_path}} {{QUANT_METHOD}}", check=True)
        os.remove(f16_path)  # free disk
        upload_path = quant_path
    else:
        upload_path = f16_path

    size_gb = os.path.getsize(upload_path) / 1024**3
    print(f"LOCALFIT_GGUF={{os.path.basename(upload_path)}} {{size_gb:.2f}}GB", flush=True)

    # ── 6. Upload to HuggingFace ──
    print("LOCALFIT_STATUS=uploading_to_hf", flush=True)
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=HF_TOKEN)
    create_repo(HF_REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
    api.upload_file(
        path_or_fileobj=upload_path,
        path_in_repo=os.path.basename(upload_path),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="localfit: {{}} GGUF via llama.cpp".format(QUANT_METHOD.upper()),
    )

    print("LOCALFIT_STATUS=done", flush=True)
    print(f"LOCALFIT_HF_REPO={{HF_REPO_ID}}", flush=True)
except Exception as e:
    print(f"LOCALFIT_ERROR={{type(e).__name__}}: {{str(e)[:400]}}", flush=True)
    traceback.print_exc()
    sys.exit(1)
'''


def _read_kaggle_output_text(tmpdir):
    """Read downloaded Kaggle log/text files for LOCALFIT_* markers."""
    chunks = []
    for root, _, files in os.walk(tmpdir):
        for filename in sorted(files):
            if not filename.endswith((".log", ".txt", ".json")):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, "r", errors="ignore") as f:
                    chunks.append(f.read())
            except Exception:
                continue
    return "\n".join(chunks)


def _hf_repo_ggufs(hf_repo_id, hf_token=None):
    """Return GGUF filenames present in a Hugging Face model repo."""
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{hf_repo_id}",
            headers={
                "User-Agent": "localfit",
                **(
                    {"Authorization": f"Bearer {hf_token}"}
                    if hf_token
                    else {}
                ),
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return [
            sibling["rfilename"]
            for sibling in data.get("siblings", [])
            if sibling.get("rfilename", "").endswith(".gguf")
        ]
    except Exception:
        return []


def _delete_kaggle_kernel(kernel_ref):
    """Best-effort deletion of a Kaggle kernel after success or failure."""
    try:
        subprocess.run(
            f'printf "yes\\n" | kaggle kernels delete {kernel_ref}',
            shell=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        pass


def _poll_kaggle_quant(kernel_ref, timeout_seconds=1800):
    """Poll Kaggle kernel output for quant completion."""
    import tempfile, shutil

    deadline = time.time() + timeout_seconds
    last_status = ""

    status_labels = {
        "installing_deps": "Installing dependencies...",
        "setting_up_llama_cpp": "Setting up llama.cpp...",
        "building_llama_quantize": "Building llama.cpp quantizer...",
        "downloading_model": "Downloading model from Hugging Face...",
        "converting_to_gguf": "Converting model to F16 GGUF...",
        "uploading_to_hf": "Uploading GGUF to HuggingFace...",
        "done": "Done!",
    }

    while time.time() < deadline:
        try:
            tmpdir = tempfile.mkdtemp(prefix="localfit-quant-poll-")
            result = subprocess.run(
                ["kaggle", "kernels", "output", kernel_ref, "-p", tmpdir],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout + result.stderr + "\n" + _read_kaggle_output_text(tmpdir)
            shutil.rmtree(tmpdir, ignore_errors=True)

            # Status updates
            for m in re.finditer(r"LOCALFIT_STATUS=(\S+)", output):
                status = m.group(1)
                if status != last_status:
                    last_status = status
                    label = status_labels.get(status, status.replace("_", " ").title())
                    if status.startswith("quantizing_"):
                        label = f"Quantizing to {status[11:].upper()}..."
                    console.print(f"  [dim]{label}[/]")

            # Done signal
            if "LOCALFIT_STATUS=done" in output:
                hf_match = re.search(r"LOCALFIT_HF_REPO=(\S+)", output)
                return hf_match.group(1) if hf_match else True

            # Error check
            err = re.search(r"LOCALFIT_ERROR=(.+)", output)
            if err:
                console.print(f"  [red]Remote error: {err.group(1)[:300]}[/]")
                return None

            # Kernel status
            ks = subprocess.run(
                ["kaggle", "kernels", "status", kernel_ref],
                capture_output=True, text=True, timeout=15,
            )
            ks_text = ks.stdout.strip().lower()
            if "error" in ks_text or "cancel" in ks_text:
                console.print(f"  [red]Kernel failed: {ks_text}[/]")
                return None
            if "complete" in ks_text and last_status != "done":
                # Kernel finished but we may have missed the status line — re-check
                pass

        except Exception as e:
            console.print(f"  [dim]Waiting... ({type(e).__name__})[/]")

        time.sleep(15)

    console.print(f"  [red]Timeout waiting for quantization (30 min)[/]")
    return None


def quantize_on_kaggle(model_repo, quant_method, hf_token, hf_repo_id):
    """Run quantization on Kaggle free T4. Returns HF repo ID on success."""
    from localfit.remote import (
        _ensure_kaggle_credentials,
        _push_kaggle_kernel,
    )

    console.print(f"\n  [bold]Quantizing on Kaggle (free T4)...[/]")
    console.print(f"  Model: [cyan]{model_repo}[/]")
    console.print(f"  Quant: [cyan]{quant_method.upper()}[/]")
    console.print(f"  HF repo: [cyan]{hf_repo_id}[/]")

    if not _ensure_kaggle_credentials():
        console.print(f"  [red]Kaggle credentials not set. Run: localfit login kaggle[/]")
        return None

    script = _generate_quant_script(model_repo, quant_method, hf_repo_id, hf_token)

    model_slug = re.sub(r"[^a-z0-9]", "-", model_repo.split("/")[-1].lower())[:25]
    kernel_ref = _push_kaggle_kernel(script, f"quant-{model_slug}")
    if not kernel_ref:
        return None

    console.print(f"  [green]✓ Notebook running:[/] {kernel_ref}")
    console.print(f"  [dim]This takes ~15-30 min. Polling for completion...[/]\n")

    result = _poll_kaggle_quant(kernel_ref)
    ggufs = _hf_repo_ggufs(hf_repo_id, hf_token) if result else []
    _delete_kaggle_kernel(kernel_ref)
    if result and ggufs:
        console.print(f"  [green]✓ Kaggle upload verified:[/] {ggufs}")
        return result
    if result and not ggufs:
        console.print(f"  [yellow]Kaggle reported completion, but no GGUF is visible on Hugging Face yet.[/]")
    return None


def quantize_on_runpod(model_repo, quant_method, hf_token, hf_repo_id, bf16_gb):
    """Run quantization on a dedicated RunPod pod. Auto-picks cheapest GPU."""
    from localfit.cloud import (
        get_runpod_key,
        fetch_gpu_options,
        _runpod_api,
        terminate_pod,
        create_pod_rest,
        get_pod,
    )

    api_key = get_runpod_key()
    if not api_key:
        console.print(f"  [red]RunPod key not set. Run: localfit login runpod[/]")
        return None

    needed_vram = bf16_gb * 0.45

    console.print(f"  [dim]Fetching live GPU pricing...[/]")
    live_gpus = fetch_gpu_options(api_key)
    # Get all GPUs that fit, sorted by price — try each until one provisions
    candidates = sorted(
        [g for g in live_gpus if g.get("vram", 0) >= needed_vram + 2],
        key=lambda x: x.get("price", 99),
    )
    if not candidates:
        console.print(f"  [red]No GPU with {needed_vram:.0f}GB+ VRAM available[/]")
        return None

    chosen_gpu = candidates[0]
    # Kill any existing localfit-quant pods to avoid duplicates + billing
    try:
        r = _runpod_api('{ myself { pods { id name } } }', api_key)
        for p in r.get("data", {}).get("myself", {}).get("pods", []):
            if "localfit-quant" in p.get("name", ""):
                console.print(f"  [yellow]Terminating old pod {p['id']}...[/]")
                terminate_pod(api_key, p["id"])
    except Exception:
        pass

    console.print(f"\n  [bold]Quantizing on RunPod...[/]")
    console.print(f"  GPU:   [cyan]{chosen_gpu['name']} ({chosen_gpu['vram']}GB)[/]")
    console.print(f"  Cost:  [cyan]~${chosen_gpu['price']:.2f}/hr[/]")
    console.print(f"  Model: [cyan]{model_repo}[/]")
    console.print(f"  Quant: [cyan]{quant_method.upper()}[/]")

    # Start the quant script as the container's real start command through the
    # REST API. This avoids relying on dockerArgs or external SSH account setup.
    quant_py = _generate_quant_script(model_repo, quant_method, hf_repo_id, hf_token)
    quant_b64 = base64.b64encode(quant_py.encode()).decode()
    model_slug = re.sub(r"[^a-zA-Z0-9-]", "-", model_repo.split("/")[-1])[:30]
    disk_gb = _estimate_remote_quant_disk_gb(bf16_gb)

    def _create_quant_pod(gpu):
        payload = _build_runpod_quant_pod_request(
            gpu_id=gpu["id"],
            cloud_type="COMMUNITY" if gpu.get("community") else "SECURE",
            model_slug=model_slug,
            script_b64=quant_b64,
            hf_token=hf_token,
            container_disk_gb=disk_gb,
        )
        payload["imageName"] = RUNPOD_QUANT_IMAGE
        return create_pod_rest(api_key, payload)

    result = _create_quant_pod(chosen_gpu)
    pod = result if isinstance(result, dict) and not result.get("error") else None

    # Retry with other GPUs if supply constraint
    supply_error = str(result).lower()
    if not pod and (
        "supply_constraint" in supply_error
        or "no instances currently available" in supply_error
        or "insufficient capacity" in supply_error
    ):
        for fallback in candidates[1:5]:
            console.print(f"  [yellow]{chosen_gpu['name']} unavailable — trying {fallback['name']}...[/]")
            chosen_gpu = fallback
            result = _create_quant_pod(fallback)
            pod = result if isinstance(result, dict) and not result.get("error") else None
            if pod:
                console.print(f"  [green]✓ Got {fallback['name']} (${fallback['price']:.2f}/hr)[/]")
                break

    if not pod:
        console.print(f"  [red]No GPUs available right now. Try again later.[/]")
        detail = result.get("error", {}).get("body") if isinstance(result, dict) else result
        if detail:
            console.print(f"  [dim]{str(detail)[:300]}[/]")
        return None

    pod_name = f"localfit-quant-{model_slug}"
    pod_id = pod.get("id")
    resolve_deadline = time.time() + 300
    while time.time() < resolve_deadline and not pod_id:
        pods_result = _runpod_api('{ myself { pods { id name desiredStatus } } }', api_key)
        for existing in pods_result.get("data", {}).get("myself", {}).get("pods", []):
            if existing.get("name") == pod_name:
                pod_id = existing.get("id")
                break
        if not pod_id:
            time.sleep(5)

    if not pod_id:
        console.print(f"  [red]RunPod accepted the launch request but no pod appeared.[/]")
        return None

    console.print(f"  [green]✓ Pod provisioned:[/] {pod_id}")
    console.print(f"  [dim]Container disk: {disk_gb}GB[/]")
    console.print(f"  [dim]Waiting for quantization + HF upload...[/]\n")

    # Poll pod status + HF repo for completion
    deadline = time.time() + 3600
    last_status = ""
    while time.time() < deadline:
        time.sleep(20)
        try:
            # Check if HF repo appeared (means upload done)
            try:
                hf_url = f"https://huggingface.co/api/models/{hf_repo_id}"
                req = urllib.request.Request(hf_url, headers={
                    "Authorization": f"Bearer {hf_token}", "User-Agent": "localfit",
                })
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = json.loads(r.read())
                    gguf = [s["rfilename"] for s in data.get("siblings", []) if ".gguf" in s["rfilename"]]
                    if gguf:
                        console.print(f"  [green]✓ GGUF uploaded to HF![/] {gguf}")
                        terminate_pod(api_key, pod_id)
                        console.print(f"  [dim]Pod terminated (billing stopped)[/]")
                        return hf_repo_id
            except Exception:
                pass

            pod_status = get_pod(api_key, pod_id)
            pod_data = (pod_status.get("data", {}) or {}).get("pod") or {}
            if not pod_data:
                # RunPod can briefly return 404 / empty pod lookups right after
                # provisioning. Re-resolve by pod name before declaring failure.
                pods_result = _runpod_api(
                    '{ myself { pods { id name desiredStatus } } }', api_key
                )
                for existing in pods_result.get("data", {}).get("myself", {}).get(
                    "pods", []
                ):
                    if existing.get("name") == pod_name:
                        pod_id = existing.get("id") or pod_id
                        break
                if pod_id:
                    pod_status = get_pod(api_key, pod_id)
                    pod_data = (pod_status.get("data", {}) or {}).get("pod") or {}
            desired = str(pod_data.get("desiredStatus") or "").lower()
            if desired and desired != last_status:
                last_status = desired
                console.print(f"  [dim]Pod {desired}...[/]")
            if not pod_data:
                console.print(f"  [red]RunPod pod disappeared before upload completed.[/]")
                return None
            if desired in {"exited", "stopped", "terminated"}:
                console.print(f"  [red]Remote quantization exited before upload completed.[/]")
                terminate_pod(api_key, pod_id)
                return None

        except Exception as e:
            console.print(f"  [dim]Polling... ({type(e).__name__})[/]")

    # Timeout — terminate to stop billing
    console.print(f"  [yellow]Timeout — terminating pod {pod_id}[/]")
    terminate_pod(api_key, pod_id)
    return None


def _fetch_safetensors_size(model_repo):
    """Check HuggingFace for a safetensors model and estimate BF16 size in GB."""
    try:
        url = f"https://huggingface.co/api/models/{model_repo}"
        req = urllib.request.Request(url, headers={"User-Agent": "localfit"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        siblings = data.get("siblings", [])
        safetensors = [s for s in siblings if s.get("rfilename", "").endswith(".safetensors")]
        if not safetensors:
            return None
        # Estimate from param count in model name, or from file count
        name = model_repo.lower()
        for size_str, gb in [("405b", 810), ("72b", 144), ("70b", 140), ("34b", 68),
                             ("32b", 64), ("27b", 54), ("14b", 28), ("13b", 26),
                             ("8b", 16), ("7b", 14), ("4b", 8), ("3b", 6),
                             ("1.5b", 3), ("1b", 2), ("0.5b", 1)]:
            if size_str in name:
                return gb
        # Fallback: rough estimate from number of safetensors shards
        # Each shard is typically 5GB → total shards × 5 ≈ BF16 size
        return max(1, len(safetensors) * 5)
    except Exception:
        return None


def _makeitfit_no_gguf(model_repo, bf16_gb, gpu_total_mb, gpu_total_gb, gpu_name, specs):
    """Handle models that have no GGUFs — only safetensors. Offer remote quantization."""
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("n", style="bold cyan", width=3)
    t.add_column("option", no_wrap=False)
    t.add_column("detail", style="dim")

    options = []
    feasible_tiers = _can_quantize(bf16_gb)

    # Kaggle option
    kaggle_tier = next((tier for tier in feasible_tiers if tier["tier"] == "kaggle"), None)
    if kaggle_tier:
        options.append(("kaggle_quant", {"quant": "q4_k_m", "tier": kaggle_tier}))
        t.add_row(
            str(len(options)),
            f"[cyan]Quantize on Kaggle (free) with Unsloth[/] → Q4_K_M GGUF",
            f"{kaggle_tier['gpu_name']} · ~{kaggle_tier['est_minutes']} min · uploads to your HF",
        )

    # RunPod option
    runpod_tier = next((tier for tier in feasible_tiers if tier["tier"] != "kaggle"), None)
    if runpod_tier:
        default_q = _TIER_DEFAULT_QUANT.get(runpod_tier["tier"], "q5_k_m")
        cost_est = runpod_tier["cost_per_hr"] * (runpod_tier["est_minutes"] / 60)
        options.append(("runpod_quant", {"quant": default_q, "tier": runpod_tier}))
        t.add_row(
            str(len(options)),
            f"[magenta]Quantize on RunPod with Unsloth[/] → {default_q.upper()} GGUF",
            f"{runpod_tier['gpu_name']} · ~{runpod_tier['est_minutes']} min · ~${cost_est:.2f}",
        )

    if not feasible_tiers:
        console.print(f"  [red]✗ Not feasible:[/] {bf16_gb}GB model is too large to quantize on available GPUs")
        console.print(f"  [dim]Even A100 80GB can't load this in 4-bit ({bf16_gb * 0.45:.0f}GB needed)[/]")
        return

    # Remote serve option
    options.append(("remote_serve", {}))
    t.add_row(
        str(len(options)),
        "[yellow]Serve remotely[/] (stream from cloud, no quantization)",
        "localfit run --remote kaggle / --cloud runpod",
    )

    console.print(t)
    console.print()

    try:
        choice = input("  Pick option: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(options):
            raise ValueError
    except ValueError:
        console.print(f"  [red]Invalid choice[/]")
        return

    action, meta = options[idx]

    if action in ("kaggle_quant", "runpod_quant"):
        # Same quant selection + HF upload flow as the GGUF path
        console.print(f"\n  [bold]Choose quantization type:[/]\n")
        for i, (method, label, _) in enumerate(QUANT_OPTIONS, 1):
            star = " [bold cyan]← recommended[/]" if method == meta["quant"] else ""
            console.print(f"  [bold cyan]{i}[/]  {label}{star}")
        console.print()
        try:
            q_choice = input("  Pick quant [1-6, default=4]: ").strip() or "4"
            q_idx = int(q_choice) - 1
            quant_method = QUANT_OPTIONS[q_idx][0]
        except (ValueError, IndexError, EOFError, KeyboardInterrupt):
            quant_method = meta["quant"]

        hf_token = _get_hf_token()
        if not hf_token:
            console.print(f"\n  [yellow]HF write token needed. Get one at: https://huggingface.co/settings/tokens[/]")
            try:
                hf_token = input("  Paste HF token: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not hf_token:
                return

        hf_username = _get_hf_username(hf_token)
        if not hf_username:
            console.print(f"  [red]Invalid HF token[/]")
            return

        model_base = re.sub(r"[^a-zA-Z0-9-]", "-", model_repo.split("/")[-1])
        hf_repo_id = f"{hf_username}/{model_base}-{quant_method.upper()}-GGUF-localfit"
        console.print(f"\n  [dim]Will upload to:[/] [cyan]{hf_repo_id}[/]\n")

        if action == "kaggle_quant":
            result = quantize_on_kaggle(model_repo, quant_method, hf_token, hf_repo_id)
        else:
            result = quantize_on_runpod(model_repo, quant_method, hf_token, hf_repo_id, bf16_gb)

        if result:
            console.print(f"\n  [bold green]✓ Done![/]")
            console.print(f"  Your model: [cyan]https://huggingface.co/{hf_repo_id}[/]")
            console.print(f"  Run it: [cyan]localfit run {hf_repo_id}[/]\n")

    elif action == "remote_serve":
        console.print(f"\n  [cyan]localfit run {model_repo} --remote kaggle[/]")
        console.print(f"  [cyan]localfit run {model_repo} --cloud[/]\n")


def cmd_makeitfit(model_query):
    """
    Main entry point for `localfit --makeitfit MODEL`.
    Analyzes fit, shows options, and orchestrates remote quantization.
    """
    from localfit.backends import fetch_hf_model, get_machine_specs

    console.print(f"\n  [bold cyan]localfit --makeitfit {model_query}[/]\n")
    console.print(f"  Checking your GPU and model metadata...")

    specs = get_machine_specs()
    gpu_total_mb = specs.get("gpu_total_mb", 0)
    gpu_total_gb = round(gpu_total_mb / 1024, 1)
    gpu_name = specs.get("gpu_name", "Unknown GPU")

    data = fetch_hf_model(model_query)
    gguf_files = data.get("gguf_files", []) if data else []
    model_repo = data["repo_id"] if data else model_query

    if not gguf_files:
        # No GGUFs — check if model exists as safetensors (needs quantization)
        bf16_gb = _fetch_safetensors_size(model_query)
        if bf16_gb:
            console.print(f"\n  [bold]Your GPU:[/] {gpu_name} ({gpu_total_gb}GB)")
            console.print(f"  [bold]Model:[/] {model_query}")
            console.print(f"  [bold]Full size (BF16):[/] ~{bf16_gb}GB")
            console.print(f"  [yellow]No GGUF quants exist yet — this model needs quantization[/]\n")
            return _makeitfit_no_gguf(model_query, bf16_gb, gpu_total_mb, gpu_total_gb, gpu_name, specs)
        else:
            console.print(f"\n  [red]✗ Model not found: {model_query}[/]")
            console.print(f"  [dim]Try: localfit show {model_query} to check availability[/]")
            return

    bf16_gb = _bf16_size_gb(gguf_files)

    # ── What fits locally ──
    fits = [f for f in gguf_files if f["size_gb"] * 1024 < gpu_total_mb - 1024]
    best_local = fits[-1] if fits else None

    console.print(f"\n  [bold]Your GPU:[/] {gpu_name} ({gpu_total_gb}GB)")
    console.print(f"  [bold]Model:[/] {model_repo}")
    if bf16_gb:
        console.print(f"  [bold]Full size (BF16):[/] ~{bf16_gb}GB\n")

    # ── Feasibility check ──
    feasible_tiers = _can_quantize(bf16_gb) if bf16_gb else []

    if bf16_gb and bf16_gb > 200:
        console.print(
            f"  [red]✗ Not feasible:[/] {bf16_gb}GB model requires 90GB+ VRAM just to load in 4-bit.\n"
            f"  [dim]Even an A100 80GB can't quantize this. Use a pre-quantized version instead.[/]"
        )
        # Still show what pre-quants exist
        if best_local:
            console.print(f"\n  [green]But good news — you can run it locally![/]")
            console.print(f"  Best fit: {best_local['quant']} ({best_local['size_gb']}GB)")
            console.print(f"\n  [dim]localfit run {model_query}[/]")
        return

    # ── Build options table ──
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("n", style="bold cyan", width=3)
    t.add_column("option", no_wrap=False)
    t.add_column("detail", style="dim")

    options = []

    # Option: already fits locally
    if best_local:
        options.append(("local", best_local))
        t.add_row(
            str(len(options)),
            f"[green]Run locally now[/] — {best_local['quant']} ({best_local['size_gb']}GB)",
            f"fits your {gpu_total_gb}GB GPU · no wait",
        )
    elif fits:
        pass
    else:
        # Nothing fits — check smallest quant
        smallest = min(gguf_files, key=lambda f: f["size_gb"])
        t.add_row(
            "–",
            f"[red]No quant fits your {gpu_total_gb}GB GPU[/]",
            f"smallest available: {smallest['quant']} ({smallest['size_gb']}GB)",
        )

    # Option: quantize on Kaggle (free)
    kaggle_tier = next((t2 for t2 in feasible_tiers if t2["tier"] == "kaggle"), None)
    if kaggle_tier:
        default_q = _TIER_DEFAULT_QUANT["kaggle"]
        options.append(("kaggle_quant", {"quant": default_q, "tier": kaggle_tier}))
        t.add_row(
            str(len(options)),
            f"[cyan]Quantize on Kaggle (free)[/] → {default_q.upper()} GGUF",
            f"{kaggle_tier['gpu_name']} · ~{kaggle_tier['est_minutes']} min · uploads to your HF",
        )
    elif bf16_gb and bf16_gb > 14:
        t.add_row(
            "–",
            f"[dim]Kaggle (T4 16GB) — model too large[/]",
            f"{bf16_gb}GB BF16 > T4 limit (~14GB) · use RunPod instead",
        )

    # Option: quantize on RunPod
    runpod_tier = next(
        (t2 for t2 in feasible_tiers if t2["tier"] != "kaggle"), None
    )
    if runpod_tier:
        default_q = _TIER_DEFAULT_QUANT.get(runpod_tier["tier"], "q5_k_m")
        cost_est = runpod_tier["cost_per_hr"] * (runpod_tier["est_minutes"] / 60)
        options.append(("runpod_quant", {"quant": default_q, "tier": runpod_tier}))
        t.add_row(
            str(len(options)),
            f"[magenta]Quantize on RunPod[/] → {default_q.upper()} GGUF",
            f"{runpod_tier['gpu_name']} · ~{runpod_tier['est_minutes']} min · ~${cost_est:.2f}",
        )
    elif bf16_gb and bf16_gb > 110:
        t.add_row(
            "–",
            f"[red]✗ Not feasible on any single GPU[/]",
            f"{bf16_gb}GB model — too large even for A100 80GB",
        )

    # Option: serve remotely (existing feature)
    options.append(("remote_serve", {}))
    t.add_row(
        str(len(options)),
        "[yellow]Serve remotely[/] (don't quantize, stream from cloud)",
        "localfit run --remote kaggle / --cloud runpod",
    )

    console.print(t)
    console.print()

    if len([o for o in options if o[0] != "remote_serve"]) == 0:
        console.print(
            f"  [red]This model cannot be made to fit on your hardware.[/]\n"
            f"  [dim]Use remote serving instead.[/]"
        )
        return

    # ── User picks ──
    try:
        choice = input("  Pick option: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(options):
            raise ValueError
    except ValueError:
        console.print(f"  [red]Invalid choice[/]")
        return

    action, meta = options[idx]

    if action == "local":
        console.print(f"\n  [green]Running locally...[/]")
        os.execlp("localfit", "localfit", "run", model_query)

    elif action in ("kaggle_quant", "runpod_quant"):
        # Ask which quant method
        console.print(f"\n  [bold]Choose quantization type:[/]\n")
        for i, (method, label, _) in enumerate(QUANT_OPTIONS, 1):
            star = " [bold cyan]← recommended[/]" if method == meta["quant"] else ""
            console.print(f"  [bold cyan]{i}[/]  {label}{star}")
        console.print()
        try:
            q_choice = input("  Pick quant [1-6, default=4]: ").strip() or "4"
            q_idx = int(q_choice) - 1
            quant_method = QUANT_OPTIONS[q_idx][0]
        except (ValueError, IndexError, EOFError, KeyboardInterrupt):
            quant_method = meta["quant"]

        # Get HF token
        hf_token = _get_hf_token()
        if not hf_token:
            console.print(
                f"\n  [yellow]HuggingFace write token needed to upload your quant.[/]"
            )
            console.print(
                f"  Get one at: https://huggingface.co/settings/tokens (write access)\n"
            )
            try:
                hf_token = input("  Paste HF token: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not hf_token:
                return
            # Save for next time
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            except Exception:
                tok_path = Path.home() / ".cache" / "huggingface" / "token"
                tok_path.parent.mkdir(parents=True, exist_ok=True)
                tok_path.write_text(hf_token)

        hf_username = _get_hf_username(hf_token)
        if not hf_username:
            console.print(f"  [red]Invalid HF token (couldn't get username)[/]")
            return

        # Build HF repo name
        model_base = re.sub(r"[^a-zA-Z0-9-]", "-", model_repo.split("/")[-1])
        hf_repo_id = f"{hf_username}/{model_base}-{quant_method.upper()}-GGUF-localfit"
        console.print(f"\n  [dim]Will upload to:[/] [cyan]{hf_repo_id}[/]\n")

        if action == "kaggle_quant":
            result = quantize_on_kaggle(model_repo, quant_method, hf_token, hf_repo_id)
        else:
            result = quantize_on_runpod(model_repo, quant_method, hf_token, hf_repo_id, bf16_gb)

        if result:
            console.print(f"\n  [bold green]✓ Done![/]")
            console.print(f"  Your model is live at: [cyan]https://huggingface.co/{hf_repo_id}[/]")
            console.print(f"\n  [bold]Run it now:[/]")
            console.print(f"  [cyan]localfit run {hf_repo_id}[/]\n")

            try:
                run_now = input("  Run it now? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if run_now == "y":
                os.execlp("localfit", "localfit", "run", hf_repo_id)

    elif action == "remote_serve":
        console.print(f"\n  For Kaggle (free 30h/week):")
        console.print(f"  [cyan]localfit run {model_query} --remote kaggle[/]")
        console.print(f"\n  For RunPod (paid, any GPU):")
        console.print(f"  [cyan]localfit run {model_query} --cloud[/]\n")
