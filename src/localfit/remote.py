"""localfit remote — serve models on Kaggle/RunPod GPUs with Cloudflare tunnel.

Usage:
    localfit serve gemma4:e4b --remote kaggle       # free T4 GPU + tunnel
    localfit serve qwen3:14b --remote kaggle        # free T4/P100 16GB
    localfit serve gemma4:27b --remote kaggle       # auto-picks T4x2 (32GB)
    localfit serve llama3:70b --remote runpod       # paid cloud GPU

Kaggle free GPUs:
    T4 x1  — 16GB VRAM  (~14GB usable)  — models up to ~13GB
    T4 x2  — 32GB VRAM  (~28GB usable)  — models up to ~27GB
    P100 x1 — 16GB VRAM (~14GB usable)  — models up to ~13GB
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from pathlib import Path
from rich.console import Console

console = Console()
CONFIG_DIR = Path.home() / ".localfit"
KAGGLE_STATE_FILE = CONFIG_DIR / "active_kaggle.json"
KAGGLE_QUOTA_FILE = CONFIG_DIR / "kaggle_quota.json"

# Default auto-stop: 10 min for testing (saves the 30h quota)
DEFAULT_DURATION_MINUTES = 10
KAGGLE_WEEKLY_QUOTA_HOURS = 30


def _get_quota_usage():
    """Get tracked Kaggle GPU quota usage this week."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if KAGGLE_QUOTA_FILE.exists():
        try:
            data = json.loads(KAGGLE_QUOTA_FILE.read_text())
            # Reset if older than 7 days
            if time.time() - data.get("week_start", 0) > 7 * 86400:
                data = {"week_start": time.time(), "used_minutes": 0, "sessions": []}
            return {"used_hours": data.get("used_minutes", 0) / 60, "data": data}
        except Exception:
            pass
    return {
        "used_hours": 0,
        "data": {"week_start": time.time(), "used_minutes": 0, "sessions": []},
    }


def _record_quota_usage(minutes):
    """Record GPU minutes used."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _get_quota_usage()["data"]
    data["used_minutes"] = data.get("used_minutes", 0) + minutes
    data["sessions"] = data.get("sessions", []) + [
        {"time": time.time(), "minutes": minutes}
    ]
    KAGGLE_QUOTA_FILE.write_text(json.dumps(data, indent=2))


# Kaggle free GPU tiers
# Kaggle API assigns P100 or T4 (both 16GB). T4x2 can't be requested (issue #821).
# Ollama auto-offloads to CPU when model > VRAM, so we allow up to ~20GB
# (16GB GPU + partial CPU offload). Slower but runs.
KAGGLE_GPUS = [
    {
        "name": "T4",
        "vram_gb": 16,
        "usable_gb": 14,  # full speed (fits entirely in VRAM)
        "max_gb": 20,  # with CPU offload (slower but works)
        "accelerator": "NvidiaTeslaT4",  # correct kagglesdk machine_shape value
        "count": 1,
        "reliable": True,
    },
    {
        "name": "T4x2",
        "vram_gb": 32,
        "usable_gb": 28,
        "max_gb": 32,
        "accelerator": "NvidiaTeslaT4",  # T4x2 can't be requested via API (issue #821) — uses T4 value, Kaggle may assign x2
        "count": 2,
        "reliable": False,  # Kaggle API can't guarantee T4x2
    },
    {
        "name": "P100",
        "vram_gb": 16,
        "usable_gb": 14,
        "max_gb": 20,
        "accelerator": "NvidiaTeslaP100",  # correct kagglesdk machine_shape value
        "count": 1,
        "reliable": True,
    },
]


def _pick_kaggle_gpu(model_size_gb):
    """Pick the best Kaggle GPU that fits the model.

    Uses usable_gb for full-speed fit, max_gb for partial CPU offload.

    Returns:
        dict with GPU info + "offload" flag, or None if nothing fits.
    """
    for gpu in KAGGLE_GPUS:
        if not gpu.get("reliable", True):
            continue
        if model_size_gb <= gpu["usable_gb"]:
            return {**gpu, "offload": False}
        if model_size_gb <= gpu.get("max_gb", gpu["usable_gb"]):
            return {**gpu, "offload": True}
    return None


def kaggle_fits(model_size_gb):
    """Check if a single model size fits any Kaggle GPU.

    Returns:
        dict: {"fits": bool, "gpu": str, "vram_gb": int, ...} or {"fits": False}
    """
    gpu = _pick_kaggle_gpu(model_size_gb)
    if gpu:
        return {
            "fits": True,
            "gpu": gpu["name"],
            "vram_gb": gpu["vram_gb"],
            "usable_gb": gpu["usable_gb"],
            "accelerator": gpu["accelerator"],
        }
    return {"fits": False}


def kaggle_check_model(gguf_files, mmproj_files=None):
    """Smart Kaggle fit check for a full model with all its quants.

    Takes the actual GGUF file list from fetch_hf_model() and returns
    the best option per Kaggle GPU tier, or explains why it won't work.

    Args:
        gguf_files: List of {"filename", "size_gb", "quant"} from fetch_hf_model()
        mmproj_files: Optional list of mmproj files (VLM vision projectors)

    Returns:
        dict with:
            "fits_kaggle": bool — at least one quant fits some Kaggle GPU
            "options": list of {"gpu": {...}, "quant": {...}, "total_gb": float}
                       sorted best-first (highest quality, smallest GPU)
            "best": the recommended option (or None)
            "reason": str — why it doesn't fit (if fits_kaggle=False)
    """
    if not gguf_files:
        return {
            "fits_kaggle": False,
            "options": [],
            "best": None,
            "reason": "No GGUF files found",
        }

    # mmproj overhead (VLM models need ~0.8-1.3GB extra for vision projector)
    mmproj_gb = 0
    mmproj_file = None
    if mmproj_files:
        # Pick smallest mmproj (usually q8_0 version)
        mmproj_files_sorted = sorted(mmproj_files, key=lambda x: x.get("size_gb", 0))
        mmproj_file = mmproj_files_sorted[0]
        mmproj_gb = mmproj_file.get("size_gb", 0.8)

    options = []
    for gpu in KAGGLE_GPUS:
        if not gpu.get("reliable", True):
            continue
        available_vram = gpu["usable_gb"] - mmproj_gb
        max_vram = gpu.get("max_gb", gpu["usable_gb"]) - mmproj_gb

        # Full speed: fits entirely in VRAM
        fits_fast = [f for f in gguf_files if f["size_gb"] <= available_vram]
        # Partial CPU offload: slower but runs
        fits_offload = [
            f
            for f in gguf_files
            if f["size_gb"] <= max_vram and f["size_gb"] > available_vram
        ]

        if fits_fast:
            best_quant = fits_fast[-1]
            total = best_quant["size_gb"] + mmproj_gb
            options.append(
                {
                    "gpu": gpu,
                    "quant": best_quant,
                    "mmproj": mmproj_file,
                    "total_gb": round(total, 1),
                    "headroom_gb": round(gpu["usable_gb"] - total, 1),
                    "offload": False,
                }
            )
        elif fits_offload:
            best_quant = fits_offload[-1]
            total = best_quant["size_gb"] + mmproj_gb
            options.append(
                {
                    "gpu": gpu,
                    "quant": best_quant,
                    "mmproj": mmproj_file,
                    "total_gb": round(total, 1),
                    "headroom_gb": round(max_vram + mmproj_gb - total, 1),
                    "offload": True,
                }
            )

    if not options:
        smallest = gguf_files[0]
        smallest_total = smallest["size_gb"] + mmproj_gb
        return {
            "fits_kaggle": False,
            "options": [],
            "best": None,
            "reason": (
                f"Smallest quant ({smallest['quant']}) = {smallest_total:.1f}GB "
                f"(model {smallest['size_gb']}GB"
                f"{f' + mmproj {mmproj_gb:.1f}GB' if mmproj_gb else ''})"
                f" > max Kaggle GPU ({KAGGLE_GPUS[-1]['usable_gb']}GB usable on "
                f"{KAGGLE_GPUS[-1]['name']})"
            ),
        }

    # Best = first option (T4 preferred over T4x2, already sorted by GPU preference)
    best = options[0]

    # But check if T4x2 gives significantly better quality
    if len(options) > 1:
        t4_quality = options[0]["quant"]["size_gb"]
        t4x2_quality = (
            options[1]["quant"]["size_gb"] if options[1]["gpu"]["name"] == "T4x2" else 0
        )
        # If T4x2 gives >30% bigger quant, recommend it instead
        if t4x2_quality > t4_quality * 1.3:
            best = options[1]

    return {
        "fits_kaggle": True,
        "options": options,
        "best": best,
        "reason": None,
    }


def _check_kaggle_cli():
    """Check if kaggle CLI is installed and configured."""
    import shutil

    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        return {"found": False, "reason": "not_installed"}

    # Check for credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text())
            if data.get("username") and data.get("key"):
                return {"found": True, "reason": "ok", "path": kaggle_bin}
        except Exception:
            pass

    # Also check env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return {"found": True, "reason": "ok", "path": kaggle_bin}

    return {"found": True, "reason": "no_credentials", "path": kaggle_bin}


def _save_kaggle_key(username, key):
    """Save Kaggle credentials to ~/.kaggle/kaggle.json."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": username, "key": key}, indent=2))
    kaggle_json.chmod(0o600)
    return kaggle_json


def _validate_kaggle_key(key):
    """Validate Kaggle API key format. Returns True if looks valid."""
    # Kaggle keys: hex string, sometimes prefixed with KGAT_
    if not key:
        return False
    # Strip common prefixes/whitespace
    key = key.strip()
    if len(key) < 20:
        return False
    return True


def _validate_kaggle_credentials(username, key):
    """Validate credentials by making a test API call."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "--max-size", "1", "-v"],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "KAGGLE_USERNAME": username, "KAGGLE_KEY": key},
        )
        # If it doesn't error with 401/403, creds are good
        if result.returncode == 0:
            return True
        if (
            "401" in result.stderr
            or "403" in result.stderr
            or "unauthorized" in result.stderr.lower()
        ):
            return False
        # Some other error — creds might still be fine
        return True
    except Exception:
        return True  # can't validate, assume ok


def save_kaggle_credentials():
    """Interactive credential setup for Kaggle. Called by `localfit login kaggle`."""
    import shutil

    console.print(f"\n  [bold]Kaggle API Setup[/]")
    console.print(f"  [dim]You need a Legacy API Key (not the new KGAT_ tokens).[/]")
    console.print()
    console.print(f"  [bold]Steps:[/]")
    console.print(f"  1. Go to [cyan]https://www.kaggle.com/settings[/]")
    console.print(f'  2. Scroll to [bold]"Legacy API Credentials"[/]')
    console.print(f'  3. Click [bold]"Create Legacy API Key"[/]')
    console.print(f"  4. Open the downloaded kaggle.json\n")

    # Check if kaggle CLI is installed
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        console.print(f"  [dim]Installing kaggle CLI...[/]")
        try:
            pipx_bin = shutil.which("pipx")
            if pipx_bin:
                result = subprocess.run(
                    [pipx_bin, "install", "kaggle"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    console.print(f"  [green]✓[/] kaggle CLI installed via pipx")
                else:
                    console.print(f"  [red]Install failed. Run: pipx install kaggle[/]")
                    return False
            else:
                console.print(
                    f"  [red]kaggle CLI not found. Install: pipx install kaggle[/]"
                )
                return False
        except Exception as e:
            console.print(f"  [red]Install failed: {e}[/]")
            return False
    else:
        console.print(f"  [green]✓[/] kaggle CLI found")

    # Check existing credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text())
            if data.get("username") and data.get("key"):
                key_preview = (
                    data["key"][:8] + "..." if len(data.get("key", "")) > 8 else "***"
                )
                is_legacy = not data["key"].startswith("KGAT_")
                status = (
                    "[green]legacy key[/]"
                    if is_legacy
                    else "[yellow]KGAT token (may not work)[/]"
                )
                console.print(
                    f"  Existing: {data['username']} · {key_preview} · {status}"
                )
                try:
                    ans = input("  Overwrite? (y/N): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    return True
                if ans not in ("y", "yes"):
                    return True
        except Exception:
            pass

    # Prompt — accept: JSON paste, username+key, or path to kaggle.json
    console.print(f"\n  [bold]Paste credentials:[/]")
    console.print(
        f"  [dim]Paste the JSON from kaggle.json, or enter username + key[/]\n"
    )

    try:
        raw = input("  Paste JSON or username: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False

    username = None
    key = None

    if raw.startswith("{"):
        # Full JSON: {"username":"xxx","key":"xxx"}
        try:
            data = json.loads(raw)
            username = data.get("username")
            key = data.get("key")
        except json.JSONDecodeError:
            console.print(f"  [red]Invalid JSON.[/]")
            return False
    elif raw.startswith("KGAT_"):
        console.print(
            f"\n  [red]That's a new-style KGAT_ token — it won't work for notebook push.[/]"
        )
        console.print(f"  [bold]You need the Legacy API Key:[/]")
        console.print(f'    https://www.kaggle.com/settings → "Create Legacy API Key"')
        console.print(f"    It downloads a kaggle.json — paste its contents here.\n")
        return False
    else:
        username = raw
        try:
            key = input("  API key (from kaggle.json): ").strip()
        except (EOFError, KeyboardInterrupt):
            return False

        if key and key.startswith("KGAT_"):
            console.print(f"\n  [red]That's a KGAT_ token, not a legacy key.[/]")
            console.print(
                f'  [bold]→ https://www.kaggle.com/settings → "Create Legacy API Key"[/]\n'
            )
            return False

    if not username or not key:
        console.print(f"  [red]Username and key are required.[/]")
        return False

    if not _validate_kaggle_key(key):
        console.print(
            f"  [red]Key looks invalid (too short). Check your kaggle.json download.[/]"
        )
        return False

    # Save
    path = _save_kaggle_key(username, key)
    console.print(f"  [green]✓[/] Saved to {path}")

    # Quick validation
    console.print(f"  [dim]Validating...[/]")
    if _validate_kaggle_credentials(username, key):
        console.print(
            f"  [green]✓[/] Credentials valid! Ready for: localfit --serve MODEL --remote kaggle\n"
        )
    else:
        console.print(
            f"  [yellow]Could not validate credentials. They may still work.[/]"
        )
        console.print(f"  [dim]Check: https://www.kaggle.com/settings → API[/]\n")

    return True


def _ensure_kaggle_credentials():
    """Ensure Kaggle credentials exist. Prompt inline if missing.

    Returns True if credentials are available, False if user cancelled.
    """
    kaggle = _check_kaggle_cli()

    if not kaggle["found"]:
        console.print(f"\n  [yellow]Kaggle CLI not found. Installing...[/]")
        import shutil

        pipx_bin = shutil.which("pipx")
        if pipx_bin:
            result = subprocess.run(
                [pipx_bin, "install", "kaggle"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                console.print(f"  [red]Failed. Run: pipx install kaggle[/]")
                return False
            console.print(f"  [green]✓[/] kaggle CLI installed")
        else:
            console.print(
                f"  [red]Need pipx or kaggle CLI. Run: pipx install kaggle[/]"
            )
            return False

    if kaggle.get("reason") == "no_credentials" or not kaggle["found"]:
        # Inline credential prompt — no need to run a separate login command
        console.print(f"\n  [bold]Kaggle API key needed[/] (one-time setup)")
        console.print(
            f"  [dim]Get yours: https://www.kaggle.com/settings → API → Create New Token[/]\n"
        )

        try:
            raw = input("  Kaggle username: ").strip()
        except (EOFError, KeyboardInterrupt):
            return False

        username = None
        key = None

        if raw.startswith("{"):
            try:
                data = json.loads(raw)
                username = data.get("username")
                key = data.get("key")
            except json.JSONDecodeError:
                console.print(f"  [red]Invalid JSON.[/]")
                return False
        else:
            username = raw
            try:
                key = input("  Kaggle API key: ").strip()
            except (EOFError, KeyboardInterrupt):
                return False

        if not username or not _validate_kaggle_key(key):
            console.print(f"  [red]Invalid credentials.[/]")
            return False

        path = _save_kaggle_key(username, key)
        console.print(f"  [green]✓[/] Saved to {path}\n")

    return True


def _get_kaggle_username():
    """Get Kaggle username from credentials."""
    # Check env first
    user = os.environ.get("KAGGLE_USERNAME")
    if user:
        return user

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text())
            return data.get("username")
        except Exception:
            pass
    return None


def _get_cloudflare_token():
    """Get Cloudflare tunnel token if configured (for stable domain)."""
    # Check env
    token = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN")
    if token:
        return token

    # Check config file
    cf_file = CONFIG_DIR / "cloudflare_token"
    if cf_file.exists():
        return cf_file.read_text().strip()
    return None


def _generate_notebook(
    model_query,
    model_repo,
    model_filename,
    kaggle_gpu,
    cf_token=None,
    mmproj_filename=None,
    max_runtime_minutes=None,
):
    """Generate a Kaggle notebook that serves any GGUF model via Ollama + Cloudflare.

    Works for ANY GGUF from HuggingFace — downloads the file, imports into Ollama
    via Modelfile, serves via Cloudflare tunnel. VLM models get PROJECTOR line.
    """
    # Build Modelfile content
    # Note: Ollama doesn't support PROJECTOR for VLM mmproj files
    # VLM models will work for text queries; vision requires llama-server
    modelfile_lines = [f"FROM /tmp/model.gguf"]
    modelfile_content = "\\n".join(modelfile_lines)

    script = f'''#!/usr/bin/env python3
"""LocalFit remote serve — auto-generated for Kaggle.

Model: {model_repo}/{model_filename}
{"VLM: " + mmproj_filename if mmproj_filename else ""}
GPU: Kaggle {kaggle_gpu["name"]} ({kaggle_gpu["vram_gb"]}GB VRAM)
Generated by: localfit serve {model_query} --remote kaggle
"""
import subprocess, sys, time, os, re

def run_cmd(cmd, **kwargs):
    if isinstance(cmd, str):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0:
        print(f"LOCALFIT_ERROR={{r.stderr[:500]}}")
        sys.stdout.flush()
    return r

# ── 1. Install Ollama + Cloudflare ──
print("LOCALFIT_STATUS=installing")
sys.stdout.flush()

# Install zstd (required by Ollama installer) + Ollama + cloudflared
os.system("apt-get update -qq && apt-get install -y -qq zstd pciutils lsof > /dev/null 2>&1")
os.system("curl -fsSL https://ollama.com/install.sh | sh")
os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i cloudflared-linux-amd64.deb")

# Ensure ollama is on PATH
os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")

# ── 2. Detect GPU ──
gpu_check = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
if gpu_check.returncode == 0:
    print(f"LOCALFIT_GPU={{gpu_check.stdout.strip()}}")
else:
    print("LOCALFIT_GPU=none")
sys.stdout.flush()

# ── 3. Start Ollama ──
print("LOCALFIT_STATUS=starting_ollama")
sys.stdout.flush()

ollama_log = open("/tmp/ollama.log", "w")
ollama_proc = subprocess.Popen(
    ["/usr/local/bin/ollama", "serve"],
    stdout=ollama_log, stderr=subprocess.STDOUT,
    env={{**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"}},
)

for i in range(30):
    time.sleep(1)
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        break
    except Exception:
        if i == 29:
            print("LOCALFIT_ERROR=ollama_timeout")
            sys.stdout.flush()
            sys.exit(1)

# ── 4. Download model from HuggingFace ──
print("LOCALFIT_STATUS=downloading_{model_filename}")
sys.stdout.flush()

run_cmd("pip install huggingface_hub -q")
from huggingface_hub import hf_hub_download

model_path = hf_hub_download("{model_repo}", "{model_filename}")
print(f"LOCALFIT_MODEL_PATH={{model_path}}")
# Symlink to /tmp for Modelfile
os.symlink(model_path, "/tmp/model.gguf")
{
        f"""
# ── 4b. Download vision projector (VLM) ──
print("LOCALFIT_STATUS=downloading_mmproj")
sys.stdout.flush()
mmproj_path = hf_hub_download("{model_repo}", "{mmproj_filename}")
os.symlink(mmproj_path, "/tmp/mmproj.gguf")
print(f"LOCALFIT_MMPROJ={{mmproj_path}}")
sys.stdout.flush()
"""
        if mmproj_filename
        else ""
    }

# ── 5. Import GGUF into Ollama via Modelfile ──
print("LOCALFIT_STATUS=importing_model")
sys.stdout.flush()

with open("/tmp/Modelfile", "w") as f:
    f.write("{modelfile_content}\\n")

r = run_cmd("/usr/local/bin/ollama create localmodel -f /tmp/Modelfile", timeout=900)
if r.returncode != 0:
    print(f"LOCALFIT_ERROR=ollama_create_failed:{{r.stderr[:300]}}")
    sys.stdout.flush()
    sys.exit(1)

print("LOCALFIT_STATUS=model_ready")
sys.stdout.flush()

# ── 6. Start Cloudflare tunnel ──
print("LOCALFIT_STATUS=starting_tunnel")
sys.stdout.flush()

tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:11434"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

endpoint = None
deadline = time.time() + 30
while time.time() < deadline:
    line = tunnel_proc.stderr.readline().decode(errors="replace")
    if "trycloudflare.com" in line:
        m = re.search(r"(https://[\\w-]+\\.trycloudflare\\.com)", line)
        if m:
            endpoint = m.group(1)
            break

if endpoint:
    print(f"LOCALFIT_ENDPOINT={{endpoint}}")
    print(f"LOCALFIT_STATUS=serving")
    with open("/kaggle/working/endpoint.txt", "w") as ef:
        ef.write(endpoint)
    # Notify localfit poller via ntfy.sh (instant, no Kaggle API delay)
    try:
        import urllib.request as _ur
        _ntfy_topic = "localfit-" + "{model_query}".replace("/", "-").replace(":", "-").replace(" ", "-")[:40]
        _ur.urlopen(f"https://ntfy.sh/{{_ntfy_topic}}", data=endpoint.encode(), timeout=5)
        print(f"LOCALFIT_NTFY_SENT={{_ntfy_topic}}")
    except Exception as _ntfy_err:
        print(f"LOCALFIT_NTFY_FAILED={{_ntfy_err}}")
        pass
    sys.stdout.flush()

    print(f"")
    print(f"========================================")
    print(f"  Model ready at: {{endpoint}}")
    print(f"========================================")
    print("")
    print("Test:")
    body = '{{"model": "localmodel", "messages": [{{"role": "user", "content": "hello"}}]}}'
    print(f"  curl {{endpoint}}/v1/chat/completions -H 'Content-Type: application/json' -d '" + body + "'")
    sys.stdout.flush()
else:
    print("LOCALFIT_ERROR=tunnel_failed")
    sys.stdout.flush()
    sys.exit(1)

# ── 7. Keep alive ──
{
        f"max_runtime = {max_runtime_minutes} * 60"
        if max_runtime_minutes
        else "max_runtime = 12 * 3600"
    }
start_time = time.time()
try:
    while True:
        time.sleep(30)
        elapsed = time.time() - start_time
        if elapsed >= max_runtime:
            print(f"LOCALFIT_STATUS=auto_stopped_after_{{int(elapsed/60)}}min")
            sys.stdout.flush()
            break
        if tunnel_proc.poll() is not None:
            print("LOCALFIT_STATUS=tunnel_died_restarting")
            sys.stdout.flush()
            tunnel_proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", "http://localhost:11434"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        if ollama_proc.poll() is not None:
            print("LOCALFIT_STATUS=ollama_died")
            sys.stdout.flush()
            break
except KeyboardInterrupt:
    pass

try:
    tunnel_proc.terminate()
    ollama_proc.terminate()
except Exception:
    pass
'''
    return script
    gpu_count = kaggle_gpu["count"]
    # For multi-GPU: use --tensor-split to spread across GPUs
    ngl_flag = "-ngl 99"
    extra_flags = ""
    if gpu_count > 1:
        # Equal split across GPUs
        split = ",".join(["1"] * gpu_count)
        extra_flags = f"--tensor-split {split}"

    # Tunnel setup: free trycloudflare.com or Zero Trust with token
    if cf_token:
        tunnel_cmd = f'["cloudflared", "service", "install", "{cf_token}"]'
        tunnel_section = f'''
# ── 6. Start Cloudflare Tunnel (Zero Trust — stable domain) ──
print("LOCALFIT_STATUS=starting_tunnel")
sys.stdout.flush()

# Install as service with your token — routes to your configured domain
run_cmd(["cloudflared", "service", "install", "{cf_token}"])
run_cmd("systemctl start cloudflared")
run_cmd("systemctl enable cloudflared")

# Verify it's running
time.sleep(5)
status = subprocess.run(["systemctl", "is-active", "cloudflared"],
                        capture_output=True, text=True)
if status.stdout.strip() == "active":
    print("LOCALFIT_STATUS=serving")
    print("LOCALFIT_ENDPOINT=cloudflare-zero-trust")
    print("")
    print("Tunnel active! Your model is available at your configured Cloudflare domain.")
    print("Check your Cloudflare Zero Trust dashboard for the URL.")
    sys.stdout.flush()
else:
    print("LOCALFIT_ERROR=cloudflare_service_failed")
    sys.stdout.flush()
    sys.exit(1)
'''
    else:
        tunnel_section = """
# ── 6. Start Cloudflare Tunnel (free trycloudflare.com) ──
print("LOCALFIT_STATUS=starting_tunnel")
sys.stdout.flush()

tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8089"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

# Read tunnel URL from cloudflared stderr
endpoint = None
deadline = time.time() + 30
while time.time() < deadline:
    line = tunnel_proc.stderr.readline().decode(errors="replace")
    if "trycloudflare.com" in line:
        m = re.search(r"(https://[\\w-]+\\.trycloudflare\\.com)", line)
        if m:
            endpoint = m.group(1)
            break

if endpoint:
    print(f"LOCALFIT_ENDPOINT={endpoint}")
    print(f"LOCALFIT_STATUS=serving")
    with open("/kaggle/working/endpoint.txt", "w") as ef:
        ef.write(endpoint)
    try:
        import urllib.request as _ur
        _ntfy_topic = "localfit-" + "{model_query}".replace("/", "-").replace(":", "-").replace(" ", "-")[:40]
        _ur.urlopen(f"https://ntfy.sh/{{_ntfy_topic}}", data=endpoint.encode(), timeout=5)
        print(f"LOCALFIT_NTFY_SENT={{_ntfy_topic}}")
    except Exception as _ntfy_err:
        print(f"LOCALFIT_NTFY_FAILED={{_ntfy_err}}")
        pass
    sys.stdout.flush()

    print(f"")
    print(f"========================================")
    print(f"  Model ready at: {endpoint}")
    print(f"========================================")
    print(f"")
    print(f"Use with Claude Code:")
    print(f"  python -m localfit.proxy --port 8090 --llama-url {endpoint}/v1/chat/completions &\\\\")
    print(f"  ANTHROPIC_AUTH_TOKEN=localfit \\\\")
    print(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:8090 \\\\")
    print(f"  ANTHROPIC_API_KEY= \\\\")
    print(f"  claude --bare --model local")
    print(f"")
    print(f"Use with curl:")
    print(f"  curl {endpoint}/v1/chat/completions \\\\")
    print(f"    -H \\"Content-Type: application/json\\" \\\\")
    print(f\'    -d \\\'{"model": "local", "messages": [{"role": "user", "content": "hello"}]}\\\'\\n\')
    sys.stdout.flush()
else:
    print("LOCALFIT_ERROR=tunnel_failed")
    sys.stdout.flush()
    sys.exit(1)
"""

    script = f'''#!/usr/bin/env python3
"""LocalFit remote serve — auto-generated for Kaggle.

Model: {model_repo}/{model_filename}
GPU: Kaggle {kaggle_gpu["name"]} ({kaggle_gpu["vram_gb"]}GB VRAM)
Generated by: localfit serve {model_query} --remote kaggle
"""
import subprocess, sys, time, os, re

def run_cmd(cmd, **kwargs):
    """Run a command, print output on failure."""
    if isinstance(cmd, str):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0:
        print(f"LOCALFIT_ERROR={{r.stderr[:500]}}")
        sys.stdout.flush()
    return r

# ── 1. Install dependencies ──
print("LOCALFIT_STATUS=installing")
sys.stdout.flush()

run_cmd("apt-get update -qq && apt-get install -y -qq cmake build-essential zstd pciutils lsof > /dev/null 2>&1")
run_cmd("pip install huggingface_hub -q")
run_cmd("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i cloudflared-linux-amd64.deb")

# ── 2. Detect GPU ──
gpu_check = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
if gpu_check.returncode == 0:
    gpu_info = gpu_check.stdout.strip()
    print(f"LOCALFIT_GPU={{gpu_info}}")
    gpu_count = len(gpu_info.strip().split("\\n"))
    print(f"LOCALFIT_GPU_COUNT={{gpu_count}}")
else:
    print("LOCALFIT_GPU=none")
    gpu_count = 0
sys.stdout.flush()

# ── 3. Install llama-cpp-python with CUDA (pre-built wheels, no build needed) ──
print("LOCALFIT_STATUS=installing_llama_cpp")
sys.stdout.flush()

# Use cu121 wheels — backward compatible with CUDA 12.x (12.1-12.8+)
# Don't use --force-reinstall or --no-cache-dir (causes source build = slow)
r = run_cmd(
    "pip install 'llama-cpp-python[server]' "
    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121",
    timeout=180,
)
if r.returncode != 0:
    # Fallback: try cu124
    r = run_cmd(
        "pip install 'llama-cpp-python[server]' "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124",
        timeout=180,
    )
if r.returncode != 0:
    # Last resort: CPU-only (will work but slow)
    run_cmd("pip install 'llama-cpp-python[server]'", timeout=180)

# Verify
try:
    import llama_cpp
    print(f"LOCALFIT_STATUS=llama_cpp_ready")
except ImportError:
    print("LOCALFIT_ERROR=llama_cpp_install_failed")
    sys.stdout.flush()
    sys.exit(1)
sys.stdout.flush()

# ── 4. Download model ──
print("LOCALFIT_STATUS=downloading_{model_filename}")
sys.stdout.flush()

from huggingface_hub import hf_hub_download
model_path = hf_hub_download("{model_repo}", "{model_filename}")
print(f"LOCALFIT_MODEL_PATH={{model_path}}")
sys.stdout.flush()
{
        f"""
# ── 4b. Download vision projector (VLM) ──
print("LOCALFIT_STATUS=downloading_mmproj")
sys.stdout.flush()
mmproj_path = hf_hub_download("{model_repo}", "{mmproj_filename}")
print(f"LOCALFIT_MMPROJ_PATH={{mmproj_path}}")
sys.stdout.flush()
"""
        if mmproj_filename
        else """
mmproj_path = None
"""
    }
# ── 5. Start llama-cpp-python server (OpenAI-compatible API) ──
print("LOCALFIT_STATUS=starting_server")
sys.stdout.flush()

server_cmd = [
    sys.executable, "-m", "llama_cpp.server",
    "--model", model_path,
    "--port", "8089",
    "--host", "0.0.0.0",
    "--n_gpu_layers", "-1",
    "--n_ctx", "32768",
]
# VLM: add vision projector (clip model)
if mmproj_path:
    server_cmd.extend(["--clip_model_path", mmproj_path])

server_log = open("/tmp/llama_server.log", "w")
server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)

# Wait for server ready (model loading can take 1-3 min)
# llama-cpp-python server uses /v1/models, not /health
for i in range(180):
    time.sleep(2)
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:8089/v1/models", timeout=2)
        data = resp.read().decode()
        if resp.status == 200:
            print("LOCALFIT_STATUS=model_ready")
            sys.stdout.flush()
            break
    except Exception:
        if i == 179:
            try:
                with open("/tmp/llama_server.log") as f:
                    lines = f.readlines()
                    print(f"LOCALFIT_ERROR=server_timeout:{{' '.join(lines[-5:])}}")
            except Exception:
                print("LOCALFIT_ERROR=server_timeout")
            sys.stdout.flush()
            sys.exit(1)
{tunnel_section}
# ── 7. Keep alive ──
# Kaggle kills the kernel when the script exits (12h max).
{
        f"max_runtime = {max_runtime_minutes} * 60  # auto-stop after {max_runtime_minutes} min"
        if max_runtime_minutes
        else "max_runtime = 12 * 3600  # 12h Kaggle limit"
    }
start_time = time.time()
try:
    while True:
        time.sleep(30)
        elapsed = time.time() - start_time
        if elapsed >= max_runtime:
            print(f"LOCALFIT_STATUS=auto_stopped_after_{{int(elapsed/60)}}min")
            sys.stdout.flush()
            break
        # Health check — restart tunnel if it dies
        if 'tunnel_proc' in dir() and tunnel_proc.poll() is not None:
            print("LOCALFIT_STATUS=tunnel_died_restarting")
            sys.stdout.flush()
            tunnel_proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", "http://localhost:8089"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            time.sleep(5)
            # Re-read URL
            for line_bytes in tunnel_proc.stderr:
                line = line_bytes.decode(errors="replace")
                if "trycloudflare.com" in line:
                    m = re.search(r"(https://[\\w-]+\\.trycloudflare\\.com)", line)
                    if m:
                        print(f"LOCALFIT_ENDPOINT={{m.group(1)}}")
                        print("LOCALFIT_STATUS=serving")
                        sys.stdout.flush()
                    break

        if server_proc.poll() is not None:
            print("LOCALFIT_STATUS=server_died")
            sys.stdout.flush()
            break
except KeyboardInterrupt:
    pass

# Cleanup
try:
    if 'tunnel_proc' in dir():
        tunnel_proc.terminate()
    server_proc.terminate()
except Exception:
    pass
'''
    return script


def _generate_notebook_ollama(
    model_query, kaggle_gpu, cf_token=None, max_runtime_minutes=None
):
    """Generate notebook using Ollama backend (fallback when no GGUF repo found)."""
    ollama_tag = model_query.replace("--", "-").strip()

    # Tunnel section (same free trycloudflare approach)
    tunnel_port = "11434"

    script = f'''#!/usr/bin/env python3
"""LocalFit remote serve — auto-generated for Kaggle.

Model: {ollama_tag} (via Ollama)
GPU: Kaggle {kaggle_gpu["name"]} ({kaggle_gpu["vram_gb"]}GB VRAM)
Generated by: localfit serve {model_query} --remote kaggle
"""
import subprocess, sys, time, os, re

def run_cmd(cmd, **kwargs):
    """Run a command, print output on failure."""
    if isinstance(cmd, str):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0:
        print(f"LOCALFIT_ERROR={{r.stderr[:500]}}")
        sys.stdout.flush()
    return r

# ── 1. Install dependencies ──
print("LOCALFIT_STATUS=installing")
sys.stdout.flush()

os.system("apt-get update -qq && apt-get install -y -qq zstd pciutils lsof > /dev/null 2>&1")
os.system("curl -fsSL https://ollama.com/install.sh | sh")
os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i cloudflared-linux-amd64.deb")
os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")

# ── 2. Detect GPU ──
gpu_check = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True)
if gpu_check.returncode == 0:
    print(f"LOCALFIT_GPU={{gpu_check.stdout.strip()}}")
else:
    print("LOCALFIT_GPU=none")
sys.stdout.flush()

# ── 3. Start Ollama ──
print("LOCALFIT_STATUS=starting_ollama")
sys.stdout.flush()

ollama_log = open("/tmp/ollama.log", "w")
ollama_proc = subprocess.Popen(
    ["/usr/local/bin/ollama", "serve"],
    stdout=ollama_log, stderr=subprocess.STDOUT,
    env={{**os.environ, "OLLAMA_HOST": "0.0.0.0:11434"}},
)

for i in range(30):
    time.sleep(1)
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        break
    except Exception:
        if i == 29:
            print("LOCALFIT_ERROR=ollama_timeout")
            sys.stdout.flush()
            sys.exit(1)

# ── 4. Pull model ──
print("LOCALFIT_STATUS=pulling_{ollama_tag}")
sys.stdout.flush()

pull = subprocess.run(["/usr/local/bin/ollama", "pull", "{ollama_tag}"],
                      capture_output=True, text=True, timeout=600)
if pull.returncode != 0:
    print(f"LOCALFIT_ERROR=pull_failed:{{pull.stderr[:300]}}")
    sys.stdout.flush()
    sys.exit(1)

print("LOCALFIT_STATUS=model_ready")
sys.stdout.flush()

# ── 5. Start Cloudflare tunnel ──
print("LOCALFIT_STATUS=starting_tunnel")
sys.stdout.flush()

tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:{tunnel_port}"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

endpoint = None
deadline = time.time() + 30
while time.time() < deadline:
    line = tunnel_proc.stderr.readline().decode(errors="replace")
    if "trycloudflare.com" in line:
        m = re.search(r"(https://[\\w-]+\\.trycloudflare\\.com)", line)
        if m:
            endpoint = m.group(1)
            break

if endpoint:
    print(f"LOCALFIT_ENDPOINT={{endpoint}}")
    print(f"LOCALFIT_STATUS=serving")
    with open("/kaggle/working/endpoint.txt", "w") as ef:
        ef.write(endpoint)
    try:
        import urllib.request as _ur
        _ntfy_topic = "localfit-" + "{model_query}".replace("/", "-").replace(":", "-").replace(" ", "-")[:40]
        _ur.urlopen(f"https://ntfy.sh/{{_ntfy_topic}}", data=endpoint.encode(), timeout=5)
        print(f"LOCALFIT_NTFY_SENT={{_ntfy_topic}}")
    except Exception as _ntfy_err:
        print(f"LOCALFIT_NTFY_FAILED={{_ntfy_err}}")
        pass
    sys.stdout.flush()
    print(f"")
    print(f"========================================")
    print(f"  Model ready at: {{endpoint}}")
    print(f"========================================")
    print(f"")
    print(f"Use with Claude Code:")
    print(f"  python -m localfit.proxy --port 8090 --llama-url {{endpoint}}/v1/chat/completions &\\\\")
    print(f"  ANTHROPIC_AUTH_TOKEN=localfit \\\\")
    print(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:8090 \\\\")
    print(f"  ANTHROPIC_API_KEY= \\\\")
    print(f"  claude --bare --model {ollama_tag}")
    sys.stdout.flush()
else:
    print("LOCALFIT_ERROR=tunnel_failed")
    sys.stdout.flush()
    sys.exit(1)

# ── 6. Keep alive ──
{f"max_runtime = {max_runtime_minutes} * 60  # auto-stop after {max_runtime_minutes} min" if max_runtime_minutes else "max_runtime = 12 * 3600  # 12h Kaggle limit"}
start_time = time.time()
try:
    while True:
        time.sleep(30)
        elapsed = time.time() - start_time
        if elapsed >= max_runtime:
            print(f"LOCALFIT_STATUS=auto_stopped_after_{{int(elapsed/60)}}min")
            sys.stdout.flush()
            break
        if tunnel_proc.poll() is not None:
            print("LOCALFIT_STATUS=tunnel_died")
            sys.stdout.flush()
            break
        if ollama_proc.poll() is not None:
            print("LOCALFIT_STATUS=ollama_died")
            sys.stdout.flush()
            break
except KeyboardInterrupt:
    pass

try:
    tunnel_proc.terminate()
    ollama_proc.terminate()
except Exception:
    pass
'''
    return script


def _create_kaggle_kernel_metadata(
    kernel_slug, username, title, accelerator="NvidiaTeslaT4"
):
    """Create kernel-metadata.json for Kaggle API push.

    Valid machine_shape values (from kagglesdk):
        "NvidiaTeslaT4"   — T4 16GB  (default)
        "NvidiaTeslaP100" — P100 16GB (sometimes faster for inference)
    Note: T4x2 (32GB) is NOT requestable via API — Kaggle assigns it automatically
    for large models when quota allows.
    The machine_shape field overrides enable_gpu for GPU type selection.
    """
    return {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": "notebook.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "machine_shape": accelerator,  # correct field name per kagglesdk
    }


def _push_kaggle_kernel(script, model_name, accelerator="NvidiaTeslaT4"):
    """Push a notebook to Kaggle and run it.

    Args:
        accelerator: "NvidiaTeslaT4" (default T4 16GB) or "NvidiaTeslaP100" (P100 16GB).
                     These are the only two values the Kaggle API reliably accepts.
                     T4x2 cannot be requested via API.

    Returns kernel ref (username/slug) or None on failure.
    """
    username = _get_kaggle_username()
    if not username:
        console.print(f"  [red]Cannot determine Kaggle username.[/]")
        return None

    import tempfile
    import re as _re_slug

    # Kaggle requires title to resolve to the slug in the id field.
    # Slug rules: lowercase, alphanumeric + hyphens, no consecutive hyphens.
    # Reuse same slug per model — Kaggle creates new version on each push.
    raw = f"localfit-{model_name[:35]}"
    kernel_slug = _re_slug.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:50]
    kernel_slug = _re_slug.sub(r"-+", "-", kernel_slug)
    title = kernel_slug

    tmpdir = tempfile.mkdtemp(prefix="localfit-kaggle-")
    metadata = _create_kaggle_kernel_metadata(kernel_slug, username, title, accelerator)

    meta_path = os.path.join(tmpdir, "kernel-metadata.json")
    script_path = os.path.join(tmpdir, "notebook.py")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(script_path, "w") as f:
        f.write(script)

    # Pass --accelerator to CLI — it maps directly to machine_shape in the API request
    # and overrides whatever is in kernel-metadata.json. Always pass it explicitly.
    push_cmd = ["kaggle", "kernels", "push", "-p", tmpdir, "--accelerator", accelerator]

    result = subprocess.run(
        push_cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "unknown error").strip()
        console.print(f"  [red]Kaggle push failed:[/] {err[:300]}")
        if "401" in err or "403" in err or "unauthorized" in err.lower():
            console.print(
                f"  [yellow]→ Invalid credentials. Run: localfit login kaggle[/]"
            )
        elif "404" in err:
            console.print(
                f"  [yellow]→ Kernel not found. This may be a new kernel — try again.[/]"
            )
        return None

    console.print(f"  [green]✓[/] Notebook pushed: {username}/{kernel_slug}")
    return f"{username}/{kernel_slug}"


def _poll_kaggle_output(kernel_ref, timeout_seconds=600, model_query=None):
    """Poll Kaggle kernel for the tunnel URL.

    Uses ntfy.sh callback (instant) + Kaggle REST API fallback.
    Returns endpoint URL or None.
    """
    import base64 as _b64

    poll_start = int(time.time())  # only accept ntfy messages after this time
    deadline = time.time() + timeout_seconds
    last_status = ""
    username = kernel_ref.split("/")[0]
    slug = kernel_ref.split("/")[1]
    # ntfy topic matches what the notebook posts to
    ntfy_topic = "localfit-" + (model_query or slug).replace("/", "-").replace(":", "-").replace(" ", "-")[:40]

    # Build auth header from kaggle.json
    try:
        kaggle_json = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
        auth = _b64.b64encode(f'{kaggle_json["username"]}:{kaggle_json["key"]}'.encode()).decode()
    except Exception:
        auth = None

    status_labels = {
        "installing": "Installing dependencies...",
        "building_llama_server": "Building llama-server (CUDA)...",
        "llama_server_built": "llama-server built!",
        "starting_ollama": "Starting Ollama...",
        "starting_server": "Starting llama-server...",
        "importing_model": "Importing model into Ollama...",
        "model_ready": "Model loaded!",
        "starting_tunnel": "Starting Cloudflare tunnel...",
        "serving": "Serving!",
    }

    while time.time() < deadline:
        try:
            # 1. Check ntfy.sh first — instant notification from notebook
            try:
                ntfy_url = f"https://ntfy.sh/{ntfy_topic}/json?poll=1&since={poll_start}"
                ntfy_req = urllib.request.Request(ntfy_url, headers={"User-Agent": "localfit"})
                with urllib.request.urlopen(ntfy_req, timeout=5) as nr:
                    for line in nr.read().decode().strip().split("\n"):
                        if not line.strip():
                            continue
                        msg = json.loads(line)
                        body = msg.get("message", "")
                        if "trycloudflare.com" in body:
                            endpoint = body.strip()
                            if endpoint.startswith("https://"):
                                console.print(f"  [green]✓ Endpoint received via callback[/]")
                                return endpoint
            except Exception:
                pass

            # 2. Fallback: Kaggle REST API for log + files
            if auth:
                url = f"https://www.kaggle.com/api/v1/kernels/output?userName={username}&kernelSlug={slug}&pageSize=100"
                req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())

                log_text = data.get("log", "")

                # Parse endpoint from log
                endpoint_match = re.search(
                    r"LOCALFIT_ENDPOINT=(https://[\w-]+\.trycloudflare\.com)", log_text
                )
                if endpoint_match:
                    return endpoint_match.group(1)

                # Show status updates from log
                for status_match in re.finditer(r"LOCALFIT_STATUS=(\S+)", log_text):
                    status = status_match.group(1)
                    if status != last_status:
                        last_status = status
                        if status.startswith("pulling_"):
                            label = f"Pulling {status[8:]}..."
                        elif status.startswith("downloading_"):
                            label = f"Downloading {status[12:]}..."
                        else:
                            label = status_labels.get(status, status.replace("_", " ").title())
                        from rich.markup import escape as _esc
                        console.print(f"  [dim]{_esc(label)}[/]")

                # Check for errors in log
                err_match = re.search(r"LOCALFIT_ERROR=(.+)", log_text)
                if err_match:
                    console.print(f"  [red]Error: {err_match.group(1)[:200]}[/]")
                    return None

                # Check files list for endpoint.txt
                for f in data.get("files", []):
                    if f.get("fileName") == "endpoint.txt" and f.get("url"):
                        try:
                            with urllib.request.urlopen(f["url"], timeout=10) as fr:
                                endpoint = fr.read().decode().strip()
                                if endpoint.startswith("https://"):
                                    return endpoint
                        except Exception:
                            pass

            # Check kernel status via REST API (works with both kaggle CLI v1 and v2)
            try:
                status_url = f"https://www.kaggle.com/api/v1/kernels/status?userName={username}&kernelSlug={slug}"
                status_req = urllib.request.Request(status_url, headers={"Authorization": f"Basic {auth}"} if auth else {})
                with urllib.request.urlopen(status_req, timeout=10) as sr:
                    status_data = json.loads(sr.read())
                    ks = status_data.get("status", "").lower()
                    if "error" in ks or "cancel" in ks:
                        fm = status_data.get("failureMessage", "")
                        console.print(f"  [red]Kernel failed: {ks} {fm[:100]}[/]")
                        return None
                    if "complete" in ks and not last_status:
                        console.print(f"  [yellow]Kernel completed without producing endpoint[/]")
                        return None
            except Exception:
                pass

        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            console.print(f"  [dim]Waiting... ({e})[/]")

        time.sleep(15)

    console.print(f"  [red]Timed out waiting for tunnel URL ({timeout_seconds}s)[/]")
    return None


def remote_serve_kaggle(model_query, max_runtime_minutes=None):
    """Full flow: check kaggle → resolve model → pick GPU → generate notebook → push → poll → URL.

    Smart GPU selection:
        model ≤ 14GB → T4 x1 (single GPU, free)
        model ≤ 28GB → T4 x2 (dual GPU, free)
        model > 28GB → too big for Kaggle → suggest RunPod
    """
    from localfit.backends import (
        fetch_hf_model,
        get_machine_specs,
        resolve_model_family,
        MODELS,
    )

    # 1. Ensure Kaggle CLI + credentials — auto-install + inline prompt if needed
    if not _ensure_kaggle_credentials():
        return None

    username = _get_kaggle_username()
    if not username:
        console.print(f"  [red]Cannot determine Kaggle username.[/]")
        return None

    cf_token = _get_cloudflare_token()

    # Ask for duration if not specified
    if max_runtime_minutes is None:
        quota = _get_quota_usage()
        remaining_h = max(0, KAGGLE_WEEKLY_QUOTA_HOURS - quota["used_hours"])
        console.print(f"\n  [bold #e07a5f]localfit remote serve[/] → Kaggle GPU (free)")
        console.print(
            f"  [dim]GPU quota: {remaining_h:.1f}h remaining of {KAGGLE_WEEKLY_QUOTA_HOURS}h weekly[/]"
        )
        console.print()
        try:
            dur_input = input(
                f"  Duration in minutes (default=10, max={int(remaining_h * 60)}): "
            ).strip()
            if dur_input:
                max_runtime_minutes = int(dur_input)
            else:
                max_runtime_minutes = DEFAULT_DURATION_MINUTES
        except (EOFError, KeyboardInterrupt, ValueError):
            max_runtime_minutes = DEFAULT_DURATION_MINUTES

    console.print(
        f"  [dim]Auto-stop: {max_runtime_minutes} min · auto-delete kernel after[/]"
    )
    console.print()

    # 2. Resolve model → find best GGUF quant → pick GPU
    model_repo = None
    model_filename = None
    model_name = model_query
    model_size_gb = None

    # Try known models first (resolve family aliases like gemma4 → gemma4-e4b)
    resolved = resolve_model_family(model_query, 32 * 1024)  # 32GB max (T4x2)
    if resolved and resolved in MODELS:
        m = MODELS[resolved]
        model_name = m.get("name", resolved)
        model_repo = m.get("hf_repo")

    # Fetch GGUF files from HuggingFace
    search_query = model_repo or model_query
    console.print(f"  [dim]Fetching {search_query}...[/]")
    data = fetch_hf_model(search_query, silent=True)

    if data and data["gguf_files"]:
        model_name = data.get("name", model_name)

        # Find best quant that fits Kaggle GPU
        # Strategy: first try full VRAM, then with CPU offload
        best_quant = None
        best_gpu = None
        is_offload = False

        for gpu in KAGGLE_GPUS:
            if not gpu.get("reliable", True):
                continue
            max_vram = gpu.get("max_gb", gpu["usable_gb"])
            # Full speed fits
            fits_fast = [
                f for f in data["gguf_files"] if f["size_gb"] <= gpu["usable_gb"]
            ]
            # Offload fits (slower)
            fits_offload = [
                f
                for f in data["gguf_files"]
                if f["size_gb"] <= max_vram and f["size_gb"] > gpu["usable_gb"]
            ]

            if fits_fast:
                best_quant = fits_fast[-1]
                best_gpu = gpu
                is_offload = False
                break
            elif fits_offload:
                best_quant = fits_offload[-1]
                best_gpu = gpu
                is_offload = True
                break

        if best_quant and best_gpu:
            model_filename = best_quant["filename"]
            model_repo = data["repo_id"]
            model_size_gb = best_quant["size_gb"]

            # Pick mmproj for VLM models (smallest = q8_0 version)
            mmproj_filename = None
            mmproj_files = data.get("mmproj_files", [])
            if mmproj_files:
                mmproj_filename = sorted(
                    mmproj_files, key=lambda x: x.get("size_gb", 0)
                )[0]["filename"]

            console.print(f"  [green]✓[/] {model_name}")
            console.print(
                f"    Quant: {best_quant['quant']} ({best_quant['size_gb']}GB)"
            )
            if mmproj_filename:
                console.print(f"    Vision: {mmproj_filename} (VLM)")
            console.print(
                f"    GPU:   Kaggle {best_gpu['name']} ({best_gpu['vram_gb']}GB VRAM)"
            )
            if is_offload:
                console.print(
                    f"    [yellow]Mode:  Partial CPU offload (model > VRAM, slower but works)[/]"
                )
            console.print(f"    Repo:  {model_repo}/{model_filename}")

            # If T4 doesn't fit but T4x2 does, show why
            t4_fits = [
                f
                for f in data["gguf_files"]
                if f["size_gb"] <= KAGGLE_GPUS[0]["usable_gb"]
            ]
            if not t4_fits and best_gpu["name"] == "T4x2":
                console.print(
                    f"    [dim]Too big for single T4 → using T4x2 (dual GPU)[/]"
                )

            # Show if there's a better quant on T4x2 that we're not using
            if best_gpu["name"] == "T4":
                t4x2_fits = [
                    f
                    for f in data["gguf_files"]
                    if f["size_gb"] <= KAGGLE_GPUS[1]["usable_gb"]
                ]
                if t4x2_fits:
                    t4x2_best = t4x2_fits[-1]
                    if t4x2_best["size_gb"] > best_quant["size_gb"] * 1.3:
                        console.print(
                            f"    [dim]Tip: T4x2 can run {t4x2_best['quant']} "
                            f"({t4x2_best['size_gb']}GB) for better quality[/]"
                        )
        else:
            # Nothing fits any Kaggle GPU
            console.print(
                f"  [red]No quant fits Kaggle GPUs (max 28GB usable on T4x2)[/]"
            )
            smallest = data["gguf_files"][0]
            console.print(
                f"  [dim]Smallest: {smallest['quant']} ({smallest['size_gb']}GB)[/]"
            )
            console.print(
                f"  [yellow]→ Use RunPod: localfit serve {model_query} --cloud[/]\n"
            )
            return None
    elif not model_repo:
        # No GGUF found on HF — try as Ollama tag
        console.print(f"  [dim]No GGUF found. Using Ollama tag: {model_query}[/]")
        # Default to T4 for Ollama models (most are <14GB)
        best_gpu = KAGGLE_GPUS[0]

    # 3. Generate notebook
    console.print(f"\n  [bold]Generating Kaggle notebook...[/]")

    if model_repo and model_filename:
        script = _generate_notebook(
            model_query,
            model_repo,
            model_filename,
            best_gpu,
            cf_token,
            mmproj_filename=mmproj_filename,
            max_runtime_minutes=max_runtime_minutes,
        )
    else:
        script = _generate_notebook_ollama(
            model_query,
            best_gpu,
            cf_token,
            max_runtime_minutes=max_runtime_minutes,
        )

    # Save locally
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CONFIG_DIR / "kaggle_notebook.py"
    local_path.write_text(script)
    console.print(f"  [dim]Saved: {local_path}[/]")

    # 4. Ask for duration + show quota
    quota = _get_quota_usage()
    remaining_h = max(0, KAGGLE_WEEKLY_QUOTA_HOURS - quota["used_hours"])

    console.print(f"\n  [bold]Ready to push to Kaggle:[/]")
    console.print(f"    Model:    {model_name}")
    if model_filename:
        console.print(f"    Quant:    {model_filename}")
    console.print(f"    GPU:      {best_gpu['name']} ({best_gpu['vram_gb']}GB)")
    console.print(
        f"    Duration: {max_runtime_minutes} min (auto-stops + auto-deletes)"
    )
    console.print(
        f"    Quota:    {remaining_h:.1f}h remaining of {KAGGLE_WEEKLY_QUOTA_HOURS}h weekly"
    )
    console.print(f"    Cost:     Free")
    if cf_token:
        console.print(f"    Tunnel:   Cloudflare Zero Trust (stable domain)")
    else:
        console.print(f"    Tunnel:   trycloudflare.com (random URL, free)")

    if remaining_h < max_runtime_minutes / 60:
        console.print(
            f"\n  [red]Warning: Not enough quota ({remaining_h:.1f}h) for {max_runtime_minutes}min![/]"
        )

    console.print()

    try:
        confirm = input("  Push and run? (Y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if confirm and confirm not in ("y", "yes", ""):
        console.print(f"  [dim]Cancelled.[/]")
        return None

    # 5. Push to Kaggle
    console.print(f"\n  [bold]Pushing to Kaggle...[/]")
    kernel_ref = _push_kaggle_kernel(script, model_name, best_gpu["accelerator"])
    if not kernel_ref:
        return None

    # Save state
    state = {
        "kernel_ref": kernel_ref,
        "model": model_name,
        "model_repo": model_repo,
        "model_filename": model_filename,
        "gpu": best_gpu["name"],
        "duration_minutes": max_runtime_minutes,
        "started_at": time.time(),
        "username": username,
    }
    KAGGLE_STATE_FILE.write_text(json.dumps(state, indent=2))

    # 6. Poll for tunnel URL
    console.print(f"\n  [bold]Waiting for GPU + model + tunnel...[/]")
    console.print(f"  [dim]Kaggle queue ~1-5 min, build ~3-5 min, download varies[/]")
    console.print(f"  [dim]Kernel: https://www.kaggle.com/code/{kernel_ref}[/]")
    console.print(f"  [dim]Logs:   https://www.kaggle.com/code/{kernel_ref}/log[/]\n")

    endpoint = _poll_kaggle_output(kernel_ref, timeout_seconds=900, model_query=model_query)  # 15 min

    if endpoint:
        # Determine model name for API
        api_model = (
            model_query if ("/" in model_query or ":" in model_query) else "localmodel"
        )
        is_vlm = bool(mmproj_filename) or (data and data.get("is_vlm"))

        # 7. Auto-test the endpoint
        console.print(f"\n  [bold]Testing endpoint...[/]")
        try:
            import urllib.request

            test_payload = json.dumps(
                {
                    "model": api_model,
                    "messages": [
                        {"role": "user", "content": "Say hello in one sentence."}
                    ],
                    "max_tokens": 30,
                    "stream": False,
                }
            ).encode()
            req = urllib.request.Request(
                f"{endpoint}/v1/chat/completions",
                data=test_payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                reply = result["choices"][0]["message"]["content"][:200]
                console.print(f"  [green]✓ Model responded:[/] {reply}")
                console.print(f"  [green bold]● ENDPOINT VERIFIED — MODEL WORKING[/]")
        except Exception as e:
            console.print(f"  [yellow]Test failed: {e}[/]")
            console.print(f"  [dim]Endpoint may still work — try manually.[/]")

        # 8. Show ready info
        _print_ready(
            endpoint, model_name, model_query, best_gpu, model_repo, is_vlm=is_vlm
        )

        state["endpoint"] = endpoint
        KAGGLE_STATE_FILE.write_text(json.dumps(state, indent=2))
        _record_quota_usage(max_runtime_minutes)

        # 9. Offer tool launch / config
        _launch_remote_tool_menu(endpoint, api_model)

        # 10. Wait for duration, then auto-delete
        console.print(
            f"\n  [dim]Auto-delete in {max_runtime_minutes} min. Ctrl+C to stop early.[/]"
        )

        # Wait for duration, then auto-delete
        try:
            remaining_secs = max_runtime_minutes * 60
            while remaining_secs > 0:
                time.sleep(min(30, remaining_secs))
                remaining_secs -= 30
                mins_left = max(0, remaining_secs // 60)
                if mins_left > 0 and mins_left % 2 == 0:
                    console.print(f"  [dim]{mins_left} min remaining...[/]")
        except KeyboardInterrupt:
            console.print(f"\n  [dim]Interrupted.[/]")

        # Auto-delete kernel
        console.print(f"\n  [bold]Auto-deleting kernel to save quota...[/]")
        try:
            subprocess.run(
                f'echo "yes" | kaggle kernels delete {kernel_ref}',
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            console.print(f"  [green]✓[/] Kernel deleted.")
        except Exception:
            console.print(
                f"  [yellow]Could not auto-delete. Delete manually or it auto-stops.[/]"
            )

        KAGGLE_STATE_FILE.unlink(missing_ok=True)
        return endpoint
    else:
        console.print(
            f"\n  [yellow]Notebook is running but tunnel URL not captured yet.[/]"
        )
        console.print(f"  [dim]Check: https://www.kaggle.com/code/{kernel_ref}[/]")
        console.print(f"  [dim]Or: localfit --remote-status[/]")
        return None


def _write_remote_opencode_config(api_base, model_name):
    """Write an OpenCode config pointing at a remote OpenAI-compatible endpoint."""
    config_path = os.path.expanduser("~/.config/opencode/opencode.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": "llamacpp/remote",
        "provider": {
            "llamacpp": {
                "name": "Remote OpenAI-compatible model (localfit)",
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": api_base},
                "models": {
                    "remote": {
                        "name": model_name,
                        "tool_call": True,
                        "reasoning": False,
                        "modalities": {
                            "input": ["text", "image"],
                            "output": ["text"],
                        },
                        "limit": {"context": 131072, "output": 8192},
                    }
                },
            }
        },
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def _launch_remote_tool_menu(endpoint, api_model):
    """Offer to launch or configure coding tools for a remote endpoint."""
    api_base = f"{endpoint}/v1"

    console.print(f"\n  [bold]Launch or configure a tool with this endpoint?[/]")
    console.print(f"  [bold]1[/]  Claude Code")
    console.print(f"  [bold]2[/]  Codex")
    console.print(f"  [bold]3[/]  OpenCode")
    console.print(f"  [bold]4[/]  OpenClaw")
    console.print(f"  [bold]5[/]  Hermes")
    console.print(f"  [bold]6[/]  Open WebUI (browser)")
    console.print(f"  [bold]7[/]  aider")
    console.print(f"  [dim]q[/]  Skip")
    console.print()

    try:
        tool_choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if tool_choice in ("", "q", "skip"):
        return

    if tool_choice in ("1", "claude"):
        from localfit.proxy import PROXY_PORT, ensure_proxy_process
        from localfit.safe_config import get_claude_launch_env

        proxy_ready = ensure_proxy_process(
            llama_url=f"{endpoint}/v1/chat/completions",
            port=PROXY_PORT,
        )
        if not proxy_ready:
            console.print(
                f"  [red]Failed to start local Claude proxy on :{PROXY_PORT}.[/]"
            )
            return

        console.print(f"\n  [bold]Launching Claude Code...[/]")
        console.print(
            f"  [cyan]python -m localfit.proxy --port {PROXY_PORT} --llama-url {endpoint}/v1/chat/completions &[/]"
        )
        console.print(f"  [cyan]ANTHROPIC_AUTH_TOKEN=localfit \\")
        console.print(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{PROXY_PORT} \\")
        console.print(f"  ANTHROPIC_API_KEY= \\")
        console.print(f"  [cyan]claude --bare --model {api_model}[/]\n")
        try:
            subprocess.Popen(
                ["claude", "--bare", "--model", api_model],
                env={
                    **os.environ,
                    **get_claude_launch_env(
                        api_base=f"http://127.0.0.1:{PROXY_PORT}"
                    ),
                },
            )
        except FileNotFoundError:
            console.print(
                f"  [red]Claude Code not installed. Run the command above manually.[/]"
            )
        return

    if tool_choice in ("2", "codex"):
        from localfit.safe_config import get_codex_launch_env

        console.print(f"\n  [bold]Launching Codex...[/]")
        console.print(f"  [cyan]OPENAI_BASE_URL={api_base} \\")
        console.print(f"  OPENAI_API_KEY=sk-no-key-required \\")
        console.print(
            f"  [cyan]codex --model {api_model} -c model_provider=openai -c features.use_responses_api=false[/]\n"
        )
        try:
            subprocess.Popen(
                [
                    "codex",
                    "--model",
                    api_model,
                    "-c",
                    "model_provider=openai",
                    "-c",
                    "features.use_responses_api=false",
                ],
                env={**os.environ, **get_codex_launch_env(api_base=api_base)},
            )
        except FileNotFoundError:
            console.print(
                f"  [red]Codex not installed. Run: npm install -g @openai/codex[/]"
            )
        return

    if tool_choice in ("3", "opencode"):
        config_path = _write_remote_opencode_config(api_base, api_model)
        console.print(f"\n  [green]✓[/] OpenCode configured")
        console.print(f"  [dim]Written: {config_path}[/]")
        console.print(f"  [dim]API: {api_base}[/]")
        try:
            subprocess.Popen(["opencode"], env={**os.environ})
        except FileNotFoundError:
            console.print(f"  [red]OpenCode not installed. Start it after install.[/]")
        return

    if tool_choice in ("4", "openclaw"):
        from localfit.safe_config import add_model_to_openclaw

        result = add_model_to_openclaw(api_base=api_base)
        if not result:
            console.print(f"  [red]OpenClaw not installed. No config file found.[/]")
            return
        console.print(f"\n  [green]✓[/] OpenClaw configured")
        console.print(f"  [dim]Written: {result['config']}[/]")
        console.print(f"  [dim]API: {api_base}[/]")
        return

    if tool_choice in ("5", "hermes"):
        from localfit.safe_config import add_model_to_hermes

        result = add_model_to_hermes(api_base=api_base, model_name=api_model)
        console.print(f"\n  [green]✓[/] Hermes configured")
        console.print(f"  [dim]Config: {result['config']}[/]")
        console.print(f"  [dim]Env: {result['env']}[/]")
        console.print(f"  [dim]API: {api_base}[/]")
        try:
            subprocess.Popen(
                ["hermes"],
                env={
                    **os.environ,
                    "OPENAI_BASE_URL": api_base,
                    "OPENAI_API_KEY": "no-key-required",
                },
            )
        except FileNotFoundError:
            console.print(
                f"  [red]Hermes not installed. Run it manually after install.[/]"
            )
        return

    if tool_choice in ("6", "webui", "open-webui", "openwebui"):
        console.print(f"\n  [bold]Launching Open WebUI...[/]")
        console.print(f"  [cyan]OPENAI_API_BASE_URL={api_base} \\")
        console.print(f"  OPENAI_API_KEY=no-key-required \\")
        console.print(f"  [cyan]open-webui serve[/]\n")
        try:
            subprocess.Popen(
                ["open-webui", "serve"],
                env={
                    **os.environ,
                    "OPENAI_API_BASE_URL": api_base,
                    "OPENAI_API_KEY": "no-key-required",
                },
            )
        except FileNotFoundError:
            console.print(
                f"  [red]Open WebUI not installed. Run: pip install open-webui[/]"
            )
        return

    if tool_choice in ("7", "aider"):
        console.print(f"\n  [bold]Launching aider...[/]")
        console.print(f"  [cyan]OPENAI_API_BASE={api_base} \\")
        console.print(f"  OPENAI_API_KEY=none \\")
        console.print(f"  [cyan]aider --model openai/{api_model}[/]\n")
        try:
            subprocess.Popen(
                ["aider", "--model", f"openai/{api_model}"],
                env={
                    **os.environ,
                    "OPENAI_API_BASE": api_base,
                    "OPENAI_API_KEY": "none",
                },
            )
        except FileNotFoundError:
            console.print(
                f"  [red]aider not installed. Run the command above manually.[/]"
            )
        return

    console.print(f"  [yellow]Unknown choice.[/]")


def _print_ready(
    endpoint,
    model_name,
    model_query,
    gpu,
    model_repo=None,
    is_vlm=False,
    provider_name="Kaggle",
    cost_text="Free (auto-stops in ~12h)",
    status_command="localfit --remote-status",
    stop_command="localfit --remote-stop",
):
    """Print the ready message with usage instructions and test code."""
    if "/" in model_query or ":" in model_query:
        api_model = model_query
    else:
        api_model = "localmodel"

    gpu_vram = gpu.get("vram_gb", gpu.get("vram", "?"))

    console.print(f"\n  [bold green]● READY — Model serving on {provider_name}![/]")
    console.print()
    console.print(f"  Model:    {model_name}")
    console.print(f"  GPU:      {provider_name} {gpu['name']} ({gpu_vram}GB)")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  Cost:     {cost_text}")
    if is_vlm:
        console.print(f"  Type:     VLM (vision + text)")
    console.print()

    console.print(f"  [bold]── Use with coding tools ──[/]")
    console.print()
    console.print(f"  [bold]Claude Code:[/]")
    console.print(
        f"  [cyan]python -m localfit.proxy --port 8090 --llama-url {endpoint}/v1/chat/completions &[/]"
    )
    console.print(f"  [cyan]ANTHROPIC_AUTH_TOKEN=localfit \\")
    console.print(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:8090 \\")
    console.print(f"  ANTHROPIC_API_KEY= \\")
    console.print(f"  [cyan]claude --bare --model {api_model}[/]")
    console.print()
    console.print(f"  [bold]Codex:[/]")
    console.print(f"  [cyan]OPENAI_BASE_URL={endpoint}/v1 \\")
    console.print(f"  OPENAI_API_KEY=sk-no-key-required \\")
    console.print(
        f"  [cyan]codex --model {api_model} -c model_provider=openai -c features.use_responses_api=false[/]"
    )
    console.print()
    console.print(f"  [bold]OpenCode:[/]")
    console.print(f"  [cyan]# Set provider baseURL to: {endpoint}/v1[/]")
    console.print()
    console.print(f"  [bold]OpenClaw:[/]")
    console.print(f"  [cyan]# Set apiBase to: {endpoint}/v1[/]")
    console.print()
    console.print(f"  [bold]Hermes:[/]")
    console.print(
        f"  [cyan]OPENAI_BASE_URL={endpoint}/v1 OPENAI_API_KEY=no-key-required hermes[/]"
    )
    console.print()
    console.print(f"  [bold]Open WebUI:[/]")
    console.print(
        f"  [cyan]OPENAI_API_BASE_URL={endpoint}/v1 OPENAI_API_KEY=no-key-required open-webui serve[/]"
    )
    console.print()

    console.print(f"  [bold]── Test: curl ──[/]")
    console.print()
    console.print(f"  [bold]Text query:[/]")
    console.print(f"  [cyan]curl {endpoint}/v1/chat/completions \\")
    console.print(f'    -H "Content-Type: application/json" \\')
    console.print(
        f'[cyan]    -d \'{{"model": "{api_model}", "messages": '
        f'[{{"role": "user", "content": "hello"}}]}}\'[/]'
    )

    if is_vlm:
        console.print()
        console.print(f"  [bold]VLM image query (base64 required by Ollama):[/]")
        console.print(
            f"  [cyan]IMG=$(curl -s https://example.com/image.jpg | base64 -w0)"
        )
        from rich.markup import escape as _esc_curl
        vlm_curl = (
            f'  curl {endpoint}/v1/chat/completions \\\n'
            f'    -H "Content-Type: application/json" \\\n'
            f'    -d \'{{"model": "{api_model}", "messages": '
            f'[{{"role": "user", "content": ['
            f'{{"type": "image_url", "image_url": {{"url": "data:image/jpeg;base64,\'$IMG\'"}}}}, '
            f'{{"type": "text", "text": "What is in this image?"}}]}}]}}\''
        )
        console.print(f"  [cyan]{_esc_curl(vlm_curl)}[/]")

    console.print()
    console.print(f"  [bold]── Test: Python ──[/]")
    console.print()
    console.print(f"  [cyan]from openai import OpenAI")
    console.print(f'  client = OpenAI(base_url="{endpoint}/v1", api_key="none")')
    console.print()
    console.print(f"  # Text query")
    console.print(f"  r = client.chat.completions.create(")
    console.print(f'      model="{api_model}",')
    console.print(f'      messages=[{{"role": "user", "content": "hello"}}]')
    console.print(f"  )")
    console.print(f"  [cyan]print(r.choices[0].message.content)[/]")

    if is_vlm:
        console.print()
        console.print(f"  [cyan]# VLM image query (Ollama needs base64, not URLs)")
        console.print(f"  import base64, httpx")
        console.print(
            f'  img_bytes = httpx.get("https://example.com/photo.jpg").content'
        )
        console.print(f"  img_b64 = base64.b64encode(img_bytes).decode()")
        console.print(f"  r = client.chat.completions.create(")
        console.print(f'      model="{api_model}",')
        console.print(f'      messages=[{{"role": "user", "content": [')
        console.print(
            f'          {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{img_b64}}"}}}},'
        )
        console.print(f'          {{"type": "text", "text": "What is in this image?"}}')
        console.print(f"      ]}}]")
        console.print(f"  )")
        console.print(f"  [cyan]print(r.choices[0].message.content)[/]")

    console.print()
    console.print(f"  [dim]Status: {status_command}[/]")
    console.print(f"  [dim]Stop:   {stop_command}[/]")


def remote_stop():
    """Stop the active remote kernel."""
    if not KAGGLE_STATE_FILE.exists():
        console.print(f"\n  [dim]No active remote session.[/]")
        return

    state = json.loads(KAGGLE_STATE_FILE.read_text())
    kernel_ref = state.get("kernel_ref")

    if kernel_ref:
        console.print(f"\n  [bold]Stopping Kaggle kernel: {kernel_ref}[/]")

        try:
            import base64 as _b64_stop
            _kj = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
            _auth = _b64_stop.b64encode(f'{_kj["username"]}:{_kj["key"]}'.encode()).decode()
            _parts = kernel_ref.split("/")
            _url = f"https://www.kaggle.com/api/v1/kernels/status?userName={_parts[0]}&kernelSlug={_parts[1]}"
            _req = urllib.request.Request(_url, headers={"Authorization": f"Basic {_auth}"})
            with urllib.request.urlopen(_req, timeout=10) as _r:
                _sd = json.loads(_r.read())
                status = _sd.get("status", "unknown")
            console.print(f"  Status: {status}")

            if "running" in status.lower():
                console.print(f"  [yellow]Kaggle doesn't support kernel cancellation via API.[/]")
                console.print(f"  [dim]Stop manually: https://www.kaggle.com/code/{kernel_ref}[/]")
                console.print(f"  [dim]Or wait — it auto-stops after duration.[/]")
            else:
                console.print(f"  [green]✓[/] Kernel already stopped.")
        except Exception as e:
            console.print(f"  [red]Error checking status: {e}[/]")

    elapsed = (time.time() - state.get("started_at", time.time())) / 3600
    console.print(f"  Runtime: ~{elapsed:.1f}h")
    KAGGLE_STATE_FILE.unlink(missing_ok=True)
    console.print(f"  [green]✓[/] Session cleared.\n")


def remote_status():
    """Show active remote session status."""
    if not KAGGLE_STATE_FILE.exists():
        console.print(f"\n  [dim]No active remote session.[/]")
        console.print(f"  [dim]Start: localfit serve MODEL --remote kaggle[/]\n")
        return

    state = json.loads(KAGGLE_STATE_FILE.read_text())
    kernel_ref = state.get("kernel_ref", "?")
    model = state.get("model", "?")
    gpu = state.get("gpu", "T4")
    endpoint = state.get("endpoint")
    elapsed = (time.time() - state.get("started_at", time.time())) / 3600
    remaining = max(0, 12 - elapsed)

    console.print(f"\n  [bold]Active remote session[/]")
    console.print(f"  Model:   {model}")
    console.print(f"  GPU:     Kaggle {gpu}")
    console.print(f"  Kernel:  {kernel_ref}")
    console.print(f"  Runtime: {elapsed:.1f}h")

    if remaining > 0:
        console.print(f"  [green]Remaining: ~{remaining:.1f}h[/]")
    else:
        console.print(f"  [red]Kaggle 12h limit likely reached[/]")

    if endpoint:
        console.print(f"  Endpoint: {endpoint}")

        # Health check
        try:
            import urllib.request

            with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=5) as resp:
                console.print(f"  [green]● Online[/]")
        except Exception:
            console.print(f"  [red]● Offline (tunnel may have expired)[/]")
    else:
        console.print(f"  [dim]Endpoint not captured. Polling...[/]")
        ep = _poll_kaggle_output(kernel_ref, timeout_seconds=30)
        if ep:
            console.print(f"  Endpoint: {ep}")
            state["endpoint"] = ep
            KAGGLE_STATE_FILE.write_text(json.dumps(state, indent=2))

    console.print(f"\n  [dim]View: https://www.kaggle.com/code/{kernel_ref}[/]")
    console.print(f"  [dim]Stop: localfit --remote-stop[/]\n")
