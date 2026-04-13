"""Backend discovery, installation, and model management."""

import json, os, shutil, subprocess, sys, time, urllib.request, urllib.parse
from pathlib import Path

# ── Kill tqdm line spam BEFORE any HF imports ──
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # fast downloads if hf_transfer installed

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# ── Platform detection ──
HOME = Path.home()
CONFIG_DIR = HOME / ".localfit"
MODELS_DIR = HOME / "models"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform == "linux"
IS_WSL = IS_LINUX and "microsoft" in (
    Path("/proc/version").read_text().lower() if Path("/proc/version").exists() else ""
)


def _find_binary(name, extra_paths=None):
    """Find a binary in PATH or known locations."""
    found = shutil.which(name)
    if found:
        return Path(found)
    for p in extra_paths or []:
        if Path(p).exists():
            return Path(p)
    return Path(name)  # fallback — will fail on check


# ── Known backends ──
BACKENDS = {
    "llamacpp": {
        "name": "llama.cpp",
        "default_port": 8089,
        "binary": _find_binary(
            "llama-server",
            [
                HOME / ".unsloth/llama.cpp/llama-server",
                Path("/usr/local/bin/llama-server"),
            ],
        ),
        "install_cmd": "curl -fsSL https://unsloth.ai/install.sh | sh",
    },
    "ollama": {
        "name": "Ollama",
        "default_port": 11434,
        "binary": _find_binary(
            "ollama",
            [
                Path("/opt/homebrew/bin/ollama"),
                Path("/usr/local/bin/ollama"),
                HOME / ".local/bin/ollama",
            ],
        ),
        "install_cmd": "curl -fsSL https://ollama.com/install.sh | sh"
        if IS_LINUX
        else "brew install ollama",
    },
    "mlx": {
        "name": "MLX (Apple Silicon)",
        "default_port": 8080,
        "binary": Path("mlx_lm"),  # python -m mlx_lm.server — no binary, pip package
        "install_cmd": "pip install mlx-lm",
    },
}

# ── Known models ──
MODELS = {
    # ── Gemma 4 — verified sizes from HF API 2026-04-12 ──
    "gemma4-26b": {
        "name": "Gemma 4 26B MoE",
        "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "hf_pattern": "*UD-Q3_K_XL*",
        "size_gb": 12.5,  # Q4 actual
        "size_q2_gb": 9.2,
        "size_q4_gb": 12.5,
        "size_q8_gb": 26.0,
        "ram_required": 16,
        "description": "Best MoE for 24GB. Vision, tool calling, 49 tok/s.",
        "ollama_tag": "gemma4:26b",
        "backend": "llamacpp",
        "server_flags": "-ngl 99 -c 32768 -np 1 -fa on -ctk q4_0 -ctv q4_0 --no-warmup --jinja",
        "source": "unsloth",
    },
    "gemma4-e4b": {
        "name": "Gemma 4 E4B",
        "hf_repo": "unsloth/gemma-4-E4B-it-GGUF",
        "hf_pattern": "*Q4_K_M*",
        "size_gb": 4.4,  # Q4 actual
        "size_q2_gb": 3.3,
        "size_q4_gb": 4.4,
        "size_q8_gb": 8.1,
        "ram_required": 8,
        "description": "Sweet spot for 16GB. Vision + audio + code.",
        "ollama_tag": "gemma4:e4b",
        "backend": "llamacpp",
        "server_flags": "-ngl 99 -c 32768 --jinja",
        "source": "unsloth",
    },
    "gemma4-31b": {
        "name": "Gemma 4 31B Dense",
        "hf_repo": "unsloth/gemma-4-31B-it-GGUF",
        "hf_pattern": "*Q3_K_XL*",
        "size_gb": 15.2,  # Q4 actual
        "size_q2_gb": 7.9,
        "size_q4_gb": 15.2,
        "size_q8_gb": 32.6,
        "ram_required": 24,
        "description": "Dense 31B. Q3 fits 16GB (7.9GB), Q4 needs 24GB.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "server_flags": "-ngl 99 -c 16384 --jinja",
        "source": "unsloth",
    },
    "gemma4-e2b": {
        "name": "Gemma 4 E2B",
        "hf_repo": "unsloth/gemma-4-E2B-it-GGUF",
        "hf_pattern": "*Q4_K_M*",
        "size_gb": 2.8,  # Q4 actual
        "size_q2_gb": 2.1,
        "size_q4_gb": 2.8,
        "size_q8_gb": 4.9,
        "ram_required": 8,
        "description": "Tiny + fast. Vision + audio. Runs on anything.",
        "ollama_tag": "gemma4:e2b",
        "backend": "llamacpp",
        "server_flags": "-ngl 99 -c 32768 --jinja",
        "source": "unsloth",
    },
    # ── Qwen 3.5 — verified sizes ──
    "qwen35-35b-a3b": {
        "name": "Qwen 3.5 35B MoE",
        "hf_repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "hf_pattern": "*UD-Q2_K_XL*",
        "size_gb": 16.3,  # Q4 actual — does NOT fit 16GB
        "size_q2_gb": 9.9,
        "size_q4_gb": 16.3,
        "size_q8_gb": 45.3,
        "ram_required": 24,
        "description": "MoE coding beast. Q2 fits 16GB (9.9GB).",
        "ollama_tag": None,
        "backend": "llamacpp",
        "server_flags": "-ngl 99 -c 32768 -np 1 -fa on -ctk q4_0 -ctv q4_0 --no-warmup --jinja --reasoning-budget 0",
        "source": "unsloth",
    },
    "qwen35-122b-a10b": {
        "name": "Qwen 3.5 122B MoE",
        "hf_repo": "unsloth/Qwen3.5-122B-A10B-GGUF",
        "hf_pattern": "*UD-IQ2_XXS*",
        "size_gb": 36.5,  # Q4 actual
        "size_q2_gb": 36.5,
        "size_q4_gb": 36.5,
        "size_q8_gb": 43.4,
        "ram_required": 48,
        "description": "Best MoE overall. Needs 48GB+ or cloud.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
        "cloud_only": True,
    },
    "qwen35-4b": {
        "name": "Qwen 3.5 4B",
        "hf_repo": "unsloth/Qwen3.5-4B-GGUF",
        "hf_pattern": "*UD-Q4_K_XL*",
        "size_gb": 2.3,  # Q4 actual
        "size_q2_gb": 1.4,
        "size_q4_gb": 2.3,
        "size_q8_gb": 5.5,
        "ram_required": 8,
        "description": "Ultrafast. Only 2.3GB GPU.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "server_flags": "-ngl 99 -c 32768 --jinja --reasoning-budget 0",
        "source": "unsloth",
    },
    "qwen35-9b": {
        "name": "Qwen 3.5 9B",
        "hf_repo": "unsloth/Qwen3.5-9B-GGUF",
        "hf_pattern": "*UD-Q4_K_XL*",
        "size_gb": 4.8,  # Q4 actual
        "size_q2_gb": 3.0,
        "size_q4_gb": 4.8,
        "size_q8_gb": 12.1,
        "ram_required": 12,
        "description": "Great mid-range. Fits 16GB with image model.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "server_flags": "-ngl 99 -c 32768 --jinja",
        "source": "unsloth",
    },
    "qwen35-27b": {
        "name": "Qwen 3.5 27B",
        "hf_repo": "unsloth/Qwen3.5-27B-GGUF",
        "hf_pattern": "*UD-Q3_K_XL*",
        "size_gb": 13.9,  # Q4 actual
        "size_q2_gb": 8.0,
        "size_q4_gb": 13.9,
        "size_q8_gb": 33.1,
        "ram_required": 24,
        "description": "Dense 27B. Q2 fits 16GB (8GB).",
        "ollama_tag": "qwen3.5:27b",
        "backend": "llamacpp",
        "source": "unsloth",
    },
    # ── Qwen 3 Coder — verified sizes ──
    "qwen3-coder-next": {
        "name": "Qwen3 Coder-Next",
        "hf_repo": "unsloth/Qwen3-Coder-Next-GGUF",
        "hf_pattern": "*UD-IQ4_NL*",
        "size_gb": 35.8,  # Q4 actual — NOT 16GB!
        "size_q2_gb": 17.6,
        "size_q4_gb": 35.8,
        "size_q8_gb": 46.2,
        "ram_required": 48,
        "description": "Latest coder. Needs 48GB+ or cloud.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
        "cloud_only": True,
    },
    "qwen3-coder-30b": {
        "name": "Qwen3-Coder 30B MoE",
        "hf_repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "hf_pattern": "*UD-Q3_K_XL*",
        "size_gb": 15.3,  # Q4 actual
        "size_q2_gb": 7.5,
        "size_q4_gb": 15.3,
        "size_q8_gb": 33.5,
        "ram_required": 24,
        "description": "MoE coder. Q2 fits 16GB (7.5GB).",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
    },
    # ── DeepSeek — verified sizes ──
    "deepseek-r1-qwen-14b": {
        "name": "DeepSeek R1 Qwen 14B",
        "hf_repo": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "hf_pattern": "*Q4_K_M*",
        "size_gb": 8.4,  # Q4 actual
        "size_q2_gb": 5.4,
        "size_q4_gb": 8.4,
        "size_q8_gb": 14.6,
        "ram_required": 16,
        "description": "Best reasoning distill. Fits 16GB.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
    },
    "deepseek-r1-qwen-32b": {
        "name": "DeepSeek R1 Qwen 32B",
        "hf_repo": "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "hf_pattern": "*Q3_K_XL*",
        "size_gb": 18.5,  # Q4 actual
        "size_q2_gb": 11.5,
        "size_q4_gb": 18.5,
        "size_q8_gb": 32.4,
        "ram_required": 24,
        "description": "Strongest reasoning. Q2 fits 16GB (11.5GB).",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
    },
    # ── GLM — verified: GLM-5 is 700B, GLM-4.7-Flash is also huge ──
    "glm-5": {
        "name": "GLM-5 (700B)",
        "hf_repo": "unsloth/GLM-5-GGUF",
        "hf_pattern": "*Q1_0*",
        "size_gb": 164.0,  # smallest GGUF is 164GB!
        "ram_required": 192,
        "description": "700B model. Cloud only (MI300X/B200).",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
        "cloud_only": True,
    },
    "glm-47-flash": {
        "name": "GLM 4.7 Flash",
        "hf_repo": "unsloth/GLM-4.7-Flash-GGUF",
        "hf_pattern": "*UD-Q4_K_XL*",
        "size_gb": 15.2,  # Q4 actual — NOT 5GB!
        "size_q2_gb": 7.8,
        "size_q4_gb": 15.2,
        "size_q8_gb": 33.2,
        "ram_required": 24,
        "description": "Q2 fits 16GB (7.8GB). Q4 needs 24GB.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
    },
    # ── Others — verified ──
    "kimi-k25": {
        "name": "Kimi K2.5",
        "hf_repo": "unsloth/Kimi-K2.5-GGUF",
        "hf_pattern": "*UD-IQ2_XXS*",
        "size_gb": 100,  # no GGUF files yet, estimate
        "ram_required": 128,
        "description": "Huge MoE. Cloud only.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
        "cloud_only": True,
    },
    "gpt-oss-20b": {
        "name": "gpt-oss 20B",
        "hf_repo": "unsloth/gpt-oss-20b-GGUF",
        "hf_pattern": "*Q4_K_M*",
        "size_gb": 10.7,  # Q4 actual — NOT 12GB
        "size_q2_gb": 10.7,
        "size_q4_gb": 10.7,
        "size_q8_gb": 12.3,
        "ram_required": 16,
        "description": "OpenAI open-source. Fits 16GB at Q2.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
    },
    "nemotron3-nano-4b": {
        "name": "Nemotron 3 Nano 4B",
        "hf_repo": "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        "hf_pattern": "*Q4_K_M*",
        "size_gb": 2.4,  # Q4 actual
        "size_q2_gb": 2.0,
        "size_q4_gb": 2.4,
        "size_q8_gb": 5.2,
        "ram_required": 8,
        "description": "NVIDIA tiny. 2.4GB, great speed.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
    },
    "devstral-24b": {
        "name": "Devstral 2 24B",
        "hf_repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        "hf_pattern": "*Q3_K_XL*",
        "size_gb": 11.9,  # Q4 actual
        "size_q2_gb": 5.2,
        "size_q4_gb": 11.9,
        "size_q8_gb": 27.0,
        "ram_required": 16,
        "description": "Mistral coder. Q2 fits 8GB (5.2GB).",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
    },
    "minimax-m25": {
        "name": "MiniMax M2.5",
        "hf_repo": "unsloth/MiniMax-M2.5-GGUF",
        "hf_pattern": "*UD-IQ2_XXS*",
        "size_gb": 51.9,  # smallest GGUF is 51.9GB
        "ram_required": 64,
        "description": "Large MoE. Cloud only.",
        "ollama_tag": None,
        "backend": "llamacpp",
        "source": "unsloth",
        "cloud_only": True,
    },
    # ── Small ──
    "gemma2-2b": {
        "name": "Gemma 2 2B",
        "hf_repo": None,
        "size_gb": 1.5,
        "ram_required": 4,
        "description": "Tiny. Runs on anything.",
        "ollama_tag": "gemma2:2b",
        "backend": "ollama",
        "source": "unsloth",
    },
}


# Family aliases → variants ordered best-to-smallest.
# "localfit --launch claude --model gemma4" picks the best that fits your GPU.
MODEL_FAMILIES = {
    "gemma4": ["gemma4-26b", "gemma4-31b", "gemma4-e4b", "gemma4-e2b"],
    "qwen35": ["qwen35-35b-a3b", "qwen35-27b", "qwen35-9b", "qwen35-4b"],
    "qwen3.5": ["qwen35-35b-a3b", "qwen35-27b", "qwen35-9b", "qwen35-4b"],
    "qwen3-coder": ["qwen3-coder-next", "qwen3-coder-30b"],
    "deepseek": ["deepseek-r1-qwen-32b", "deepseek-r1-qwen-14b"],
    "glm": ["glm-5", "glm-47-flash"],
    "devstral": ["devstral-24b"],
    "nemotron": ["nemotron3-nano-4b"],
}


def resolve_model_family(query, gpu_total_mb):
    """Resolve a family alias or colon syntax to the best variant that fits.

    Supports:
        "gemma4"      → best variant for your GPU
        "gemma4:e4b"  → exact variant (Ollama-style)
        "gemma4:26b"  → exact variant

    Returns the model ID string, or None if no match.
    """
    query = query.lower().strip()

    # Handle colon syntax: "gemma4:e4b" → "gemma4-e4b"
    if ":" in query:
        as_dash = query.replace(":", "-")
        if as_dash in MODELS:
            return as_dash
        # Try family + variant substring match
        family, variant = query.split(":", 1)
        variants = MODEL_FAMILIES.get(family, [])
        for mid in variants:
            if variant in mid:
                return mid
        return None

    # Family alias: auto-pick best that fits
    variants = MODEL_FAMILIES.get(query)
    if not variants:
        return None

    for mid in variants:
        m = MODELS[mid]
        if m["size_gb"] * 1024 <= gpu_total_mb:
            return mid

    # Nothing fits — return smallest with a warning flag
    return variants[-1]


def _parse_footprint_mb(pid):
    """Get process memory footprint in MB using macOS footprint command."""
    if not IS_MAC:
        return 0
    try:
        fp = subprocess.run(
            ["/usr/bin/footprint", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in fp.stdout.splitlines():
            if "Footprint:" in line:
                parts = line.split("Footprint:")[1].strip().split()
                val = float(parts[0])
                unit = parts[1] if len(parts) > 1 else "KB"
                if "GB" in unit:
                    return int(val * 1024)
                elif "MB" in unit:
                    return int(val)
                elif "KB" in unit:
                    return max(1, int(val / 1024))
                return int(val)
    except Exception:
        pass
    return 0


def get_system_ram_gb():
    """Get total system RAM in GB (macOS, Linux, WSL)."""
    try:
        if IS_MAC:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return int(out.stdout.strip()) // (1024**3)
        else:
            # Linux / WSL
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // (1024 * 1024)
    except:
        pass
    return 0


def get_machine_specs():
    """Get full machine specs: chip, cores, RAM, GPU memory breakdown."""
    specs = {
        "chip": "Unknown",
        "cpu_cores": 0,
        "gpu_cores": 0,
        "ram_gb": get_system_ram_gb(),
        "gpu_total_mb": 0,
        "gpu_used_mb": 0,
        "gpu_free_mb": 0,
        "gpu_processes": [],  # list of {name, pid, rss_mb}
        "mem_pressure": "unknown",
    }

    if IS_MAC:
        # Chip name
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            specs["chip"] = out.stdout.strip()
            if not specs["chip"] or "Apple" not in specs["chip"]:
                # Fallback for Apple Silicon
                out2 = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in out2.stdout.splitlines():
                    if "Chip" in line and ":" in line:
                        specs["chip"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        # CPU / GPU core counts
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            specs["cpu_cores"] = int(out.stdout.strip())
        except Exception:
            pass
        try:
            out = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in out.stdout.splitlines():
                if "Total Number of Cores" in line:
                    specs["gpu_cores"] = int(line.split(":")[-1].strip())
                    break
        except Exception:
            pass

        # Metal GPU budget — use real ioreg value, then check sysctl override
        # 1. Try ioreg for real Metal VRAM,totalMB
        try:
            import re as _re_ioreg

            _ioreg_out = subprocess.run(
                ["ioreg", "-l"], capture_output=True, text=True, timeout=10
            )
            for _line in _ioreg_out.stdout.splitlines():
                if "VRAM,totalMB" in _line:
                    _m = _re_ioreg.search(r'"VRAM,totalMB"=(\d+)', _line)
                    if _m:
                        specs["gpu_total_mb"] = int(_m.group(1))
                    break
        except Exception:
            pass

        # 2. Check if user overrode with iogpu.wired_limit_mb
        if specs["gpu_total_mb"] == 0:
            try:
                out = subprocess.run(
                    ["sysctl", "-n", "iogpu.wired_limit_mb"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                custom_limit = int(out.stdout.strip())
                if custom_limit > 0:
                    specs["gpu_total_mb"] = custom_limit
            except Exception:
                pass

        # 3. Fallback to estimate
        if specs["gpu_total_mb"] == 0:
            specs["gpu_total_mb"] = int(specs["ram_gb"] * 1024 * 0.67)

        # Find GPU-heavy processes (llama-server, ollama, any ML inference)
        gpu_proc_names = [
            "llama-server",
            "ollama",
            "ollama_llama_server",
            "mlx_lm",
            "whisper",
        ]
        try:
            out = subprocess.run(
                ["ps", "axo", "pid,comm"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in out.stdout.splitlines()[1:]:
                parts = line.split()
                if len(parts) < 2:
                    continue
                pid, comm = parts[0], parts[1]
                name = os.path.basename(comm)
                if not any(gp in name for gp in gpu_proc_names):
                    continue
                mem_mb = _parse_footprint_mb(pid)
                if mem_mb < 10:
                    # Fallback to RSS
                    try:
                        rss = subprocess.run(
                            ["ps", "-o", "rss=", "-p", pid],
                            capture_output=True,
                            text=True,
                        )
                        if rss.stdout.strip():
                            mem_mb = int(rss.stdout.strip()) // 1024
                    except Exception:
                        pass

                if mem_mb > 100:
                    specs["gpu_processes"].append(
                        {
                            "name": name,
                            "pid": int(pid),
                            "rss_mb": mem_mb,
                        }
                    )
        except Exception:
            pass

        specs["gpu_used_mb"] = sum(p["rss_mb"] for p in specs["gpu_processes"])
        specs["gpu_free_mb"] = max(0, specs["gpu_total_mb"] - specs["gpu_used_mb"])

        # Memory pressure
        try:
            out = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            level = int(out.stdout.strip())
            specs["mem_pressure"] = {
                0: "normal",
                1: "warn",
                2: "critical",
                4: "critical",
            }.get(level, "unknown")
        except Exception:
            pass

    elif IS_LINUX:
        # Linux / WSL
        try:
            with open("/proc/cpuinfo") as f:
                specs["cpu_cores"] = sum(
                    1 for line in f if line.startswith("processor")
                )
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                        specs["gpu_free_mb"] = avail_kb // 1024
        except Exception:
            pass

        # Check for NVIDIA GPU
        has_gpu = False
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                parts = out.stdout.strip().split(",")
                specs["chip"] = parts[0].strip()
                specs["gpu_total_mb"] = int(parts[1].strip())
                specs["gpu_used_mb"] = int(parts[2].strip())
                specs["gpu_free_mb"] = int(parts[3].strip())
                has_gpu = True
        except FileNotFoundError:
            pass

        if not has_gpu:
            # Try AMD via rocm-smi
            try:
                out = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0:
                    for line in out.stdout.splitlines()[1:]:
                        parts = line.split(",")
                        if len(parts) >= 3:
                            specs["gpu_total_mb"] = int(parts[0]) // (1024 * 1024)
                            specs["gpu_used_mb"] = int(parts[1]) // (1024 * 1024)
                            specs["gpu_free_mb"] = (
                                specs["gpu_total_mb"] - specs["gpu_used_mb"]
                            )
                            has_gpu = True
            except FileNotFoundError:
                pass

        if not has_gpu:
            # Detect Intel/other iGPU via PowerShell (WSL2) or lspci
            gpu_name = None
            gpu_vram_mb = 0
            if IS_WSL:
                try:
                    out = subprocess.run(
                        [
                            "powershell.exe",
                            "-Command",
                            "Get-CimInstance Win32_VideoController | Select-Object -First 1 Name, AdapterRAM | ConvertTo-Json",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        gpu_info = json.loads(out.stdout.strip())
                        gpu_name = gpu_info.get("Name", "")
                        adapter_ram = gpu_info.get("AdapterRAM", 0)
                        if adapter_ram:
                            gpu_vram_mb = adapter_ram // (1024 * 1024)
                except Exception:
                    pass
            if not gpu_name:
                try:
                    out = subprocess.run(
                        ["lspci"], capture_output=True, text=True, timeout=5
                    )
                    import re as _re_lspci

                    for line in out.stdout.splitlines():
                        if "VGA" in line or "3D" in line or "Display" in line:
                            gpu_name = _re_lspci.split(r":\s+", line, maxsplit=2)[
                                -1
                            ].strip()
                            break
                except Exception:
                    pass

            # CPU-only mode: use system RAM as the budget for model fitting
            specs["cpu_only"] = True
            specs["chip"] = gpu_name or f"CPU ({specs.get('cpu_cores', '?')} cores)"
            specs["gpu_name"] = gpu_name
            specs["gpu_vram_mb"] = gpu_vram_mb
            # Models run in system RAM on CPU — use available RAM as budget
            specs["gpu_total_mb"] = specs["ram_gb"] * 1024
            specs["gpu_free_mb"] = specs["gpu_total_mb"]

    return specs


def cleanup_gpu_memory(force=False):
    """Free GPU memory by unloading idle models and killing stale processes.

    Returns dict with what was cleaned up.
    """
    cleaned = {"ollama_unloaded": [], "processes_killed": [], "freed_mb": 0}

    # 1. Unload Ollama models (set keep_alive=0)
    if check_backend_running("ollama"):
        try:
            models = get_running_models("ollama")
            for m in models:
                urllib.request.urlopen(
                    urllib.request.Request(
                        "http://127.0.0.1:11434/api/generate",
                        data=json.dumps({"model": m, "keep_alive": 0}).encode(),
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=5,
                )
                cleaned["ollama_unloaded"].append(m)
        except Exception:
            pass

    # 2. Kill stale llama-server processes (if force or not our session)
    if force:
        try:
            out = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True,
                text=True,
            )
            for pid in out.stdout.strip().splitlines():
                pid = pid.strip()
                if pid:
                    rss = subprocess.run(
                        ["ps", "-o", "rss=", "-p", pid],
                        capture_output=True,
                        text=True,
                    )
                    mb = int(rss.stdout.strip()) // 1024 if rss.stdout.strip() else 0
                    subprocess.run(["kill", pid], timeout=3)
                    cleaned["processes_killed"].append(
                        {"pid": int(pid), "freed_mb": mb}
                    )
                    cleaned["freed_mb"] += mb
        except Exception:
            pass

    # Give time for memory to be released
    if cleaned["ollama_unloaded"] or cleaned["processes_killed"]:
        time.sleep(2)

    return cleaned


def get_top_memory_processes(min_mb=80, limit=12):
    """Get top memory-consuming processes with accurate footprint.

    Categorizes processes as:
    - 'ml': ML inference servers (llama-server, ollama)
    - 'app': User apps (Chrome, Slack, etc.)
    - 'system': System processes (WindowServer, kernel_task)
    """
    SYSTEM_PROCS = {
        "WindowServer",
        "WindowManager",
        "kernel_task",
        "launchd",
        "mds",
        "mds_stores",
        "opendirectoryd",
        "fseventsd",
        "corebrightnessd",
        "bluetoothd",
        "nearbyd",
        "systemstats",
        "loginwindow",
        "Dock",
        "Finder",
        "SystemUIServer",
        "ControlCenter",
        "NotificationCenter",
        "Terminal",
        "iTerm2",
        "zsh",
        "bash",
        "sh",
    }
    # System procs safe to kill (macOS auto-restarts them lean, freeing bloated memory)
    # Maps name → description for the debloat wizard
    SYSTEM_RESTARTABLE = {
        "CoreLocationAgent": "Location services cache — often leaks to 8GB+",
        "CacheDeleteExtension": "Storage cleanup daemon — bloats during disk scans",
        "remindd": "Reminders sync daemon — known memory leak on macOS 15",
        "suggestd": "Siri suggestions indexer — heavy background ML",
        "photoanalysisd": "Photos face/scene ML analysis — runs after imports",
        "mediaanalysisd": "Media ML classifier — visual lookup, Live Text",
        "nsurlsessiond": "Background network downloads — iCloud sync cache",
        "cloudd": "iCloud Drive sync daemon — bloats with many files",
        "bird": "CloudKit/iCloud container daemon",
        "callservicesd": "FaceTime/phone call routing daemon",
        "SafariLaunchAgent": "Safari preload — keeps old pages in memory",
        "SoftwareUpdateNotificationManager": "macOS update checker — safe to kill",
        "com.apple.WebKit.Networking": "WebKit network process — cache bloat",
    }
    ML_PROCS = {
        "llama-server",
        "ollama",
        "ollama_llama_server",
        "mlx_lm",
        "whisper",
        "vllm",
        "tgi",
    }

    procs = []
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,rss=,comm="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Pre-filter by RSS to avoid calling footprint on hundreds of tiny processes
        candidates = []
        for line in out.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            pid, rss_kb, comm = parts[0], parts[1], " ".join(parts[2:])
            try:
                rss_mb = int(rss_kb) // 1024
            except ValueError:
                continue
            if rss_mb < min_mb // 4:  # loose pre-filter
                continue
            candidates.append((pid, rss_mb, comm))

        # Sort by RSS descending, only footprint top N candidates (fast)
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: limit * 3]  # check 3x limit, take top N

        for pid, rss_mb, comm in candidates:
            name = os.path.basename(comm.split()[0]) if comm else "?"

            # Use RSS directly (fast) — footprint is 0.3s per process
            fp_mb = rss_mb

            if fp_mb < min_mb:
                continue

            # Categorize
            if name in ML_PROCS or any(ml in name for ml in ML_PROCS):
                category = "ml"
            elif name in SYSTEM_PROCS:
                category = "system"
            elif name in SYSTEM_RESTARTABLE or any(
                sr in name for sr in SYSTEM_RESTARTABLE
            ):
                category = "bloat"
            else:
                category = "app"

            procs.append(
                {
                    "pid": int(pid),
                    "name": name,
                    "mb": fp_mb,
                    "category": category,
                    "killable": category not in ("system",),
                }
            )
    except Exception:
        pass

    # Normalize names for grouping
    def _group_name(name):
        # Group all Chrome helpers under "Chrome"
        if "Google" in name or "Chrome" in name:
            return "Chrome"
        return name

    grouped = {}
    for p in procs:
        key = _group_name(p["name"])
        if key in grouped:
            grouped[key]["mb"] += p["mb"]
            grouped[key]["count"] += 1
            grouped[key]["pids"].append(p["pid"])
        else:
            grouped[key] = {**p, "name": key, "count": 1, "pids": [p["pid"]]}

    result = sorted(grouped.values(), key=lambda x: x["mb"], reverse=True)
    return result[:limit]


def print_machine_specs(specs=None):
    """Print a compact machine specs panel using Rich."""
    if specs is None:
        specs = get_machine_specs()

    ram = specs["ram_gb"]
    gpu_total = specs["gpu_total_mb"]
    gpu_used = specs["gpu_used_mb"]
    gpu_free = specs["gpu_free_mb"]

    # Color code free GPU memory
    if gpu_free > 14000:
        free_color = "green"
    elif gpu_free > 8000:
        free_color = "yellow"
    else:
        free_color = "red"

    pressure_color = {"normal": "green", "warn": "yellow", "critical": "red"}.get(
        specs["mem_pressure"], "dim"
    )

    is_cpu_only = specs.get("cpu_only", False)

    lines = [
        f"  [bold]{specs['chip']}[/]  ·  {specs['cpu_cores']} CPU"
        + (f" · {specs['gpu_cores']} GPU cores" if specs["gpu_cores"] else ""),
    ]

    if is_cpu_only:
        gpu_name = specs.get("gpu_name", "None")
        igpu_vram = specs.get("gpu_vram_mb", 0)
        lines.append(
            f"  RAM: [bold]{ram}GB[/] total  ·  [yellow]CPU-only mode[/] (no dedicated GPU)"
        )
        if gpu_name:
            igpu_info = f"  iGPU: {gpu_name}"
            if igpu_vram:
                igpu_info += f" ({igpu_vram // 1024}GB shared)"
            lines.append(igpu_info)
        lines.append(
            f"  Available for models: [{free_color}]{gpu_free // 1024}GB[/{free_color}] system RAM"
            + f"  [dim](models run in RAM on CPU — slower but works)[/]"
        )
    else:
        gpu_label = "Metal GPU budget" if IS_MAC else "GPU VRAM"
        lines.append(
            f"  RAM: [bold]{ram}GB[/] total  ·  {gpu_label}: [bold]{gpu_total // 1024}GB[/]"
            + (
                f"  ·  pressure: [{pressure_color}]{specs['mem_pressure']}[/{pressure_color}]"
                if specs["mem_pressure"] != "unknown"
                else ""
            )
        )
        lines.append(
            f"  GPU VRAM: [{free_color}]{gpu_free // 1024}GB free[/{free_color}]"
            + f"  ·  {gpu_used // 1024}GB used  ·  {gpu_total // 1024}GB total"
        )

    if specs["gpu_processes"]:
        procs = "  GPU processes: " + ", ".join(
            f"[cyan]{p['name']}[/] ({p['rss_mb'] // 1024}GB)"
            for p in specs["gpu_processes"]
        )
        lines.append(procs)

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Machine Specs[/]",
            border_style="dim",
            padding=(0, 1),
        )
    )


def _detect_model_info(server_config, model_id=None):
    """Detect model name, quant level, and file size from model path or model_id."""
    info = {"name": None, "quant": None, "size_gb": None}

    # Try model_id first
    if model_id and model_id in MODELS:
        m = MODELS[model_id]
        info["name"] = m["name"].split(" Q")[0] if " Q" in m["name"] else m["name"]
        info["size_gb"] = m["size_gb"]
        # Extract quant from name
        for part in m["name"].split():
            if part.startswith("Q") and "_" in part:
                info["quant"] = part
                break

    # Try to parse from model path
    model_path = server_config.get("model_path", "") or ""
    if model_path:
        import re

        basename = os.path.basename(model_path)

        # Detect quant from filename (e.g., Q3_K_XL, Q4_K_M, Q8_0)
        quant_match = re.search(
            r"(Q\d+_K(?:_[A-Z]+)?|Q\d+_\d+|IQ\d+_[A-Z]+)", basename, re.IGNORECASE
        )
        if quant_match:
            info["quant"] = quant_match.group(1).upper()

        # Detect model name from path
        name_patterns = [
            (r"gemma[-_]?4[-_]?(\d+[bB])", "Gemma 4"),
            (r"qwen[-_]?3\.?5[-_]?(\d+[bB])", "Qwen 3.5"),
            (r"llama[-_]?3[-_.]?(\d+[bB])", "Llama 3"),
            (r"mistral[-_]?(\d+[bB])", "Mistral"),
            (r"phi[-_]?(\d+)", "Phi"),
        ]
        for pattern, prefix in name_patterns:
            m = re.search(pattern, basename, re.IGNORECASE)
            if m:
                info["name"] = f"{prefix} {m.group(1).upper()}"
                break

        # Detect file size
        if os.path.exists(model_path):
            try:
                size_bytes = os.path.getsize(model_path)
                info["size_gb"] = round(size_bytes / (1024**3), 1)
            except OSError:
                pass

    return info


def _build_dashboard_layout(model_id=None):
    """Build the full dashboard as a single Rich renderable (for clear-screen rendering)."""
    from rich.columns import Columns
    from rich.text import Text
    from rich.rule import Rule

    specs = get_machine_specs()
    diag = diagnose_gpu_health(model_id)
    top_procs = get_top_memory_processes(min_mb=80, limit=8)
    swap_mb = get_swap_usage_mb()

    # ── Status Bar (full-width colored line) ──
    status_map = {
        "healthy": ("green", "HEALTHY"),
        "degraded": ("yellow", "DEGRADED"),
        "critical": ("red", "CRITICAL"),
        "unknown": ("dim", "UNKNOWN"),
    }
    sc, sl = status_map.get(diag["status"], ("dim", "?"))
    status_bar = Rule(title=f"[bold {sc}] {sl} [/bold {sc}]", style=sc)

    # ── Header ──
    header = Text()
    header.append(f"  {specs['chip']}  ·  {specs['ram_gb']}GB RAM  ·  ", style="bold")
    header.append(f"{specs.get('gpu_cores', '?')} GPU cores", style="bold")

    # ── Model Info Line ──
    model_info_obj = _detect_model_info(diag["server_config"], model_id)
    model_line = None
    if model_info_obj["name"]:
        parts = []
        parts.append(f"[bold cyan]{model_info_obj['name']}[/bold cyan]")
        if model_info_obj["quant"]:
            parts.append(f"[yellow]{model_info_obj['quant']}[/yellow]")
        if model_info_obj["size_gb"]:
            parts.append(f"[dim]{model_info_obj['size_gb']}GB[/dim]")
        model_line = Text.from_markup("  " + " · ".join(parts))

    # ── Status Cards (equal height, horizontal row) ──
    CARD_HEIGHT = 6  # content lines per card (excluding border)

    gpu_on = diag["on_gpu"]
    compute_lines = []
    if diag["server_config"].get("running"):
        icon = "[green]●[/]" if gpu_on else "[red]●[/]"
        compute_lines.append(f"{icon} {'GPU (Metal)' if gpu_on else 'CPU — SLOW!'}")
        compute_lines.append(f"  Layers: {diag['gpu_layers']}/99")
        compute_lines.append(f"  Util: {diag['gpu_util_pct']}%")
        compute_lines.append(
            f"  Model: {diag['server_config'].get('footprint_mb', 0)} MB"
        )
        if not gpu_on:
            compute_lines.append("[dim]GPU = 20x faster[/]")
            compute_lines.append("[dim]Use -ngl 99[/]")
    else:
        compute_lines.append("[dim]Server not running[/]")

    kv_lines = []
    kv_ok = diag["kv_quantized"]
    kv_icon = "[green]●[/]" if kv_ok else "[red]●[/]"
    kv_lines.append(f"{kv_icon} {'Quantized' if kv_ok else 'Full (2x mem!)'}")
    if diag["kv_type"]:
        kv_lines.append(f"  Type: {diag['kv_type']}")
    kv_lines.append(f"  Size: ~{diag['kv_cache_est_mb']} MB")
    kv_lines.append(f"  Ctx: {diag['context_size'] // 1024}K")
    fa_icon = "[green]●[/]" if diag["flash_attn"] else "[yellow]●[/]"
    kv_lines.append(f"{fa_icon} FlashAttn: {'on' if diag['flash_attn'] else 'off'}")

    pressure_color = {"normal": "green", "warn": "yellow", "critical": "red"}.get(
        diag["mem_pressure"], "dim"
    )
    swap_color = "red" if diag["swap_thrashing"] else "green"
    gpu_headroom = diag["gpu_total_mb"] - diag["gpu_alloc_mb"]
    hr_color = (
        "green" if gpu_headroom > 2048 else "yellow" if gpu_headroom > 0 else "red"
    )
    mem_lines = [
        f"  Pressure: [{pressure_color}]{diag['mem_pressure']}[/{pressure_color}]",
        f"  Swap: [{swap_color}]{swap_mb // 1024}GB[/{swap_color}]",
        f"  GPU: {diag['gpu_alloc_mb'] // 1024}/{diag['gpu_total_mb'] // 1024}GB",
        f"  Free: [{hr_color}]{gpu_headroom // 1024}GB[/{hr_color}]",
    ]
    if diag["swap_thrashing"]:
        mem_lines.append("[dim]Swap = 100x slower[/]")

    # Pad all cards to the same height
    for card_lines in (compute_lines, kv_lines, mem_lines):
        while len(card_lines) < CARD_HEIGHT:
            card_lines.append("")

    cards = Columns(
        [
            Panel(
                "\n".join(compute_lines),
                title="[bold]Compute[/]",
                border_style="cyan",
                width=26,
                padding=(0, 1),
            ),
            Panel(
                "\n".join(kv_lines),
                title="[bold]KV Cache[/]",
                border_style="cyan",
                width=26,
                padding=(0, 1),
            ),
            Panel(
                "\n".join(mem_lines),
                title="[bold]Memory[/]",
                border_style="cyan",
                width=26,
                padding=(0, 1),
            ),
        ],
        padding=1,
    )

    # ── VRAM Usage Bar ──
    gpu_budget_mb = (
        diag["gpu_total_mb"]
        if diag["gpu_total_mb"] > 0
        else (specs["ram_gb"] * 1024 * 75 // 100)
    )
    model_mb = diag.get("model_size_mb", 0) or (
        diag["server_config"].get("footprint_mb", 0)
    )
    kv_mb = diag["kv_cache_est_mb"]
    apps_mb = max(0, diag["gpu_alloc_mb"] - model_mb - kv_mb)
    free_mb = max(0, gpu_budget_mb - model_mb - kv_mb - apps_mb)

    BAR_WIDTH = 50
    total_for_bar = max(1, gpu_budget_mb)
    seg_model = max(0, int(BAR_WIDTH * model_mb / total_for_bar))
    seg_kv = max(0, int(BAR_WIDTH * kv_mb / total_for_bar))
    seg_apps = max(0, int(BAR_WIDTH * apps_mb / total_for_bar))
    seg_free = max(0, BAR_WIDTH - seg_model - seg_kv - seg_apps)

    vram_bar = Text()
    vram_bar.append("  VRAM ", style="bold")
    vram_bar.append("\u2588" * seg_model, style="cyan")
    vram_bar.append("\u2588" * seg_kv, style="magenta")
    vram_bar.append("\u2588" * seg_apps, style="yellow")
    vram_bar.append("\u2591" * seg_free, style="dim")
    vram_bar.append(f"  {gpu_budget_mb // 1024}GB", style="dim")

    vram_legend = Text.from_markup(
        "       [cyan]\u2588[/] Model"
        f" ({model_mb // 1024}G)"
        "  [magenta]\u2588[/] KV Cache"
        f" ({kv_mb // 1024}G)"
        "  [yellow]\u2588[/] Apps"
        f" ({apps_mb // 1024}G)"
        "  [dim]\u2591[/] Free"
        f" ({free_mb // 1024}G)"
    )

    # ── Process Table ──
    table = Table(
        show_header=True,
        header_style="bold",
        border_style="dim",
        padding=(0, 1),
        expand=False,
        width=82,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Process", min_width=18)
    table.add_column("Memory", justify="right", width=8)
    table.add_column("Type", width=6)
    table.add_column("", min_width=14)

    total_reclaimable = 0
    for i, p in enumerate(top_procs, 1):
        mb = p["mb"]
        name = p["name"]
        count = p.get("count", 1)
        label = f"{name}" + (f" \u00d7{count}" if count > 1 else "")

        cat_style = {
            "ml": "[cyan]ML[/]",
            "app": "[yellow]app[/]",
            "system": "[dim]sys[/]",
            "bloat": "[red]bloat[/]",
        }
        cat = cat_style.get(p["category"], "[dim]?[/]")

        bar_width = min(14, max(1, mb // 300))
        bar_color = "red" if mb > 2000 else "yellow" if mb > 500 else "green"
        _block = "\u2588"
        bar = f"[{bar_color}]{_block * bar_width}[/{bar_color}]"

        size_str = f"{mb / 1024:.1f}G" if mb >= 1024 else f"{mb}M"
        table.add_row(str(i), label, size_str, cat, bar)

        if p["category"] in ("app", "bloat") and p["killable"]:
            total_reclaimable += mb

    # ── Fixes ──
    fix_lines = []
    if diag["issues"]:
        for issue in diag["issues"]:
            fix_lines.append(f"  [red]\u25cf[/] {issue}")
        fix_lines.append("")

    # Bloat fixes
    for p in top_procs:
        if p["category"] == "bloat" and p["mb"] > 500:
            freed = p["mb"] // 1024
            fix_lines.append(
                f"  [green]\u2192[/] Kill {p['name']} [dim](~{freed}GB — auto-restarts lean)[/]"
            )
    # App fixes
    for p in top_procs:
        if p["category"] == "app" and p["mb"] > 500:
            count = p.get("count", 1)
            freed = p["mb"] // 1024
            name = p["name"]
            if name == "Chrome":
                fix_lines.append(
                    f"  [green]\u2192[/] Close Chrome tabs [dim]({count} procs = ~{freed}GB)[/]"
                )
            elif "claude" in name.lower():
                fix_lines.append(
                    f"  [green]\u2192[/] Close Claude windows [dim]({count} = ~{freed}GB)[/]"
                )
            elif freed >= 1:
                fix_lines.append(f"  [green]\u2192[/] Quit {name} [dim](~{freed}GB)[/]")

    if total_reclaimable > 2000:
        fix_lines.append("")
        fix_lines.append(
            f"  [bold]Reclaimable: ~{total_reclaimable // 1024}GB[/]  \u00b7  [dim]localcoder --cleanup[/]"
        )

    fixes_panel = None
    if fix_lines:
        border = (
            "red"
            if diag["status"] == "critical"
            else "yellow"
            if diag["status"] == "degraded"
            else "dim"
        )
        fixes_panel = Panel(
            "\n".join(fix_lines),
            title="[bold]Fixes[/]",
            border_style=border,
            padding=(0, 1),
        )

    # ── Glossary (noob-friendly, using a borderless Rich Table for alignment) ──
    glossary_table = Table(
        show_header=False,
        show_edge=False,
        show_lines=False,
        box=None,
        padding=(0, 1),
        expand=False,
    )
    glossary_table.add_column("Term", style="dim", width=14, no_wrap=True)
    glossary_table.add_column("Description", style="dim")

    glossary_entries = [
        (
            "KV Cache",
            "Stores conversation history in GPU. Grows with context length.\n"
            "128K ctx = 630MB (q4_0) or 1.2GB (f16). Use -ctk q4_0 to halve it.",
        ),
        (
            "Quantization",
            "Compresses model weights: Q3=small  Q4=sweet spot  Q8=best quality.\n"
            "Rule: ~0.7GB per 1B params at Q4. 26B Q3 = 12GB, Q4 = 18GB.",
        ),
        ("GPU Layers", "-ngl 99 = all on GPU (fast). Partial offload = 5-10x slower."),
        ("Flash Attn", "-fa on = memory-efficient attention. Always enable it."),
        ("Swap", "RAM overflow to disk. 100x slower. Keep under 2GB."),
        ("MoE", "Mixture of Experts -- only 4B of 26B active per token."),
        (
            "Metal Limit",
            "macOS reserves ~25% RAM. Override: sudo sysctl iogpu.wired_limit_mb=N",
        ),
    ]
    for term, desc in glossary_entries:
        glossary_table.add_row(term, desc)

    glossary = Panel(
        glossary_table,
        title="[bold dim]What do these mean?[/]",
        border_style="dim",
        padding=(0, 1),
    )

    return (
        status_bar,
        header,
        model_line,
        cards,
        vram_bar,
        vram_legend,
        table,
        fixes_panel,
        glossary,
        diag,
    )


def _build_status_bar(diag, specs):
    """Build a pinned bottom status bar like Claude Code / btop."""
    from rich.text import Text

    swap_mb = get_swap_usage_mb()
    gpu_alloc = diag.get("gpu_alloc_mb", 0)
    gpu_total = diag.get("gpu_total_mb", 0)
    pressure = diag.get("mem_pressure", "?")

    # Color-code values
    pc = {"normal": "green", "warn": "yellow", "critical": "red"}.get(pressure, "dim")
    sc = "red" if swap_mb > 4000 else "yellow" if swap_mb > 1000 else "green"
    gc = (
        "red"
        if gpu_alloc > gpu_total
        else "yellow"
        if gpu_alloc > gpu_total * 0.8
        else "green"
    )

    bar = Text()
    bar.append(" GPU ", style="bold white on blue")
    bar.append(" ")
    bar.append(f"{gpu_alloc // 1024}/{gpu_total // 1024}GB", style=gc)
    bar.append("  ")
    bar.append(" SWAP ", style="bold white on blue")
    bar.append(" ")
    bar.append(f"{swap_mb // 1024}GB", style=sc)
    bar.append("  ")
    bar.append(" MEM ", style="bold white on blue")
    bar.append(" ")
    bar.append(f"{pressure}", style=pc)
    bar.append("    ")

    # Shortcuts
    bar.append(" h ", style="bold black on white")
    bar.append(" health ", style="dim")
    bar.append(" c ", style="bold black on white")
    bar.append(" cleanup ", style="dim")
    bar.append(" d ", style="bold black on white")
    bar.append(" debloat ", style="dim")
    bar.append(" s ", style="bold black on white")
    bar.append(" simulate ", style="dim")

    return bar


def print_health_dashboard(model_id=None):
    """Render GPU health dashboard — clear screen, fixed width, status bar at bottom."""
    import shutil

    term_w, term_h = shutil.get_terminal_size()

    # Use a fixed-width console to prevent stretching
    from rich.console import Console as _Console

    out = _Console(width=min(90, term_w), highlight=False)

    # Phase 1: Loading spinner
    out.clear()
    loading = out.status(
        "[bold cyan]  Scanning GPU, processes, server...[/]", spinner="dots"
    )
    loading.start()

    specs = get_machine_specs()
    diag = diagnose_gpu_health(model_id)

    loading.stop()

    # Phase 2: Build layout
    result = _build_dashboard_layout(model_id)
    (
        status_bar_top,
        header,
        model_line,
        cards,
        vram_bar,
        vram_legend,
        table,
        fixes_panel,
        glossary,
        _diag,
    ) = result

    # Phase 3: Clear and render all at once
    out.clear()

    out.print(status_bar_top)
    out.print(header)
    if model_line:
        out.print(model_line)
    out.print()
    out.print(cards)
    out.print(vram_bar)
    out.print(vram_legend)
    out.print()
    out.print(table)
    if fixes_panel:
        out.print(fixes_panel)

    # Glossary only if space
    if term_h > 42:
        out.print(glossary)

    # Status bar at bottom (no ANSI cursor tricks — just print it)
    status_bar_widget = _build_status_bar(diag, specs)
    out.print()
    out.print(status_bar_widget)
    out.print()

    return diag


def check_backend_installed(backend_id):
    """Check if a backend binary/package exists."""
    if backend_id == "mlx":
        return check_mlx_available()
    b = BACKENDS[backend_id]
    binary = b["binary"]
    if binary.exists():
        return True
    if shutil.which(binary.name):
        return True
    return False


def check_backend_running(backend_id):
    """Check if backend server is responding."""
    b = BACKENDS[backend_id]
    port = b["default_port"]
    try:
        url = f"http://127.0.0.1:{port}/v1/models"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return True
    except:
        return False


def get_running_models(backend_id):
    """Get list of models from a running backend."""
    b = BACKENDS[backend_id]
    port = b["default_port"]
    try:
        url = f"http://127.0.0.1:{port}/v1/models"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        return [m.get("id", "") for m in data.get("data", [])]
    except:
        return []


def discover_all():
    """Discover all backends and their models."""
    results = []
    for bid, b in BACKENDS.items():
        installed = check_backend_installed(bid)
        running = check_backend_running(bid) if installed else False
        models = get_running_models(bid) if running else []
        results.append(
            {
                "id": bid,
                "name": b["name"],
                "installed": installed,
                "running": running,
                "models": models,
                "port": b["default_port"],
            }
        )
    return results


def install_backend(backend_id):
    """Install a backend (macOS, Linux, WSL)."""
    b = BACKENDS[backend_id]
    console.print(f"\n  [bold]Installing {b['name']}...[/]")
    console.print(f"  [dim]{b['install_cmd']}[/]\n")

    r = subprocess.run(["bash", "-c", b["install_cmd"]], timeout=600)
    if r.returncode == 0:
        # Re-discover binary path after install
        BACKENDS[backend_id]["binary"] = _find_binary(
            "llama-server" if backend_id == "llamacpp" else "ollama",
            [BACKENDS[backend_id]["binary"]],
        )
    return r.returncode == 0


def download_model_hf(model_id):
    """Download a model from HuggingFace."""
    m = MODELS[model_id]
    if not m.get("hf_repo"):
        console.print(f"  [red]No HuggingFace repo for {model_id}[/]")
        return None

    local_dir = MODELS_DIR / model_id
    console.print(f"\n  [bold]Downloading {m['name']}...[/]")
    console.print(f"  [dim]From: {m['hf_repo']}[/]")
    console.print(f"  [dim]To:   {local_dir}[/]")
    console.print(f"  [dim]Size: ~{m['size_gb']} GB[/]\n")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print(f"  [yellow]Installing huggingface_hub...[/]")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "huggingface_hub",
                "--break-system-packages",
                "-q",
            ],
            timeout=120,
        )
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            console.print(
                f"  [red]Failed to install huggingface_hub. Run: pip install huggingface_hub[/]"
            )
            return None

    try:
        # Suppress tqdm line spam
        import logging as _dl

        _dl.getLogger("huggingface_hub").setLevel(_dl.ERROR)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TQDM_DISABLE"] = "1"

        # Download model file
        patterns = m.get("hf_pattern", "*").split(",") if m.get("hf_pattern") else None
        # Also download mmproj (vision) if available
        if patterns:
            patterns = [p.strip() for p in patterns] + ["mmproj-BF16*"]

        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TimeRemainingColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[dim]{task.fields[size]}[/]"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"  {m['name']}", total=None, size=f"~{m['size_gb']}GB"
            )
            snapshot_download(
                repo_id=m["hf_repo"],
                local_dir=str(local_dir),
                allow_patterns=patterns,
            )
            progress.update(task, completed=100, total=100)

        console.print(f"  [green]✓ Downloaded to {local_dir}[/]")
        return str(local_dir)
    except Exception as e:
        console.print(f"  [red]Download failed: {e}[/]")
        return None


def download_model_ollama(model_id):
    """Pull a model via Ollama."""
    m = MODELS[model_id]
    tag = m.get("ollama_tag")
    if not tag:
        return False
    console.print(f"\n  [bold]Pulling {tag} via Ollama...[/]")
    r = subprocess.run(["ollama", "pull", tag], timeout=1800)
    return r.returncode == 0


def find_model_file(model_id):
    """Find the GGUF file for a model."""
    local_dir = MODELS_DIR / model_id
    if not local_dir.exists():
        # Check HF cache
        cache_dir = HOME / ".cache/huggingface/hub"
        m = MODELS.get(model_id, {})
        if m.get("hf_repo"):
            repo_dir = cache_dir / f"models--{m['hf_repo'].replace('/', '--')}"
            if repo_dir.exists():
                for f in repo_dir.rglob("*.gguf"):
                    if "mmproj" not in f.name:
                        return str(f)
        return None

    # Find the GGUF file in local dir
    for f in local_dir.rglob("*.gguf"):
        if "mmproj" not in f.name:
            return str(f)
    return None


def find_mmproj_file(model_id):
    """Find the vision projector file for a model."""
    local_dir = MODELS_DIR / model_id
    search_dirs = [local_dir]

    # Also check HF cache
    m = MODELS.get(model_id, {})
    if m.get("hf_repo"):
        cache_dir = (
            HOME
            / ".cache/huggingface/hub"
            / f"models--{m['hf_repo'].replace('/', '--')}"
        )
        search_dirs.append(cache_dir)

    for d in search_dirs:
        if d.exists():
            for f in d.rglob("*mmproj*"):
                return str(f)
    return None


def start_llama_server(model_id, port=8089):
    """Start llama-server with a model. Auto-installs if missing."""
    m = MODELS.get(model_id, {})
    model_file = find_model_file(model_id)
    if not model_file:
        console.print(f"  [red]Model file not found for {model_id}[/]")
        return None

    # Find llama-server — check multiple locations
    binary = str(BACKENDS["llamacpp"]["binary"])
    if not os.path.exists(binary):
        binary = shutil.which("llama-server")
    if not binary:
        # Try ~/.local/bin (where our pre-built installer puts it)
        local_bin = os.path.join(str(HOME), ".local", "bin", "llama-server")
        if os.path.exists(local_bin):
            binary = local_bin
    if not binary:
        # Auto-install
        from localfit.prerequisites import ensure_llama_server

        binary = ensure_llama_server(auto_install=True)
    if not binary:
        console.print(f"  [red]llama-server not found. Run: localfit --check[/]")
        return None

    # Get flags — auto-size context to fit available VRAM
    specs = get_machine_specs()
    gpu_total_mb = specs.get("gpu_total_mb", 0)
    model_size_mb = int(m.get("size_gb", 0) * 1024)

    # Find mmproj early so we can account for its VRAM cost in context sizing
    mmproj = find_mmproj_file(model_id)
    mmproj_mb = 0
    if mmproj:
        try:
            mmproj_mb = int(os.path.getsize(mmproj) / (1024 * 1024))
        except OSError:
            mmproj_mb = 1400  # ~1.4GB conservative estimate

    if specs.get("cpu_only"):
        default_flags = "-ngl 0 -c 8192 --jinja"
    else:
        # Dynamic context: use available VRAM after model + mmproj load
        # KV cache cost ≈ 60MB per 1K context (Q4_0 quantized KV)
        free_mb = max(
            0, gpu_total_mb - model_size_mb - mmproj_mb - 512
        )  # 512MB headroom
        kv_per_1k = 60  # MB per 1K tokens with -ctk q4_0 -ctv q4_0
        max_ctx = int(free_mb / kv_per_1k * 1024)
        # Clamp to reasonable range
        max_ctx = max(4096, min(max_ctx, 131072))
        # Round to nearest power-of-2-ish
        for nice in [131072, 65536, 32768, 16384, 8192, 4096]:
            if max_ctx >= nice:
                max_ctx = nice
                break
        default_flags = f"-ngl 99 -c {max_ctx} --jinja"

    flags = m.get("server_flags", default_flags).split()

    # Override context size dynamically if model has hardcoded flags
    if not specs.get("cpu_only") and gpu_total_mb > 0:
        for i, f in enumerate(flags):
            if f == "-c" and i + 1 < len(flags):
                # Recalculate optimal context for this hardware
                free_mb = max(0, gpu_total_mb - model_size_mb - mmproj_mb - 512)
                kv_per_1k = 60
                optimal_ctx = int(free_mb / kv_per_1k * 1024)
                optimal_ctx = max(4096, min(optimal_ctx, 131072))
                for nice in [131072, 65536, 32768, 16384, 8192, 4096]:
                    if optimal_ctx >= nice:
                        optimal_ctx = nice
                        break
                flags[i + 1] = str(optimal_ctx)
                break

    # Override -ngl for CPU-only even if model has GPU flags
    if specs.get("cpu_only"):
        for i, f in enumerate(flags):
            if f == "-ngl" and i + 1 < len(flags):
                flags[i + 1] = "0"

    cmd = [binary, "-m", model_file, "--port", str(port)] + flags

    # Set LD_LIBRARY_PATH for pre-built binaries (libs are next to binary)
    env = os.environ.copy()
    bin_dir = os.path.dirname(binary)
    ld = env.get("LD_LIBRARY_PATH", "")
    if bin_dir not in ld:
        env["LD_LIBRARY_PATH"] = bin_dir + (":" + ld if ld else "")

    # Add mmproj if available (required for VLMs to process images)
    if mmproj:
        cmd += ["--mmproj", mmproj]
        console.print(
            f"  [magenta]Vision projector: {os.path.basename(mmproj)} ({mmproj_mb}MB)[/]"
        )

    console.print(
        f"  [dim]Starting: {' '.join(os.path.basename(c) if '/' in c else c for c in cmd[:6])}...[/]"
    )

    import tempfile

    stderr_log = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, prefix="llama-"
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_log, env=env)

    # Wait for server — check both health endpoint and process liveness
    for i in range(90):
        # Check if process crashed
        if proc.poll() is not None:
            stderr_log.close()
            try:
                err = open(stderr_log.name).read()[-500:]
            except Exception:
                err = ""
            console.print(f"  [red]Server crashed (exit code {proc.returncode})[/]")
            if err:
                console.print(f"  [dim]{err}[/]")
            os.unlink(stderr_log.name)
            return None

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=1):
                stderr_log.close()
                os.unlink(stderr_log.name)
                console.print(f"  [green]✓ Server ready on port {port}[/]")
                console.print(f"  [dim]API: http://127.0.0.1:{port}/v1[/]")
                return proc
        except Exception:
            if i % 10 == 9:
                console.print(f"  [dim]Loading model... ({i + 1}s)[/]")
            time.sleep(1)

    stderr_log.close()
    try:
        err = open(stderr_log.name).read()[-500:]
    except Exception:
        err = ""
    console.print(f"  [red]Server failed to start (timeout 90s)[/]")
    if err:
        console.print(f"  [dim]{err}[/]")
    os.unlink(stderr_log.name)
    proc.kill()
    return None


def get_gpu_memory_info():
    """Get GPU memory total and available (macOS Metal)."""
    info = {"total_mb": 0, "free_mb": 0, "used_by_llama_mb": 0}
    ram = get_system_ram_gb()
    if IS_MAC:
        # Metal GPU limit is ~67% of unified memory
        info["total_mb"] = int(ram * 1024 * 0.67)
        info["free_mb"] = info["total_mb"]

        # Check if llama-server is using GPU
        try:
            out = subprocess.run(
                ["pgrep", "-f", "llama-server"], capture_output=True, text=True
            )
            if out.stdout.strip():
                pid = out.stdout.strip().split()[0]
                rss = subprocess.run(
                    ["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True
                )
                if rss.stdout.strip():
                    info["used_by_llama_mb"] = int(rss.stdout.strip()) // 1024
                    info["free_mb"] = max(
                        0, info["total_mb"] - info["used_by_llama_mb"]
                    )
        except:
            pass
    else:
        info["total_mb"] = ram * 1024
        info["free_mb"] = info["total_mb"]
    return info


def get_llama_server_config():
    """Parse running llama-server process flags and API state."""
    config = {
        "running": False,
        "pid": None,
        "model_path": None,
        "ngl": 0,  # GPU layers (-ngl)
        "n_ctx": 0,  # Context size (-c)
        "kv_quant": None,  # KV cache quantization type (-ctk/-ctv)
        "flash_attn": False,  # Flash attention (-fa)
        "footprint_mb": 0,  # Process memory footprint
        "flags": [],
    }

    try:
        out = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True,
            text=True,
        )
        pids = out.stdout.strip().splitlines()
        if not pids:
            return config
        config["running"] = True
        config["pid"] = int(pids[0].strip())

        # Get full command line
        cmd_out = subprocess.run(
            ["ps", "-o", "args=", "-p", str(config["pid"])],
            capture_output=True,
            text=True,
        )
        args = cmd_out.stdout.strip().split()
        config["flags"] = args

        # Parse flags
        for i, arg in enumerate(args):
            if arg == "-ngl" and i + 1 < len(args):
                config["ngl"] = int(args[i + 1])
            elif arg == "-c" and i + 1 < len(args):
                config["n_ctx"] = int(args[i + 1])
            elif arg == "-ctk" and i + 1 < len(args):
                config["kv_quant"] = args[i + 1]
            elif arg == "-fa":
                config["flash_attn"] = True
            elif arg == "-m" and i + 1 < len(args):
                config["model_path"] = args[i + 1]

        # Get process memory footprint
        if IS_MAC:
            config["footprint_mb"] = _parse_footprint_mb(config["pid"])
        else:
            try:
                rss = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(config["pid"])],
                    capture_output=True,
                    text=True,
                )
                if rss.stdout.strip():
                    config["footprint_mb"] = int(rss.stdout.strip()) // 1024
            except Exception:
                pass

    except Exception:
        pass

    return config


def get_metal_gpu_stats():
    """Get real GPU stats — Metal on macOS, nvidia-smi on Linux."""
    stats = {
        "total_mb": 0,
        "alloc_mb": 0,
        "in_use_mb": 0,
        "free_vram_bytes": 0,
        "utilization_pct": 0,
        "temperature_c": None,
        "fan_pct": None,
        "power_w": None,
        "gpu_name": None,
    }

    if IS_MAC:
        try:
            import re

            out = subprocess.run(
                ["ioreg", "-l"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in out.stdout.splitlines():
                if "VRAM,totalMB" in line:
                    m = re.search(r'"VRAM,totalMB"=(\d+)', line)
                    if m:
                        stats["total_mb"] = int(m.group(1))
                if "PerformanceStatistics" in line and "Alloc system memory" in line:
                    m = re.search(r'"Alloc system memory"=(\d+)', line)
                    if m:
                        stats["alloc_mb"] = int(m.group(1)) // (1024 * 1024)
                    m2 = re.search(r'"In use system memory"=(\d+)', line)
                    if m2:
                        stats["in_use_mb"] = int(m2.group(1)) // (1024 * 1024)
                    m3 = re.search(r'"Device Utilization %"=(\d+)', line)
                    if m3:
                        stats["utilization_pct"] = int(m3.group(1))

            # Thermal state on macOS (approximate — no direct GPU temp on Apple Silicon)
            try:
                therm = subprocess.run(
                    ["pmset", "-g", "therm"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if "CPU_Scheduler_Limit" in therm.stdout:
                    # Thermal throttling active
                    stats["temperature_c"] = 95  # approximate
            except Exception:
                pass

        except Exception:
            pass

    elif IS_LINUX:
        # nvidia-smi for NVIDIA GPUs
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,fan.speed,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                parts = [p.strip() for p in out.stdout.strip().split(",")]
                if len(parts) >= 8:
                    stats["gpu_name"] = parts[0]
                    stats["total_mb"] = int(float(parts[1]))
                    stats["alloc_mb"] = int(float(parts[2]))
                    stats["in_use_mb"] = int(float(parts[2]))
                    stats["utilization_pct"] = int(float(parts[5]))
                    try:
                        stats["temperature_c"] = int(float(parts[6]))
                    except (ValueError, IndexError):
                        pass
                    try:
                        stats["fan_pct"] = int(float(parts[7].replace("%", "")))
                    except (ValueError, IndexError):
                        pass
                    try:
                        stats["power_w"] = float(parts[8])
                    except (ValueError, IndexError):
                        pass
        except FileNotFoundError:
            # No NVIDIA GPU — check for AMD via rocm-smi
            try:
                out = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0:
                    for line in out.stdout.splitlines()[1:]:
                        parts = line.split(",")
                        if len(parts) >= 3:
                            stats["total_mb"] = int(parts[0]) // (1024 * 1024)
                            stats["alloc_mb"] = int(parts[1]) // (1024 * 1024)
            except FileNotFoundError:
                pass

        # If still no GPU detected, try Intel iGPU via PowerShell (WSL2)
        if stats["total_mb"] == 0 and IS_WSL:
            try:
                out = subprocess.run(
                    [
                        "powershell.exe",
                        "-Command",
                        "Get-CimInstance Win32_VideoController | Select-Object -First 1 Name, AdapterRAM | ConvertTo-Json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if out.returncode == 0 and out.stdout.strip():
                    gpu_info = json.loads(out.stdout.strip())
                    stats["gpu_name"] = gpu_info.get("Name", "Intel GPU")
                    adapter_ram = gpu_info.get("AdapterRAM", 0)
                    if adapter_ram:
                        stats["total_mb"] = adapter_ram // (1024 * 1024)
                    stats["cpu_only"] = True
            except Exception:
                pass

    return stats


def get_disk_info():
    """Get disk space and ALL model storage info across tools."""
    info = {
        "disk_total_gb": 0,
        "disk_free_gb": 0,
        "hf_cache_gb": 0,
        "ollama_cache_gb": 0,
        "comfyui_gb": 0,
        "total_cache_gb": 0,
        "models": [],  # list of {name, size_gb, path, source}
        "docker_gb": 0,
    }
    try:
        # Disk space
        st = os.statvfs(HOME)
        info["disk_total_gb"] = round((st.f_blocks * st.f_frsize) / (1024**3))
        info["disk_free_gb"] = round((st.f_bavail * st.f_frsize) / (1024**3))

        # ── HuggingFace cache ──
        hf_cache = HOME / ".cache/huggingface/hub"
        if hf_cache.exists():
            total = 0
            for blob in hf_cache.rglob("*"):
                if blob.is_file() and not blob.is_symlink():
                    total += blob.stat().st_size
            info["hf_cache_gb"] = round(total / (1024**3))

            # Individual GGUF models
            for gguf in hf_cache.rglob("*.gguf"):
                name = gguf.name
                if "mmproj" in name.lower():
                    continue
                real = gguf.resolve()
                try:
                    sz = real.stat().st_size / (1024**3)
                    info["models"].append(
                        {
                            "name": name,
                            "size_gb": round(sz, 1),
                            "path": str(real),
                            "source": "HuggingFace",
                        }
                    )
                except OSError:
                    pass

            # Image model dirs in HF cache
            for model_dir in hf_cache.iterdir():
                if model_dir.name.startswith("models--") and (
                    "flux" in model_dir.name.lower()
                    or "klein" in model_dir.name.lower()
                    or "z-image" in model_dir.name.lower()
                ):
                    total_sz = sum(
                        f.stat().st_size
                        for f in model_dir.rglob("*")
                        if f.is_file() and not f.is_symlink()
                    )
                    sz_gb = round(total_sz / (1024**3), 1)
                    if sz_gb > 0.1:
                        clean_name = model_dir.name.replace("models--", "").replace(
                            "--", "/"
                        )
                        info["models"].append(
                            {
                                "name": f"{clean_name} (image)",
                                "size_gb": sz_gb,
                                "path": str(model_dir),
                                "source": "HuggingFace",
                            }
                        )

        # ── Ollama models ──
        ollama_dir = HOME / ".ollama/models"
        if ollama_dir.exists():
            total = 0
            for blob in (
                (ollama_dir / "blobs").rglob("*")
                if (ollama_dir / "blobs").exists()
                else []
            ):
                if blob.is_file():
                    total += blob.stat().st_size
            info["ollama_cache_gb"] = round(total / (1024**3))

            # List individual Ollama models from manifests
            manifests = ollama_dir / "manifests" / "registry.ollama.ai" / "library"
            if manifests.exists():
                for model_dir in manifests.iterdir():
                    if model_dir.is_dir():
                        for tag in model_dir.iterdir():
                            if tag.is_file():
                                try:
                                    mdata = json.loads(tag.read_text())
                                    sz = sum(
                                        l.get("size", 0)
                                        for l in mdata.get("layers", [])
                                    )
                                    sz_gb = round(sz / (1024**3), 1)
                                    if sz_gb > 0.1:
                                        info["models"].append(
                                            {
                                                "name": f"{model_dir.name}:{tag.name}",
                                                "size_gb": sz_gb,
                                                "path": str(tag),
                                                "source": "Ollama",
                                            }
                                        )
                                except Exception:
                                    pass

        # ── ComfyUI / SD-WebUI / Forge / Fooocus ──
        for ui_dir in [
            HOME / "ComfyUI/models",
            HOME / "comfyui/models",
            HOME / "stable-diffusion-webui/models",
            HOME / "stable-diffusion-forge/models",
            HOME / "Fooocus/models",
            Path("/opt/ComfyUI/models"),
        ]:
            if ui_dir.exists():
                total = 0
                for f in ui_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in (
                        ".safetensors",
                        ".ckpt",
                        ".bin",
                        ".gguf",
                        ".pt",
                    ):
                        sz = f.stat().st_size
                        total += sz
                        sz_gb = round(sz / (1024**3), 1)
                        if sz_gb > 0.1:
                            info["models"].append(
                                {
                                    "name": f.name,
                                    "size_gb": sz_gb,
                                    "path": str(f),
                                    "source": ui_dir.parent.name,
                                }
                            )
                info["comfyui_gb"] += round(total / (1024**3))

        info["total_cache_gb"] = (
            info["hf_cache_gb"] + info["ollama_cache_gb"] + info["comfyui_gb"]
        )
        info["models"].sort(key=lambda x: x["size_gb"], reverse=True)

        # Docker (if running)
        try:
            out = subprocess.run(
                ["docker", "system", "df", "--format", "{{.Size}}"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if out.returncode == 0:
                for line in out.stdout.strip().splitlines():
                    line = line.strip().upper()
                    if "GB" in line:
                        info["docker_gb"] += float(line.replace("GB", ""))
                    elif "MB" in line:
                        info["docker_gb"] += float(line.replace("MB", "")) / 1024
                info["docker_gb"] = round(info["docker_gb"])
        except (FileNotFoundError, Exception):
            pass
    except Exception:
        pass
    return info


def get_swap_usage_mb():
    """Get swap usage in MB."""
    try:
        if IS_MAC:
            out = subprocess.run(
                ["sysctl", "-n", "vm.swapusage"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            # "total = 10240.00M  used = 8538.06M  free = 1701.94M"
            for part in out.stdout.split():
                if (
                    part.endswith("M")
                    and "used"
                    not in out.stdout.split()[out.stdout.split().index(part) - 1]
                ):
                    continue
            import re

            m = re.search(r"used\s*=\s*([\d.]+)M", out.stdout)
            if m:
                return int(float(m.group(1)))
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("SwapTotal:"):
                        total = int(line.split()[1]) // 1024
                    if line.startswith("SwapFree:"):
                        free = int(line.split()[1]) // 1024
                        return total - free
    except Exception:
        pass
    return 0


def diagnose_gpu_health(model_id=None):
    """Full GPU health diagnostic. Returns dict with status and recommendations.

    Checks:
    1. Is model running on GPU or CPU?
    2. Is KV cache optimized?
    3. Is context size appropriate?
    4. Is swap thrashing happening?
    5. Are flags optimal?
    """
    diag = {
        "status": "unknown",  # "healthy", "degraded", "critical"
        "on_gpu": False,
        "gpu_layers": 0,
        "total_layers": 99,
        "kv_quantized": False,
        "kv_type": None,
        "flash_attn": False,
        "context_size": 0,
        "kv_cache_est_mb": 0,
        "model_size_mb": 0,
        "gpu_total_mb": 0,
        "gpu_alloc_mb": 0,
        "gpu_util_pct": 0,
        "swap_used_mb": 0,
        "swap_thrashing": False,
        "mem_pressure": "unknown",
        "issues": [],
        "fixes": [],
        "server_config": {},
    }

    # Get server config
    srv = get_llama_server_config()
    diag["server_config"] = srv

    if not srv["running"]:
        diag["status"] = "unknown"
        diag["issues"].append("llama-server not running")
        return diag

    # GPU layer offload
    diag["gpu_layers"] = srv["ngl"]
    diag["on_gpu"] = srv["ngl"] >= 90  # -ngl 99 means all on GPU
    diag["context_size"] = srv["n_ctx"]
    diag["flash_attn"] = srv["flash_attn"]

    if not diag["on_gpu"]:
        diag["issues"].append(
            f"Only {srv['ngl']} layers on GPU — model partially on CPU"
        )
        diag["fixes"].append("Restart with -ngl 99 to offload all layers to GPU")

    # KV cache
    diag["kv_type"] = srv["kv_quant"]
    diag["kv_quantized"] = srv["kv_quant"] in ("q4_0", "q8_0", "q4_1", "f16")
    if not diag["kv_quantized"]:
        diag["issues"].append(
            "KV cache not quantized — using full precision (2x memory)"
        )
        diag["fixes"].append(
            "Add -ctk q4_0 -ctv q4_0 to quantize KV cache (saves ~50% KV memory)"
        )

    if not diag["flash_attn"]:
        diag["issues"].append("Flash attention disabled — slower and more memory")
        diag["fixes"].append("Add -fa on to enable flash attention")

    # Estimate KV cache memory
    # For Gemma 4 26B: 5 global layers × 128K context × 2 (K+V) × hidden_dim
    # With q4_0: ~630MB. Without quantization: ~1.2GB
    if diag["context_size"] > 0:
        # Rough estimate: 128K ctx with q4_0 KV ≈ 630MB, without ≈ 1200MB
        ctx_ratio = diag["context_size"] / 131072
        if diag["kv_quantized"]:
            diag["kv_cache_est_mb"] = int(630 * ctx_ratio)
        else:
            diag["kv_cache_est_mb"] = int(1200 * ctx_ratio)

    # Model size
    if model_id and model_id in MODELS:
        diag["model_size_mb"] = int(MODELS[model_id]["size_gb"] * 1024)

    # Metal GPU stats
    metal = get_metal_gpu_stats()
    diag["gpu_total_mb"] = metal["total_mb"]
    diag["gpu_alloc_mb"] = metal["alloc_mb"]
    diag["gpu_util_pct"] = metal["utilization_pct"]

    # Swap check
    diag["swap_used_mb"] = get_swap_usage_mb()
    diag["swap_thrashing"] = diag["swap_used_mb"] > 4000  # >4GB swap = bad

    if diag["swap_thrashing"]:
        diag["issues"].append(
            f"Swap thrashing: {diag['swap_used_mb'] // 1024}GB in swap — major slowdown"
        )
        diag["fixes"].append(
            "Reduce context size (-c 32768) or use smaller quant to free GPU memory"
        )

    # Memory pressure
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        level = int(out.stdout.strip())
        diag["mem_pressure"] = {
            0: "normal",
            1: "warn",
            2: "critical",
            4: "critical",
        }.get(level, "unknown")
    except Exception:
        pass

    if diag["mem_pressure"] == "critical":
        diag["issues"].append("Critical memory pressure — system may kill processes")
        diag["fixes"].append("Run: localcoder --cleanup")

    # Context size warnings
    if diag["context_size"] > 65536 and not diag["kv_quantized"]:
        diag["issues"].append(
            f"Large context ({diag['context_size'] // 1024}K) without KV quantization"
        )
        diag["fixes"].append("Either reduce context or add -ctk q4_0 -ctv q4_0")

    # Check if Metal limit could be raised
    if IS_MAC and diag["gpu_total_mb"] > 0:
        ram_mb = get_system_ram_gb() * 1024
        current_limit = diag["gpu_total_mb"]
        max_safe = int(ram_mb * 0.90)  # leave 10% for system
        if current_limit < max_safe and diag["swap_thrashing"]:
            new_limit = max_safe
            diag["fixes"].append(
                f"Raise Metal GPU limit: sudo sysctl iogpu.wired_limit_mb={new_limit}"
                f" (current: {current_limit}MB, max safe: {new_limit}MB)"
            )

    # Overall status
    if not diag["issues"]:
        diag["status"] = "healthy"
    elif (
        diag["swap_thrashing"]
        or not diag["on_gpu"]
        or diag["mem_pressure"] == "critical"
    ):
        diag["status"] = "critical"
    else:
        diag["status"] = "degraded"

    return diag


def print_gpu_health(diag=None, model_id=None):
    """Print GPU health diagnostic panel."""
    if diag is None:
        diag = diagnose_gpu_health(model_id)

    status_style = {
        "healthy": ("green", "✓ Healthy"),
        "degraded": ("yellow", "⚠ Degraded"),
        "critical": ("red", "✗ Critical"),
        "unknown": ("dim", "? Unknown"),
    }
    color, label = status_style.get(diag["status"], ("dim", "?"))

    lines = []

    # GPU offload status
    if diag["server_config"].get("running"):
        gpu_icon = "[green]●[/] GPU" if diag["on_gpu"] else "[red]●[/] CPU (SLOW!)"
        lines.append(
            f"  Compute: {gpu_icon}  ·  {diag['gpu_layers']} layers offloaded  ·  GPU util: {diag['gpu_util_pct']}%"
        )

        # KV cache
        kv_icon = "[green]●[/]" if diag["kv_quantized"] else "[red]●[/]"
        kv_info = (
            f"quantized ({diag['kv_type']})"
            if diag["kv_quantized"]
            else "full precision (2x memory!)"
        )
        lines.append(
            f"  KV cache: {kv_icon} {kv_info}  ·  ~{diag['kv_cache_est_mb']}MB"
            f"  ·  context: {diag['context_size'] // 1024}K tokens"
        )

        # Flash attention
        fa_icon = "[green]●[/]" if diag["flash_attn"] else "[yellow]●[/]"
        lines.append(
            f"  Flash attn: {fa_icon} {'on' if diag['flash_attn'] else 'off'}"
            f"  ·  footprint: {diag['server_config'].get('footprint_mb', 0)}MB"
        )

    # Memory
    swap_color = "red" if diag["swap_thrashing"] else "green"
    pressure_color = {"normal": "green", "warn": "yellow", "critical": "red"}.get(
        diag["mem_pressure"], "dim"
    )
    lines.append(
        f"  Memory: [{pressure_color}]{diag['mem_pressure']}[/{pressure_color}]"
        f"  ·  swap: [{swap_color}]{diag['swap_used_mb'] // 1024}GB[/{swap_color}]"
        f"  ·  GPU alloc: {diag['gpu_alloc_mb'] // 1024}GB / {diag['gpu_total_mb'] // 1024}GB"
    )

    # Issues
    if diag["issues"]:
        lines.append("")
        for issue in diag["issues"]:
            lines.append(f"  [red]✗[/] {issue}")
    if diag["fixes"]:
        lines.append("")
        for fix in diag["fixes"]:
            lines.append(f"  [green]→[/] {fix}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold]GPU Health [{color}]{label}[/{color}][/]",
            border_style=color,
            padding=(0, 1),
        )
    )

    return diag


def auto_optimize_server(model_id=None):
    """Check if server needs optimization and apply fixes.

    Returns True if server was restarted with better flags.
    """
    diag = diagnose_gpu_health(model_id)

    if diag["status"] == "healthy":
        return False

    needs_restart = False
    srv = diag["server_config"]
    model_info = MODELS.get(model_id, {}) if model_id else {}
    optimal_flags = model_info.get("server_flags", "").split() if model_info else []

    # Check if current flags are suboptimal
    if not diag["on_gpu"] and "-ngl" not in " ".join(srv.get("flags", [])):
        needs_restart = True
    if not diag["kv_quantized"] and "-ctk" not in " ".join(srv.get("flags", [])):
        needs_restart = True
    if not diag["flash_attn"] and "-fa" not in " ".join(srv.get("flags", [])):
        needs_restart = True

    if needs_restart and model_id:
        console.print(
            "\n  [yellow]Server running with suboptimal flags — restarting with optimizations...[/]"
        )
        # Kill current server
        if srv.get("pid"):
            try:
                subprocess.run(["kill", str(srv["pid"])], timeout=5)
                time.sleep(2)
            except Exception:
                pass
        # Start with optimal flags
        proc = start_llama_server(model_id)
        if proc:
            console.print("  [green]✓ Server restarted with optimal GPU flags[/]")
            return True
        else:
            console.print("  [red]Failed to restart server[/]")

    # If swap thrashing, try to free memory without restart
    if diag["swap_thrashing"] and not needs_restart:
        console.print(
            "\n  [yellow]Swap thrashing detected — cleaning up GPU memory...[/]"
        )
        cleanup_gpu_memory(force=False)

    return False


# ── macOS Debloat categories for ML workloads ──
DEBLOAT_CATEGORIES = {
    "ml_hogs": {
        "name": "ML & Analysis Daemons",
        "desc": "Apple's background ML that competes with your model for GPU",
        "safe": True,
        "services": {
            "com.apple.photoanalysisd": "Photos face/scene ML — uses GPU + 2-8GB RAM",
            "com.apple.mediaanalysisd": "Visual Lookup, Live Text ML — GPU heavy",
            "com.apple.suggestd": "Siri suggestions indexer — background ML",
            "com.apple.intelligenced": "Apple Intelligence (Sequoia) — GPU heavy",
            "com.apple.mlruntime": "Core ML runtime — shared GPU compute",
        },
    },
    "location_bloat": {
        "name": "Location & Sync Bloat",
        "desc": "Known memory leakers on macOS 14/15",
        "safe": True,
        "services": {
            "com.apple.CoreLocationAgent": "Location cache — leaks to 8GB+ (notorious)",
            "com.apple.remindd": "Reminders sync — memory leak on macOS 15",
            "com.apple.cloudd": "iCloud Drive sync — bloats with many files",
            "com.apple.bird": "CloudKit container daemon",
        },
    },
    "telemetry": {
        "name": "Telemetry & Analytics",
        "desc": "Crash reports, analytics, diagnostics — zero impact to disable",
        "safe": True,
        "services": {
            "com.apple.analyticsd": "Analytics collection",
            "com.apple.ReportCrash": "Crash report generation",
            "com.apple.spindump": "CPU sampling diagnostics",
            "com.apple.DiagnosticReportCleanup": "Diagnostic cleanup",
            "com.apple.ap.adprivacyd": "Ad privacy daemon",
            "com.apple.ap.adservicesd": "Ad services",
            "com.apple.triald": "A/B testing framework",
        },
    },
    "siri_ai": {
        "name": "Siri & Apple AI",
        "desc": "Siri, assistant, Apple Intelligence",
        "safe": True,
        "services": {
            "com.apple.Siri.agent": "Siri main service",
            "com.apple.assistantd": "Assistant daemon",
            "com.apple.parsec.fbf": "Siri search suggestions",
            "com.apple.tipsd": "Tips and suggestions",
            "com.apple.ScreenTimeAgent": "Screen time tracking",
        },
    },
}


def debloat_wizard():
    """Interactive debloat wizard for ML workloads.

    Shows categories of services that can be disabled to free GPU/memory.
    User picks categories, we disable via launchctl.
    Creates restore script.
    """
    import shutil

    console.clear()
    console.print()
    console.print("  [bold]localfit debloat wizard[/]")
    console.print(
        "  [dim]Disable macOS services that compete with your model for GPU & memory[/]"
    )
    console.print(
        "  [dim]All changes are reversible — a restore script is saved automatically[/]\n"
    )

    # Show current bloated processes
    top_procs = get_top_memory_processes(min_mb=200, limit=5)
    bloat_procs = [p for p in top_procs if p["category"] == "bloat"]
    if bloat_procs:
        console.print("  [yellow]Currently bloated system processes:[/]")
        for p in bloat_procs:
            mb = p["mb"]
            size = f"{mb / 1024:.1f}GB" if mb >= 1024 else f"{mb}MB"
            desc = SYSTEM_RESTARTABLE.get(p["name"], "")
            console.print(f"    [red]●[/] {p['name']}  [bold]{size}[/]  [dim]{desc}[/]")
        console.print()

    # Show categories
    cats = list(DEBLOAT_CATEGORIES.items())
    for i, (key, cat) in enumerate(cats, 1):
        n_services = len(cat["services"])
        console.print(f"  [bold]{i}.[/] {cat['name']}  [dim]({n_services} services)[/]")
        console.print(f"     [dim]{cat['desc']}[/]")
        for svc, desc in list(cat["services"].items())[:3]:
            console.print(f"     [dim]  · {svc.split('.')[-1]}: {desc}[/]")
        if n_services > 3:
            console.print(f"     [dim]  + {n_services - 3} more[/]")
        console.print()

    console.print(
        f"  [bold]k.[/] Kill bloated processes now  [dim](one-time, they may restart)[/]"
    )
    console.print(f"  [bold]a.[/] All categories  [dim](maximum GPU headroom)[/]")
    console.print(f"  [bold]r.[/] Restore all  [dim](re-enable everything)[/]")
    console.print(f"  [bold]q.[/] Quit\n")

    try:
        ans = input("  Choose (e.g. 1,2 or a): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if ans == "q" or not ans:
        return

    if ans == "r":
        _debloat_restore()
        return

    if ans == "k":
        _kill_bloated_processes()
        return

    # Parse selection
    selected_cats = []
    if ans == "a":
        selected_cats = list(DEBLOAT_CATEGORIES.keys())
    else:
        for part in ans.replace(" ", "").split(","):
            try:
                idx = int(part) - 1
                if 0 <= idx < len(cats):
                    selected_cats.append(cats[idx][0])
            except ValueError:
                pass

    if not selected_cats:
        console.print("  [dim]No categories selected.[/]")
        return

    # Confirm
    total_services = sum(len(DEBLOAT_CATEGORIES[c]["services"]) for c in selected_cats)
    cat_names = ", ".join(DEBLOAT_CATEGORIES[c]["name"] for c in selected_cats)
    console.print(f"\n  [yellow]Will disable {total_services} services: {cat_names}[/]")
    try:
        confirm = input("  Proceed? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if confirm != "y":
        return

    # Disable services
    disabled = []
    restore_cmds = []
    for cat_key in selected_cats:
        cat = DEBLOAT_CATEGORIES[cat_key]
        for svc, desc in cat["services"].items():
            # Try both user and system domains
            for domain in [f"gui/{os.getuid()}", "system"]:
                cmd = ["launchctl", "disable", f"{domain}/{svc}"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                # Also bootout if currently loaded
                subprocess.run(
                    ["launchctl", "bootout", f"{domain}/{svc}"],
                    capture_output=True,
                    text=True,
                )
                restore_cmds.append(f"launchctl enable {domain}/{svc}")
            disabled.append(svc)
            console.print(f"  [green]✓[/] {svc.split('.')[-1]}  [dim]{desc}[/]")

    # Also kill currently bloated processes
    for p in bloat_procs:
        for pid in p.get("pids", [p["pid"]]):
            try:
                import signal

                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        console.print(f"  [green]✓[/] Killed {p['name']} (was {p['mb'] // 1024}GB)")

    # Save restore script
    restore_path = CONFIG_DIR / "restore_debloat.sh"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(restore_path, "w") as f:
        f.write("#!/bin/bash\n# localfit debloat restore script\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        for cmd in restore_cmds:
            f.write(f"{cmd}\n")
        f.write('\necho "All services restored. Reboot recommended."\n')
    os.chmod(restore_path, 0o755)

    console.print(f"\n  [green]Disabled {len(disabled)} services.[/]")
    console.print(f"  [dim]Restore script: {restore_path}[/]")
    console.print(f"  [dim]Run: localcoder --debloat  then choose 'r' to restore[/]\n")


def _kill_bloated_processes():
    """Kill all currently bloated system processes (one-time)."""
    import signal

    procs = get_top_memory_processes(min_mb=300)
    bloat = [p for p in procs if p["category"] == "bloat"]
    if not bloat:
        console.print("  [dim]No bloated processes found.[/]")
        return

    freed = 0
    for p in bloat:
        for pid in p.get("pids", [p["pid"]]):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        mb = p["mb"]
        freed += mb
        console.print(f"  [green]✓[/] Killed {p['name']} ({mb // 1024}GB)")

    console.print(
        f"\n  [green]Freed ~{freed // 1024}GB[/]  [dim](processes may restart smaller)[/]"
    )


def _debloat_restore():
    """Restore all debloated services."""
    restore_path = CONFIG_DIR / "restore_debloat.sh"
    if not restore_path.exists():
        console.print("  [dim]No restore script found — nothing to restore.[/]")
        return

    console.print("  [yellow]Restoring all disabled services...[/]")
    r = subprocess.run(
        ["bash", str(restore_path)], capture_output=True, text=True, timeout=30
    )
    if r.returncode == 0:
        console.print("  [green]All services restored. Reboot recommended.[/]")
        restore_path.unlink()
    else:
        console.print(f"  [red]Some services failed to restore: {r.stderr[:200]}[/]")


# LocalLLaMA community favorites for coding — from Best LLMs 2025 megathread
# Updated from r/LocalLLaMA actual user recommendations, not benchmarks
COMMUNITY_CODING_MODELS = {
    # <=8GB VRAM
    "lfm2-8b-a1b": {
        "name": "LFM2 8B-A1B",
        "hf": "liquid/LFM2-8B-A1B-GGUF",
        "vram": "8GB",
        "note": "Crazy fast MoE, great general + tool calling",
    },
    "qwen3-4b": {
        "name": "Qwen 3 4B",
        "hf": "unsloth/Qwen3-4B-GGUF",
        "vram": "4GB",
        "note": "Best tool calling at 4B size",
    },
    # 12-24GB VRAM (most LocalLLaMA users)
    "qwen3-coder-30b": {
        "name": "Qwen 3 Coder 30B-A3B",
        "hf": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "vram": "12-24GB",
        "note": "Top agentic coder, MoE",
    },
    "nemotron-30b-a3b": {
        "name": "Nemotron 30B-A3B",
        "hf": "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        "vram": "12-24GB",
        "note": "NVIDIA MoE, fastest generation",
    },
    "gemma4-26b": {
        "name": "Gemma 4 26B-A4B",
        "hf": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "vram": "12-16GB",
        "note": "Best tool calling + vision, 49 tok/s",
    },
    "devstral-24b": {
        "name": "Devstral Small 24B",
        "hf": "lmstudio-community/Devstral-Small-2-24B-Instruct-2512-GGUF",
        "vram": "12-24GB",
        "note": "Reliable daily driver for coding",
    },
    "glm-4.6v-flash": {
        "name": "GLM 4.6V Flash",
        "hf": "THUDM/glm-4.6v-flash-9b-gguf",
        "vram": "8-12GB",
        "note": "Best small model of the year (r/LocalLLaMA)",
    },
    # 24-48GB VRAM
    "gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "hf": "unsloth/gpt-oss-20b-GGUF",
        "vram": "24GB",
        "note": "Best accuracy under 48GB",
    },
    "qwen3.5-35b-a3b": {
        "name": "Qwen 3.5 35B-A3B",
        "hf": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "vram": "12-24GB",
        "note": "1.5M downloads, MoE coding beast",
    },
    # 48-96GB VRAM
    "glm-4.5-air": {
        "name": "GLM 4.5 Air",
        "hf": "THUDM/glm-4.5-9b-air-gguf",
        "vram": "48-96GB",
        "note": "Flat-out amazing for codegen (r/LocalLLaMA)",
    },
    # 96GB+
    "gpt-oss-120b": {
        "name": "GPT-OSS 120B",
        "hf": "unsloth/gpt-oss-120b-GGUF",
        "vram": "96GB+",
        "note": "Most recommended for agentic coding",
    },
    "devstral-123b": {
        "name": "Devstral 123B",
        "hf": "mistralai/Devstral-2-123B-GGUF",
        "vram": "96GB+",
        "note": "Compact 123B, fits 2x RTX Pro",
    },
    "minimax-m2": {
        "name": "MiniMax M2.1",
        "hf": "unsloth/MiniMax-M2.1-GGUF",
        "vram": "96GB+",
        "note": "Frontier performance, fantastic agentic coding",
    },
}


_hf_model_cache = {"data": None, "ts": 0}


def _fetch_all_hf_models():
    """Fetch GGUF models from all top providers in parallel. Cached for 10 minutes.

    One call, returns everything — trending, liked, latest. No duplicate fetches.
    """
    import concurrent.futures

    # Return cache if fresh
    if _hf_model_cache["data"] and time.time() - _hf_model_cache["ts"] < 600:
        return _hf_model_cache["data"]

    providers = ["unsloth", "bartowski", "lmstudio-community"]
    all_raw = []

    def _fetch_one(author):
        """Fetch from one provider — downloads sort gets us everything we need."""
        try:
            url = f"https://huggingface.co/api/models?author={author}&sort=downloads&direction=-1&limit=20"
            req = urllib.request.Request(url, headers={"User-Agent": "localfit/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                return json.loads(resp.read())
        except Exception:
            return []

    # Parallel fetch — all 3 providers at once (~1 API call time instead of 3)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_fetch_one, p): p for p in providers}
            for future in concurrent.futures.as_completed(futures, timeout=10):
                author = futures[future]
                try:
                    for m in future.result():
                        m["_author"] = author
                        all_raw.append(m)
                except Exception:
                    pass
    except Exception:
        return []

    # Deduplicate by base model name, prefer unsloth > bartowski > lmstudio
    provider_rank = {"unsloth": 0, "bartowski": 1, "lmstudio-community": 2}
    seen = {}
    for m in all_raw:
        tags = m.get("tags", [])
        if "gguf" not in tags:
            continue
        dl = m.get("downloads", 0)
        if dl < 1000:
            continue

        rid = m["id"]
        base = (
            rid.split("/")[-1]
            .replace("-GGUF", "")
            .replace("-Instruct", "")
            .replace("-it", "")
            .lower()
        )
        author = m.get("_author", "")
        rank = provider_rank.get(author, 9)

        if base not in seen or rank < seen[base]["_rank"]:
            name = (
                rid.split("/")[-1]
                .replace("-GGUF", "")
                .replace("-Instruct", "")
                .replace("-it", "")
            )
            tags = m.get("tags", [])

            # Detect modalities from tags
            caps = []
            if "image-text-to-text" in tags:
                caps.append("vision")
            if any("audio" in t for t in tags):
                caps.append("audio")
            if (
                any("code" in t.lower() or "coder" in t.lower() for t in tags)
                or "coder" in name.lower()
            ):
                caps.append("code")
            if (
                any("moe" in t.lower() for t in tags)
                or "A3B" in name
                or "A4B" in name
                or "A10B" in name
            ):
                caps.append("MoE")

            # Estimate smallest quant size from model name
            # Rule: ~0.5GB per 1B params at Q2, MoE active params only
            import re as _re_est

            est_smallest_gb = None

            # Handle E-series naming (e.g. E2B = ~2B params, E4B = ~4B params)
            e_match = _re_est.search(r"E(\d+)[bB]", name)
            if e_match:
                e_params = int(e_match.group(1))
                # E-series at Q4: ~1.5-2GB per 1B params (includes overhead)
                est_smallest_gb = round(e_params * 1.5, 1)
            else:
                param_match = _re_est.search(r"(\d+)[bB]", name)
                active_match = _re_est.search(r"A(\d+)[bB]", name)
                if param_match:
                    total_b = int(param_match.group(1))
                    active_b = int(active_match.group(1)) if active_match else total_b
                    # For MoE: estimate from total params, not active
                    # Q2 quant ≈ 0.35 GB per 1B total params
                    est_smallest_gb = round(total_b * 0.35, 1)

            seen[base] = {
                "repo_id": rid,
                "label": name,
                "downloads": dl,
                "likes": m.get("likes", 0),
                "author": author,
                "caps": caps,
                "est_smallest_gb": est_smallest_gb,
                "_rank": rank,
                "_base": base,
            }

    result = list(seen.values())
    _hf_model_cache["data"] = result
    _hf_model_cache["ts"] = time.time()
    return result


def fetch_unsloth_top_models(limit=12):
    """Top GGUF models sorted by downloads. Cached, parallel fetch."""
    models = _fetch_all_hf_models()
    models_sorted = sorted(models, key=lambda x: x["downloads"], reverse=True)
    return models_sorted[:limit]


def fetch_hf_trending_models(limit=5, sort="downloads"):
    """GGUF models sorted by downloads or likes. Cached, parallel fetch."""
    models = _fetch_all_hf_models()
    if sort == "likes":
        models_sorted = sorted(models, key=lambda x: x.get("likes", 0), reverse=True)
    else:
        models_sorted = sorted(models, key=lambda x: x["downloads"], reverse=True)
    return models_sorted[:limit]


# Legacy compat — old code referenced this directly
def _fetch_unsloth_top_compat(limit=12):
    return fetch_unsloth_top_models(limit)


def fetch_hf_model(query, silent=False):
    """Fetch GGUF model info from HuggingFace.

    Accepts:
    - Full URL: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
    - Repo ID: unsloth/gemma-4-26B-A4B-it-GGUF
    - Search term: gemma 4 26b gguf

    Args:
        silent: If True, skip interactive "Pick (1-5)" prompt and auto-select
                best match (most downloads). Used by run_menu.py.

    Returns dict with model name, GGUF files with sizes, or None.
    """
    import re as _re

    repo_id = None

    # Parse URL
    if "huggingface.co" in query:
        # https://huggingface.co/org/model or /org/model/...
        m = _re.search(r"huggingface\.co/([^/]+/[^/\s?#]+)", query)
        if m:
            repo_id = m.group(1)
    elif "/" in query and " " not in query:
        # Direct repo ID: unsloth/gemma-4-26B-A4B-it-GGUF
        repo_id = query
    elif query.count("_") == 1 and " " not in query:
        # Some UI paths may sanitize org/model to org_model
        repo_id = query.replace("_", "/", 1)
    elif "ollama.com" in query:
        # Ollama URL — extract model name for search
        m = _re.search(r"ollama\.com/library/([^/\s?#]+)", query)
        if m:
            query = m.group(1) + " gguf"

    # If no repo_id, search HuggingFace and show matches
    if not repo_id:
        try:
            search_url = f"https://huggingface.co/api/models?search={urllib.parse.quote(query + ' gguf')}&sort=downloads&direction=-1&limit=10"
            req = urllib.request.Request(
                search_url, headers={"User-Agent": "localfit/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                results = json.loads(resp.read())
            # Filter GGUF repos
            gguf_results = [
                r
                for r in results
                if any("gguf" in t.lower() for t in r.get("tags", []))
            ]

            if len(gguf_results) > 1 and not silent:
                # Use prompt_toolkit for interactive selection
                try:
                    from prompt_toolkit import prompt as pt_prompt
                    from prompt_toolkit.completion import WordCompleter

                    from rich.markup import escape as _esc_search

                    console.print(
                        f"\n  [bold]Found {len(gguf_results)} matches for '{query}':[/]\n"
                    )
                    for i, r in enumerate(gguf_results[:5], 1):
                        dl = r.get("downloads", 0)
                        dl_str = (
                            f"{dl // 1000}K"
                            if dl < 1_000_000
                            else f"{dl / 1_000_000:.1f}M"
                        )
                        console.print(
                            f"  [bold cyan]{i}[/]  {_esc_search(r['id']):<50} [dim]{dl_str} dl[/]"
                        )
                    console.print()

                    # Autocomplete with repo IDs
                    completer = WordCompleter(
                        [str(i) for i in range(1, len(gguf_results[:5]) + 1)]
                        + [r["id"] for r in gguf_results[:5]],
                        ignore_case=True,
                    )
                    pick = pt_prompt("  Pick (1-5): ", completer=completer).strip()

                    if not pick:
                        repo_id = gguf_results[0]["id"]
                    elif pick.isdigit():
                        idx = int(pick) - 1
                        if 0 <= idx < len(gguf_results):
                            repo_id = gguf_results[idx]["id"]
                    elif "/" in pick:
                        repo_id = pick
                    else:
                        repo_id = gguf_results[0]["id"]

                except (ImportError, EOFError, KeyboardInterrupt):
                    repo_id = gguf_results[0]["id"] if gguf_results else None
            elif gguf_results:
                repo_id = gguf_results[0]["id"]

            if not repo_id and results:
                repo_id = results[0]["id"]
        except Exception:
            return None

    if not repo_id:
        return None

    # Fetch model metadata + file sizes (with fallback to search)
    data = None
    try:
        api_url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
        req = urllib.request.Request(api_url, headers={"User-Agent": "localfit/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        # Direct lookup failed — try searching with the repo name as query
        try:
            import re as _re2

            search_term = repo_id.split("/")[-1].replace("-", " ").replace("_", " ")
            # Strip version numbers for better search
            search_term = _re2.sub(r"\b\d{4}\b", "", search_term).strip()
            search_url = f"https://huggingface.co/api/models?search={urllib.parse.quote(search_term)}&sort=downloads&direction=-1&limit=3"
            req = urllib.request.Request(
                search_url, headers={"User-Agent": "localfit/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                results = json.loads(resp.read())
            if results:
                repo_id = results[0]["id"]
                api_url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
                req = urllib.request.Request(
                    api_url, headers={"User-Agent": "localfit/1.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
        except Exception:
            pass

    if not data:
        return None

    # Extract GGUF files with sizes
    gguf_files = []
    for s in data.get("siblings", []):
        name = s.get("rfilename", "")
        size = s.get("size", 0)
        if not name.endswith(".gguf") or size < 500_000_000:  # skip tiny/split files
            continue
        if "mmproj" in name.lower():
            continue  # skip vision projectors
        # Handle split files: sum all parts to get real total size
        import re as _re_split

        split_m = _re_split.search(r"-(\d{5})-of-(\d{5})\.gguf$", name)
        if split_m:
            part_num = int(split_m.group(1))
            total_parts = int(split_m.group(2))
            if part_num != total_parts:
                continue  # only process last part (we'll calculate total below)
            # Calculate real total by summing all parts
            base_prefix = name[: split_m.start()]
            real_total = 0
            for sib in data.get("siblings", []):
                sib_name = sib.get("rfilename", "")
                if sib_name.startswith(base_prefix) and sib_name.endswith(".gguf"):
                    real_total += sib.get("size", 0)
            size = real_total  # override with real total

        # Parse quant from filename
        quant = "unknown"
        qm = _re.search(
            r"(BF16|F16|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d+|IQ\d+_[A-Z]+|MXFP\d+)",
            name,
            _re.IGNORECASE,
        )
        if qm:
            quant = qm.group(1).upper()

        gguf_files.append(
            {
                "filename": name,
                "size_bytes": size,
                "size_gb": round(size / (1024**3), 1),
                "quant": quant,
            }
        )

    # Sort by size ascending
    gguf_files.sort(key=lambda x: x["size_bytes"])

    # Extract mmproj (vision projector) files
    mmproj_files = []
    for s in data.get("siblings", []):
        name = s.get("rfilename", "")
        size = s.get("size", 0)
        if "mmproj" in name.lower() and name.endswith((".gguf", ".mmproj")):
            mmproj_files.append(
                {
                    "filename": name,
                    "size_bytes": size,
                    "size_gb": round(size / (1024**3), 1),
                }
            )
    mmproj_files.sort(key=lambda x: x["size_bytes"])

    # Detect if model is a VLM (vision-language model)
    tags = data.get("tags", [])
    is_vlm = (
        "image-text-to-text" in tags
        or any("vision" in t.lower() for t in tags)
        or len(mmproj_files) > 0
    )

    return {
        "repo_id": repo_id,
        "name": data.get("id", repo_id).split("/")[-1],
        "tags": tags,
        "downloads": data.get("downloads", 0),
        "gguf_files": gguf_files,
        "mmproj_files": mmproj_files,
        "is_vlm": is_vlm,
    }


def _get_cloud_gpus():
    """Get cloud GPU pricing from RunPod API.

    Returns (gpu_list, is_live) — is_live=True if prices come from RunPod API.
    Returns empty list if no API key (user must run: localfit --login runpod).
    """
    try:
        from localfit.cloud import get_runpod_key, fetch_gpu_options

        key = get_runpod_key()
        if key:
            gpus = fetch_gpu_options(key)
            if gpus:
                return gpus, True
    except Exception:
        pass
    # No API key or API failed — return empty, caller shows login prompt
    return [], False


def _arrow_pick(items, default_idx=0):
    """Arrow-key interactive picker. Returns selected index or None.

    Items with selectable=False are skipped during navigation (section headers).
    Supports up/down/j/k arrows, Enter to select, q/Escape to cancel.
    """
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style

        selectable = [i for i, it in enumerate(items) if it.get("selectable", True)]
        if not selectable:
            return None
        if default_idx not in selectable:
            default_idx = selectable[0]
        pos = [selectable.index(default_idx)]

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _up(e):
            pos[0] = max(0, pos[0] - 1)

        @kb.add("down")
        @kb.add("j")
        def _down(e):
            pos[0] = min(len(selectable) - 1, pos[0] + 1)

        @kb.add("enter")
        def _enter(e):
            e.app.exit(result=selectable[pos[0]])

        @kb.add("q")
        @kb.add("escape")
        def _quit(e):
            e.app.exit(result=None)

        def _render():
            out = []
            cur = selectable[pos[0]]
            for i, it in enumerate(items):
                lbl = it["label"]
                if not it.get("selectable", True):
                    out.append(("class:sep", f"\n  {lbl}\n"))
                elif i == cur:
                    out.append(("class:sel", f"  ▸ {lbl}\n"))
                else:
                    out.append(("", f"    {lbl}\n"))
            out.append(("class:hint", "\n  ↑↓ navigate · Enter select · q quit"))
            return out

        app = Application(
            layout=Layout(
                Window(FormattedTextControl(_render), dont_extend_height=True)
            ),
            key_bindings=kb,
            style=Style.from_dict(
                {"sel": "bold cyan", "sep": "#888888 bold", "hint": "#666666"}
            ),
            full_screen=False,
        )
        return app.run()
    except (ImportError, Exception):
        # Fallback: numbered input
        console.print("\n  [dim]Enter number to select, or q to quit[/]\n")
        try:
            ans = input("  > ").strip()
            if ans.isdigit():
                idx = int(ans) - 1
                if 0 <= idx < len(items) and items[idx].get("selectable", True):
                    return idx
        except (EOFError, KeyboardInterrupt):
            pass
        return None


def _check_ollama_registry(model_name):
    """Check if a model exists in Ollama's registry and get its size.

    Tries common Ollama tag patterns for a given model name.
    Returns dict with model info or None.
    """
    import urllib.request as _ur
    import json as _json

    # Normalize: "microsoft/Fara-7B" -> "fara", "maternion/fara" -> "maternion/fara"
    name = model_name.strip().lower()
    # Extract base name for searching
    base = name.split("/")[-1].replace("-gguf", "").replace("_", "-")
    # Remove size suffixes for broader matching
    for suffix in ["-7b", "-8b", "-13b", "-14b", "-27b", "-32b", "-70b", "-72b"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    # Try the query as-is (e.g. "maternion/fara"), plus common patterns
    candidates = [name]
    if "/" not in name:
        candidates.append(name)  # plain name like "fara"
    # Also try base name in case user passed HF repo
    if base != name:
        candidates.append(base)

    for tag in candidates:
        # Ollama registry API: check if manifest exists
        # Format: library/MODEL for official, USER/MODEL for community
        parts = tag.split("/")
        if len(parts) == 1:
            url = f"https://registry.ollama.ai/v2/library/{parts[0]}/manifests/latest"
        elif len(parts) == 2:
            url = (
                f"https://registry.ollama.ai/v2/{parts[0]}/{parts[1]}/manifests/latest"
            )
        else:
            continue

        try:
            req = _ur.Request(
                url,
                headers={
                    "Accept": "application/vnd.docker.distribution.manifest.v2+json"
                },
            )
            resp = _ur.urlopen(req, timeout=5)
            data = _json.loads(resp.read())
            layers = data.get("layers", [])
            total_size = sum(l.get("size", 0) for l in layers)
            has_projector = any("projector" in l.get("mediaType", "") for l in layers)
            # Check if it's a VLM by looking at layer types
            model_layers = [l for l in layers if "model" in l.get("mediaType", "")]
            proj_layers = [l for l in layers if "projector" in l.get("mediaType", "")]

            return {
                "tag": tag,
                "size_gb": round(total_size / (1024**3), 1),
                "has_projector": has_projector or len(proj_layers) > 0,
                "is_vlm": has_projector or len(proj_layers) > 0,
                "layers": len(layers),
            }
        except Exception:
            continue

    return None


def will_it_fit(query):
    """Universal fit checker — paste ANY HuggingFace URL (LLM or image model).

    The viral feature: works for GGUF, safetensors, diffusers, MLX, everything.
    """
    import re as _re

    specs = get_machine_specs()
    metal = get_metal_gpu_stats()
    gpu_total = metal.get("total_mb") or specs.get("gpu_total_mb", 0)
    gpu_used = metal.get("alloc_mb", 0)
    gpu_free = gpu_total - gpu_used
    gpu_gb = gpu_total / 1024
    free_gb = gpu_free / 1024
    chip = specs.get("chip", "GPU")
    ram_gb = specs.get("ram_gb", 0)

    # Parse repo ID from URL
    repo_id = None
    if "huggingface.co" in query:
        m = _re.search(r"huggingface\.co/([^/]+/[^/\s?#]+)", query)
        if m:
            repo_id = m.group(1)
    elif "/" in query and " " not in query:
        repo_id = query.strip("/")

    if not repo_id:
        # Try to find in our MODELS catalog by name or ID
        query_lower = query.lower().strip()
        for mid, minfo in MODELS.items():
            if (
                query_lower == mid.lower()
                or query_lower == minfo.get("name", "").lower()
                or query_lower in minfo.get("name", "").lower()
                or mid.lower() in query_lower
            ):
                if minfo.get("hf_repo"):
                    repo_id = minfo["hf_repo"]
                    break

    if not repo_id:
        # Try Ollama tag lookup
        for mid, minfo in MODELS.items():
            if minfo.get("ollama_tag") and query_lower == minfo["ollama_tag"].lower():
                if minfo.get("hf_repo"):
                    repo_id = minfo["hf_repo"]
                    break

    if not repo_id:
        # Fall back to LLM simulate (does its own HF search)
        return simulate_hf_model(query)

    console.print(f"\n  [bold]Checking: {repo_id}[/]")
    console.print(
        f"  [dim]{chip} · {ram_gb}GB RAM · GPU: {gpu_gb:.0f}GB total, {free_gb:.0f}GB free[/]\n"
    )

    # Fetch model metadata from HF API
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "localfit/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            meta = json.loads(resp.read())
    except Exception as e:
        console.print(f"  [red]Could not fetch: {e}[/]")
        return

    pipeline_tag = meta.get("pipeline_tag", "")
    library = meta.get("library_name", "")
    tags = [t.lower() for t in meta.get("tags", [])]
    siblings = meta.get("siblings", [])
    total_bytes = meta.get("usedStorage", 0)
    total_gb = round(total_bytes / (1024**3), 1)
    downloads = meta.get("downloads", 0)

    # Detect model type
    is_diffusion = (
        library == "diffusers"
        or pipeline_tag in ("text-to-image", "image-to-image", "text-to-video")
        or any(
            t in tags for t in ["diffusers", "flux", "stable-diffusion", "diffusion"]
        )
    )
    is_gguf = any("gguf" in t for t in tags) or any(
        s.get("rfilename", "").endswith(".gguf") for s in siblings
    )
    is_mlx = "mlx" in repo_id.lower() or any("mlx" in t for t in tags)
    is_video = pipeline_tag == "text-to-video" or any(
        "wan" in t or "video" in t for t in tags
    )

    # Count file types
    safetensors = [
        s for s in siblings if s.get("rfilename", "").endswith(".safetensors")
    ]
    gguf_files = [s for s in siblings if s.get("rfilename", "").endswith(".gguf")]

    if is_diffusion and not is_gguf:
        _check_diffusion_fit(
            repo_id,
            meta,
            pipeline_tag,
            total_gb,
            safetensors,
            gpu_gb,
            free_gb,
            chip,
            is_mlx,
            is_video,
            downloads,
        )
    elif is_gguf:
        # Delegate to existing GGUF simulator
        simulate_hf_model(query)
    else:
        # Generic safetensors model
        _check_generic_fit(repo_id, total_gb, gpu_gb, free_gb, chip, downloads)


def _check_diffusion_fit(
    repo_id,
    meta,
    pipeline_tag,
    total_gb,
    safetensors,
    gpu_gb,
    free_gb,
    chip,
    is_mlx,
    is_video,
    downloads,
):
    """Fit analysis for diffusion/image models."""

    dl_str = (
        f"{downloads // 1000}K"
        if downloads < 1_000_000
        else f"{downloads / 1_000_000:.1f}M"
    )

    # Detect model family
    repo_lower = repo_id.lower()
    family = "Unknown"
    if "flux.2-klein" in repo_lower or "flux2-klein" in repo_lower:
        family = "FLUX.2 Klein"
    elif "flux.2" in repo_lower or "flux2" in repo_lower:
        family = "FLUX.2"
    elif "flux.1" in repo_lower or "flux1" in repo_lower:
        family = "FLUX.1"
    elif "z-image" in repo_lower:
        family = "Z-Image"
    elif "qwen-image" in repo_lower or "qwen_image" in repo_lower:
        family = "Qwen-Image"
    elif "fibo" in repo_lower:
        family = "FIBO"
    elif "stable-diffusion" in repo_lower or "sdxl" in repo_lower:
        family = "Stable Diffusion"
    elif "wan" in repo_lower:
        family = "Wan (Video)"

    task = pipeline_tag or "text-to-image"

    # Estimate peak VRAM (diffusion models have peak during denoising)
    # Rule of thumb: model weights + ~30% for activations/KV during generation
    peak_vram_gb = total_gb * 1.3

    # With quantization
    q4_gb = round(total_gb * 0.3, 1)  # Q4 ≈ 30% of BF16
    q8_gb = round(total_gb * 0.55, 1)  # Q8 ≈ 55% of BF16

    # Backends available
    backends = []
    if IS_MAC:
        backends.append(
            ("mflux (MLX)", "Mac native, fastest", q4_gb if not is_mlx else total_gb)
        )
        backends.append(("sd.cpp (Metal)", "GGUF quantized", q4_gb))
    backends.append(("diffusers (BF16)", "Full precision", peak_vram_gb))
    backends.append(
        ("diffusers + cpu_offload", "Offload to RAM", round(total_gb * 0.4, 1))
    )
    if not IS_MAC:
        backends.append(("sd.cpp (CUDA/Vulkan)", "GGUF quantized", q4_gb))

    # Print analysis
    console.print(f"  [bold cyan]{family}[/]  {task}  [dim]{dl_str} downloads[/]")
    if is_video:
        console.print(f"  [magenta]Video generation model[/]")
    if is_mlx:
        console.print(f"  [cyan]MLX optimized[/] — native Apple Silicon")
    console.print(
        f"  [dim]Total on disk: {total_gb}GB ({len(safetensors)} safetensors files)[/]\n"
    )

    # Fit table
    table = Table(
        show_header=True, header_style="bold", border_style="dim", padding=(0, 1)
    )
    table.add_column("Backend", width=25)
    table.add_column("VRAM", justify="right", width=10)
    table.add_column("Fits?", width=18)
    table.add_column("Speed", width=20)
    table.add_column("", width=16)

    best = None
    for name, desc, vram in backends:
        fits = vram <= free_gb
        fits_total = vram <= gpu_gb

        if fits:
            status = "[green]✓ fits[/]"
            if not best:
                best = (name, vram)
        elif fits_total:
            status = "[yellow]⚠ tight[/]"
            if not best:
                best = (name, vram)
        else:
            status = "[red]✗ too big[/]"

        # Speed estimate for diffusion
        if "cpu_offload" in name:
            speed = "slow (~60-120s/img)"
        elif "mflux" in name and IS_MAC:
            speed = "fast (~5-20s/img)"
        elif "sd.cpp" in name:
            speed = "fast (~5-15s/img)"
        else:
            speed = "medium (~15-45s/img)"

        bar_pct = min(1.0, vram / gpu_gb) if gpu_gb else 0
        bar_w = int(bar_pct * 16)
        bc = "green" if fits else "yellow" if fits_total else "red"
        bar = f"[{bc}]{'█' * bar_w}[/{bc}][dim]{'░' * (16 - bar_w)}[/]"

        table.add_row(name, f"{vram}GB", status, speed, bar)

    console.print(table)

    # Recommendation
    if best:
        console.print(f"\n  [green]→ Best local: {best[0]} ({best[1]}GB)[/]")
        console.print(
            f"    Fits your {gpu_gb:.0f}GB GPU with {free_gb - best[1]:.0f}GB spare"
        )

        # Can pair with LLM?
        llm_room = free_gb - best[1]
        if llm_room >= 5:
            console.print(f"    [green]✓[/] Room for LLM too (E4B 4.6GB)")
        elif llm_room >= 3:
            console.print(f"    [yellow]⚠[/] Tight — small LLM only (E2B 2.7GB)")
        else:
            console.print(f"    [red]✗[/] No room for LLM — image only")
    else:
        console.print(f"\n  [red]✗ Does not fit locally[/]")
        console.print(
            f"    Model needs {q4_gb}GB (Q4) but you have {free_gb:.0f}GB free"
        )
        console.print(f"\n  [cyan]☁ Try cloud:[/]")
        console.print(
            f"    localfit --serve {repo_id} --remote kaggle  [dim](free T4)[/]"
        )
        console.print(
            f"    localfit --serve {repo_id} --remote runpod  [dim](paid, any GPU)[/]"
        )

    # GGUF alternative?
    if not any(
        "gguf" in s.get("rfilename", "").lower() for s in (meta.get("siblings") or [])
    ):
        console.print(
            f"\n  [dim]💡 Smaller with GGUF: search for {repo_id.split('/')[-1]}-GGUF on HuggingFace[/]"
        )
        console.print(
            f"  [dim]   sd.cpp supports GGUF quantized diffusion models (Q4 = ~30% size)[/]"
        )


def _check_generic_fit(repo_id, total_gb, gpu_gb, free_gb, chip, downloads):
    """Generic fit check for non-GGUF, non-diffusion models."""
    dl_str = (
        f"{downloads // 1000}K"
        if downloads < 1_000_000
        else f"{downloads / 1_000_000:.1f}M"
    )
    console.print(f"  [dim]{dl_str} downloads · {total_gb}GB on disk[/]\n")

    fits = total_gb <= free_gb
    fits_total = total_gb <= gpu_gb

    if fits:
        console.print(
            f"  [green]✓ Fits your {gpu_gb:.0f}GB GPU ({free_gb:.0f}GB free)[/]"
        )
    elif fits_total:
        console.print(
            f"  [yellow]⚠ Tight — {total_gb}GB model, {free_gb:.0f}GB free[/]"
        )
    else:
        console.print(f"  [red]✗ Too big — {total_gb}GB model, {gpu_gb:.0f}GB GPU[/]")
        console.print(f"\n  [dim]Try a quantized/GGUF version or cloud GPU[/]")


def simulate_hf_model(query):
    """Fetch a model from HuggingFace and show which quants fit.

    The "holy shit" feature: paste a URL, see instant fit analysis for every quant.
    """
    specs = get_machine_specs()
    metal = get_metal_gpu_stats()
    gpu_total = metal.get("total_mb") or specs["gpu_total_mb"]
    gpu_used = metal.get("alloc_mb", 0)

    console.clear()
    model = fetch_hf_model(query)

    if not model:
        console.print(f"\n  [red]Model not found: {query}[/]")
        console.print(
            f"  [dim]Try a HuggingFace URL or search term like 'llama 3.1 70b gguf'[/]\n"
        )
        return

    console.clear()
    console.print()
    is_vlm = model.get("is_vlm", False)
    mmproj_files = model.get("mmproj_files", [])
    mmproj_size_gb = mmproj_files[0]["size_gb"] if mmproj_files else 0
    vlm_badge = "  [magenta bold]👁 Vision Model (VLM)[/]" if is_vlm else ""
    console.print(f"  [bold]{model['repo_id']}[/]{vlm_badge}")
    is_cpu_only = specs.get("cpu_only", False)
    if is_cpu_only:
        console.print(
            f"  [dim]{specs['chip']}  ·  {specs['ram_gb']}GB RAM  ·  [yellow]CPU-only[/] (models use system RAM, ~3-12 tok/s)[/]\n"
        )
    else:
        console.print(
            f"  [dim]{specs['chip']}  ·  {specs['ram_gb']}GB RAM  ·  GPU budget: {gpu_total // 1024}GB  ·  In use: {gpu_used // 1024}GB[/]\n"
        )

    # VLM warning: vision models need an mmproj file to process images
    if is_vlm:
        if mmproj_files:
            mmproj_name = mmproj_files[0]["filename"]
            console.print(
                f"  [magenta]This is a vision model — it needs a vision projector (mmproj) to see images.[/]"
            )
            console.print(
                f"  [magenta]mmproj:[/] {mmproj_name} [dim]({mmproj_size_gb}GB — added to sizes below)[/]"
            )
            console.print(
                f"  [dim]Without the mmproj, the model is text-only and will ignore image inputs.[/]\n"
            )
        else:
            console.print(
                f"  [red bold]Warning:[/] [magenta]This is a vision model but no mmproj file was found in the repo.[/]"
            )
            console.print(
                f"  [dim]The model will be text-only without a vision projector. Check the original model repo.[/]\n"
            )

    if not model["gguf_files"]:
        # Try to find GGUF version from Unsloth/bartowski
        repo_name = model["repo_id"].split("/")[-1]
        console.print(
            f"  [yellow]No GGUF files in this repo. Searching for GGUF version...[/]"
        )
        gguf_model = fetch_hf_model(f"{repo_name} GGUF")
        if gguf_model and gguf_model["gguf_files"]:
            console.print(f"  [green]Found:[/] {gguf_model['repo_id']}\n")
            model = gguf_model
            console.print(f"  [bold]{model['repo_id']}[/]")
            console.print(
                f"  [dim]{specs['chip']}  ·  {specs['ram_gb']}GB RAM  ·  GPU budget: {gpu_total // 1024}GB[/]\n"
            )
        else:
            console.print(
                f"  [red]No GGUF version found. This model needs conversion to GGUF first.[/]"
            )
            console.print(
                f"  [dim]Try: unsloth/{repo_name}-GGUF or bartowski/{repo_name}-GGUF[/]\n"
            )
            return

    # Show all quants with fit status
    size_col_label = "Total" if is_vlm and mmproj_size_gb else "Size"
    table = Table(
        title=f"Available Quants ({len(model['gguf_files'])})",
        show_header=True,
        header_style="bold",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Quant", width=14)
    table.add_column(size_col_label, justify="right", width=12)
    table.add_column("Fits?" if not is_cpu_only else "Fits RAM?", width=18)
    table.add_column("Est. Speed", width=12)
    table.add_column("", width=18)

    best_fit_idx = None
    for i, f in enumerate(model["gguf_files"], 1):
        size_gb = f["size_gb"]
        # VLMs need mmproj loaded too — include it in the total
        total_gb = size_gb + mmproj_size_gb if is_vlm else size_gb
        total_mb = int(total_gb * 1024)
        fits = total_mb < gpu_total
        fits_free = total_mb < (gpu_total - gpu_used)

        if fits_free:
            if is_cpu_only:
                status = "[green]✓ fits RAM[/] [dim](CPU)[/]"
            else:
                status = "[green]✓ fits[/]"
            if (
                best_fit_idx is None
                or f["size_gb"] > model["gguf_files"][best_fit_idx - 1]["size_gb"]
            ):
                best_fit_idx = i
        elif fits:
            status = "[yellow]⚠ tight[/]"
            if best_fit_idx is None:
                best_fit_idx = i
        else:
            status = "[red]✗ too big[/]"

        # Speed estimate — CPU-only is much slower
        if is_cpu_only:
            # CPU inference: roughly 1-12 tok/s depending on model size and CPU
            tps = max(1, int(12 * 4 / max(1, total_gb))) if fits else 0
            speed = f"~{tps} tok/s [dim](CPU)[/]" if fits else "[red]✗[/]"
        else:
            tps = (
                min(120, max(1, int(49 * 12 / max(1, total_gb))))
                if fits
                else max(1, int(5 * 16 / max(1, total_gb)))
            )
            speed = f"~{tps} tok/s" if fits else f"[red]~{tps} tok/s[/]"

        # Visual bar
        bar_pct = min(1.0, total_mb / gpu_total) if gpu_total else 0
        bar_w = int(bar_pct * 16)
        bar_color = "green" if fits_free else "yellow" if fits else "red"
        bar = f"[{bar_color}]{'█' * bar_w}[/{bar_color}][dim]{'░' * (16 - bar_w)}[/]"

        # Size display: show "model + mmproj = total" for VLMs
        if is_vlm and mmproj_size_gb:
            size_str = f"{size_gb}+{mmproj_size_gb}G"
        else:
            size_str = f"{size_gb}GB"

        table.add_row(str(i), f["quant"], size_str, status, speed, bar)

    console.print(table)

    # Recommendation
    is_unsloth = "unsloth" in model["repo_id"].lower()
    if best_fit_idx:
        bf = model["gguf_files"][best_fit_idx - 1]
        bf_total = bf["size_gb"] + mmproj_size_gb if is_vlm else bf["size_gb"]
        if is_cpu_only:
            cpu_tps = max(1, int(12 * 4 / max(1, bf_total)))
            console.print(
                f"\n  [green bold]→ Best local: #{best_fit_idx} {bf['quant']} ({bf_total}GB)[/]"
            )
            console.print(
                f"    [dim]Fits your {specs['ram_gb']}GB RAM — runs on CPU at ~{cpu_tps} tok/s (no GPU acceleration)[/]"
            )
            console.print(
                f"    [yellow]→ For 5-10x faster inference, use a cloud GPU:[/] localfit --login runpod"
            )
        else:
            console.print(
                f"\n  [green bold]→ Best local: #{best_fit_idx} {bf['quant']} ({bf_total}GB)[/]"
            )
            console.print(
                f"    [dim]Highest quality that fits your {gpu_total // 1024}GB GPU[/]"
            )
        if is_unsloth:
            console.print(
                f"    [dim]Unsloth quants use imatrix calibration — better quality than standard GGUF[/]"
            )

        # VLM: show how to use with Ollama and llama-server
        if is_vlm and mmproj_files:
            mmproj_name = mmproj_files[0]["filename"]
            console.print()
            console.print(f"  [magenta bold]Vision setup required:[/]")
            console.print(
                f"    [magenta]Ollama:[/] Create a Modelfile with both model GGUF and mmproj:"
            )
            console.print(f"    [cyan]FROM ./{bf['filename']}[/]")
            console.print(f"    [cyan]PROJECTOR ./{mmproj_name}[/]")
            console.print(f"    [dim]Then: ollama create mymodel -f Modelfile[/]")
            console.print(
                f"    [magenta]llama-server:[/] [cyan]--mmproj {mmproj_name}[/]"
            )
            console.print(
                f"    [red]Without the mmproj, the model cannot process images![/]"
            )

        # Check if cloud gives much better quality
        from localfit.matcher import get_quant_quality, get_quality_label

        local_quality = get_quant_quality(bf["quant"])
        best_overall = model["gguf_files"][-1]  # highest quality quant available
        best_quality = get_quant_quality(best_overall["quant"])

        if best_quality - local_quality > 15:
            # Significant quality upgrade available on cloud
            console.print()
            console.print(f"  [yellow]☁ Better quality on cloud:[/]")
            try:
                live_gpus, _is_live = _get_cloud_gpus()
                local_label, local_color = get_quality_label(local_quality)
                shown = 0
                for cg in sorted(live_gpus, key=lambda x: x["price"]):
                    cloud_fits = [
                        f for f in model["gguf_files"] if f["size_gb"] < cg["vram"]
                    ]
                    if cloud_fits:
                        cloud_best = cloud_fits[-1]
                        cq = get_quant_quality(cloud_best["quant"])
                        if cq > local_quality + 10:
                            cl, cc = get_quality_label(cq)
                            console.print(
                                f"    {cg['name']:<15} [{cc}]{cloud_best['quant']} ({cl})[/{cc}]  "
                                f"vs local [{local_color}]{bf['quant']} ({local_label})[/{local_color}]  "
                                f"${cg['price']:.2f}/hr"
                            )
                            shown += 1
                            if shown >= 3:
                                break
                if shown:
                    console.print(
                        f"    [dim]localfit --serve {model['repo_id']} --cloud[/]"
                    )
                elif not live_gpus:
                    console.print(
                        f"    [dim]localfit --login runpod for cloud GPU pricing[/]"
                    )
            except Exception:
                pass

            # Also show Kaggle free option if model fits
            try:
                from localfit.remote import kaggle_fits

                best_overall = model["gguf_files"][-1]
                kf = kaggle_fits(best_overall["size_gb"])
                if kf["fits"] and best_overall["size_gb"] > bf["size_gb"]:
                    console.print(
                        f"  [green]★ Or free on Kaggle {kf['gpu']} ({kf['vram_gb']}GB):[/]"
                    )
                    console.print(
                        f"    [cyan]localfit serve {model['repo_id']} --remote kaggle[/]"
                    )
            except Exception:
                pass
    else:
        smallest = model["gguf_files"][0]
        smallest_gb = smallest["size_gb"]

        if is_cpu_only:
            console.print(
                f"\n  [red bold]✗ No quant fits your {specs['ram_gb']}GB RAM (CPU-only mode).[/]"
            )
            console.print(
                f"  Smallest: {smallest['quant']} = {smallest_gb}GB (your RAM: {specs['ram_gb']}GB)"
            )
        else:
            console.print(
                f"\n  [red bold]✗ No quant fits your {gpu_total // 1024}GB GPU.[/]"
            )
            console.print(
                f"  Smallest: {smallest['quant']} = {smallest_gb}GB (your GPU: {gpu_total // 1024}GB)"
            )

        # Show Kaggle free GPU option first
        try:
            from localfit.remote import kaggle_fits

            # Check each quant against Kaggle GPUs
            kaggle_options = []
            for f in model["gguf_files"]:
                kf = kaggle_fits(f["size_gb"])
                if kf["fits"]:
                    kaggle_options.append((f, kf))
            if kaggle_options:
                best_kaggle = kaggle_options[-1]  # highest quality that fits
                kf_quant, kf_gpu = best_kaggle
                console.print()
                console.print(f"  [green bold]★ FREE on Kaggle:[/]")
                console.print(
                    f"    {kf_gpu['gpu']:<6} ({kf_gpu['vram_gb']}GB)  "
                    f"{kf_quant['quant']} ({kf_quant['size_gb']}GB)  "
                    f"[green]free · 12h limit[/]"
                )
                console.print(
                    f"    [cyan]localfit serve {model['repo_id']} --remote kaggle[/]"
                )
        except Exception:
            pass

        # Show cloud options with cost estimates
        console.print()
        try:
            live_gpus, _is_live = _get_cloud_gpus()

            # Find GPUs that can fit this model
            cloud_fits = []
            for cg in sorted(live_gpus, key=lambda x: x["price"]):
                # Check which quant fits this GPU
                fits = [f for f in model["gguf_files"] if f["size_gb"] < cg["vram"]]
                if fits:
                    best = fits[-1]  # highest quality that fits
                    from localfit.matcher import get_quant_quality, get_quality_label

                    q = get_quant_quality(best["quant"])
                    label, color = get_quality_label(q)
                    cloud_fits.append((cg, best, label, color))

            if not live_gpus:
                console.print(f"  [yellow]☁ Run on cloud instead:[/]")
                console.print(
                    f"    [dim]localfit --login runpod to see cloud GPU pricing[/]"
                )
            elif cloud_fits:
                console.print(f"  [yellow]☁ Run on cloud instead:[/]")
                for cg, best, label, color in cloud_fits[:4]:
                    console.print(
                        f"    {cg['name']:<15} {cg['vram']}GB  "
                        f"[{color}]{best['quant']} ({label})[/{color}]  "
                        f"${cg['price']:.2f}/hr  "
                        f"[dim]1h=${cg['price']:.2f} · 8h=${cg['price'] * 8:.2f}[/]"
                    )
                console.print(
                    f"  [dim]  localfit --serve {model['repo_id']} --cloud[/]"
                )
            else:
                console.print(
                    f"  [red]  Too big for any single GPU. Needs multi-GPU setup.[/]"
                )
                console.print(
                    f"  [dim]  Smallest quant ({smallest['quant']}) = {smallest_gb:.0f}GB — no GPU has this much VRAM.[/]"
                )
        except Exception:
            console.print(
                f"  [red]  Too big for any single GPU. Needs multi-GPU setup.[/]"
            )

        # Recommend similar models that DO fit
        console.print(
            f"\n  [bold]Models with similar quality that fit your {gpu_total // 1024}GB GPU:[/]"
        )
        try:
            # Find the model family from the name
            model_name = model.get("name", model.get("repo_id", "")).lower()
            alternatives = []

            # Search for alternatives from trending models
            all_models = _fetch_all_hf_models()
            for am in sorted(all_models, key=lambda x: x["downloads"], reverse=True):
                est = am.get("est_smallest_gb")
                if est and est * 1024 < gpu_total:
                    alternatives.append(am)
                if len(alternatives) >= 5:
                    break

            if alternatives:
                for alt in alternatives[:3]:
                    dl = alt["downloads"]
                    dl_str = (
                        f"{dl // 1000}K" if dl < 1_000_000 else f"{dl / 1_000_000:.1f}M"
                    )
                    caps = " ".join(c for c in alt.get("caps", []))
                    from rich.markup import escape as _esc2

                    console.print(
                        f"  [green]→[/] {_esc2(alt['label']):<28} ~{alt.get('est_smallest_gb', '?')}GB  {caps}  [dim]{dl_str} dl[/]"
                    )
                console.print(f"  [dim]  localfit --fetch REPO to check quants[/]")
            else:
                console.print(f"  [dim]  No alternatives found.[/]")
        except Exception:
            pass

    # --- Interactive action menu ---
    from localfit.matcher import get_quant_quality as _gqq, get_quality_label as _gql

    menu = []
    rec_idx = None
    best_local_q = 0

    # Local quants that fit (best quality first)
    _local_q = []
    for _fi, _ff in enumerate(model["gguf_files"]):
        _smb = int(_ff["size_gb"] * 1024)
        if _smb < gpu_total:
            _q = _gqq(_ff["quant"])
            _free = _smb < (gpu_total - gpu_used)
            _local_q.append((_fi, _ff, _q, _free))
    _local_q.sort(key=lambda x: x[2], reverse=True)

    if _local_q:
        best_local_q = _local_q[0][2]
        menu.append({"label": "── Serve locally ──", "selectable": False})
        for _li, (_fi, _ff, _q, _free) in enumerate(_local_q):
            _ql, _ = _gql(_q)
            _tps = min(120, max(1, int(49 * 12 / max(1, _ff["size_gb"]))))
            _st = "✓" if _free else "⚠"
            _rec = " ★ recommended" if _li == 0 else ""  # best quality = first
            menu.append(
                {
                    "label": f"{_st} {_ff['quant']:<12} {_ff['size_gb']:>6}GB  ({_ql}) ~{_tps} tok/s{_rec}",
                    "selectable": True,
                    "action": "serve_local",
                    "quant": _ff,
                }
            )
            if _li == 0:
                rec_idx = len(menu) - 1  # default to best quality

    # Cloud options (always fetch live prices)
    _cg_list, _cg_live = _get_cloud_gpus()
    _cloud_menu = []
    for _cg in sorted(_cg_list, key=lambda x: x["price"]):
        _cfits = [f for f in model["gguf_files"] if f["size_gb"] < _cg.get("vram", 0)]
        if _cfits:
            _cb = _cfits[-1]
            _cq = _gqq(_cb["quant"])
            _cl, _ = _gql(_cq)
            if _cq > best_local_q + 10 or not _local_q:
                _cloud_menu.append((_cg, _cb, _cl, _cq))

    if not _cg_list and not _cg_live:
        # No RunPod API key — show login hint
        menu.append({"label": "── Cloud (login for pricing) ──", "selectable": False})
        menu.append(
            {
                "label": "☁ Login to RunPod for cloud GPU pricing: localfit --login runpod",
                "selectable": False,
            }
        )

    if _cloud_menu:
        _hdr = (
            "── Serve on cloud ──" if not _local_q else "── Cloud (better quality) ──"
        )
        menu.append({"label": _hdr, "selectable": False})
        _seen_q = set()
        for _cg, _cb, _cl, _cq in _cloud_menu[:5]:
            _qk = _cb["quant"]
            if _qk in _seen_q:
                continue
            _seen_q.add(_qk)
            menu.append(
                {
                    "label": f"☁ {_cg['name']:<15} {_cg.get('vram', '?')}GB  {_cb['quant']} ({_cl})  ${_cg['price']:.2f}/hr",
                    "selectable": True,
                    "action": "serve_cloud",
                    "gpu": _cg,
                    "quant": _cb,
                }
            )
            if rec_idx is None:
                rec_idx = len(menu) - 1

    # Alternative models that fit locally
    if not _local_q or best_local_q < 60:
        try:
            _alt_all = _fetch_all_hf_models()
            _alts = []
            for _am in sorted(_alt_all, key=lambda x: x["downloads"], reverse=True):
                _est = _am.get("est_smallest_gb")
                if (
                    _est
                    and _est * 1024 < gpu_total
                    and _am.get("repo_id", "") != model["repo_id"]
                ):
                    _alts.append(_am)
                if len(_alts) >= 5:
                    break
            if _alts:
                menu.append(
                    {
                        "label": "── Similar models (fit your GPU) ──",
                        "selectable": False,
                    }
                )
                for _alt in _alts[:3]:
                    _dl = _alt["downloads"]
                    _dls = (
                        f"{_dl // 1000}K"
                        if _dl < 1_000_000
                        else f"{_dl / 1_000_000:.1f}M"
                    )
                    _caps = " ".join(c for c in _alt.get("caps", []))
                    menu.append(
                        {
                            "label": f"→ {_alt['label']:<28} ~{_alt.get('est_smallest_gb', '?')}GB  {_caps}  {_dls} dl",
                            "selectable": True,
                            "action": "switch_model",
                            "repo_id": _alt.get("repo_id", _alt.get("id", "")),
                        }
                    )
        except Exception:
            pass

    if not any(it.get("selectable", True) for it in menu):
        return

    # Show arrow-key picker
    _default = (
        rec_idx
        if rec_idx is not None
        else next((i for i, m in enumerate(menu) if m.get("selectable", True)), 0)
    )
    console.print()
    picked = _arrow_pick(menu, default_idx=_default)
    if picked is None:
        return

    _chosen = menu[picked]
    _action = _chosen.get("action")

    if _action == "serve_local":
        _qf = _chosen["quant"]
        _simulate_with_real_size(
            _qf,
            model["repo_id"],
            specs,
            gpu_total,
            gpu_used,
            mmproj_size_gb=mmproj_size_gb,
        )
        console.print(
            f"\n  [bold]Next:[/]  [cyan]d[/] download  ·  [cyan]s[/] download + serve  ·  [cyan]q[/] quit\n"
        )
        try:
            _ans = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if _ans in ("d", "s"):
            _dl_path = _download_gguf(model["repo_id"], _qf["filename"])
            # Download mmproj for VLMs
            _mmproj_path = None
            if is_vlm and mmproj_files:
                _mmproj_name = mmproj_files[0]["filename"]
                console.print(
                    f"  [dim]Downloading vision projector {_mmproj_name}...[/]"
                )
                _mmproj_path = _download_gguf(model["repo_id"], _mmproj_name)
                if not _mmproj_path:
                    console.print(
                        f"  [yellow]mmproj download failed — model will be text-only[/]"
                    )
            if _ans == "s" and _dl_path:
                _binary = str(BACKENDS["llamacpp"]["binary"])
                if not os.path.exists(_binary):
                    _binary = shutil.which("llama-server") or _binary
                console.print(f"\n  [bold]Starting llama-server...[/]")
                _cmd = [
                    _binary,
                    "-m",
                    _dl_path,
                    "--port",
                    "8089",
                    "-ngl",
                    "99",
                    "-c",
                    "32768",
                    "-fa",
                    "on",
                    "-ctk",
                    "q4_0",
                    "-ctv",
                    "q4_0",
                    "--jinja",
                    "--no-warmup",
                ]
                if _mmproj_path:
                    _cmd += ["--mmproj", _mmproj_path]
                _proc = subprocess.Popen(
                    _cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                for _w in range(60):
                    try:
                        _rq = urllib.request.Request("http://127.0.0.1:8089/health")
                        with urllib.request.urlopen(_rq, timeout=1):
                            console.print(f"  [green]✓ Server ready on :8089[/]")
                            if _mmproj_path:
                                console.print(
                                    f"  [magenta]✓ Vision projector loaded — image input enabled[/]"
                                )
                            console.print(f"  [dim]API: http://127.0.0.1:8089/v1[/]")
                            console.print(
                                f"  [dim]Claude Code: localfit --launch claude[/]"
                            )
                            break
                    except Exception:
                        time.sleep(1)
                else:
                    console.print(f"  [yellow]Server starting... check :8089[/]")
                try:
                    _proc.wait()
                except KeyboardInterrupt:
                    _proc.kill()

    elif _action == "serve_cloud":
        _gpu = _chosen["gpu"]
        _qf = _chosen["quant"]
        console.print(
            f"\n  [yellow]☁ {_gpu['name']} — {_qf['quant']} ({_qf['size_gb']}GB) ${_gpu['price']:.2f}/hr[/]"
        )
        console.print(
            f"\n  [bold]Start RunPod pod?[/]  [cyan]y[/] yes  ·  [cyan]q[/] cancel\n"
        )
        try:
            _ans = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if _ans == "y":
            from localfit.cloud import cloud_serve

            cloud_serve(model["repo_id"])

    elif _action == "switch_model":
        simulate_hf_model(_chosen["repo_id"])


def _simulate_with_real_size(
    gguf, repo_id, specs, gpu_total, gpu_used, mmproj_size_gb=0
):
    """Show detailed fit analysis for a specific GGUF file with real size."""
    size_gb = gguf["size_gb"]
    total_gb = size_gb + mmproj_size_gb
    total_mb = int(total_gb * 1024)
    size_mb = int(size_gb * 1024)
    fits = total_mb < gpu_total
    fits_free = total_mb < (gpu_total - gpu_used)

    kv_per_1k = max(2, int(size_gb * 0.4))
    tps = min(120, max(1, int(49 * 12 / max(1, total_gb)))) if fits else max(1, int(5))

    vlm_note = f" + mmproj {mmproj_size_gb}G" if mmproj_size_gb else ""
    console.print(
        f"\n  [bold]{repo_id}[/]  ·  [cyan]{gguf['quant']}[/]  ·  [bold]{size_gb}GB{vlm_note}[/]"
    )

    # Memory bar
    bw = 50
    mb = int(min(1.0, total_mb / gpu_total) * bw) if gpu_total else 0
    ub = int(min(1.0, gpu_used / gpu_total) * bw) if gpu_total else 0
    fb = max(0, bw - mb - ub)
    console.print(
        f"\n  [cyan]{'█' * ub}[/][{'green' if fits else 'red'}]{'█' * mb}[/][dim]{'░' * fb}[/]  {gpu_total // 1024}GB"
    )
    console.print(
        f"  [cyan]■[/] used:{gpu_used // 1024}G  [{'green' if fits else 'red'}]■[/] model:{total_gb}G  [dim]░[/] free:{max(0, gpu_total - gpu_used - total_mb) // 1024}G"
    )

    # Context table
    console.print()
    for ctx in [8192, 32768, 65536, 131072]:
        kv = kv_per_1k * (ctx // 1024)
        tot = total_mb + kv
        h = gpu_total - tot
        icon = "[green]✓[/]" if h > 2000 else "[yellow]⚠[/]" if h > 0 else "[red]✗[/]"
        kv_s = f"{kv}M" if kv < 1024 else f"{kv / 1024:.1f}G"
        console.print(
            f"  {icon} {ctx // 1024}K ctx  →  model {total_gb}G + KV {kv_s} = {tot / 1024:.1f}G"
        )

    is_cpu_only = specs.get("cpu_only", False)
    if is_cpu_only:
        cpu_tps = max(1, int(12 * 4 / max(1, total_gb))) if fits else 0
        console.print(
            f"\n  Est. speed: [bold]~{cpu_tps} tok/s[/] [dim](CPU-only, no GPU acceleration)[/]"
        )
        if fits:
            console.print(f"  [yellow]→ For 5-10x faster: localfit --login runpod[/]")
    else:
        console.print(
            f"\n  Est. speed: [bold]~{tps} tok/s[/]"
            + ("" if fits else "  [red](CPU swap)[/]")
        )

    if not fits:
        if is_cpu_only:
            console.print(
                f"  [yellow]→ Too big for {specs['ram_gb']}GB RAM. Try a smaller quant or cloud GPU: localfit --login runpod[/]"
            )
        elif IS_MAC:
            console.print(
                f"  [yellow]→ Try a smaller quant or: sudo sysctl iogpu.wired_limit_mb={int(specs['ram_gb'] * 1024 * 0.9)}[/]"
            )
        else:
            console.print(
                f"  [yellow]→ Try a smaller quant or cloud GPU: localfit --login runpod[/]"
            )


def _download_gguf(repo_id, filename):
    """Download a GGUF file (or split GGUF) from HuggingFace.

    Handles:
    - Single files: download directly
    - Split files (e.g. model-00001-of-00011.gguf): download ALL parts
    - Disk space check before downloading
    - Cleanup offer if not enough space
    """
    import re as _re

    # Detect split file pattern: name-00011-of-00011.gguf
    split_match = _re.search(r"-(\d{5})-of-(\d{5})\.gguf$", filename)
    is_split = bool(split_match)

    if is_split:
        total_parts = int(split_match.group(2))
        # Build list of all part filenames
        base = filename[: split_match.start()]
        folder = os.path.dirname(filename)
        part_files = []
        for i in range(1, total_parts + 1):
            part_name = (
                f"{folder}/{os.path.basename(base)}-{i:05d}-of-{total_parts:05d}.gguf"
            )
            if folder:
                part_files.append(part_name)
            else:
                part_files.append(
                    f"{os.path.basename(base)}-{i:05d}-of-{total_parts:05d}.gguf"
                )
        console.print(f"  [dim]Split model: {total_parts} parts[/]")
    else:
        part_files = [filename]

    # Check disk space
    di = get_disk_info()
    disk_free_gb = di.get("disk_free_gb", 0)

    # Estimate download size (use the reported size from HF)
    # For split files, we need total across all parts
    # The size in gguf_files is the size of the LAST part — estimate total
    est_size_gb = 0
    try:
        url_check = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
        req = urllib.request.Request(url_check, headers={"User-Agent": "localfit/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        for s in data.get("siblings", []):
            fn = s.get("rfilename", "")
            if is_split:
                # Sum all parts in this split
                base_name = os.path.basename(filename).rsplit("-", 2)[0]
                if base_name in fn and fn.endswith(".gguf"):
                    est_size_gb += s.get("size", 0) / (1024**3)
            elif fn == filename:
                est_size_gb = s.get("size", 0) / (1024**3)
    except Exception:
        est_size_gb = 15  # rough fallback

    if est_size_gb > disk_free_gb:
        console.print(f"\n  [red]Not enough disk space![/]")
        console.print(f"  Need: {est_size_gb:.1f}GB  Free: {disk_free_gb}GB")
        console.print()

        # Offer cleanup
        cache_gb = di.get("hf_cache_gb", 0)
        if cache_gb > est_size_gb:
            console.print(f"  [yellow]HuggingFace cache: {cache_gb}GB[/]")
            console.print(f"  [dim]Run: localfit --cleanup to free space[/]")
            console.print()
            try:
                ans = input("  Clean up cache now? (y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if ans in ("y", "yes"):
                # List models and let user pick what to delete
                models = di.get("models", [])
                if models:
                    console.print(f"\n  [bold]Installed models:[/]")
                    for i, m in enumerate(models, 1):
                        console.print(f"  {i}. {m['name']}  ({m['size_gb']}GB)")
                    console.print(f"  a. Delete ALL cached models")
                    console.print()
                    try:
                        choice = input("  Delete which? (number/a/n): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        return None
                    if choice == "a":
                        import shutil

                        hf_cache = HOME / ".cache/huggingface/hub"
                        if hf_cache.exists():
                            shutil.rmtree(hf_cache)
                            console.print(f"  [green]✓ Cleared {cache_gb}GB[/]")
                    elif choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(models):
                            m = models[idx]
                            # Find and delete the model's cache dir
                            import shutil

                            model_path = Path(m["path"])
                            # Walk up to find the models-- directory
                            for parent in model_path.parents:
                                if parent.name.startswith("models--"):
                                    shutil.rmtree(parent)
                                    console.print(
                                        f"  [green]✓ Deleted {m['name']} ({m['size_gb']}GB)[/]"
                                    )
                                    break
                else:
                    console.print(f"  [dim]No models to clean up.[/]")
                    return None

                # Re-check disk
                di = get_disk_info()
                disk_free_gb = di.get("disk_free_gb", 0)
                if est_size_gb > disk_free_gb:
                    console.print(
                        f"  [red]Still not enough: need {est_size_gb:.1f}GB, have {disk_free_gb}GB[/]"
                    )
                    return None
            else:
                return None

    # Download with Rich progress bar (suppress ALL other progress bars)
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        DownloadColumn,
        TransferSpeedColumn,
        TimeRemainingColumn,
        TextColumn,
    )

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        import logging

        # Suppress ALL HF progress output
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        # Also silence tqdm globally
        os.environ["TQDM_DISABLE"] = "1"
        try:
            import tqdm

            tqdm.tqdm.__init__.__defaults__ = (
                (None,) * len(tqdm.tqdm.__init__.__defaults__)
                if hasattr(tqdm.tqdm.__init__, "__defaults__")
                and tqdm.tqdm.__init__.__defaults__
                else None
            )
            tqdm.tqdm.disable = True
        except Exception:
            pass

        if is_split:
            pattern = (
                os.path.dirname(filename) + "/*"
                if os.path.dirname(filename)
                else os.path.basename(filename).rsplit("-", 2)[0] + "*"
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=30),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Downloading {total_parts} parts ({est_size_gb:.1f}GB)",
                    total=None,
                )
                path = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[pattern],
                )
                progress.update(task, completed=100, total=100)

            first_part = part_files[0]
            full_path = os.path.join(path, first_part)
            if os.path.exists(full_path):
                console.print(f"  [green]✓ Downloaded {total_parts} parts[/]")
                return full_path
            for f in Path(path).rglob("*-00001-of-*.gguf"):
                console.print(f"  [green]✓ Downloaded {total_parts} parts[/]")
                return str(f)
            console.print(f"  [red]Download completed but can't find first part[/]")
            return path
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=30),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"  {os.path.basename(filename)}",
                    total=int(est_size_gb * 1024**3) if est_size_gb else None,
                )
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                progress.update(
                    task,
                    completed=int(est_size_gb * 1024**3) if est_size_gb else 100,
                    total=int(est_size_gb * 1024**3) if est_size_gb else 100,
                )

            console.print(f"  [green]✓ Downloaded[/]")
            return path

    except ImportError:
        console.print(f"  [red]pip install huggingface_hub required for downloads[/]")
        return None
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            console.print(f"  [red]Gated model — run: huggingface-cli login[/]")
        else:
            console.print(f"  [red]Download failed: {e}[/]")
        return None
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            console.print(f"  [red]Gated model — run: huggingface-cli login[/]")
        else:
            console.print(f"  [red]Download failed: {e}[/]")
        return None


def simulate_model_fit(model_query):
    """Predict if a model will fit BEFORE downloading."""
    import re as _re

    specs = get_machine_specs()
    metal = get_metal_gpu_stats()

    gpu_total = metal.get("total_mb") or specs["gpu_total_mb"]
    gpu_used = metal.get("alloc_mb", 0)
    gpu_free = max(0, gpu_total - gpu_used)

    # Find in known models
    model_id = None
    model_info = None
    query = model_query.lower().replace("-", "").replace("_", "").replace(" ", "")
    for mid, m in MODELS.items():
        mid_clean = mid.lower().replace("-", "").replace("_", "")
        name_clean = (
            m["name"].lower().replace("-", "").replace("_", "").replace(" ", "")
        )
        if query in mid_clean or query in name_clean:
            model_id = mid
            model_info = m
            break

    if not model_info:
        param_match = _re.search(r"(\d+)b", query)
        quant_match = _re.search(r"q(\d)", query)
        if param_match:
            params_b = int(param_match.group(1))
            quant = int(quant_match.group(1)) if quant_match else 4
            bpw = {2: 2.5, 3: 3.5, 4: 4.5, 5: 5.5, 6: 6.5, 8: 8.5}.get(quant, 4.5)
            size_gb = round(params_b * bpw / 8, 1)
            model_info = {"name": f"{params_b}B Q{quant}", "size_gb": size_gb}
        else:
            console.print(f"\n  [red]Unknown model: {model_query}[/]")
            console.print(
                f"  [dim]Known: {', '.join(MODELS.keys())}  or  '70b q4'[/]\n"
            )
            return

    name = model_info["name"]
    size_gb = model_info["size_gb"]
    size_mb = int(size_gb * 1024)
    kv_per_1k = max(2, int(size_gb * 0.4))  # MB per 1K ctx

    fits_gpu = size_mb < gpu_total
    fits_free = size_mb < gpu_free
    base_tps = (
        min(120, max(1, int(49 * 12 / max(1, size_gb))))
        if fits_gpu
        else max(1, int(10 * 16 / max(1, size_gb)))
    )

    # Render
    console.clear()
    console.print()

    if fits_free:
        console.print(
            f"  [green bold]✓ {name} WILL FIT[/]  ·  {size_gb}GB model  ·  {gpu_free // 1024}GB free"
        )
    elif fits_gpu:
        console.print(
            f"  [yellow bold]⚠ {name} TIGHT FIT[/]  ·  {size_gb}GB  ·  close apps first"
        )
    else:
        console.print(
            f"  [red bold]✗ {name} WON'T FIT[/]  ·  {size_gb}GB model  ·  {gpu_total // 1024}GB limit"
        )
        console.print()
        console.print(f"  [yellow]☁ Run it on cloud instead:[/]")
        # Show cost estimates for different durations
        cloud_gpus = [
            ("RTX 3090", 24, 0.50, 35),
            ("RTX 4090", 24, 0.59, 45),
            ("A100 80GB", 80, 2.69, 90),
        ]
        for gname, gvram, gprice, gtps in cloud_gpus:
            if size_gb < gvram:
                console.print(
                    f"    [cyan]{gname}[/] ({gvram}GB)  ~{gtps} tok/s  "
                    f"[dim]1h=${gprice:.2f}  2h=${gprice * 2:.2f}  8h=${gprice * 8:.2f}[/]"
                )
        console.print(f"  [dim]  localfit --serve {name} --cloud --budget 2h[/]")

    console.print(
        f"  [dim]{specs['chip']}  ·  {specs['ram_gb']}GB RAM  ·  GPU budget: {gpu_total // 1024}GB[/]\n"
    )

    # Memory bar
    bw = 60
    mb = int(min(1.0, size_mb / gpu_total) * bw) if gpu_total else 0
    ub = int(min(1.0, gpu_used / gpu_total) * bw) if gpu_total else 0
    fb = max(0, bw - mb - ub)
    console.print(
        f"  GPU Memory:  [cyan]{'█' * ub}[/][{'green' if fits_gpu else 'red'}]{'█' * mb}[/][dim]{'░' * fb}[/]"
    )
    console.print(
        f"  [cyan]■[/] used:{gpu_used // 1024}G  [{'green' if fits_gpu else 'red'}]■[/] model:{size_gb}G  [dim]░[/] free:{max(0, gpu_total - gpu_used - size_mb) // 1024}G\n"
    )

    # Performance
    perf = Table(
        show_header=True, header_style="bold", border_style="dim", padding=(0, 1)
    )
    perf.add_column("", width=18)
    perf.add_column("Value", width=16)
    perf.add_column("", width=38)
    perf.add_row(
        "Model",
        f"{size_gb} GB",
        "Fits GPU" if fits_gpu else "[red]Exceeds GPU → swap[/]",
    )
    perf.add_row(
        "Compute",
        "GPU" if fits_gpu else "[red]CPU[/]",
        "All layers on GPU" if fits_gpu else "[red]5-10x slower[/]",
    )
    perf.add_row(
        "Speed", f"~{base_tps} tok/s", "" if fits_gpu else "[red]swap thrashing[/]"
    )
    perf.add_row(
        "Download", f"~{max(1, int(size_gb * 12))}s", f"at 100MB/s ({size_gb}GB)"
    )
    console.print(perf)
    console.print()

    # Context table
    ct = Table(
        title="Context Length vs Memory",
        show_header=True,
        header_style="bold",
        border_style="dim",
        padding=(0, 1),
    )
    ct.add_column("Context", width=8)
    ct.add_column("KV Cache", width=8, justify="right")
    ct.add_column("Total", width=8, justify="right")
    ct.add_column("Verdict", width=25)
    for ctx in [4096, 8192, 32768, 65536, 131072]:
        kv = kv_per_1k * (ctx // 1024)
        tot = size_mb + kv
        h = gpu_total - tot
        s = (
            "[green]✓ fits[/]"
            if h > 2000
            else f"[yellow]⚠ tight[/]"
            if h > 0
            else f"[red]✗ OOM ({-h // 1024}GB over)[/]"
        )
        ct.add_row(
            f"{ctx // 1024}K",
            f"{kv}M" if kv < 1024 else f"{kv / 1024:.1f}G",
            f"{tot / 1024:.1f}G",
            s,
        )
    console.print(ct)

    console.print()
    if not fits_gpu:
        for mid, m in sorted(
            MODELS.items(), key=lambda x: x[1]["size_gb"], reverse=True
        ):
            if m["size_gb"] * 1024 < gpu_total:
                console.print(
                    f"  [green]→ Try:[/] {m['name']} ({m['size_gb']}GB) — {m.get('description', '')}"
                )
                break
        console.print(
            f"  [green]→ Or:[/] sudo sysctl iogpu.wired_limit_mb={int(specs['ram_gb'] * 1024 * 0.9)}"
        )
    elif not fits_free:
        console.print(
            f"  [yellow]→[/] localcoder --cleanup  [dim](free {gpu_used // 1024}GB)[/]"
        )
    else:
        console.print(
            f"  [green]→[/] localcoder{' -m ' + model_id if model_id else ''}  [dim](ready to run)[/]"
        )
    console.print()


def recommend_model(ram_gb, cpu_only=False):
    """Recommend the best model for given RAM. Handles CPU-only systems."""
    if cpu_only:
        # CPU-only: recommend small models that run well on CPU, suggest cloud for bigger ones
        # Use >= 12 threshold for E4B (5.5GB model + KV cache + OS overhead fits in 12GB+ RAM)
        if ram_gb >= 12:
            return "gemma4-e4b", (
                "Gemma 4 E4B Q4 (~5.5GB) — runs on CPU with ~3-8 tok/s. Audio + image + code.\n"
                "  [yellow]→ For faster inference, try cloud GPU:[/] localfit --login runpod"
            )
        elif ram_gb >= 8:
            return "gemma4-e2b", (
                "Gemma 4 E2B Q4 (~4GB) — runs on CPU with ~5-12 tok/s.\n"
                "  [yellow]→ For faster inference, try cloud GPU:[/] localfit --login runpod"
            )
        else:
            return "gemma4-e2b", (
                "Gemma 4 E2B Q2 (~2GB) — minimal quality loss, runs on CPU.\n"
                "  [yellow]→ For better models, try cloud GPU:[/] localfit --login runpod"
            )

    if ram_gb >= 48:
        return (
            "gemma4-26b",
            "26B Q4_K_M (best quality) + vision + 128K context. Plenty of headroom.",
        )
    elif ram_gb >= 36:
        return "qwen35b-a3b", "Qwen 3.5 35B-A3B Q3_K_XL — best coding quality at 36GB+."
    elif ram_gb >= 24:
        return (
            "gemma4-26b",
            "Gemma 4 26B Q3_K_XL — 49 tok/s, best overall for 24GB. Also try Qwen 35B Q2.",
        )
    elif ram_gb >= 16:
        return (
            "gemma4-e4b",
            "E4B is the sweet spot for 16GB. Audio + image + code, 57 tok/s.",
        )
    elif ram_gb >= 8:
        return "qwen35-4b", "Qwen 3.5 4B — ultrafast at 50 tok/s, only 2.7GB GPU."
    else:
        return "gemma4-e2b", "E2B is the only option under 8GB."


def can_run_simultaneously(ram_gb, model1_gb, model2_gb):
    """Check if two models can run at the same time."""
    gpu_limit = ram_gb * 0.67  # Metal limit ~67% of unified memory
    return (model1_gb + model2_gb) < gpu_limit


def estimate_vram_gb(
    params_b,
    quant="Q4_K_M",
    ctx_k=32,
    kv_quant="q4_0",
    is_moe=False,
    active_params_b=None,
):
    """Estimate VRAM needed for a model at a given quant level.

    Args:
        params_b: Total parameter count in billions
        quant: Quantization type (Q4_K_M, Q3_K_XL, etc.)
        ctx_k: Context length in thousands (32 = 32K)
        kv_quant: KV cache quant (f16, q4_0, q8_0)
        is_moe: Whether model is Mixture of Experts
        active_params_b: For MoE, the active params per token
    Returns:
        dict with weight_gb, kv_gb, overhead_gb, total_gb
    """
    bpw = QUANT_BPW.get(quant, 4.8)
    weight_gb = (params_b * bpw) / 8  # billion params * bits / 8 = GB

    # KV cache estimate
    # Rough: 2 bytes per param per token for f16, halved for q4_0
    hidden_dim = int(params_b**0.5 * 1024)  # rough estimate
    if params_b <= 4:
        hidden_dim = 2048
    elif params_b <= 9:
        hidden_dim = 4096
    elif params_b <= 14:
        hidden_dim = 5120
    elif params_b <= 32:
        hidden_dim = 6144
    elif params_b <= 70:
        hidden_dim = 8192
    else:
        hidden_dim = 8192

    n_layers = max(1, int(params_b / (hidden_dim * hidden_dim / 1e9 * 2)))
    if n_layers < 16:
        n_layers = 32  # fallback
    if n_layers > 128:
        n_layers = 80

    kv_bytes_per_token = n_layers * hidden_dim * 2 * 2  # K + V, 2 bytes each for f16
    if kv_quant == "q4_0":
        kv_bytes_per_token //= 4
    elif kv_quant == "q8_0":
        kv_bytes_per_token //= 2

    kv_gb = (kv_bytes_per_token * ctx_k * 1024) / (1024**3)

    # Overhead (CUDA/Metal runtime, activations, etc.)
    overhead_gb = 0.5 if params_b < 10 else 1.0 if params_b < 30 else 1.5

    # MoE: only active experts loaded in compute buffer
    compute_gb = 0
    if is_moe and active_params_b:
        compute_gb = (
            (active_params_b * bpw) / 8 * 0.1
        )  # ~10% extra for active computation

    total = weight_gb + kv_gb + overhead_gb + compute_gb
    return {
        "weight_gb": round(weight_gb, 1),
        "kv_gb": round(kv_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "total_gb": round(total, 1),
    }


def get_model_quant_options(model_id):
    """Get all quant options for a model with VRAM estimates.

    Returns list of {quant, vram_gb, fits, fits_with_image, file_pattern}
    sorted by quality (best first).
    """
    m = MODELS.get(model_id)
    if not m:
        return []

    # Detect hardware
    specs = get_machine_specs()
    gpu_mb = get_metal_gpu_stats().get("total_mb", 0) or specs.get("gpu_total_mb", 0)
    gpu_gb = gpu_mb / 1024
    usable_gb = gpu_gb - 1.5  # reserve for OS

    # Parse model params from name/size
    params_b = m.get("params_b", 0)
    if not params_b:
        # Estimate from name
        name = m.get("name", "") + model_id
        for s in [
            "397",
            "235",
            "122",
            "120",
            "70",
            "32",
            "31",
            "30",
            "27",
            "26",
            "24",
            "20",
            "14",
            "12",
            "9",
            "8",
            "7",
            "4",
            "3",
            "2",
            "1",
            "0.6",
        ]:
            if (
                s + "B" in name
                or s + "b" in name.lower().replace("-", "")
                or s + "B" in model_id.upper()
            ):
                params_b = float(s)
                break
        if not params_b:
            params_b = m.get("size_gb", 5) * 2  # rough guess

    is_moe = (
        "moe" in m.get("description", "").lower()
        or "a3b" in model_id.lower()
        or "a4b" in model_id.lower()
        or "a10b" in model_id.lower()
    )
    active_b = None
    if is_moe:
        for tag in ["a3b", "a4b", "a10b", "a17b", "a22b", "a35b"]:
            if tag in model_id.lower() or tag in m.get("name", "").lower():
                active_b = float(tag.replace("a", "").replace("b", ""))
                break

    # Common quants to show
    quant_order = [
        "BF16",
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q4_K_M",
        "Q4_K_XL",
        "IQ4_NL",
        "Q3_K_XL",
        "Q3_K_M",
        "Q3_K_S",
        "Q2_K_XL",
        "Q2_K",
        "IQ2_XXS",
    ]

    options = []
    for q in quant_order:
        est = estimate_vram_gb(
            params_b,
            q,
            ctx_k=32,
            kv_quant="q4_0",
            is_moe=is_moe,
            active_params_b=active_b,
        )
        fits = est["total_gb"] <= usable_gb
        fits_tight = est["total_gb"] <= usable_gb + 2
        # Check if can also run image model (klein-4b ~8GB)
        fits_with_image = (est["total_gb"] + 8) <= usable_gb

        options.append(
            {
                "quant": q,
                "vram_gb": est["total_gb"],
                "weight_gb": est["weight_gb"],
                "kv_gb": est["kv_gb"],
                "fits": "yes" if fits else "tight" if fits_tight else "no",
                "fits_with_image": fits_with_image,
                "params_b": params_b,
                "is_moe": is_moe,
            }
        )

    return options


def get_all_models_fit_report():
    """Get a fit report for ALL models in catalog for this hardware.

    Returns list of {model_id, name, best_quant, vram_gb, fits, description, source,
                     can_pair_with_image, where}
    sorted by: fits_local first, then by quality desc.
    """
    specs = get_machine_specs()
    gpu_mb = get_metal_gpu_stats().get("total_mb", 0) or specs.get("gpu_total_mb", 0)
    gpu_gb = gpu_mb / 1024
    usable_gb = gpu_gb - 1.5
    is_mac = IS_MAC
    has_mlx = False
    if is_mac:
        try:
            has_mlx = check_mlx_available()
        except Exception:
            pass

    report = []
    for mid, m in MODELS.items():
        options = get_model_quant_options(mid)
        # Find best quant that fits
        best_fit = None
        best_tight = None
        for opt in options:
            if opt["fits"] == "yes" and not best_fit:
                best_fit = opt
            elif opt["fits"] == "tight" and not best_tight:
                best_tight = opt

        best = best_fit or best_tight
        if best:
            where = "local"
            if is_mac and has_mlx:
                where = (
                    "local (MLX)" if m.get("backend") != "llamacpp" else "local (Metal)"
                )
            report.append(
                {
                    "model_id": mid,
                    "name": m["name"],
                    "best_quant": best["quant"],
                    "vram_gb": best["vram_gb"],
                    "fits": best["fits"],
                    "can_pair_with_image": best["fits_with_image"],
                    "description": m.get("description", ""),
                    "source": m.get("source", ""),
                    "where": where,
                    "is_moe": best.get("is_moe", False),
                    "cloud_only": False,
                }
            )
        else:
            # Doesn't fit locally — cloud only
            smallest = (
                options[-1]
                if options
                else {"quant": "IQ2_XXS", "vram_gb": m.get("size_gb", 99)}
            )
            report.append(
                {
                    "model_id": mid,
                    "name": m["name"],
                    "best_quant": smallest["quant"],
                    "vram_gb": smallest.get("vram_gb", m.get("size_gb", 99)),
                    "fits": "cloud",
                    "can_pair_with_image": False,
                    "description": m.get("description", ""),
                    "source": m.get("source", ""),
                    "where": "cloud (Kaggle/RunPod)",
                    "is_moe": smallest.get("is_moe", False)
                    if isinstance(smallest, dict)
                    else False,
                    "cloud_only": True,
                }
            )

    # Sort: fits_local first (yes > tight > cloud), then by vram desc (bigger = better quality)
    order = {"yes": 0, "tight": 1, "cloud": 2}
    report.sort(key=lambda x: (order.get(x["fits"], 3), -x["vram_gb"]))
    return report


def stop_conflicting_backends(target_backend):
    """Stop other backends to free GPU memory."""
    if target_backend == "ollama":
        if check_backend_running("llamacpp"):
            console.print(
                f"  [yellow]Stopping llama-server to free GPU for Ollama...[/]"
            )
            try:
                subprocess.run(["pkill", "-f", "llama-server"], timeout=5)
                time.sleep(2)
            except:
                pass
        # Kill any mlx_lm.server
        try:
            subprocess.run(["pkill", "-f", "mlx_lm.server"], timeout=5)
        except:
            pass
    elif target_backend == "llamacpp":
        if check_backend_running("ollama"):
            console.print(f"  [yellow]Unloading Ollama models to free GPU...[/]")
            try:
                models = get_running_models("ollama")
                for m in models:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            "http://127.0.0.1:11434/api/generate",
                            data=json.dumps({"model": m, "keep_alive": 0}).encode(),
                            headers={"Content-Type": "application/json"},
                        ),
                        timeout=5,
                    )
                time.sleep(2)
            except:
                pass
        try:
            subprocess.run(["pkill", "-f", "mlx_lm.server"], timeout=5)
        except:
            pass
    elif target_backend == "mlx":
        # Kill llama-server and ollama models to free unified memory
        if check_backend_running("llamacpp"):
            console.print(
                f"  [yellow]Stopping llama-server to free memory for MLX...[/]"
            )
            try:
                subprocess.run(["pkill", "-f", "llama-server"], timeout=5)
                time.sleep(2)
            except:
                pass
        if check_backend_running("ollama"):
            console.print(
                f"  [yellow]Unloading Ollama models to free memory for MLX...[/]"
            )
            try:
                models = get_running_models("ollama")
                for m in models:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            "http://127.0.0.1:11434/api/generate",
                            data=json.dumps({"model": m, "keep_alive": 0}).encode(),
                            headers={"Content-Type": "application/json"},
                        ),
                        timeout=5,
                    )
                time.sleep(2)
            except:
                pass


def start_ollama_serve():
    """Ensure Ollama is serving."""
    if check_backend_running("ollama"):
        return True
    try:
        subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(2)
        return check_backend_running("ollama")
    except:
        return False


# ── MLX backend (Apple Silicon) ──


def check_mlx_available():
    """Check if mlx-lm is pip-installed and we're on Apple Silicon."""
    if not IS_MAC:
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx_lm; print('ok')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def find_mlx_community_model(model_name):
    """
    Search mlx-community on HuggingFace for a quantized version of model_name.
    Returns the repo ID (e.g. 'mlx-community/Qwen2.5-7B-Instruct-8bit') or None.
    Tries common bit-widths: 8bit, 4bit, 6bit, bf16.
    """
    # Normalise: strip org prefix, lowercase for search
    base = model_name.split("/")[-1]
    # Strip common suffixes that differ between HF and mlx-community naming
    base_clean = base.lower().replace("_", "-")

    candidates = []
    for bits in ["8bit", "4bit", "6bit", "bf16", "3bit"]:
        candidates.append(f"mlx-community/{base}-{bits}")
        # Some repos use different casing
        candidates.append(f"mlx-community/{base.replace('-', '_')}-{bits}")

    for repo in candidates:
        try:
            url = f"https://huggingface.co/api/models/{repo}"
            req = urllib.request.Request(url, headers={"User-Agent": "localfit"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return repo
        except Exception:
            continue

    # Fallback: search HuggingFace API
    try:
        search_url = (
            f"https://huggingface.co/api/models?"
            f"author=mlx-community&search={urllib.parse.quote(base_clean)}&limit=5"
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": "localfit"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            results = json.loads(resp.read())
        for r in results:
            rid = r.get("id", "")
            if rid.startswith("mlx-community/") and base_clean[:8] in rid.lower():
                return rid
    except Exception:
        pass

    return None


def start_mlx_server(model_repo, port=8080, context=32768):
    """
    Start mlx_lm.server for a HuggingFace model repo.
    Returns the subprocess or None on failure.
    """
    import tempfile

    console.print(f"  [cyan]Backend: MLX (Apple Silicon native)[/]")
    console.print(f"  [dim]Model:   {model_repo}[/]")
    console.print(f"  [dim]Port:    {port}[/]")

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.server",
        "--model",
        model_repo,
        "--port",
        str(port),
        "--log-level",
        "WARNING",
    ]

    stderr_log = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, prefix="mlx-"
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_log)

    # Wait up to 120s for server to come up (model download can take a while)
    console.print(f"  [dim]Loading model (may download on first run)...[/]")
    for i in range(120):
        if proc.poll() is not None:
            stderr_log.flush()
            stderr_log.close()
            try:
                err = open(stderr_log.name).read()[-600:]
            except Exception:
                err = ""
            console.print(f"  [red]MLX server crashed:[/]\n{err}")
            return None
        try:
            url = f"http://127.0.0.1:{port}/v1/models"
            req = urllib.request.Request(
                url, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=1) as resp:
                data = json.loads(resp.read())
                # Find the model_id that matches what we loaded.
                # For local paths, mlx_lm uses the absolute path as the id.
                # For HF repos, it uses the repo id. Match by suffix.
                model_abs = (
                    str(Path(model_repo).resolve())
                    if Path(model_repo).exists()
                    else None
                )
                ids = [m.get("id", "") for m in data.get("data", [])]
                matched_id = None
                for mid in ids:
                    if model_abs and (mid == model_abs or mid.endswith(model_repo)):
                        matched_id = mid
                        break
                    if not model_abs and model_repo in mid:
                        matched_id = mid
                        break
                if not matched_id and ids:
                    # Fall back: use exact path or first id
                    matched_id = model_abs if model_abs else ids[0]
                console.print(f"  [green]✓ MLX server ready on :{port}[/]")
                console.print(
                    f"  [dim]API: http://127.0.0.1:{port}/v1  model: {matched_id or model_repo}[/]"
                )
                # Attach resolved model_id so callers can use it
                proc._mlx_model_id = matched_id or model_repo
                return proc
        except Exception:
            time.sleep(1)

    proc.kill()
    console.print(f"  [red]MLX server did not start in 120s[/]")
    return None


def convert_to_mlx(hf_model_repo, q_bits=4, upload_repo=None):
    """
    Convert any HuggingFace model to MLX format locally on Apple Silicon.
    Downloads BF16 model, quantizes to q_bits, saves to ~/.cache/localfit/mlx/.

    Requirements:
      - Apple Silicon Mac
      - mlx-lm installed
      - Enough RAM: model_size_bf16 ≈ params_B × 2GB (e.g. 7B needs ~14GB)

    Args:
        hf_model_repo: HuggingFace repo ID (e.g. "bytedance-research/UI-TARS-7B-DPO")
        q_bits: quantization bits (4 = 4-bit, 8 = 8-bit, None = no quant / bfloat16)
        upload_repo: if set, upload result to this HF repo ID after conversion

    Returns local path to converted model, or None on failure.
    """
    if not IS_MAC:
        console.print(f"  [red]MLX conversion requires Apple Silicon Mac[/]")
        return None
    if not check_mlx_available():
        console.print(f"  [yellow]mlx-lm not installed. Run: pip install mlx-lm[/]")
        return None

    model_name = hf_model_repo.split("/")[-1]
    mlx_cache = HOME / ".cache" / "localfit" / "mlx"
    mlx_cache.mkdir(parents=True, exist_ok=True)
    suffix = f"{q_bits}bit" if q_bits else "bf16"
    out_dir = mlx_cache / f"{model_name}-{suffix}-mlx"

    if out_dir.exists() and any(out_dir.glob("*.safetensors")):
        console.print(f"  [green]✓ Already converted:[/] {out_dir}")
        return str(out_dir)

    console.print(f"\n  [cyan]Converting {hf_model_repo} to MLX {suffix}...[/]")
    console.print(
        f"  [dim]Downloads full model then quantizes — needs ~{_estimate_bf16_ram(hf_model_repo)}GB RAM[/]"
    )
    console.print(f"  [dim]Output: {out_dir}[/]\n")

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        hf_model_repo,
        "--mlx-path",
        str(out_dir),
    ]
    if q_bits:
        cmd += ["-q", "--q-bits", str(q_bits)]
    if upload_repo:
        cmd += ["--upload-repo", upload_repo]

    result = subprocess.run(cmd, timeout=3600)  # 1h max
    if result.returncode != 0:
        console.print(f"  [red]MLX conversion failed[/]")
        return None

    console.print(f"  [green]✓ Converted to MLX {suffix}[/]")
    if upload_repo:
        console.print(f"  [green]✓ Uploaded to HF:[/] [cyan]{upload_repo}[/]")
    return str(out_dir)


def _estimate_bf16_ram(hf_model_repo):
    """Rough RAM estimate for BF16 model download during MLX conversion."""
    name = hf_model_repo.lower()
    for size, gb in [
        ("405b", 810),
        ("70b", 140),
        ("32b", 64),
        ("27b", 54),
        ("14b", 28),
        ("13b", 26),
        ("8b", 16),
        ("7b", 14),
        ("4b", 8),
        ("3b", 6),
        ("1.5b", 3),
        ("1b", 2),
    ]:
        if size in name:
            return gb
    return "?"


def select_best_backend(model_name, specs=None):
    """
    Pick the best inference backend for model_name on this machine.
    Returns (backend_id, mlx_repo_or_none).

    Priority on Mac:
      1. MLX — if mlx-lm installed AND mlx-community has this model
      2. llama.cpp — GGUF via llama-server (Metal accelerated)
      3. Ollama — if installed
    On Linux:
      1. llama.cpp (CUDA)
      2. Ollama
    """
    if specs is None:
        specs = get_machine_specs()

    if IS_MAC and check_mlx_available():
        mlx_repo = find_mlx_community_model(model_name)
        if mlx_repo:
            return ("mlx", mlx_repo)

    if check_backend_installed("llamacpp"):
        return ("llamacpp", None)

    if check_backend_installed("ollama"):
        return ("ollama", None)

    return (None, None)
