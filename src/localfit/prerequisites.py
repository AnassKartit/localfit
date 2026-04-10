"""Prerequisite detection and auto-install for localfit."""
import os
import shutil
import subprocess
import platform
import sys

from rich.console import Console

console = Console()
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_WSL = IS_LINUX and "microsoft" in (open("/proc/version").read().lower() if os.path.exists("/proc/version") else "")


def check_all():
    """Check all prerequisites. Returns dict of what's found/missing."""
    return {
        "llama_server": check_llama_server(),
        "cuda": check_cuda(),
        "metal": check_metal(),
        "node": check_node(),
        "claude_code": check_claude_code(),
        "hermes_agent": check_hermes_agent(),
        "open_webui": check_open_webui(),
        "python_version": check_python_version(),
        "huggingface_hub": check_huggingface_hub(),
    }


def check_llama_server():
    """Check if llama-server is installed."""
    paths = [
        shutil.which("llama-server"),
        os.path.expanduser("~/.unsloth/llama.cpp/llama-server"),
        "/usr/local/bin/llama-server",
    ]
    for p in paths:
        if p and os.path.exists(p):
            return {"found": True, "path": p}
    return {"found": False, "path": None}


def check_cuda():
    """Check if CUDA toolkit is available (Linux/NVIDIA)."""
    if IS_MAC:
        return {"found": False, "reason": "mac", "version": None}

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if os.path.exists(nvcc):
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=5)
            version = None
            for line in out.stdout.splitlines():
                if "release" in line.lower():
                    version = line.strip()
            return {"found": True, "path": nvcc, "version": version}
        except Exception:
            return {"found": True, "path": nvcc, "version": None}

    # Check nvidia-smi (driver exists but no toolkit)
    if shutil.which("nvidia-smi"):
        return {"found": False, "reason": "driver_only", "version": None}

    return {"found": False, "reason": "none", "version": None}


def check_metal():
    """Check if Apple Metal is available (macOS)."""
    if not IS_MAC:
        return {"found": False, "reason": "not_mac"}
    # Metal is always available on Apple Silicon
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=3)
        chip = out.stdout.strip()
        if "Apple" in chip:
            return {"found": True, "chip": chip}
    except Exception:
        pass
    return {"found": True, "chip": "Unknown Mac"}


def check_node():
    """Check if Node.js is installed (needed for Claude Code)."""
    node = shutil.which("node")
    if node:
        try:
            out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=3)
            return {"found": True, "version": out.stdout.strip(), "path": node}
        except Exception:
            return {"found": True, "version": "?", "path": node}
    return {"found": False}


def check_claude_code():
    """Check if Claude Code CLI is installed."""
    claude = shutil.which("claude")
    if claude:
        try:
            out = subprocess.run([claude, "--version"], capture_output=True, text=True, timeout=5)
            return {"found": True, "version": out.stdout.strip(), "path": claude}
        except Exception:
            return {"found": True, "version": "?", "path": claude}
    return {"found": False}


def check_python_version():
    """Check Python version."""
    v = sys.version_info
    return {"found": True, "version": f"{v.major}.{v.minor}.{v.micro}", "ok": v >= (3, 10)}


def check_open_webui():
    """Check if Open WebUI is installed and how."""
    # 1. Check if running already
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:8080/api/version", timeout=2) as r:
            data = __import__("json").loads(r.read())
            return {"found": True, "running": True, "version": data.get("version"), "method": "running"}
    except Exception:
        pass

    # 2. Check pip/uv installed
    open_webui_bin = shutil.which("open-webui")
    if open_webui_bin:
        return {"found": True, "running": False, "path": open_webui_bin, "method": "pip"}

    # 3. Check if uv can run it (python 3.11 available)
    try:
        r = subprocess.run(
            ["uv", "run", "--python", "3.11", "--with", "open-webui", "--", "open-webui", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            version = r.stdout.strip().split()[-1] if r.stdout.strip() else "?"
            return {"found": True, "running": False, "method": "uv", "version": version}
    except Exception:
        pass

    # 4. Check Docker
    try:
        r = subprocess.run(
            ["docker", "images", "ghcr.io/open-webui/open-webui", "--format", "{{.Tag}}"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return {"found": True, "running": False, "method": "docker", "version": r.stdout.strip().split()[0]}
    except Exception:
        pass

    # 5. Check if uv exists (can install on the fly)
    if shutil.which("uv"):
        return {"found": False, "can_install": True, "method": "uv"}

    return {"found": False, "can_install": False}


def check_cloudflared():
    """Check if cloudflared tunnel is installed."""
    cf = shutil.which("cloudflared")
    if cf:
        try:
            r = subprocess.run([cf, "--version"], capture_output=True, text=True, timeout=5)
            version = r.stdout.strip().split()[-1] if r.stdout.strip() else "?"
            return {"found": True, "path": cf, "version": version}
        except Exception:
            return {"found": True, "path": cf}
    # Check brew
    brew_path = "/opt/homebrew/bin/cloudflared"
    if os.path.exists(brew_path):
        return {"found": True, "path": brew_path}
    return {"found": False}


def ensure_cloudflared():
    """Install cloudflared if missing."""
    cf = check_cloudflared()
    if cf["found"]:
        return cf.get("path", "cloudflared")
    console.print(f"  [yellow]cloudflared not installed.[/]")
    if IS_MAC:
        console.print(f"  [cyan]brew install cloudflared[/]")
    else:
        console.print(f"  [cyan]curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared[/]")
    try:
        ans = input("  Install now? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if ans in ("y", "yes"):
        if IS_MAC:
            r = subprocess.run(["brew", "install", "cloudflared"], timeout=120)
        else:
            r = subprocess.run(["bash", "-c",
                "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"],
                timeout=120)
        if r.returncode == 0:
            cf = check_cloudflared()
            if cf["found"]:
                console.print(f"  [green]✓ Installed[/]")
                return cf["path"]
    return None


def check_hermes_agent():
    """Check if Hermes Agent is installed."""
    hermes = shutil.which("hermes")
    if hermes:
        return {"found": True, "path": hermes}
    paths = [
        os.path.expanduser("~/.local/bin/hermes"),
        os.path.expanduser("~/hermes-agent/run_agent.py"),
    ]
    for p in paths:
        if os.path.exists(p):
            return {"found": True, "path": p}
    return {"found": False}


def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        return {"found": True, "version": huggingface_hub.__version__}
    except ImportError:
        return {"found": False}


def print_status():
    """Print prerequisite status with install instructions."""
    checks = check_all()

    console.print(f"\n  [bold]Prerequisites[/]\n")

    # GPU
    if IS_MAC:
        metal = checks["metal"]
        if metal["found"]:
            console.print(f"  [green]✓[/] Metal GPU: {metal.get('chip', 'available')}")
        else:
            console.print(f"  [red]✗[/] No Apple Silicon detected")
    else:
        cuda = checks["cuda"]
        if cuda["found"]:
            console.print(f"  [green]✓[/] CUDA: {cuda.get('version', 'found')}")
        elif cuda.get("reason") == "driver_only":
            console.print(f"  [yellow]![/] NVIDIA driver found but CUDA toolkit missing")
            console.print(f"    [dim]Install: sudo apt install nvidia-cuda-toolkit[/]")
            console.print(f"    [dim]Or: https://developer.nvidia.com/cuda-downloads[/]")
        else:
            # Try to detect Intel/other iGPU on WSL
            gpu_name = None
            if IS_WSL:
                try:
                    _out = subprocess.run(
                        ["powershell.exe", "-Command",
                         "(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if _out.returncode == 0 and _out.stdout.strip():
                        gpu_name = _out.stdout.strip()
                except Exception:
                    pass
            if gpu_name:
                console.print(f"  [yellow]![/] {gpu_name} (iGPU — no CUDA, CPU-only inference)")
                console.print(f"    [dim]Models will run on CPU using system RAM (~3-12 tok/s)[/]")
                console.print(f"    [dim]For faster inference: localfit --login runpod (cloud GPU)[/]")
            else:
                console.print(f"  [red]✗[/] No GPU detected (CPU-only mode)")

    # llama-server
    ls = checks["llama_server"]
    if ls["found"]:
        console.print(f"  [green]✓[/] llama-server: {ls['path']}")
    else:
        console.print(f"  [red]✗[/] llama-server not found")
        if IS_MAC:
            console.print(f"    [dim]Install: curl -fsSL https://unsloth.ai/install.sh | sh[/]")
        else:
            cuda = checks["cuda"]
            if cuda["found"]:
                console.print(f"    [dim]Install:[/]")
                console.print(f"    [dim]  git clone https://github.com/ggml-org/llama.cpp[/]")
                console.print(f"    [dim]  cmake llama.cpp -B build -DGGML_CUDA=ON[/]")
                console.print(f"    [dim]  cmake --build build -j --target llama-server[/]")
                console.print(f"    [dim]  sudo cp build/bin/llama-server /usr/local/bin/[/]")
            else:
                console.print(f"    [dim]Install (CPU only):[/]")
                console.print(f"    [dim]  git clone https://github.com/ggml-org/llama.cpp[/]")
                console.print(f"    [dim]  cmake llama.cpp -B build[/]")
                console.print(f"    [dim]  cmake --build build -j --target llama-server[/]")

    # Claude Code
    cc = checks["claude_code"]
    if cc["found"]:
        console.print(f"  [green]✓[/] Claude Code: {cc.get('version', 'found')}")
    else:
        node = checks["node"]
        if node["found"]:
            console.print(f"  [dim]·[/] Claude Code: not installed  [dim](npm install -g @anthropic-ai/claude-code)[/]")
        else:
            console.print(f"  [dim]·[/] Claude Code: not installed  [dim](needs Node.js first)[/]")
            if IS_MAC:
                console.print(f"    [dim]Install Node: brew install node[/]")
            else:
                console.print(f"    [dim]Install Node: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash && sudo apt install -y nodejs[/]")

    # huggingface_hub
    hf = checks["huggingface_hub"]
    if hf["found"]:
        console.print(f"  [green]✓[/] huggingface_hub: {hf.get('version', '?')}")
    else:
        console.print(f"  [dim]·[/] huggingface_hub: not installed  [dim](pip install huggingface_hub)[/]")

    # Python
    py = checks["python_version"]
    if py["ok"]:
        console.print(f"  [green]✓[/] Python {py['version']}")
    else:
        console.print(f"  [red]✗[/] Python {py['version']} (need 3.10+)")

    console.print()
    return checks


def _get_llama_release_asset():
    """Pick the right pre-built llama.cpp binary for this platform."""
    import json
    import urllib.request

    machine = platform.machine().lower()
    cuda = check_cuda()

    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"User-Agent": "localfit/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None, None

    tag = data["tag_name"]
    assets = {a["name"]: a["browser_download_url"] for a in data["assets"]}

    # Pick the best binary for this platform
    if IS_MAC:
        if "arm" in machine or "aarch64" in machine:
            key = f"llama-{tag}-bin-macos-arm64.tar.gz"
        else:
            key = f"llama-{tag}-bin-macos-x64.tar.gz"
    elif IS_LINUX:
        if cuda["found"]:
            # Prefer CUDA build — check for 12.x first
            for suffix in ["cuda-12.4-x64", "cuda-13.1-x64"]:
                key = f"llama-{tag}-bin-ubuntu-{suffix}.tar.gz"
                if key in assets:
                    break
            else:
                key = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
        elif "arm" in machine or "aarch64" in machine:
            key = f"llama-{tag}-bin-ubuntu-arm64.tar.gz"
        else:
            key = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
    else:
        return None, None

    url = assets.get(key)
    return key, url


def _install_from_prebuilt(url, asset_name):
    """Download and install pre-built llama.cpp binaries."""
    import tarfile
    import tempfile
    import urllib.request

    install_dir = os.path.join(str(os.path.expanduser("~")), ".local", "bin")
    os.makedirs(install_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, asset_name)

        # Download with progress
        console.print(f"  [dim]Downloading {asset_name}...[/]")
        try:
            urllib.request.urlretrieve(url, archive_path)
        except Exception as e:
            console.print(f"  [red]Download failed: {e}[/]")
            return None

        # Extract
        console.print(f"  [dim]Extracting...[/]")
        try:
            with tarfile.open(archive_path) as tf:
                tf.extractall(tmpdir)
        except Exception as e:
            console.print(f"  [red]Extract failed: {e}[/]")
            return None

        # Find and copy binaries + shared libraries
        # Libraries MUST be in the same directory as binaries for backend loading
        installed = []
        for root, dirs, files in os.walk(tmpdir):
            for fname in files:
                src = os.path.join(root, fname)
                # Copy binaries
                if fname in ("llama-server", "llama-cli", "llama-mtmd-cli"):
                    dst = os.path.join(install_dir, fname)
                    shutil.copy2(src, dst)
                    os.chmod(dst, 0o755)
                    installed.append(fname)
                # Copy shared libraries (.so, .dylib) — same dir as binaries
                elif fname.endswith((".so", ".dylib")) or ".so." in fname:
                    dst = os.path.join(install_dir, fname)
                    shutil.copy2(src, dst)
                    os.chmod(dst, 0o755)

        if "llama-server" in installed:
            server_path = os.path.join(install_dir, "llama-server")
            console.print(f"  [green]✓ Installed: {server_path}[/]")
            for b in installed:
                if b != "llama-server":
                    console.print(f"  [green]✓ Installed: {os.path.join(install_dir, b)}[/]")

            # Ensure ~/.local/bin is on PATH and libs are findable
            _ensure_local_bin_on_path(install_dir)
            _setup_lib_path(install_dir)
            return server_path

    return None


def _ensure_local_bin_on_path(local_bin):
    """Add ~/.local/bin to PATH in shell profile if not already there."""
    if local_bin in os.environ.get("PATH", ""):
        return

    os.environ["PATH"] = local_bin + ":" + os.environ.get("PATH", "")

    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        profile = os.path.expanduser("~/.zshrc")
    elif os.path.exists(os.path.expanduser("~/.bashrc")):
        profile = os.path.expanduser("~/.bashrc")
    else:
        profile = os.path.expanduser("~/.profile")

    try:
        content = open(profile).read() if os.path.exists(profile) else ""
        if ".local/bin" not in content:
            with open(profile, "a") as f:
                f.write('\n# Added by localfit\nexport PATH="$HOME/.local/bin:$PATH"\n')
            console.print(f"  [dim]Added ~/.local/bin to PATH in {profile}[/]")
    except Exception:
        console.print(f"  [yellow]Add ~/.local/bin to your PATH manually[/]")


def _setup_lib_path(lib_dir):
    """Ensure shared libraries in lib_dir are findable at runtime."""
    if not os.path.isdir(lib_dir):
        return

    # Set for current process
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld if ld else "")

    # Persist in shell profile
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        profile = os.path.expanduser("~/.zshrc")
    elif os.path.exists(os.path.expanduser("~/.bashrc")):
        profile = os.path.expanduser("~/.bashrc")
    else:
        profile = os.path.expanduser("~/.profile")

    try:
        content = open(profile).read() if os.path.exists(profile) else ""
        marker = "# localfit llama.cpp libs"
        if marker not in content:
            with open(profile, "a") as f:
                f.write(f'\n{marker}\nexport LD_LIBRARY_PATH="{lib_dir}:$LD_LIBRARY_PATH"\n')
            console.print(f"  [dim]Added LD_LIBRARY_PATH to {profile}[/]")
    except Exception:
        console.print(f"  [yellow]Add to your shell profile: export LD_LIBRARY_PATH=\"{lib_dir}:$LD_LIBRARY_PATH\"[/]")


def ensure_llama_server(auto_install=False):
    """Ensure llama-server is available. Downloads pre-built binary if missing.

    No cmake, no compilation — just download and run. Works on:
    - macOS (ARM + Intel)
    - Linux x64 (CPU, CUDA, ROCm)
    - WSL2 (same as Linux)

    Args:
        auto_install: If True, skip confirmation prompt (used when called from --serve).
    """
    ls = check_llama_server()
    if ls["found"]:
        return ls["path"]

    console.print(f"\n  [yellow]llama-server not found.[/]")

    # Find the right pre-built binary
    asset_name, url = _get_llama_release_asset()

    if not url:
        console.print(f"  [red]Could not find a pre-built binary for your platform.[/]")
        console.print(f"  [dim]Build manually: https://github.com/ggml-org/llama.cpp#build[/]")
        return None

    cuda = check_cuda()
    variant = "CUDA" if cuda["found"] else "CPU"
    size_note = ""
    if "cuda" in (asset_name or ""):
        size_note = " (~240MB)"
    elif "macos" in (asset_name or ""):
        size_note = " (~38MB)"
    else:
        size_note = " (~30MB)"

    console.print(f"  Install llama.cpp pre-built binary ({variant})?{size_note}")
    console.print(f"  [dim]{asset_name}[/]")
    console.print(f"  [dim]Installs to ~/.local/bin/ (no sudo needed)[/]")
    console.print()

    if auto_install:
        console.print(f"  [dim]Auto-installing...[/]")
    else:
        try:
            ans = input("  Install now? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if ans not in ("y", "yes"):
            return None

    path = _install_from_prebuilt(url, asset_name)
    if path:
        # Verify it works
        try:
            r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                console.print(f"  [green]✓ llama-server works![/]  {r.stdout.strip()}")
        except Exception:
            pass
        return path

    # Fallback: offer manual build instructions
    console.print(f"\n  [yellow]Pre-built install failed. Manual build:[/]")
    console.print(f"  [cyan]git clone https://github.com/ggml-org/llama.cpp[/]")
    cuda_flag = " -DGGML_CUDA=ON" if cuda["found"] else ""
    console.print(f"  [cyan]cmake llama.cpp -B build{cuda_flag}[/]")
    console.print(f"  [cyan]cmake --build build -j --target llama-server[/]")
    console.print(f"  [cyan]cp build/bin/llama-server ~/.local/bin/[/]")
    return None


def ensure_claude_code():
    """Ensure Claude Code is installed. Offer to install if missing. Returns path or None."""
    cc = check_claude_code()
    if cc["found"]:
        return cc["path"]

    node = check_node()
    if not node["found"]:
        console.print(f"  [yellow]Claude Code needs Node.js.[/]")
        if IS_MAC:
            console.print(f"  [cyan]brew install node[/]")
        else:
            console.print(f"  [cyan]curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash[/]")
            console.print(f"  [cyan]sudo apt install -y nodejs[/]")
        return None

    console.print(f"  [yellow]Claude Code not installed.[/]")
    if IS_MAC:
        console.print(f"  [cyan]brew install --cask claude-code[/]")
    else:
        console.print(f"  [cyan]npm install -g @anthropic-ai/claude-code[/]")

    try:
        ans = input("  Install now? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if ans in ("y", "yes"):
        if IS_MAC:
            r = subprocess.run(["brew", "install", "--cask", "claude-code"], timeout=300)
        else:
            r = subprocess.run(["npm", "install", "-g", "@anthropic-ai/claude-code"], timeout=300)
        if r.returncode == 0:
            cc = check_claude_code()
            if cc["found"]:
                console.print(f"  [green]✓ Installed: {cc['path']}[/]")
                return cc["path"]
        console.print(f"  [red]Install failed.[/]")

    return None
