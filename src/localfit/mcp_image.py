"""localfit MCP server — image generation + editing tools for Claude Code.

Exposes generate_image and edit_image as MCP tools. Talks to either:
  - Local image_server (http://127.0.0.1:8189)
  - Remote endpoint (Kaggle/RunPod tunnel URL)

Usage:
  python -m localfit.mcp_image                           # local default
  python -m localfit.mcp_image --endpoint http://...     # remote tunnel
  python -m localfit.mcp_image --model schnell           # pick model

Claude Code config (~/.claude/settings.json):
  "mcpServers": {
    "localfit-image": {
      "command": "python3",
      "args": ["-m", "localfit.mcp_image"]
    }
  }
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("localfit-image")

# Resolved at startup
_endpoint = "http://127.0.0.1:8189"
_output_dir = str(Path.home() / "Pictures" / "localfit")


def _api_call(path: str, payload: dict | None = None, timeout: int = 300) -> dict:
    """Call the image server API."""
    url = f"{_endpoint}{path}"
    if payload is not None:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _detect_image_viewer():
    """Detect best way to display images in current terminal."""
    import shutil
    import subprocess

    # Check for timg (best terminal image viewer — supports kitty, iTerm2, sixel)
    if shutil.which("timg"):
        return "timg"
    # Check for kitty terminal (native image protocol)
    if os.environ.get("TERM_PROGRAM") == "kitty" or "kitty" in os.environ.get("TERM", ""):
        if shutil.which("kitten"):
            return "kitty"
    # Check for iTerm2 (imgcat)
    if os.environ.get("TERM_PROGRAM") == "iTerm.app":
        if shutil.which("imgcat"):
            return "imgcat"
    # Check for wezterm
    if os.environ.get("TERM_PROGRAM") == "WezTerm":
        if shutil.which("wezterm"):
            return "wezterm"
    # Check for chafa (wide terminal support)
    if shutil.which("chafa"):
        return "chafa"
    # Fallback: open with system viewer
    return "open"


def _display_image(path: str) -> str:
    """Display image — open in system viewer (can't write to terminal from MCP stdio)."""
    import subprocess

    try:
        # MCP uses stdio for transport — can't write timg escape codes to stdout
        # Instead, open in system viewer or save path for user
        import platform
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "(opened in Preview)"
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "(opened in viewer)"
    except Exception:
        pass
    return ""


def _save_image(b64_data: str, name: str) -> str:
    """Save base64 PNG to output dir, return path."""
    os.makedirs(_output_dir, exist_ok=True)
    ts = int(time.time())
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in name)[:50]
    filename = f"{safe_name}_{ts}.png"
    path = os.path.join(_output_dir, filename)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return path


def _estimate_time(model: str, size: str, steps: int) -> str:
    """Estimate generation time based on model, size, steps."""
    # Based on real benchmarks
    estimates = {
        "klein-4b": {"base": 6, "per_step": 1.5},      # Mac mflux
        "flux2-klein-4b": {"base": 6, "per_step": 1.5},
        "schnell": {"base": 15, "per_step": 5},
        "z-image-turbo": {"base": 20, "per_step": 15},
        "z-image": {"base": 30, "per_step": 20},
        "dev": {"base": 30, "per_step": 10},
    }
    w, h = (int(x) for x in size.split("x")) if "x" in size else (512, 512)
    scale = (w * h) / (512 * 512)

    est = estimates.get(model, {"base": 20, "per_step": 10})
    secs = (est["base"] + est["per_step"] * steps) * max(1.0, scale * 0.7)
    if secs < 60:
        return f"~{int(secs)}s"
    return f"~{int(secs/60)}m{int(secs%60)}s"


@mcp.tool()
def generate_image(
    prompt: str,
    size: str = "512x512",
    steps: int = 4,
    seed: int | None = None,
) -> list:
    """Generate an image from a text prompt using Flux models locally or on remote GPU.

    Speed guide (Mac M4 Pro):
      klein-4b: ~24s for 1024x1024 (fastest)
      schnell: ~79s for 512x512
      z-image-turbo: ~90s+ (slow, high quality)

    Args:
        prompt: Text description of the image to generate
        size: Image dimensions as WxH (e.g. "256x256", "512x512", "1024x1024")
        steps: Number of inference steps (2-4 fast, 8+ quality)
        seed: Random seed for reproducibility (optional)

    Returns:
        Path to the saved PNG image file with generation time
    """
    # Validate size (Flux needs minimum 256 and multiples of 64)
    w, h = (int(x) for x in size.split("x")) if "x" in size else (512, 512)
    if w < 256 or h < 256:
        size = "256x256"
        w, h = 256, 256
    w = (w // 64) * 64
    h = (h // 64) * 64
    size = f"{w}x{h}"

    # Get current model for ETA
    try:
        health = _api_call("/health")
        model = health.get("model", "unknown")
    except Exception:
        model = "unknown"

    eta = _estimate_time(model, size, steps)

    payload = {"prompt": prompt, "size": size, "steps": steps}
    if seed is not None:
        payload["seed"] = seed

    t0 = time.time()
    result = _api_call("/v1/images/generations", payload)
    elapsed = time.time() - t0

    if "error" in result:
        return f"Error: {result['error'].get('message', 'unknown')}"

    b64 = result["data"][0]["b64_json"]
    kb = len(b64) // 1024
    img_bytes = base64.b64decode(b64)
    path = _save_image(b64, prompt[:30])

    # Return Image object (MCP SDK handles base64 encoding for Claude Code display)
    # Plus text metadata
    return [
        Image(data=img_bytes, format="png"),
        f"Model: {model} | Size: {size} | Steps: {steps} | Time: {elapsed:.1f}s (est {eta}) | {kb}KB\nSaved: {os.path.abspath(path)}",
    ]


@mcp.tool()
def edit_image(
    image_path: str,
    prompt: str,
    strength: float = 0.7,
    steps: int = 8,
) -> list:
    """Edit an existing image using a text prompt (image-to-image).

    Args:
        image_path: Path to the source image to edit
        prompt: Text description of the desired edit
        strength: How much to change (0.0 = no change, 1.0 = complete redraw). Default 0.7
        steps: Number of inference steps

    Returns:
        Path to the saved edited PNG image file
    """
    # Read source image and convert to base64
    abs_path = os.path.expanduser(image_path)
    if not os.path.exists(abs_path):
        return f"Error: Image not found at {abs_path}"

    with open(abs_path, "rb") as f:
        src_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "prompt": prompt,
        "image": src_b64,
        "strength": strength,
        "steps": steps,
    }

    result = _api_call("/v1/images/edits", payload, timeout=600)

    if "error" in result:
        return f"Error: {result['error'].get('message', 'unknown')}"

    b64 = result["data"][0]["b64_json"]
    img_bytes = base64.b64decode(b64)
    path = _save_image(b64, f"edit_{prompt[:20]}")

    return [
        Image(data=img_bytes, format="png"),
        f"Edited image saved to {os.path.abspath(path)}",
    ]


@mcp.tool()
def list_image_models() -> str:
    """List available image generation models on the connected server.

    Returns:
        JSON string of available models
    """
    try:
        result = _api_call("/v1/models")
        models = result.get("data", [])
        lines = ["Available image models:"]
        for m in models:
            lines.append(f"  - {m.get('id', 'unknown')}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error connecting to image server at {_endpoint}: {e}"


@mcp.tool()
def check_resources() -> str:
    """Check GPU/memory resources, loaded model, and estimate generation times.
    ALWAYS call this before generate_image to pick optimal settings.

    Returns:
        System resources, loaded model, and ETA estimates for different configs
    """
    import platform
    import subprocess

    lines = []

    # Platform
    lines.append(f"Platform: {platform.system()} {platform.machine()}")

    # Mac GPU info
    if platform.system() == "Darwin":
        try:
            import subprocess as sp
            mem = sp.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            ram_gb = int(mem.stdout.strip()) // (1024**3)
            chip = sp.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            lines.append(f"Chip: {chip.stdout.strip()}")
            lines.append(f"RAM: {ram_gb}GB (unified = GPU VRAM)")
        except Exception:
            pass

    # Check image server
    try:
        health = _api_call("/health")
        model = health.get("model", "none")
        lines.append(f"Image server: RUNNING")
        lines.append(f"Loaded model: {model}")
    except Exception:
        model = "none"
        lines.append(f"Image server: NOT RUNNING")
        lines.append(f"Start with: python -m localfit.image_server 8189 klein-4b 4")
        return "\n".join(lines)

    # ETA estimates for loaded model
    lines.append(f"")
    lines.append(f"Estimated generation times ({model}):")
    for size, label in [("256x256", "small"), ("512x512", "medium"), ("1024x1024", "large")]:
        for steps in [2, 4]:
            eta = _estimate_time(model, size, steps)
            lines.append(f"  {size} {steps} steps: {eta} ({label})")

    # Recommendations
    lines.append(f"")
    if model in ("z-image-turbo", "z-image", "dev"):
        lines.append(f"TIP: {model} is slow. Switch to klein-4b for faster generation:")
        lines.append(f"  pkill -f image_server && python -m localfit.image_server 8189 klein-4b 4")
    elif model in ("klein-4b", "flux2-klein-4b"):
        lines.append(f"TIP: {model} is the fastest model. Use 256x256 + 2 steps for instant results.")
    elif model == "schnell":
        lines.append(f"TIP: schnell is moderate speed. Use 256x256 + 2 steps, or switch to klein-4b for 3x faster.")

    return "\n".join(lines)


@mcp.tool()
def show_image(image_path: str, width: int = 40, height: int = 20) -> str:
    """Display an image in the terminal using timg. Use this after generate_image to show the result.

    Args:
        image_path: Path to the image file to display
        width: Display width in columns (default 40)
        height: Display height in rows (default 20)

    Returns:
        Status message confirming display
    """
    import subprocess
    import shutil

    abs_path = os.path.expanduser(image_path)
    if not os.path.exists(abs_path):
        return f"Error: File not found: {abs_path}"

    # timg must write to a real terminal, not MCP stdio
    # So we launch it as a detached process writing to /dev/tty
    if shutil.which("timg"):
        try:
            subprocess.run(
                f'timg -g {width}x{height} -p q "{abs_path}" > /dev/tty 2>&1',
                shell=True, timeout=10,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return f"Displayed {abs_path} via timg ({width}x{height})"
        except Exception as e:
            return f"timg failed: {e}. Try: timg -g {width}x{height} '{abs_path}'"
    else:
        # Fallback: open in system viewer
        try:
            subprocess.Popen(["open", abs_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opened {abs_path} in system viewer (install timg for terminal display: brew install timg)"
        except Exception:
            return f"No image viewer found. File at: {abs_path}"


@mcp.tool()
def image_server_status() -> str:
    """Check if the image generation server is running and what model is loaded.

    Returns:
        Server status including model name and endpoint URL
    """
    try:
        result = _api_call("/health")
        model = result.get("model", "unknown")
        return (
            f"Status: Running\n"
            f"Model: {model}\n"
            f"Endpoint: {_endpoint}\n"
            f"API: /v1/images/generations (OpenAI)\n"
            f"API: /sdapi/v1/txt2img (AUTOMATIC1111)\n"
            f"Open WebUI: Settings > Images > URL: {_endpoint}"
        )
    except Exception as e:
        return f"Status: NOT RUNNING\nEndpoint: {_endpoint}\nError: {e}"


def main():
    global _endpoint, _output_dir

    parser = argparse.ArgumentParser(description="localfit image MCP server")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("LOCALFIT_IMAGE_ENDPOINT", "http://127.0.0.1:8189"),
        help="Image server URL (local or remote tunnel)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "LOCALFIT_IMAGE_OUTPUT", str(Path.home() / "Pictures" / "localfit")
        ),
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model hint (for display only — model is loaded by the server)",
    )
    args = parser.parse_args()

    _endpoint = args.endpoint.rstrip("/")
    _output_dir = args.output_dir

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
