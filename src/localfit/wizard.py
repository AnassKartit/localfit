"""localfit wizard — guided flow from model to tool. Params skip steps."""

import os
import sys
import json
import time

from rich.console import Console

console = Console(highlight=False)


def run_wizard(model=None, backend=None, budget=None, tool=None, tunnel=False):
    """
    The single entry point for all localfit flows.
    Each param pre-fills a step. Missing params show a menu.

    localfit                          → model=None, tool=None (full wizard)
    localfit run gemma4:e4b           → model="gemma4:e4b" (skip model pick)
    localfit launch openwebui         → tool="openwebui" (skip tool pick)
    localfit launch openwebui --model gemma4:e4b --remote kaggle --budget 1h
                                      → everything filled, zero menus
    """
    from localfit.home_menu import show_home_menu
    from localfit.backends import get_machine_specs, IS_MAC, check_mlx_available

    specs = get_machine_specs()
    gpu_total_mb = specs.get("gpu_total_mb", 0)
    gpu_gb = round(gpu_total_mb / 1024, 1)
    chip = specs.get("chip", "GPU")

    # ── Step 1: Model ──
    # If model provided, skip. Otherwise show model picker.
    if not model:
        # Return to normal home screen — it handles model picking
        return None  # signal: use the existing _boot_screen flow

    # ── Step 2: Backend (local vs remote) ──
    # Check what options exist for this model
    from localfit.run_menu import collect_options
    local_opts, remote_opts, recommended, metadata = collect_options(model, specs)

    all_opts = local_opts + remote_opts
    if not all_opts:
        console.print(f"\n  [red]No options found for {model}[/]")
        return None

    # If backend specified (--remote kaggle/runpod), filter to that
    if backend:
        if backend in ("kaggle", "runpod", "cloud"):
            # Skip local options, go straight to remote
            pass  # handled later
        else:
            backend = None  # invalid, show menu

    # If only one local option fits and no --remote, auto-pick it
    fits = [o for o in local_opts if o.get("fits")]
    if fits and not backend and not tool:
        # Show the run menu to let user pick quant
        best = fits[-1]  # highest quality that fits
        console.print(f"\n  [green]✓ Best for your {chip}:[/] [bold]{best['backend']} {best['name'].split('/')[-1]}[/] ({best['size']})")
        if len(fits) > 1:
            console.print(f"  [dim]{len(fits)} options fit your {gpu_gb}GB GPU[/]")

    # If model + remote + budget all provided, skip to serve
    if model and backend and budget:
        return _auto_serve(model, backend, budget, tool, specs)

    # ── Step 3: Show menu with options ──
    if not backend:
        # Show run menu (same as localfit run MODEL)
        from localfit.run_menu import show_run_menu
        hw_label = f"{chip}  {gpu_gb}GB"

        choice = show_run_menu(model, hw_label, local_opts, remote_opts, recommended)
        if not choice or choice == "quit" or choice == "back":
            return None

        opt = all_opts[int(choice) - 1] if isinstance(choice, int) and 1 <= choice <= len(all_opts) else None
        if not opt:
            return None

        action = opt.get("action")
        if action == "kaggle":
            backend = "kaggle"
        elif action == "runpod":
            backend = "runpod"
        else:
            # Local option selected — serve it
            return _serve_and_pick_tool(model, opt, tool, specs)

    # ── Remote flow ──
    if backend in ("kaggle", "runpod", "cloud"):
        return _remote_serve_and_pick_tool(model, backend, budget, tool, specs)

    return None


def _auto_serve(model, backend, budget, tool, specs):
    """All params provided — zero menus, just serve + launch."""
    console.print(f"\n  [bold]{model}[/] → {backend} ({budget})")

    if backend == "kaggle":
        # Parse budget
        b = budget.lower().strip()
        if b.endswith("m"):
            duration = int(b[:-1])
        elif b.endswith("h"):
            duration = int(float(b[:-1]) * 60)
        else:
            duration = 60

        from localfit.remote import remote_serve_kaggle
        remote_serve_kaggle(model, max_runtime_minutes=duration)

    elif backend in ("runpod", "cloud"):
        from localfit.cloud import cloud_serve
        cloud_serve(model)

    # After serve, launch tool if specified
    if tool:
        _launch_tool_direct(tool, model)


def _serve_and_pick_tool(model, opt, tool, specs):
    """Serve a local model, then show tool picker (or auto-launch if tool specified)."""
    action = opt.get("action")

    if action == "mlx":
        from localfit.backends import start_mlx_server, stop_conflicting_backends
        stop_conflicting_backends("mlx")
        proc = start_mlx_server(opt["repo"], port=8080)
        if not proc:
            console.print(f"  [red]MLX server failed[/]")
            return None
        port = 8080
        api_model = proc._mlx_model_id

    elif action == "gguf":
        from localfit.backends import _download_gguf
        from localfit.prerequisites import ensure_llama_server
        import subprocess

        path = _download_gguf(opt["repo"], opt["filename"])
        if not path:
            return None
        binary = ensure_llama_server()
        if not binary:
            return None

        # Friendly model alias (e.g. "gemma4:e4b" instead of "gemma-4-E4B-it-UD-Q8_K_XL.gguf")
        friendly_name = model.replace("-", " ").replace("_", " ") if model else opt.get("quant", "local")
        ngl = "99"
        ctx = "32768"
        cmd = [binary, "-m", path, "--port", "8089", "-ngl", ngl, "-c", ctx, "--jinja", "--alias", friendly_name]
        if opt.get("hf_data", {}).get("is_vlm") and opt.get("hf_data", {}).get("mmproj_files"):
            mmproj = opt["hf_data"]["mmproj_files"][0]["filename"]
            mmproj_path = _download_gguf(opt["repo"], mmproj)
            if mmproj_path:
                cmd += ["--mmproj", mmproj_path]

        import tempfile
        stderr_log = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, prefix="llama-")
        env = os.environ.copy()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_log, env=env)

        # Wait for server
        import urllib.request
        for i in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=1)
                break
            except Exception:
                time.sleep(1)

        port = 8089
        api_model = opt.get("quant", model)
    else:
        return None

    console.print(f"\n  [green]✓ Serving on :{port}[/]")

    # If tool specified, launch directly
    if tool:
        _launch_tool_direct(tool, api_model, port)
        return {"port": port, "model": api_model}

    # Otherwise show tool picker
    _show_tool_menu(api_model, port)
    return {"port": port, "model": api_model}


def _remote_serve_and_pick_tool(model, backend, budget, tool, specs):
    """Serve on remote, then tool picker."""
    if backend == "kaggle":
        if not budget:
            console.print(f"\n  [bold]Budget?[/] [dim](default: 1h)[/]")
            try:
                budget = input("  > ").strip() or "1h"
            except (EOFError, KeyboardInterrupt):
                return None

        b = budget.lower().strip()
        if b.endswith("m"):
            duration = int(b[:-1])
        elif b.endswith("h"):
            duration = int(float(b[:-1]) * 60)
        else:
            duration = 60

        from localfit.remote import remote_serve_kaggle
        remote_serve_kaggle(model, max_runtime_minutes=duration)

    elif backend in ("runpod", "cloud"):
        from localfit.cloud import cloud_serve
        cloud_serve(model)

    if tool:
        _launch_tool_direct(tool, model)


def _show_tool_menu(api_model, port=8089):
    """Show TOOLS menu using show_home_menu — same arrow keys as everywhere."""
    from localfit.home_menu import show_home_menu
    from localfit.backends import get_machine_specs

    specs = get_machine_specs()
    api_url = f"http://127.0.0.1:{port}/v1"

    system = {
        "subtitle": f"{specs.get('chip', 'GPU')}  {specs.get('gpu_total_mb', 0) // 1024}GB",
        "verdict": "SERVING",
        "color": "green",
        "gpu": f"{api_model} loaded on :{port}",
        "swap": "",
        "disk": "",
        "model": f"[green]{api_model}[/] on :{port}",
        "machine": f"{specs.get('chip', '?')} · {specs.get('ram_gb', '?')}GB",
    }

    tools = [
        ("Open WebUI", "webui", "ChatGPT-style browser UI"),
        ("Claude Code", "claude", "AI coding assistant"),
        ("OpenCode", "opencode", "Terminal coding tool"),
        ("Codex", "codex", "OpenAI Codex CLI"),
        ("aider", "aider", "AI pair programming"),
    ]

    items = [
        {
            "index": 1, "section": "ACTIVE", "label": api_model,
            "meta": "local", "detail": f"Serving on :{port} · {api_url}",
            "repo": f"local:{port}", "source": "local", "accent": "green",
            "badge": "●", "action": "noop", "selectable": False,
        },
    ]

    for i, (name, tool_id, desc) in enumerate(tools, 2):
        items.append({
            "index": i, "section": "TOOLS", "label": name,
            "meta": "", "detail": desc,
            "repo": tool_id, "source": "", "accent": "cyan",
            "badge": "→", "action": "launch_tool", "selectable": True,
        })

    os.system("clear")
    while True:
        result = show_home_menu(system, items)
        if not result or result.get("action") in ("quit", "back", None):
            return
        if result.get("action") == "launch_tool":
            tool_id = result.get("repo")
            if tool_id:
                _launch_tool_direct(tool_id, api_model, port)
                # Stay on menu — user can launch more tools or quit


def _launch_tool_direct(tool, model, port=8089):
    """Launch a tool connected to a running model. Env vars only, no config writes."""
    import subprocess

    api_base = f"http://127.0.0.1:{port}/v1"
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = api_base
    env["OPENAI_API_BASE_URL"] = api_base
    env["OPENAI_API_KEY"] = "no-key-required"

    console.print(f"  [green]✓[/] Launching {tool} → {api_base}\n")

    if tool in ("webui", "openwebui", "open-webui", "chat"):
        import webbrowser
        webui_dir = os.path.expanduser("~/.localfit/open-webui")
        os.makedirs(webui_dir, exist_ok=True)
        env["DATA_DIR"] = webui_dir
        env["ENABLE_OPENAI_API"] = "True"
        env["ENABLE_OLLAMA_API"] = "False"
        env["DEFAULT_MODELS"] = model or "local"
        db_path = os.path.join(webui_dir, "webui.db")
        if not os.path.exists(db_path):
            env["WEBUI_AUTH"] = "False"

        try:
            subprocess.Popen(
                ["uv", "run", "--python", "3.11", "--with", "open-webui", "--", "open-webui", "serve"],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            try:
                subprocess.Popen(["open-webui", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                console.print(f"  [red]Open WebUI not installed: pip install open-webui[/]")
                return
        console.print(f"  [green]✓[/] Open WebUI → http://localhost:8080")
        time.sleep(3)
        webbrowser.open("http://localhost:8080")

    elif tool in ("claude",):
        try:
            from localfit.proxy import PROXY_PORT, ensure_proxy_process
            from localfit.safe_config import get_claude_launch_env
            ensure_proxy_process(llama_url=f"{api_base}/chat/completions", port=PROXY_PORT)
            subprocess.Popen(["claude", "--bare", "--model", model or "local"],
                env={**os.environ, **get_claude_launch_env(api_base=f"http://127.0.0.1:{PROXY_PORT}")})
        except Exception as e:
            console.print(f"  [red]{e}[/]")

    elif tool in ("opencode",):
        subprocess.Popen(["opencode"], env=env)

    elif tool in ("codex",):
        subprocess.Popen(["codex", "--model", model or "local", "-c", "model_provider=openai",
            "-c", "features.use_responses_api=false"], env=env)

    elif tool in ("aider",):
        env["OPENAI_API_BASE"] = api_base
        subprocess.Popen(["aider", "--model", f"openai/{model or 'local'}"], env=env)

    else:
        console.print(f"  [yellow]Unknown tool: {tool}[/]")
