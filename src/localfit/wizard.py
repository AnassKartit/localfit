"""localfit wizard — guided flow from model to tool. Params skip steps."""

import os
import sys
import json
import time

from rich.console import Console

console = Console(highlight=False)


def _pick(title, options, header="", subtitle="↑↓ move · Enter select · q cancel"):
    """Arrow-key picker with optional header. options = list of (label, description, value).
    Returns value or None if cancelled."""
    import tty, termios

    sel = 0
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            sys.stdout.write("\033[H\033[J")  # clear
            if header:
                sys.stdout.write(f"\r\n{header}\r\n")
            sys.stdout.write(f"\r\n  \033[1m{title}\033[0m\r\n\r\n")
            for i, (label, desc, _) in enumerate(options):
                if i == sel:
                    sys.stdout.write(f"  \033[32m> {label}\033[0m\r\n")
                    if desc:
                        sys.stdout.write(f"    \033[2m{desc}\033[0m\r\n")
                else:
                    sys.stdout.write(f"    \033[2m{label}\033[0m\r\n")
            sys.stdout.write(f"\r\n  \033[2m{subtitle}\033[0m")
            sys.stdout.flush()

            ch = sys.stdin.read(1)
            if ch == "\r" or ch == "\n":
                return options[sel][2]
            elif ch == "q":
                return None
            elif ch == "\x1b":
                sys.stdin.read(1)
                arrow = sys.stdin.read(1)
                if arrow == "A":
                    sel = (sel - 1) % len(options)
                elif arrow == "B":
                    sel = (sel + 1) % len(options)
            elif ch.isdigit():
                idx = int(ch) - 1
                if 0 <= idx < len(options):
                    return options[idx][2]
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def _build_header(specs, swap_mb):
    """Build a compact dashboard header for all wizard screens."""
    chip = specs.get("chip", "GPU")
    gpu_gb = round(specs.get("gpu_total_mb", 0) / 1024, 1)
    ram_gb = specs.get("ram_gb", 0)
    usable_gb = round(gpu_gb - 2, 1)

    # Swap color
    if swap_mb > 8000:
        swap = f"\033[31m{swap_mb // 1024}GB swap CRITICAL\033[0m"
    elif swap_mb > 2000:
        swap = f"\033[33m{swap_mb // 1024}GB swap\033[0m"
    else:
        swap = f"\033[32m{swap_mb // 1024}GB swap\033[0m"

    # Disk
    disk_free = "?"
    try:
        st = os.statvfs(os.path.expanduser("~"))
        disk_free = f"{round((st.f_bavail * st.f_frsize) / (1024**3))}GB free"
    except Exception:
        pass

    # Running models
    import urllib.request as _ur

    running_lines = []
    for _port, _be in [(8089, "llama.cpp"), (11434, "Ollama")]:
        try:
            _r = _ur.urlopen(f"http://127.0.0.1:{_port}/v1/models", timeout=1)
            _d = json.loads(_r.read())
            for _m in _d.get("data", []):
                running_lines.append(f"  \033[32m●\033[0m {_m['id']} ({_be} :{_port})")
        except Exception:
            pass
    try:
        _r = _ur.urlopen("http://127.0.0.1:8189/health", timeout=1)
        _d = json.loads(_r.read())
        if _d.get("status") == "ok" and _d.get("model", "not loaded") != "not loaded":
            running_lines.append(f"  \033[36m●\033[0m {_d['model']} (image :8189)")
    except Exception:
        pass

    lines = []
    lines.append(
        f"  \033[1mlocalfit\033[0m  {chip} · \033[1m{gpu_gb}GB\033[0m GPU ({usable_gb}GB usable) · {ram_gb}GB RAM · {swap} · {disk_free}"
    )
    if running_lines:
        lines.append("")
        lines.append("  \033[1mRunning:\033[0m")
        lines.extend(running_lines)

    return "\r\n".join(lines)


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
    from localfit.backends import (
        get_machine_specs,
        IS_MAC,
        get_metal_gpu_stats,
        get_swap_usage_mb,
    )

    specs = get_machine_specs()
    gpu_total_mb = specs.get("gpu_total_mb", 0)
    gpu_gb = round(gpu_total_mb / 1024, 1)
    chip = specs.get("chip", "GPU")
    swap_mb = get_swap_usage_mb()

    # ── Build dashboard header for all steps ──
    header = _build_header(specs, swap_mb)

    # ── Step 1: What do you want? ──
    if not model:
        # Check what's already running
        _running = []
        import urllib.request as _ur

        for _port, _backend in [(8089, "llama.cpp"), (11434, "Ollama")]:
            try:
                _r = _ur.urlopen(f"http://127.0.0.1:{_port}/v1/models", timeout=1)
                _d = json.loads(_r.read())
                for _m in _d.get("data", []):
                    _running.append((_m["id"], _backend, _port))
            except Exception:
                pass
        _img_running = None
        try:
            _r = _ur.urlopen("http://127.0.0.1:8189/health", timeout=1)
            _d = json.loads(_r.read())
            if (
                _d.get("status") == "ok"
                and _d.get("model", "not loaded") != "not loaded"
            ):
                _img_running = _d["model"]
        except Exception:
            pass

        _step1 = []
        if _running:
            for _mid, _be, _port in _running:
                _step1.append(
                    (
                        f"Use {_mid} ({_be} :{_port})",
                        "Already running — connect a tool",
                        ("use_running", _mid, _be, _port),
                    )
                )
        _step1.append(
            (
                "Run an LLM",
                "Chat, code, agents — pick a model for your GPU",
                ("pick_llm",),
            )
        )
        _step1.append(
            (
                "Run an Image model",
                f"Generate images locally{' — ' + _img_running + ' running' if _img_running else ''}",
                ("pick_image",),
            )
        )
        _step1.append(
            (
                "Run Both (LLM + Image)",
                "Full local AI stack",
                ("pick_both",),
            )
        )
        if swap_mb > 2000:
            _step1.append(
                (
                    f"Cleanup ({swap_mb // 1024}GB swap, free memory)",
                    "Unload models, clear cache, fix swap",
                    ("cleanup",),
                )
            )
        _step1.append(
            (
                "Advanced menu...",
                "Full dashboard with all options",
                ("advanced",),
            )
        )

        choice = _pick("What do you want to do?", _step1, header=header)
        if not choice:
            return None

        if choice[0] == "use_running":
            _, _mid, _be, _port = choice
            return _pick_tool_and_launch(
                _mid, f"http://127.0.0.1:{_port}/v1", specs, header
            )

        if choice[0] == "cleanup":
            return "cleanup"

        if choice[0] == "advanced":
            return "advanced"

        if choice[0] in ("pick_llm", "pick_both"):
            model = _pick_llm_model(specs, gpu_gb, chip, header)
            if not model:
                return None

        if choice[0] == "pick_image":
            return _pick_and_start_image(specs, gpu_gb, header)

        if choice[0] == "pick_both":
            _pick_and_start_image(specs, gpu_gb, header)

    # ── Step 2: Where to run? ──
    from localfit.run_menu import collect_options

    local_opts, remote_opts, recommended, metadata = collect_options(model, specs)

    fits = [o for o in local_opts if o.get("fits")]
    _where_opts = []
    if fits:
        best = fits[-1]
        _where_opts.append(
            (
                f"Local ({best['size']}, {best['backend']})",
                f"Runs on your {chip} — fastest",
                ("local", best),
            )
        )
    _where_opts.append(
        (
            "Kaggle (free, T4 GPU)",
            "Free 30h/week, no credit card",
            ("kaggle", None),
        )
    )
    try:
        from localfit.cloud import get_runpod_key

        if get_runpod_key():
            _where_opts.append(
                (
                    "RunPod (paid, any GPU)",
                    "Auto-stop, pay per minute",
                    ("runpod", None),
                )
            )
    except Exception:
        pass

    if len(_where_opts) == 1:
        where = _where_opts[0][2]
    else:
        where = _pick(f"Where to run {model}?", _where_opts)
    if not where:
        return None

    if where[0] == "local":
        return _serve_and_pick_tool(model, where[1], tool, specs)
    elif where[0] in ("kaggle", "runpod"):
        return _remote_serve_and_pick_tool(model, where[0], budget, tool, specs)

    return None


def _pick_llm_model(specs, gpu_gb, chip):
    """Step 2a: Pick an LLM model based on hardware."""
    from localfit.backends import MODELS, recommend_model

    best_id, best_reason = recommend_model(int(gpu_gb))
    usable = gpu_gb - 2  # reserve 2GB for system

    options = []
    # Recommended first
    if best_id in MODELS:
        m = MODELS[best_id]
        options.append(
            (
                f"★ {m['name']} (recommended)",
                f"{best_reason}",
                best_id,
            )
        )

    # Other models that fit
    for mid, m in MODELS.items():
        if mid == best_id:
            continue
        size = m.get("size_gb", 99)
        if size <= usable:
            options.append(
                (
                    f"{m['name']} (~{size}GB)",
                    m.get("description", ""),
                    mid,
                )
            )

    # Search option
    options.append(
        (
            "Search HuggingFace...",
            "Find any model by name",
            "__search__",
        )
    )

    choice = _pick(f"Pick an LLM ({chip}, {gpu_gb}GB GPU)", options)
    if choice == "__search__":
        os.system("clear")
        try:
            term = input("\n  Search model: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return term if term else None
    return choice


def _pick_and_start_image(specs, gpu_gb):
    """Pick and start an image model."""
    from localfit.image_models import IMAGE_MODELS

    options = []
    for k, v in IMAGE_MODELS.items():
        vram = v.get("vram_gb", 0)
        if vram <= gpu_gb:
            options.append(
                (
                    f"{k} ({vram}GB)",
                    f"{v['pipeline']} · {v['task']}",
                    k,
                )
            )

    choice = _pick("Pick an image model", options)
    if not choice:
        return None

    os.system("clear")
    console.print(f"\n  [cyan]Starting {choice}...[/]")
    from localfit.cli import _start_image_server

    _start_image_server(choice)
    return "image_started"


def _pick_tool_and_launch(model_name, api_base, specs):
    """Pick a tool to connect to a running model."""
    options = [
        ("Open WebUI (chat in browser)", "Best for chatting", "webui"),
        ("localcoder (AI coding agent)", "Write code from terminal", "localcoder"),
        ("Claude Code (via proxy)", "Use Claude Code with local model", "claude"),
        ("OpenCode", "CLI coding assistant", "opencode"),
        ("Aider", "AI pair programming", "aider"),
        ("Just serve (no tool)", "API on " + api_base, "none"),
    ]

    choice = _pick(f"Connect {model_name} to:", options)
    if not choice or choice == "none":
        console.print(f"\n  [green]✓ API ready:[/] {api_base}")
        return "served"

    from localfit.cli import _launch_tool_with_endpoint

    _launch_tool_with_endpoint(choice, api_base, model_name)
    return "launched"

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
        console.print(
            f"\n  [green]✓ Best for your {chip}:[/] [bold]{best['backend']} {best['name'].split('/')[-1]}[/] ({best['size']})"
        )
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

        opt = (
            all_opts[int(choice) - 1]
            if isinstance(choice, int) and 1 <= choice <= len(all_opts)
            else None
        )
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
        friendly_name = (
            model.replace("-", " ").replace("_", " ")
            if model
            else opt.get("quant", "local")
        )
        ngl = "99"
        ctx = "32768"
        cmd = [
            binary,
            "-m",
            path,
            "--port",
            "8089",
            "-ngl",
            ngl,
            "-c",
            ctx,
            "--jinja",
            "--alias",
            friendly_name,
        ]
        if opt.get("hf_data", {}).get("is_vlm") and opt.get("hf_data", {}).get(
            "mmproj_files"
        ):
            mmproj = opt["hf_data"]["mmproj_files"][0]["filename"]
            mmproj_path = _download_gguf(opt["repo"], mmproj)
            if mmproj_path:
                cmd += ["--mmproj", mmproj_path]

        import tempfile

        stderr_log = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, prefix="llama-"
        )
        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=stderr_log, env=env
        )

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

    from localfit.backends import get_metal_gpu_stats, get_swap_usage_mb, get_disk_info

    metal = get_metal_gpu_stats()
    swap_mb = get_swap_usage_mb()
    di = get_disk_info()

    gpu_total = specs.get("gpu_total_mb", 0)
    if metal.get("total_mb"):
        gpu_total = metal["total_mb"]
    gpu_used = metal.get("used_mb", 0)
    gpu_free = max(0, gpu_total - gpu_used)

    swap_gb = round(swap_mb / 1024, 1)
    swap_text = f"{swap_gb}GB" if swap_mb > 100 else "minimal"
    if swap_mb > 2000:
        swap_text += " · [yellow]close apps to free[/]"

    disk_free = di.get("disk_free_gb", 0)
    disk_cache = di.get("hf_cache_gb", 0)

    system = {
        "subtitle": f"{specs.get('chip', 'GPU')}  {gpu_total // 1024}GB",
        "verdict": "SERVING",
        "color": "green",
        "gpu": f"{gpu_used // 1024}GB/{gpu_total // 1024}GB used · {gpu_free // 1024}GB free",
        "swap": swap_text,
        "disk": f"{disk_free}GB free · cache {disk_cache}GB",
        "model": f"[green]{api_model}[/] on :{port}",
        "machine": f"{specs.get('chip', '?')} · {specs.get('ram_gb', '?')}GB · {specs.get('cpu_cores', '?')} cores",
    }

    tools = [
        ("Open WebUI", "webui", "ChatGPT-style browser UI"),
        ("Claude Code", "claude", "AI coding in terminal"),
        ("OpenCode", "opencode", "Terminal coding tool"),
        ("Codex", "codex", "OpenAI Codex CLI"),
        ("aider", "aider", "AI pair programming"),
    ]

    actions = [
        (
            "Image Gen (Flux)",
            "start_image",
            "Start Flux 2 Klein 4B for image generation",
        ),
        ("Quantize → HF", "quantize", "Create GGUF quant + upload to HuggingFace"),
        ("Convert to MLX", "convert_mlx", "Create MLX version for Apple Silicon"),
        ("Make it fit", "makeitfit", "Too big? Quantize remotely on Kaggle/RunPod"),
        ("Stop model", "stop_model", "Kill server + free GPU"),
        ("Switch model", "switch_model", "Pick a different model"),
        ("Browse models", "browse", "Trending + compatible for your GPU"),
    ]

    items = [
        {
            "index": 1,
            "section": "ACTIVE",
            "label": api_model,
            "meta": "local",
            "detail": f"Serving on :{port} · {api_url}",
            "repo": f"local:{port}",
            "source": "local",
            "accent": "green",
            "badge": "●",
            "action": "noop",
            "selectable": False,
        },
    ]

    idx = 2
    for name, tool_id, desc in tools:
        items.append(
            {
                "index": idx,
                "section": "TOOLS",
                "label": name,
                "meta": "",
                "detail": desc,
                "repo": tool_id,
                "source": "",
                "accent": "cyan",
                "badge": "→",
                "action": "launch_tool",
                "selectable": True,
            }
        )
        idx += 1

    for name, action_id, desc in actions:
        items.append(
            {
                "index": idx,
                "section": "ACTIONS",
                "label": name,
                "meta": "",
                "detail": desc,
                "repo": action_id,
                "source": "",
                "accent": "yellow",
                "badge": "⚙",
                "action": action_id,
                "selectable": True,
            }
        )
        idx += 1

    os.system("clear")
    while True:
        result = show_home_menu(system, items)
        if not result or result.get("action") in ("quit", "back", None):
            return "quit"
        if result.get("action") == "launch_tool":
            tool_id = result.get("repo")
            if tool_id:
                _launch_tool_direct(tool_id, api_model, port)
                # Stay on menu — user can launch more tools
        elif result.get("action") == "start_image":
            import subprocess as _sp

            console.print(f"\n  [bold]Starting Flux 2 Klein 4B image server...[/]")
            console.print(f"  [dim]First run downloads ~2GB model[/]")
            _sp.Popen(
                [
                    sys.executable,
                    "-m",
                    "localfit.image_server",
                    "8189",
                    "flux2-klein-4b",
                ],
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
            )
            time.sleep(2)
            console.print(
                f"  [green]✓ Image API: http://127.0.0.1:8189/v1/images/generations[/]"
            )
            console.print(
                f"  [dim]Configure in Open WebUI: Settings → Images → URL: http://127.0.0.1:8189[/]"
            )
        elif result.get("action") == "quantize":
            from localfit.makeitfit import cmd_makeitfit

            os.system("clear")
            cmd_makeitfit(api_model)
        elif result.get("action") == "convert_mlx":
            from localfit.backends import convert_to_mlx

            os.system("clear")
            convert_to_mlx(api_model, q_bits=4)
        elif result.get("action") == "makeitfit":
            from localfit.makeitfit import cmd_makeitfit

            os.system("clear")
            cmd_makeitfit(api_model)
        elif result.get("action") == "stop_model":
            import subprocess

            subprocess.run(["pkill", "-f", "llama-server"], timeout=5)
            subprocess.run(["pkill", "-f", "mlx_lm"], timeout=5)
            console.print(f"  [yellow]Model stopped. GPU freed.[/]")
            return "stopped"
        elif result.get("action") == "switch_model":
            import subprocess

            subprocess.run(["pkill", "-f", "llama-server"], timeout=5)
            subprocess.run(["pkill", "-f", "mlx_lm"], timeout=5)
            return "switch"
        elif result.get("action") == "browse":
            return "browse"


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
                [
                    "uv",
                    "run",
                    "--python",
                    "3.11",
                    "--with",
                    "open-webui",
                    "--",
                    "open-webui",
                    "serve",
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            try:
                subprocess.Popen(
                    ["open-webui", "serve"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                console.print(
                    f"  [red]Open WebUI not installed: pip install open-webui[/]"
                )
                return
        console.print(f"  [green]✓[/] Open WebUI → http://localhost:8080")
        time.sleep(3)
        webbrowser.open("http://localhost:8080")

    elif tool in ("claude",):
        try:
            from localfit.proxy import PROXY_PORT, ensure_proxy_process
            from localfit.safe_config import get_claude_launch_env

            ensure_proxy_process(
                llama_url=f"{api_base}/chat/completions", port=PROXY_PORT
            )
            subprocess.Popen(
                ["claude", "--bare", "--model", model or "local"],
                env={
                    **os.environ,
                    **get_claude_launch_env(api_base=f"http://127.0.0.1:{PROXY_PORT}"),
                },
            )
        except Exception as e:
            console.print(f"  [red]{e}[/]")

    elif tool in ("opencode",):
        subprocess.Popen(["opencode"], env=env)

    elif tool in ("codex",):
        subprocess.Popen(
            [
                "codex",
                "--model",
                model or "local",
                "-c",
                "model_provider=openai",
                "-c",
                "features.use_responses_api=false",
            ],
            env=env,
        )

    elif tool in ("aider",):
        env["OPENAI_API_BASE"] = api_base
        subprocess.Popen(["aider", "--model", f"openai/{model or 'local'}"], env=env)

    else:
        console.print(f"  [yellow]Unknown tool: {tool}[/]")
