"""localfit CLI — GPU toolkit for local LLMs."""

import argparse, json, os, sys, time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape
from rich.text import Text

console = Console()

# ── Ollama-style subcommand mapping ──
# Converts: localfit run MODEL → localfit --serve MODEL
# So users who know Ollama feel right at home.
_SUBCOMMANDS = {
    "run": "--serve",
    "pull": "--pull",
    "list": "--ps",
    "ps": "--ps",
    "stop": "--kill",
    "show": "--fetch",
    "serve": "--serve",
    "bench": "--bench",
    "arena": "--arena",
    "health": "--health",
    "cleanup": "--cleanup",
    "specs": "--specs",
    "simulate": "--simulate",
    "trending": "--trending",
    "check": "--check",
    "debloat": "--debloat",
    "doctor": "--doctor",
    "restore": "--restore",
    "login": "--login",
    "makeitfit": "--makeitfit",
    "launch": "--launch",
}


def _rewrite_subcommands():
    """Rewrite Ollama-style subcommands to flag-style before argparse sees them.

    localfit run gemma4:e4b              → localfit --serve gemma4:e4b
    localfit run gemma4:e4b --remote kaggle → localfit --serve gemma4:e4b --remote kaggle
    localfit pull gemma4:e4b             → localfit --pull gemma4:e4b
    localfit list                        → localfit --ps
    localfit login kaggle                → localfit --login kaggle
    """
    if len(sys.argv) < 2:
        return

    first = sys.argv[1]

    # Skip if it's already a flag (--xxx) or help
    if first.startswith("-"):
        return

    flag = _SUBCOMMANDS.get(first)
    if flag:
        sys.argv[1] = flag


def main():
    _rewrite_subcommands()

    parser = argparse.ArgumentParser(
        description="localfit — GPU toolkit for local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""commands (Ollama-compatible):
  localfit run MODEL                  download + run (auto-detects GPU fit)
  localfit pull MODEL                 download a model
  localfit show MODEL                 show quants + fit analysis
  localfit list                       list installed models
  localfit ps                         show running models
  localfit stop                       stop running model

what Ollama can't do:
  localfit                            GPU dashboard + trending models with fit check
  localfit run MODEL --remote kaggle  doesn't fit? run on free Kaggle T4/T4x2 GPU
  localfit run MODEL --cloud          run on RunPod cloud GPU (paid)
  localfit health                     GPU health: VRAM, swap, temp, processes
  localfit simulate                   will this model fit? (interactive)
  localfit bench                      benchmark all installed models
  localfit trending                   browse trending models with fit/cloud tags
  localfit specs                      full machine specs
  localfit cleanup                    free GPU memory
  localfit login kaggle               save Kaggle key (free cloud GPU)
  localfit login runpod               save RunPod key (paid cloud GPU)

tool integration:
  localfit --launch claude            serve model + launch Claude Code
  localfit --launch opencode          serve model + launch OpenCode
  localfit --launch aider             serve model + launch aider
  localfit --config TOOL              auto-configure any tool for local model
""",
    )
    parser.add_argument(
        "--simulate",
        nargs="?",
        const="__interactive__",
        metavar="MODEL",
        help="Will this model fit? Interactive picker or '70b q4'",
    )
    parser.add_argument(
        "--fetch",
        type=str,
        metavar="URL_OR_NAME",
        help="Fetch model from HuggingFace and check fit",
    )
    parser.add_argument(
        "--bench", action="store_true", help="Benchmark all installed models"
    )
    parser.add_argument("--arena", action="store_true", help="Show model leaderboard")
    parser.add_argument(
        "--force", action="store_true", help="Re-run benchmarks even if cached"
    )
    parser.add_argument("--health", action="store_true", help="GPU health dashboard")
    parser.add_argument("--cleanup", action="store_true", help="Free GPU memory")
    parser.add_argument("--specs", action="store_true", help="Machine specs")
    parser.add_argument(
        "--serve",
        "--run",
        type=str,
        metavar="MODEL",
        help="Download + start model (Ollama-style: gemma4:e4b)",
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="TOOL",
        help="Auto-configure a tool (claude, codex, opencode, aider, cursor)",
    )
    parser.add_argument(
        "--launch",
        type=str,
        metavar="TOOL",
        help="Start model + launch tool in one command (claude, codex, opencode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="MODEL",
        help="Model to serve (used with --launch, e.g. unsloth/Qwen3.5-35B-A3B)",
    )
    parser.add_argument(
        "--debloat", action="store_true", help="Disable macOS services stealing GPU"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check prerequisites (llama-server, CUDA, Claude Code, etc.)",
    )
    parser.add_argument(
        "--trending",
        action="store_true",
        help="Browse trending models from HuggingFace with fit check",
    )
    parser.add_argument(
        "--cloud", action="store_true", help="Serve model on RunPod cloud GPU"
    )
    parser.add_argument(
        "--remote",
        type=str,
        metavar="PROVIDER",
        help="Serve model on remote GPU (kaggle=free T4, runpod=paid)",
    )
    parser.add_argument(
        "--remote-stop", action="store_true", help="Stop active remote Kaggle session"
    )
    parser.add_argument(
        "--remote-status", action="store_true", help="Show active remote session status"
    )
    parser.add_argument(
        "--duration",
        type=int,
        metavar="MINUTES",
        default=None,
        help="Auto-stop remote kernel after N minutes (saves GPU quota)",
    )
    parser.add_argument(
        "--budget", type=str, default="2h", help="Cloud time budget (e.g. 1h, 2h, 4h)"
    )
    parser.add_argument("--stop", action="store_true", help="Stop cloud pod + billing")
    parser.add_argument(
        "--cloud-status", action="store_true", help="Show active cloud pod"
    )
    parser.add_argument(
        "--pull",
        type=str,
        metavar="MODEL",
        help="Download a model (gemma4:e4b, qwen35:a3b)",
    )
    parser.add_argument(
        "--ps",
        "--list",
        action="store_true",
        help="Show running models and installed models",
    )
    parser.add_argument("--kill", action="store_true", help="Stop local llama-server")
    parser.add_argument(
        "--makeitfit",
        type=str,
        metavar="MODEL",
        help="Make any model fit your GPU — quantize remotely with Unsloth, upload to HF",
    )
    parser.add_argument(
        "--login", type=str, metavar="SERVICE", help="Save API key (runpod)"
    )
    parser.add_argument(
        "--menubar", action="store_true", help="Start macOS menu bar app"
    )
    parser.add_argument(
        "--install-menubar",
        action="store_true",
        help="Register menu bar to start on login",
    )
    parser.add_argument(
        "--uninstall-menubar",
        action="store_true",
        help="Remove menu bar from login items",
    )
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Expose via Cloudflare Tunnel (shareable public URL)",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore all tool configs to pre-localfit state",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check if localfit broke any tool configs and self-heal",
    )
    args = parser.parse_args()

    # ── Restore / Doctor ──
    if args.restore:
        from localfit.safe_config import list_backups, self_heal

        backups = list_backups()
        if not backups:
            console.print(
                f"\n  [dim]No backups found. localfit hasn't modified any tool configs.[/]\n"
            )
        else:
            console.print(f"\n  [bold]Backups created by localfit:[/]\n")
            for b in backups[:10]:
                console.print(f"  {b.name}")
            console.print(f"\n  [bold]Restoring all to pre-localfit state...[/]")
            fixed = self_heal()
            if fixed:
                for f in fixed:
                    console.print(
                        f"  [green]✓[/] Restored {f['tool']} from {f['restored_from']}"
                    )
            else:
                console.print(f"  [green]✓[/] All configs healthy — nothing to restore")
        return

    if args.doctor:
        from localfit.safe_config import check_health, self_heal

        console.print(f"\n  [bold]localfit doctor — checking tool configs...[/]\n")
        issues = check_health()
        if not issues:
            console.print(f"  [green]✓[/] All tool configs healthy")
            console.print(f"  [dim]Checked: Hermes, OpenClaw, Open WebUI[/]\n")
        else:
            console.print(f"  [red]Found {len(issues)} issue(s):[/]")
            for i in issues:
                console.print(f"  [red]✗[/] {i['tool']}: {i['issue']} ({i['file']})")
            console.print(f"\n  [bold]Auto-fixing...[/]")
            fixed = self_heal()
            for f in fixed:
                console.print(f"  [green]✓[/] Fixed {f['tool']} — restored from backup")
            if not fixed:
                console.print(f"  [yellow]No backups available to restore from[/]")
        return

    # ── Menu bar ──
    if args.menubar:
        from localfit.menubar import main as menubar_main

        menubar_main()
        return

    if args.install_menubar:
        from localfit.launchagent import install

        if install():
            console.print(f"  [green]✓[/] Menu bar registered to start on login")
        else:
            console.print(f"  [red]Failed to register[/]")
        return

    if args.uninstall_menubar:
        from localfit.launchagent import uninstall

        if uninstall():
            console.print(f"  [green]✓[/] Menu bar removed from login items")
        else:
            console.print(f"  [red]Failed to unregister[/]")
        return

    # ── Login ──
    if args.login:
        service = args.login.lower()
        if service == "runpod":
            from localfit.cloud import save_runpod_key

            console.print(f"\n  [bold]RunPod — cloud GPUs for serving & quantization[/]\n")
            console.print(f"  New to RunPod? Create an account:")
            console.print(f"  [cyan]https://runpod.io?ref=901ol203[/]")
            console.print(f"  [dim]Referral helps fund localfit development · same price for you[/]\n")
            console.print(f"  Already have an account? Get your API key at:")
            console.print(f"  [cyan]https://www.runpod.io/console/user/settings[/]\n")
            try:
                key = input("  Paste API key: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if key:
                save_runpod_key(key)
        elif service in ("hf", "huggingface"):
            console.print(
                f"\n  Get your HuggingFace token at: https://huggingface.co/settings/tokens"
            )
            console.print(
                f"  [dim]This speeds up downloads and unlocks gated models (Llama, Mistral)[/]"
            )
            try:
                token = input("  Paste token: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if token:
                # Save via huggingface_hub if available
                try:
                    from huggingface_hub import login

                    login(token=token, add_to_git_credential=False)
                    console.print(f"  [green]✓ HuggingFace token saved[/]")
                except ImportError:
                    # Save manually
                    hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
                    os.makedirs(os.path.dirname(hf_token_path), exist_ok=True)
                    with open(hf_token_path, "w") as f:
                        f.write(token)
                    console.print(f"  [green]✓ Token saved to {hf_token_path}[/]")
        elif service == "kaggle":
            from localfit.remote import save_kaggle_credentials

            save_kaggle_credentials()
        else:
            console.print(
                f"  [red]Unknown service: {args.login}. Supported: runpod, kaggle, huggingface[/]"
            )
        return

    # ── Cloud stop ──
    if args.stop:
        from localfit.cloud import cloud_stop

        cloud_stop()
        return

    # ── Cloud status ──
    if getattr(args, "cloud_status", False):
        from localfit.cloud import cloud_status

        cloud_status()
        return

    # ── Remote stop (Kaggle) ──
    if getattr(args, "remote_stop", False):
        from localfit.remote import remote_stop

        remote_stop()
        return

    # ── Remote status (Kaggle) ──
    if getattr(args, "remote_status", False):
        from localfit.remote import remote_status

        remote_status()
        return

    # ── Trending gallery ──
    if args.trending:
        _trending_gallery()
        return

    # ── Check prerequisites ──
    if args.check:
        from localfit.prerequisites import print_status

        print_status()
        return

    # ── Specs ──
    if args.specs:
        from localfit.backends import get_machine_specs, print_machine_specs

        print_machine_specs()
        return

    # ── Health ──
    if args.health:
        from localfit.backends import print_health_dashboard

        print_health_dashboard()
        return

    # ── Cleanup ──
    if args.cleanup:
        from localfit.backends import (
            cleanup_gpu_memory,
            get_machine_specs,
            print_machine_specs,
        )

        console.print("\n  [bold yellow]Cleaning up GPU memory...[/]")
        result = cleanup_gpu_memory(force=True)
        if result["ollama_unloaded"]:
            console.print(
                f"  [green]Unloaded: {', '.join(result['ollama_unloaded'])}[/]"
            )
        if result["processes_killed"]:
            for p in result["processes_killed"]:
                console.print(f"  [green]Killed PID {p['pid']}[/]")
        if not result["ollama_unloaded"] and not result["processes_killed"]:
            console.print("  [dim]Nothing to clean up.[/]")
        console.print()
        print_machine_specs()
        return

    # ── Debloat ──
    if args.debloat:
        from localfit.backends import debloat_wizard

        debloat_wizard()
        return

    # ── Bench / Arena ──
    if args.bench:
        from localfit.bench import run_full_bench

        run_full_bench(force=args.force)
        return
    if args.arena:
        from localfit.bench import show_leaderboard

        show_leaderboard()
        return

    # ── Fetch from HuggingFace ──
    if args.fetch:
        from localfit.backends import simulate_hf_model

        simulate_hf_model(args.fetch)
        return

    # ── Make it fit ──
    if args.makeitfit:
        from localfit.makeitfit import cmd_makeitfit
        cmd_makeitfit(args.makeitfit)
        return

    # ── Simulate ──
    if args.simulate:
        if args.simulate == "__interactive__":
            _interactive_simulate()
        else:
            from localfit.backends import simulate_model_fit

            simulate_model_fit(args.simulate)
        return

    # ── Pull (download model) ──
    if args.pull:
        from localfit.backends import (
            MODELS,
            download_model_hf,
            resolve_model_family,
            get_machine_specs,
        )

        specs = get_machine_specs()
        resolved = resolve_model_family(args.pull, specs["gpu_total_mb"])
        if resolved and resolved in MODELS:
            m = MODELS[resolved]
            console.print(f"\n  [bold]Pulling {m['name']}...[/]")
            download_model_hf(resolved)
        else:
            # Try HuggingFace search
            from localfit.backends import simulate_hf_model

            simulate_hf_model(args.pull)
        return

    # ── PS (list models) ──
    if args.ps:
        from localfit.backends import (
            get_disk_info,
            get_llama_server_config,
            _detect_model_info,
            MODELS,
        )

        srv = get_llama_server_config()
        if srv["running"]:
            mi = _detect_model_info(srv, None)
            name = mi.get("name", "?")
            quant = mi.get("quant", "")
            console.print(
                f"\n  [green]●[/] [bold]{name} {quant}[/]  [dim]port {srv.get('port', 8089)}  ctx {srv.get('n_ctx', 0) // 1024}K[/]"
            )
        else:
            console.print(f"\n  [dim]No model running[/]")

        di = get_disk_info()
        if di.get("models"):
            console.print(f"\n  [dim]Installed ({len(di['models'])} models):[/]")
            for m in di["models"]:
                console.print(
                    f"  {m['name'].replace('.gguf', ''):<35} {m['size_gb']}GB"
                )
        console.print()
        return

    # ── Kill (stop server) ──
    if args.kill:
        import signal
        from localfit.backends import get_llama_server_config

        srv = get_llama_server_config()
        if srv["running"]:
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
            console.print(f"  [green]✓[/] Server stopped")
        else:
            console.print(f"  [dim]No server running[/]")
        return

    # ── Serve / Run ──
    if args.serve:
        if args.cloud:
            from localfit.cloud import cloud_serve

            budget = float(args.budget.replace("h", ""))
            cloud_serve(args.serve, budget_hours=budget)
            return
        if args.remote:
            provider = args.remote.lower().strip()
            if provider == "kaggle":
                from localfit.remote import remote_serve_kaggle

                remote_serve_kaggle(args.serve, max_runtime_minutes=args.duration)
                return
            elif provider == "runpod":
                # Route to existing cloud.py
                from localfit.cloud import cloud_serve

                budget = float(args.budget.replace("h", ""))
                cloud_serve(args.serve, budget_hours=budget)
                return
            else:
                console.print(f"  [red]Unknown remote provider: {provider}[/]")
                console.print(f"  [dim]Supported: kaggle (free T4), runpod (paid)[/]")
                return
        # Use wizard — shows run menu (MLX/GGUF/Remote), serves, then tool picker
        from localfit.wizard import run_wizard
        _show_logo_intro()
        run_wizard(model=args.serve, tool=args.launch)
        return

    # ── Remote serve without --serve (shortcut) — skip if --launch handles it ──
    if args.remote and not args.serve and not args.launch:
        console.print(f"\n  [red]--remote requires a model.[/]")
        console.print(f"  [dim]Usage: localfit run MODEL --remote kaggle[/]")
        console.print(f"  [dim]   or: localfit launch openwebui --model MODEL --remote kaggle[/]\n")
        return

    # ── Launch tool (serve model + launch tool in one command) ──
    if args.launch:
        model = args.model or args.serve
        provider = getattr(args, "remote", None)

        # Remote launch: serve on Kaggle/RunPod, then launch tool with that endpoint
        if provider:
            if not model:
                console.print(f"\n  [red]--remote requires a model.[/]")
                console.print(f"  Usage: [cyan]localfit launch {args.launch} --model gemma4:e4b --remote kaggle[/]\n")
                return

            # Show budget info
            if provider == "kaggle":
                from localfit.remote import _get_quota_usage, KAGGLE_WEEKLY_QUOTA_HOURS
                quota = _get_quota_usage()
                remaining_h = max(0, KAGGLE_WEEKLY_QUOTA_HOURS - quota["used_hours"])
                console.print(f"\n  [bold]Kaggle GPU quota:[/] {remaining_h:.0f}h remaining (of {KAGGLE_WEEKLY_QUOTA_HOURS}h/week)")
                console.print(f"  [dim]Note: tracked locally — check kaggle.com/settings for exact quota[/]")
            elif provider == "runpod":
                try:
                    from localfit.cloud import get_runpod_key, _runpod_api
                    rk = get_runpod_key()
                    if rk:
                        r = _runpod_api('{ myself { clientBalance } }', rk)
                        balance = r.get("data", {}).get("myself", {}).get("clientBalance", 0)
                        console.print(f"\n  [bold]RunPod balance:[/] ${balance:.2f}")
                except Exception:
                    pass

            # Parse budget
            duration = getattr(args, "duration", None)
            if not duration:
                budget = getattr(args, "budget", "1h")
                if budget.startswith("$"):
                    # Money → calculate time on cheapest GPU
                    try:
                        from localfit.cloud import get_runpod_key, fetch_gpu_options
                        rk = get_runpod_key()
                        gpus = fetch_gpu_options(rk) if rk else []
                        if gpus:
                            cheapest = gpus[0]["price"]
                            duration = int(float(budget.strip("$")) / cheapest * 60)
                            console.print(f"  ${budget.strip('$')} ≈ {duration}min on {gpus[0]['name']} (${cheapest}/hr)")
                    except Exception:
                        duration = 60
                else:
                    # Time string: 30m, 1h, 2h
                    b = budget.lower().strip()
                    if b.endswith("m"):
                        duration = int(b[:-1])
                    elif b.endswith("h"):
                        duration = int(float(b[:-1]) * 60)
                    else:
                        duration = 60

            console.print(f"  Duration: {duration}min\n")

            # Serve remotely
            if provider == "kaggle":
                from localfit.remote import remote_serve_kaggle
                remote_serve_kaggle(model, max_runtime_minutes=duration)
                # After serve returns, the tool menu was already shown
            elif provider in ("runpod", "cloud"):
                from localfit.cloud import cloud_serve
                cloud_serve(model)
            return

        # Local launch: serve locally + launch tool
        from localfit.prerequisites import (
            check_llama_server,
            ensure_llama_server,
            ensure_claude_code,
        )

        ls = check_llama_server()
        if not ls["found"]:
            path = ensure_llama_server()
            if not path:
                return
        if args.launch == "claude":
            from localfit.prerequisites import check_claude_code

            cc = check_claude_code()
            if not cc["found"]:
                ensure_claude_code()
                cc = check_claude_code()
                if not cc["found"]:
                    return

        _launch_tool(args.launch, model, tunnel=args.tunnel)
        return

    # ── Config tool ──
    if args.config:
        _config_tool(args.config)
        return

    # ── Default: animated logo → dashboard ──
    _show_logo_intro()
    _boot_screen()


def _boot_caps_text(caps):
    text = Text()
    styles = {
        "vision": "magenta",
        "audio": "bright_cyan",
        "code": "cyan",
        "MoE": "green",
    }
    for cap in caps or []:
        if text:
            text.append(" ")
        text.append(cap, style=styles.get(cap, "dim"))
    return text


def _boot_section_panel(title, rows, border_style="bright_white", subtitle=None):
    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column(ratio=1)
    if rows:
        for row in rows:
            body.add_row(row)
    else:
        body.add_row(Text("Nothing to show", style="dim"))
    return Panel(
        body,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        padding=(0, 1),
    )


LOGO_BIG = [
    "██╗      ██████╗  ██████╗ █████╗ ██╗     ███████╗██╗████████╗",
    "██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██╔════╝██║╚══██╔══╝",
    "██║     ██║   ██║██║     ███████║██║     █████╗  ██║   ██║   ",
    "██║     ██║   ██║██║     ██╔══██║██║     ██╔══╝  ██║   ██║   ",
    "███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║     ██║   ██║   ",
    "╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝   ╚═╝   ",
]

LOGO_SMALL = "▌ localfit ▐"

LOGO_GRADIENT = ["#00ffff", "#00e5ff", "#00ccff", "#00b3ff", "#0099ff", "#0080ff"]


def _show_logo_intro():
    """Animated intro: big logo with color sweep → shrinks to small title → dashboard."""
    import time
    from rich.text import Text
    from rich.align import Align
    from rich.live import Live

    gradient = [
        "#00ffff", "#00e5ff", "#00ccff", "#00b3ff", "#0099ff",
        "#0080ff", "#0066ff", "#1a4dff", "#3333ff", "#4d1aff",
        "#6600ff", "#8000ff", "#9900ff", "#b300ff", "#cc00ff",
    ]

    try:
        console.clear()

        term_w = console.width
        term_h = console.height or 24
        logo_w = max(len(l) for l in LOGO_BIG)

        # Small terminal: skip big logo, show compact version
        if term_w < logo_w + 4 or term_h < 14:
            small = Text("\n")
            small.append("  ▌ ", style="bright_cyan")
            small.append("local", style="bold bright_cyan")
            small.append("fit", style="bold bright_magenta")
            small.append(" ▐  ", style="bright_magenta")
            small.append("GPU toolkit for local LLMs\n", style="dim")
            console.print(Align.center(small))
            time.sleep(0.3)
            return

        # Vertical padding: 1/3 from top (looks better than exact center)
        logo_h = len(LOGO_BIG) + 3
        v_pad = max(1, (term_h - logo_h) // 3)
        top_pad = "\n" * v_pad

        with Live(console=console, refresh_per_second=30, transient=True) as live:
            max_col = logo_w

            # Phase 1: color sweep across big logo (left → right)
            for sweep in range(max_col + len(gradient) + 5):
                text = Text(top_pad)
                for row_idx, line in enumerate(LOGO_BIG):
                    for col_idx, ch in enumerate(line):
                        dist = sweep - col_idx - row_idx
                        if 0 <= dist < len(gradient):
                            text.append(ch, style=gradient[dist])
                        elif dist >= len(gradient):
                            text.append(ch, style="bright_white")
                        else:
                            text.append(ch, style="grey23")
                    text.append("\n")
                text.append("\n          GPU toolkit for local LLMs\n", style="dim italic")
                live.update(Align.center(text))
                time.sleep(0.012)

            # Phase 2: glow pulse — breathe bright → dim → bright (2 cycles)
            glow_levels = [
                "bright_white", "#ddddff", "#aaaaee", "#8888dd", "#6666cc",
                "#8888dd", "#aaaaee", "#ddddff", "bright_white",
                "#ddddff", "#aaaaee", "#8888dd", "#aaaaee", "#ddddff", "bright_white",
            ]
            for glow in glow_levels:
                text = Text("\n\n")
                for line in LOGO_BIG:
                    text.append(line + "\n", style=glow)
                text.append("\n          GPU toolkit for local LLMs\n", style="dim italic")
                live.update(Align.center(text))
                time.sleep(0.05)

            time.sleep(0.3)

            # Phase 3: shrink — remove lines from top and bottom
            for trim in range(1, len(LOGO_BIG)):
                text = Text("\n")
                remaining = LOGO_BIG[trim:]
                for i, line in enumerate(remaining):
                    text.append(line + "\n", style=LOGO_GRADIENT[min(i + trim, len(LOGO_GRADIENT) - 1)])
                live.update(Align.center(text))
                time.sleep(0.07)

            # Phase 4: collapse to single line
            small = Text()
            small.append("\n  ▌ ", style="bright_cyan")
            small.append("local", style="bold bright_cyan")
            small.append("fit", style="bold bright_magenta")
            small.append(" ▐  ", style="bright_magenta")
            small.append("GPU toolkit for local LLMs", style="dim")
            small.append("\n")
            live.update(Align.center(small))
            time.sleep(0.4)

    except Exception:
        # Fallback: just print small logo
        console.print()
        console.print("  [bold bright_cyan]local[/][bold bright_magenta]fit[/]  [dim]GPU toolkit for local LLMs[/]")
        console.print()


def _boot_screen():
    """Default command — show system check + interactive model browser."""
    from localfit.backends import (
        MODELS,
        _detect_model_info,
        diagnose_gpu_health,
        fetch_unsloth_top_models,
        get_disk_info,
        get_llama_server_config,
        get_machine_specs,
        get_metal_gpu_stats,
        get_swap_usage_mb,
        print_health_dashboard,
        recommend_model,
        simulate_hf_model,
    )
    from localfit.home_menu import show_home_menu

    with console.status("[bold]  Scanning system...", spinner="dots"):
        specs = get_machine_specs()
        metal = get_metal_gpu_stats()
        srv = get_llama_server_config()
        swap_mb = get_swap_usage_mb()
        diag = diagnose_gpu_health()
        di = get_disk_info()
        try:
            all_models = fetch_unsloth_top_models(40)
        except Exception:
            all_models = []

    is_cpu_only = specs.get("cpu_only", False)
    gpu_total = specs["gpu_total_mb"] if is_cpu_only else (metal.get("total_mb") or specs["gpu_total_mb"])

    model_mb = 0
    model_name = "none running"
    model_source = ""
    if srv["running"]:
        mi = _detect_model_info(srv, None)
        model_mb = int((mi.get("size_gb") or 0) * 1024)
        # Build friendly name from model info
        name = mi.get("name") or ""
        if not name or name == "?":
            # Try to get from /v1/models API
            try:
                import urllib.request as _ur
                with _ur.urlopen("http://127.0.0.1:8089/v1/models", timeout=2) as _r:
                    _md = json.loads(_r.read())
                    _mid = _md.get("data", [{}])[0].get("id", "")
                    if _mid:
                        # Clean up: gemma-4-E4B-it-UD-Q8_K_XL.gguf → Gemma 4 E4B Q8_K_XL
                        name = _mid.replace(".gguf", "").replace("-it-UD-", " ").replace("-it-", " ").replace("-", " ")
            except Exception:
                pass
        model_name = name or "model"
        if mi.get("quant") and mi["quant"] not in model_name:
            model_name += f" {mi['quant']}"
        if mi.get("size_gb"):
            model_name += f" {mi['size_gb']}GB"
        model_source = "local"
    else:
        # Check for active remote session
        try:
            _kaggle_state = Path.home() / ".localfit" / "active_kaggle.json"
            if _kaggle_state.exists():
                _ks = json.loads(_kaggle_state.read_text())
                _ep = _ks.get("endpoint", "")
                _km = _ks.get("model", "?")
                if _ep:
                    model_name = f"{_km} (Kaggle)"
                    model_source = "kaggle"
                else:
                    model_name = f"{_km} (starting...)"
                    model_source = "kaggle"
        except Exception:
            pass
    kv_mb = diag.get("kv_cache_est_mb", 0)
    used_mb = model_mb + kv_mb
    free_mb = max(0, gpu_total - used_mb)
    model_fits = used_mb < gpu_total if used_mb > 0 else True

    if model_fits and swap_mb < 2000:
        border_color, verdict = "green", "READY"
    elif model_fits:
        border_color, verdict = "yellow", "READY"
    else:
        border_color, verdict = "red", "SLOW"

    dfree = di.get("disk_free_gb", 0)
    dcache = di.get("hf_cache_gb", 0)
    if is_cpu_only:
        gpu_line = f"{used_mb // 1024}/{gpu_total // 1024}GB used · CPU-only" if used_mb else f"{specs['ram_gb']}GB available · CPU-only"
        machine_line = f"{specs['chip']} · {specs['ram_gb']}GB · {specs['cpu_cores']} CPU cores"
        subtitle = f"{specs['chip']}  {specs['ram_gb']}GB RAM"
    else:
        if used_mb:
            gpu_line = f"{used_mb // 1024}/{gpu_total // 1024}GB used · {free_mb // 1024}GB free"
        elif model_source in ("kaggle", "runpod"):
            gpu_line = f"{gpu_total // 1024}GB total · model on remote GPU"
        else:
            gpu_line = f"{gpu_total // 1024}GB total · no model loaded"
        machine_line = f"{specs['chip']} · {specs['ram_gb']}GB · {specs.get('gpu_cores', '?')} GPU cores"
        subtitle = f"{specs['chip']}  {gpu_total // 1024}GB unified"

    if srv["running"]:
        model_line = f"{model_name} · {'GPU' if srv['ngl'] >= 90 else 'CPU'} · ctx {srv['n_ctx'] // 1024}K"
    else:
        model_line = "none running · localfit --serve MODEL"

    system = {
        "verdict": verdict,
        "color": border_color,
        "gpu": gpu_line,
        "swap": f"{swap_mb // 1024}GB" + (" · close apps to fix" if swap_mb > 2000 else ""),
        "disk": f"{dfree}GB free · cache {dcache}GB",
        "model": model_line,
        "machine": machine_line,
        "subtitle": subtitle,
    }

    items = []
    next_index = 1

    def _source(repo):
        if repo and "/" in repo:
            return repo.split("/", 1)[0]
        return ""

    def _fmt_dl(downloads):
        if not downloads:
            return "0 dl"
        return f"{downloads // 1000}K dl" if downloads < 1_000_000 else f"{downloads / 1_000_000:.1f}M dl"

    def _clean_reason(reason):
        return (
            reason.replace("[yellow]", "")
            .replace("[/]", "")
            .replace("\n", " ")
            .replace("  ", " ")
            .strip()
        )

    def _add_item(
        section,
        label,
        meta="",
        detail="",
        repo=None,
        source="",
        caps=None,
        accent="bright_white",
        badge="",
        selectable=True,
        action="inspect",
    ):
        nonlocal next_index
        item = {
            "index": next_index if selectable else None,
            "section": section,
            "accent": accent,
            "badge": badge,
            "label": label,
            "caps": caps or [],
            "meta": meta,
            "detail": detail,
            "repo": repo,
            "source": source,
            "action": action,
            "selectable": selectable,
        }
        items.append(item)
        if selectable:
            next_index += 1

    os.system("clear" if os.name != "nt" else "cls")

    # Show locally running models as ACTIVE
    if srv["running"]:
        mi = _detect_model_info(srv, None)
        local_model = mi.get("name") or "unknown"
        local_detail = f"Running on :8089 · {mi.get('size_gb', '?')}GB"
        if mi.get("quant"):
            local_model += f" {mi['quant']}"
        _add_item(
            section="ACTIVE",
            label=local_model,
            meta="local",
            detail=local_detail,
            repo="local:8089",
            source="local",
            accent="green",
            badge="●",
            selectable=True,
            action="launch_tool_local",
        )

    # Show active remote sessions
    for session in _get_active_remote_sessions():
        _add_item(
            section="ACTIVE",
            label=session["label"],
            meta=session["meta"],
            detail=session["detail"],
            repo=session["provider"].lower(),
            source=session["provider"],
            accent="magenta",
            badge="●",
            selectable=True,
            action="launch_tool_remote",
        )

    installed_total = sum(m["size_gb"] for m in di.get("models", []))
    installed_models = di.get("models", [])[:5]
    for i, model in enumerate(installed_models):
        note = "Installed locally in Hugging Face cache"
        if i == 0:
            note = f"Top local cache entries · {len(di.get('models', []))} models · {installed_total:.0f}GB"
        _add_item(
            section="INSTALLED",
            label=model["name"].replace(".gguf", ""),
            meta=f"{model['size_gb']}GB",
            detail=note,
            repo=model["name"].replace(".gguf", ""),
            source="local cache",
            accent="green",
            badge="✓",
            selectable=True,
        )

    rec_id, rec_reason = recommend_model(specs["ram_gb"], cpu_only=is_cpu_only)
    rec_model = MODELS.get(rec_id)
    if rec_model:
        rec_repo = rec_model.get("hf_repo") or rec_model.get("ollama_tag") or rec_id
        _add_item(
            section="RECOMMENDED",
            label=rec_model["name"],
            meta=f"~{rec_model['size_gb']}GB",
            detail=_clean_reason(rec_reason),
            repo=rec_repo,
            source=_source(rec_repo) or "localfit",
            accent="yellow",
            badge="★",
            selectable=True,
        )
        if rec_id == "gemma4-e4b" and "gemma4-e2b" in MODELS:
            e2b = MODELS["gemma4-e2b"]
            e2b_repo = e2b.get("hf_repo") or e2b.get("ollama_tag") or "gemma4-e2b"
            _add_item(
                section="RECOMMENDED",
                label=e2b["name"],
                meta=f"~{e2b['size_gb']}GB",
                detail=_clean_reason(e2b.get("description", "Smaller fallback for tight memory budgets.")),
                repo=e2b_repo,
                source=_source(e2b_repo) or "localfit",
                accent="yellow",
                badge="→",
                selectable=True,
            )

    installed_names = {
        m["name"].lower().replace(".gguf", "").replace("-", "").replace("_", "")
        for m in di.get("models", [])
    }
    compatible = []
    cloud_only = []
    for model in all_models:
        est = model.get("est_smallest_gb")
        if est is None:
            cloud_only.append(model)
        elif est * 1024 <= gpu_total:
            compatible.append(model)
        else:
            cloud_only.append(model)

    for model in compatible[:5]:
        est = model.get("est_smallest_gb", "?")
        base = model["label"].lower().replace("-", "").replace("_", "")
        installed = any(base in inst for inst in installed_names)
        if installed:
            detail = f"Compatible with your machine · already installed · {_fmt_dl(model['downloads'])}"
            badge = "✓"
        elif is_cpu_only:
            cpu_note = "CPU-only" if est == "?" or not isinstance(est, (int, float)) or est <= specs["ram_gb"] * 0.6 else "CPU slow"
            detail = f"Fits your RAM budget · {_fmt_dl(model['downloads'])} · {cpu_note}"
            badge = "→"
        else:
            detail = f"Fits your local GPU budget · {_fmt_dl(model['downloads'])}"
            badge = "→"
        _add_item(
            section="COMPATIBLE",
            label=model["label"],
            meta=f"~{est}GB",
            detail=detail,
            repo=model["repo_id"],
            source=_source(model["repo_id"]),
            caps=model.get("caps", []),
            accent="green" if installed else "bright_white",
            badge=badge,
            selectable=True,
        )

    top_trending = sorted(all_models, key=lambda x: x["downloads"], reverse=True)[:5]
    compatible_repos = {m["repo_id"] for m in compatible[:5]}
    trending_extra = [m for m in top_trending if m["repo_id"] not in compatible_repos]
    for model in trending_extra[:3]:
        est = model.get("est_smallest_gb")
        if est and est * 1024 > gpu_total:
            detail = f"Hot model · {_fmt_dl(model['downloads'])} · needs cloud for this machine"
        elif est:
            detail = f"Hot model · {_fmt_dl(model['downloads'])} · fits locally"
        else:
            detail = f"Hot model · {_fmt_dl(model['downloads'])}"
        _add_item(
            section="TRENDING",
            label=model["label"],
            meta=f"~{est}GB" if est else "? GB",
            detail=detail,
            repo=model["repo_id"],
            source=_source(model["repo_id"]),
            caps=model.get("caps", []),
            accent="bright_white",
            badge="▲",
            selectable=True,
        )

    trending_extra_repos = {m["repo_id"] for m in trending_extra[:3]}
    cloud_unique = [m for m in cloud_only if m["repo_id"] not in trending_extra_repos]
    for model in cloud_unique[:3]:
        est = model.get("est_smallest_gb")
        size_str = f"~{est}GB" if est else "? GB"
        _add_item(
            section="CLOUD",
            label=model["label"],
            meta=size_str,
            detail=f"Too large for local run · {_fmt_dl(model['downloads'])} · use RunPod or Kaggle",
            repo=model["repo_id"],
            source=_source(model["repo_id"]),
            caps=model.get("caps", []),
            accent="yellow",
            badge="☁",
            selectable=True,
        )
    if cloud_unique:
        _add_item(
            section="CLOUD",
            label="localfit --login runpod to enable cloud serving",
            meta="",
            detail="Login unlocks live cloud GPU pricing and one-command serve.",
            repo=None,
            source="runpod",
            accent="yellow",
            badge="",
            selectable=False,
        )

    if not items:
        _add_item(
            section="RECOMMENDED",
            label="Search for a model",
            meta="",
            detail="Press s to open the full model picker.",
            repo=None,
            accent="yellow",
            badge="★",
            selectable=False,
        )

    while True:
        result = show_home_menu(system, items)
        if not result or result["action"] == "quit":
            return
        if result["action"] == "simulate":
            _interactive_simulate()
            continue
        if result["action"] == "bench":
            from localfit.bench import run_full_bench

            run_full_bench()
            continue
        if result["action"] == "health":
            print_health_dashboard()
            continue
        if result["action"] == "remote_status":
            if result.get("repo") == "runpod":
                from localfit.cloud import cloud_status
                cloud_status()
            else:
                from localfit.remote import remote_status
                remote_status()
            continue
        if result["action"] in ("launch_tool_local", "launch_tool_remote"):
            os.system("clear")
            _remote_ep = None
            if result["action"] == "launch_tool_local":
                console.print(f"\n  [green]● Model running locally on :8089[/]\n")
            else:
                try:
                    _st = json.loads((Path.home() / ".localfit" / "active_kaggle.json").read_text())
                    _remote_ep = _st.get("endpoint", "")
                    console.print(f"\n  [magenta]● {_st.get('model','?')} (Kaggle remote)[/]")
                    console.print(f"  [cyan]{_remote_ep}[/]\n")
                except Exception:
                    console.print(f"\n  [yellow]Remote session active[/]\n")

            picked = _arrow_tool_picker()
            if picked:
                if _remote_ep:
                    # Check if endpoint is alive before launching
                    import urllib.request as _ur
                    try:
                        _ur.urlopen(f"{_remote_ep}/v1/models", timeout=5)
                        console.print(f"  [green]● Endpoint alive[/]")
                    except Exception:
                        console.print(f"  [red]● Endpoint dead — Kaggle session may have expired[/]")
                        console.print(f"  [dim]Restart: localfit run gemma4:e4b --remote kaggle[/]")
                        input("\n  Press Enter to go back...")
                        continue
                    _model = _st.get("model", "gemma4:e4b") if "_st" in dir() else "gemma4:e4b"
                    _launch_tool_with_endpoint(picked, f"{_remote_ep}/v1", _model)
                else:
                    _launch_tool(picked, None)
            continue
        if result["action"] == "inspect" and result.get("repo"):
            outcome = _serve_model(result["repo"])
            if outcome == "quit":
                return
            continue


def _interactive_simulate():
    """Interactive model picker."""
    from localfit.backends import (
        get_machine_specs,
        simulate_model_fit,
        MODELS,
        fetch_unsloth_top_models,
        simulate_hf_model,
    )

    specs = get_machine_specs()
    os.system("clear" if os.name != "nt" else "cls")
    console.print(
        f"\n  [bold]Will it fit?[/]  ·  {specs['chip']}  ·  {specs['ram_gb']}GB RAM\n"
    )

    # Known models
    models_list = list(MODELS.items())
    for i, (mid, m) in enumerate(models_list, 1):
        fits = "✓" if m["size_gb"] * 1024 < specs["gpu_total_mb"] else "✗"
        icon = "[green]✓[/]" if fits == "✓" else "[red]✗[/]"
        console.print(
            f"  {icon} [bold]{i:>2}.[/] {m['name']:<30} {m['size_gb']:>5}GB  [dim]{m.get('description', '')[:40]}[/]"
        )

    # Live from HuggingFace
    try:
        with console.status("[dim]  Fetching...[/]", spinner="dots"):
            hf_models = fetch_unsloth_top_models(limit=10)
        console.print(f"\n  [dim]── Top models · powered by Unsloth GGUF ──[/]")
        fav_start = len(models_list) + 1
        for j, m in enumerate(hf_models, fav_start):
            dl = m["downloads"]
            dl_str = f"{dl // 1000}K" if dl < 1_000_000 else f"{dl / 1_000_000:.1f}M"
            console.print(
                f"     [bold]{j:>2}.[/] {escape(m['label']):<28} [dim]{dl_str} dl[/]"
            )
    except Exception:
        hf_models = []

    console.print(
        f"\n  [bold] s.[/] Search HuggingFace  [bold] c.[/] Custom ('70b q4')  [bold] q.[/] Quit\n"
    )

    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "q" or not choice:
        return
    elif choice == "s":
        q = input("  Search or paste URL: ").strip()
        if q:
            simulate_hf_model(q)
    elif choice == "c":
        q = input("  Model (e.g. '70b q4'): ").strip()
        if q:
            simulate_model_fit(q)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models_list):
                simulate_model_fit(models_list[idx][0])
            elif idx < len(models_list) + len(hf_models):
                simulate_hf_model(hf_models[idx - len(models_list)]["repo_id"])
        except ValueError:
            if "/" in choice or "huggingface" in choice:
                simulate_hf_model(choice)
            else:
                simulate_model_fit(choice)


def _offer_remote(model_query, model_size_gb=None, hf_data=None):
    """Smart options when a model doesn't fit locally.

    Shows ALL options — no other tool does this today:
    1. Smaller quant that fits GPU
    2. Extreme quant (IQ2, Q2_K) — worse quality but runs
    3. Partial GPU offload — some layers GPU, rest CPU
    4. CPU-only — if enough RAM
    5. Kaggle free GPU
    6. RunPod paid GPU
    7. "Try anyway" — YOLO mode, swap to disk

    Args:
        model_query: The model the user tried to run
        model_size_gb: Known size (for known models)
        hf_data: Optional fetch_hf_model() result (avoids re-fetching)
    """
    from localfit.backends import get_machine_specs

    specs = get_machine_specs()
    gpu_total_mb = specs["gpu_total_mb"]
    gpu_total_gb = gpu_total_mb // 1024
    ram_gb = specs["ram_gb"]
    is_cpu_only = specs.get("cpu_only", False)

    try:
        from localfit.remote import kaggle_check_model, remote_serve_kaggle, kaggle_fits
    except ImportError:
        kaggle_check_model = None

    options = []  # list of (key, label, action_fn)
    num = 1

    gguf_files = hf_data.get("gguf_files", []) if hf_data else []
    mmproj_files = hf_data.get("mmproj_files", []) if hf_data else []
    is_vlm = hf_data.get("is_vlm", False) if hf_data else False
    mmproj_overhead_gb = mmproj_files[0]["size_gb"] if (is_vlm and mmproj_files) else 0
    mmproj_overhead_mb = int(mmproj_overhead_gb * 1024)

    # ── 1. Smaller quant that fits GPU ──
    if gguf_files and not is_cpu_only:
        fits_gpu = [
            f
            for f in gguf_files
            if (f["size_gb"] * 1024 + mmproj_overhead_mb) < gpu_total_mb
        ]
        if fits_gpu:
            best_local = fits_gpu[-1]
            total_gb = best_local["size_gb"] + mmproj_overhead_gb
            vlm_note = f" + mmproj {mmproj_overhead_gb}GB" if mmproj_overhead_gb else ""
            console.print(
                f"  [bold green]{num}[/]  Run locally — "
                f"{best_local['quant']} ({best_local['size_gb']}GB{vlm_note}) "
                f"[green]fits your {gpu_total_gb}GB GPU[/]"
            )
            if best_local["quant"] in (
                "Q2_K",
                "Q2_K_S",
                "IQ2_M",
                "IQ2_S",
                "IQ2_XS",
                "IQ2_XXS",
            ):
                console.print(
                    f"     [dim]Extreme quant — lower quality but runs full speed[/]"
                )
            options.append(("local_quant", best_local))
            num += 1

    # ── 2. Partial GPU offload ──
    if gguf_files and not is_cpu_only and gpu_total_gb >= 4:
        # Find a quant bigger than GPU but smaller than RAM
        partial_candidates = [
            f
            for f in gguf_files
            if f["size_gb"] * 1024 >= gpu_total_mb and f["size_gb"] <= ram_gb * 0.8
        ]
        if partial_candidates:
            best_partial = partial_candidates[-1]  # biggest that fits RAM
            # Estimate layers on GPU vs CPU
            gpu_frac = gpu_total_gb / max(1, best_partial["size_gb"])
            gpu_pct = int(gpu_frac * 100)
            est_ngl = max(1, int(gpu_frac * 40))  # rough: 40 layers typical for 7B
            # Estimate speed: partial offload is ~30-60% of full GPU speed
            est_tps = max(1, int(30 * gpu_frac))
            console.print(
                f"  [bold yellow]{num}[/]  Partial GPU offload — "
                f"{best_partial['quant']} ({best_partial['size_gb']}GB)"
            )
            console.print(
                f"     [dim]~{gpu_pct}% on GPU, rest on CPU · "
                f"-ngl {est_ngl} · ~{est_tps} tok/s · needs {best_partial['size_gb']}GB RAM[/]"
            )
            options.append(("partial", best_partial))
            num += 1

    # ── 3. CPU-only ──
    if gguf_files:
        fits_ram = [f for f in gguf_files if f["size_gb"] <= ram_gb * 0.7]
        if fits_ram:
            best_cpu = fits_ram[-1]
            # Only show if we haven't already offered it as local_quant
            already_offered = any(
                k == "local_quant" and v["filename"] == best_cpu["filename"]
                for k, v in options
            )
            if not already_offered:
                est_cpu_tps = max(1, int(12 * 4 / max(1, best_cpu["size_gb"])))
                console.print(
                    f"  [bold blue]{num}[/]  CPU-only — "
                    f"{best_cpu['quant']} ({best_cpu['size_gb']}GB) "
                    f"[dim]~{est_cpu_tps} tok/s · slow but works[/]"
                )
                options.append(("cpu", best_cpu))
                num += 1

    # ── 4. Kaggle free GPU ──
    if kaggle_check_model and gguf_files:
        kaggle_result = kaggle_check_model(gguf_files, mmproj_files)
        if kaggle_result["fits_kaggle"]:
            best = kaggle_result["best"]
            gpu = best["gpu"]
            quant = best["quant"]
            console.print(
                f"  [bold green]{num}[/]  Kaggle remote [green](free)[/] — "
                f"{gpu['name']} ({gpu['vram_gb']}GB) · "
                f"{quant['quant']} ({quant['size_gb']}GB) · 12h"
            )
            options.append(("kaggle", kaggle_result))
            num += 1
        else:
            console.print(f"  [dim]     Kaggle: {kaggle_result['reason']}[/]")
    elif model_size_gb:
        kf = kaggle_fits(model_size_gb) if kaggle_fits else {"fits": False}
        if kf.get("fits"):
            console.print(
                f"  [bold green]{num}[/]  Kaggle remote [green](free)[/] — "
                f"{kf['gpu']} ({kf['vram_gb']}GB) · 12h"
            )
            options.append(("kaggle", kf))
            num += 1

    # ── 5. RunPod paid ──
    size = model_size_gb or (gguf_files[-1]["size_gb"] if gguf_files else 0)
    if size:
        price = (
            "~$0.20/hr RTX 4090"
            if size <= 24
            else "~$0.75/hr A6000"
            if size <= 48
            else "~$1.50/hr A100"
            if size <= 80
            else "~$2.50/hr H100"
        )
        console.print(
            f"  [bold cyan]{num}[/]  RunPod cloud [yellow](paid)[/] — {price}"
        )
        options.append(("runpod", None))
        num += 1

    # ── 6. YOLO mode ──
    if gguf_files:
        smallest = gguf_files[0]
        if smallest["size_gb"] > ram_gb * 0.7:
            console.print(
                f"  [bold red]{num}[/]  Try anyway [red](will swap to disk, ~0.5 tok/s)[/] — "
                f"{smallest['quant']} ({smallest['size_gb']}GB)"
            )
            console.print(
                f"     [dim]Your RAM: {ram_gb}GB · model: {smallest['size_gb']}GB · "
                f"will thrash swap. You asked for it.[/]"
            )
            options.append(("yolo", smallest))
            num += 1

    # ── Tips: techniques that could help ──
    tips = []

    # KV cache quantization — can save 2-4GB at 32K context
    if gguf_files and not is_cpu_only:
        # Check if a quant ALMOST fits (within ~3GB of GPU)
        almost_fits = [
            f
            for f in gguf_files
            if f["size_gb"] * 1024 > gpu_total_mb
            and f["size_gb"] * 1024 < gpu_total_mb + 3 * 1024
        ]
        if almost_fits:
            tips.append(
                f"[bold]KV cache quantization[/]: {almost_fits[-1]['quant']} "
                f"({almost_fits[-1]['size_gb']}GB) is close to fitting. "
                f"With [cyan]-ctk q4_0 -ctv q4_0[/] (4-bit KV cache), "
                f"you save ~2GB VRAM at 32K context. "
                f"Or reduce context: [cyan]-c 8192[/] saves ~1.5GB."
            )
            tips.append(
                f"[dim]Coming soon: Google's TurboQuant (3-bit KV, zero accuracy loss) "
                f"— ICLR 2026 — will save even more when it lands in llama.cpp.[/]"
            )

    # No small quants exist — suggest creating one
    if gguf_files:
        smallest = gguf_files[0]
        has_extreme = any(
            q["quant"]
            in (
                "Q2_K",
                "Q2_K_S",
                "IQ2_M",
                "IQ2_S",
                "IQ2_XS",
                "IQ2_XXS",
                "IQ1_M",
                "IQ1_S",
            )
            for q in gguf_files
        )
        if not has_extreme and smallest["size_gb"] * 1024 > gpu_total_mb:
            tips.append(
                f"[bold]Extreme quant not available[/]: Smallest is {smallest['quant']} "
                f"({smallest['size_gb']}GB). An IQ2_M/Q2_K quant would be ~{smallest['size_gb'] * 0.4:.1f}GB "
                f"but doesn't exist on HuggingFace yet."
            )
            tips.append(
                f"[dim]Create one: download the model, then run: "
                f"[cyan]llama-quantize {smallest['filename']} output-IQ2_M.gguf IQ2_M[/][/]"
            )

    # No GGUF at all
    if not gguf_files and hf_data:
        tips.append(
            f"[bold]No GGUF quants available[/] for this model. "
            f"It may only have safetensors/PyTorch weights."
        )
        tips.append(
            f"[dim]Convert: [cyan]python llama.cpp/convert_hf_to_gguf.py {hf_data.get('repo_id', 'MODEL')}[/] "
            f"then quantize with llama-quantize. "
            f"Or check: unsloth/{hf_data.get('name', 'MODEL')}-GGUF[/]"
        )

    if tips:
        console.print(f"\n  [dim]── tips ──[/]")
        for tip in tips:
            console.print(f"  {tip}")

    console.print(f"\n  [dim]q[/]  Cancel")
    console.print()

    if not options:
        return

    try:
        ans = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if ans == "q" or ans == "":
        return

    # Map answer to option
    try:
        idx = int(ans) - 1
        if 0 <= idx < len(options):
            key, data = options[idx]
        else:
            return
    except ValueError:
        return

    if key == "local_quant":
        # Serve locally with the smaller quant
        console.print(f"\n  [bold]Running {data['quant']} locally...[/]")
        _serve_model_gguf(hf_data, data)
    elif key == "partial":
        # Serve with partial GPU offload
        gpu_frac = gpu_total_gb / max(1, data["size_gb"])
        est_ngl = max(1, int(gpu_frac * 40))
        console.print(f"\n  [bold]Partial offload: -ngl {est_ngl}...[/]")
        _serve_model_gguf(hf_data, data, ngl=str(est_ngl))
    elif key == "cpu":
        console.print(f"\n  [bold]Running on CPU (slow but works)...[/]")
        _serve_model_gguf(hf_data, data, ngl="0", ctx="8192")
    elif key == "kaggle":
        from localfit.remote import remote_serve_kaggle

        remote_serve_kaggle(model_query)
    elif key == "runpod":
        from localfit.cloud import cloud_serve

        cloud_serve(model_query, budget_hours=2)
    elif key == "yolo":
        console.print(f"\n  [bold red]YOLO mode. Your fans are about to scream.[/]")
        _serve_model_gguf(hf_data, data, ngl="0", ctx="4096")


def _serve_model_gguf(hf_data, quant_file, ngl="99", ctx="32768"):
    """Download and serve a specific GGUF quant. Used by _offer_options."""
    from localfit.backends import _download_gguf
    from localfit.prerequisites import ensure_llama_server
    import subprocess

    binary = ensure_llama_server()
    if not binary:
        return

    console.print(
        f"  [dim]Downloading {quant_file['quant']} ({quant_file['size_gb']}GB)...[/]"
    )
    path = _download_gguf(hf_data["repo_id"], quant_file["filename"])
    if not path:
        console.print(f"  [red]Download failed[/]")
        return

    # Download mmproj for VLM models
    mmproj_path = None
    mmproj_files = hf_data.get("mmproj_files", [])
    if hf_data.get("is_vlm") and mmproj_files:
        mmproj_name = mmproj_files[0]["filename"]
        console.print(
            f"  [dim]Downloading vision projector {mmproj_name} ({mmproj_files[0]['size_gb']}GB)...[/]"
        )
        mmproj_path = _download_gguf(hf_data["repo_id"], mmproj_name)
        if not mmproj_path:
            console.print(
                f"  [yellow]mmproj download failed — model will be text-only[/]"
            )

    console.print(f"  [dim]Starting server...[/]")
    cmd = [binary, "-m", path, "--port", "8089", "-ngl", ngl, "-c", ctx, "--jinja"]
    if mmproj_path:
        cmd += ["--mmproj", mmproj_path]

    # Pre-launch safety: check free memory before Popen to prevent OOM crash
    from localfit.backends import get_machine_specs

    _specs = get_machine_specs()
    _free_mb = _specs.get("gpu_free_mb", 0)
    _model_mb = int(quant_file["size_gb"] * 1024)
    _mmproj_mb = (
        int(hf_data.get("mmproj_files", [{}])[0].get("size_gb", 0) * 1024)
        if mmproj_path
        else 0
    )
    _needed_mb = _model_mb + _mmproj_mb + 1024  # 1GB headroom
    if _free_mb > 0 and _free_mb < _needed_mb:
        console.print(f"\n  [red bold]✗ Not enough free memory to launch safely[/]")
        console.print(
            f"  [red]Need {_needed_mb // 1024}GB free, only {_free_mb // 1024}GB available[/]"
        )
        console.print(
            f"  [dim]Close other apps or use a smaller quant to avoid crashing your system[/]"
        )
        return

    # Set LD_LIBRARY_PATH for pre-built binaries
    env = os.environ.copy()
    lib_dir = os.path.join(os.path.dirname(binary), "lib")
    if os.path.isdir(lib_dir):
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_dir + (":" + ld if ld else "")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )
    console.print(f"  [green]✓ Running on :8089[/]  [dim]-ngl {ngl} -c {ctx}[/]")
    if mmproj_path:
        console.print(f"  [magenta]✓ Vision projector loaded — image input enabled[/]")
    console.print(f"  [dim]API: http://127.0.0.1:8089/v1[/]")
    _print_local_ready_hints(port=8089)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()

    # Check Kaggle fit
    kaggle_result = None
    if hf_data and hf_data.get("gguf_files"):
        kaggle_result = kaggle_check_model(
            hf_data["gguf_files"],
            hf_data.get("mmproj_files"),
        )
    elif model_size_gb:
        kf = kaggle_fits(model_size_gb)
        if kf["fits"]:
            kaggle_result = {
                "fits_kaggle": True,
                "best": {
                    "gpu": {"name": kf["gpu"], "vram_gb": kf["vram_gb"]},
                    "quant": {"quant": "best fit", "size_gb": model_size_gb},
                },
            }
        else:
            kaggle_result = {
                "fits_kaggle": False,
                "reason": f"{model_size_gb}GB too big for Kaggle",
            }

    # Present options
    options = []

    if kaggle_result and kaggle_result["fits_kaggle"]:
        best = kaggle_result["best"]
        gpu = best["gpu"]
        quant = best["quant"]
        console.print()
        console.print(f"  [bold green]1[/]  Run on Kaggle [green](free)[/]")
        console.print(
            f"     {gpu['name']} ({gpu['vram_gb']}GB) · "
            f"{quant.get('quant', '?')} ({quant.get('size_gb', '?')}GB) · "
            f"12h limit"
        )
        options.append("kaggle")
    else:
        if kaggle_result and kaggle_result.get("reason"):
            console.print(f"\n  [dim]Kaggle: {kaggle_result['reason']}[/]")

    # RunPod option
    console.print(
        f"  [bold cyan]{'2' if options else '1'}[/]  Run on RunPod [yellow](paid)[/]"
    )
    if model_size_gb:
        if model_size_gb <= 24:
            console.print(f"     ~$0.20/hr on RTX 4090 (24GB)")
        elif model_size_gb <= 48:
            console.print(f"     ~$0.75/hr on A6000 (48GB)")
        elif model_size_gb <= 80:
            console.print(f"     ~$1.50/hr on A100 (80GB)")
        else:
            console.print(f"     ~$2.50/hr on H100 (80GB)")
    options.append("runpod")

    console.print(f"  [dim]q[/]  Cancel")
    console.print()

    try:
        ans = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if ans in ("1", "kaggle", "k") and "kaggle" in options:
        remote_serve_kaggle(model_query)
    elif ans in ("2", "runpod", "r") or (ans == "1" and "kaggle" not in options):
        from localfit.cloud import cloud_serve

        cloud_serve(model_query, budget_hours=2)
    elif ans in ("q", ""):
        return


def _detect_local_api_model(port=8089):
    """Best-effort lookup of the current local API model ID."""
    import json as _json
    import urllib.request as _ur

    try:
        with _ur.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as resp:
            data = _json.loads(resp.read())
        if data.get("data"):
            return data["data"][0].get("id") or "local"
    except Exception:
        pass
    return "local"


def _base_model_query_from_gguf_name(name):
    """Strip quant suffixes from a local GGUF filename for backend discovery."""
    import re as _re

    base = os.path.basename(name).replace(".gguf", "")
    patterns = [
        r"-(?:UD-)?(?:IQ|Q)[A-Z0-9_]+$",
        r"-(?:BF16|FP16|F16)$",
    ]
    for pattern in patterns:
        stripped = _re.sub(pattern, "", base, flags=_re.IGNORECASE)
        if stripped != base:
            return stripped
    return base


def _normalize_model_query_for_discovery(model_query):
    """Map installed quant filenames back to a better discovery query."""
    from localfit.backends import get_disk_info

    if "/" in model_query or " " in model_query:
        return model_query

    query = model_query.replace(".gguf", "")
    for model in get_disk_info().get("models", []):
        local_name = model["name"].replace(".gguf", "")
        if query.lower() == local_name.lower():
            return _base_model_query_from_gguf_name(local_name)
    return model_query


def _get_active_remote_sessions():
    """Read active remote sessions launched by localfit from local state files."""
    sessions = []
    config_dir = os.path.expanduser("~/.localfit")

    try:
        with open(os.path.join(config_dir, "active_pod.json")) as f:
            pod = json.load(f)
        sessions.append(
            {
                "provider": "RunPod",
                "label": pod.get("model", "Remote pod"),
                "meta": pod.get("gpu", "GPU"),
                "detail": (
                    f"Started {int((time.time() - pod.get('started_at', time.time())) / 60)} min ago"
                    + (f" · {pod.get('quant')}" if pod.get("quant") else "")
                    + (f" · {pod.get('tunnel_url')}/v1" if pod.get("tunnel_url") else "")
                ),
            }
        )
    except Exception:
        pass

    try:
        with open(os.path.join(config_dir, "active_kaggle.json")) as f:
            kaggle = json.load(f)
        sessions.append(
            {
                "provider": "Kaggle",
                "label": kaggle.get("model", "Remote session"),
                "meta": kaggle.get("gpu", "T4"),
                "detail": (
                    f"Started {int((time.time() - kaggle.get('started_at', time.time())) / 60)} min ago"
                    + (f" · {kaggle.get('endpoint')}" if kaggle.get("endpoint") else "")
                ),
            }
        )
    except Exception:
        pass

    return sessions


def _arrow_tool_picker(title="Launch a tool", endpoint_info=None):
    """Tool picker using the same Rich Live pattern as the home menu."""
    from rich.text import Text
    from rich.panel import Panel
    from rich.live import Live
    from localfit.run_menu import _read_key

    tools = [
        ("Open WebUI", "webui", "ChatGPT-style browser UI"),
        ("Claude Code", "claude", "AI coding in terminal"),
        ("OpenCode", "opencode", "Terminal coding tool"),
        ("Codex", "codex", "OpenAI Codex CLI"),
        ("aider", "aider", "AI pair programming"),
    ]

    selected = 0

    try:
        with Live(console=console, auto_refresh=False, transient=True) as live:
            while True:
                text = Text()
                text.append(f"  {title}\n\n", style="bold")
                for i, (name, _, desc) in enumerate(tools):
                    if i == selected:
                        text.append(f"  › {name:<15}", style="bold cyan on grey23")
                        text.append(f" {desc}\n", style="dim on grey23")
                    else:
                        text.append(f"    {name:<15}", style="")
                        text.append(f" {desc}\n", style="dim")
                subtitle = endpoint_info or ""
                if subtitle:
                    subtitle += "  "
                subtitle += "[dim]↑↓/jk move · enter select · 1-9 jump · q back[/]"
                live.update(
                    Panel(text, border_style="cyan", width=min(console.width - 4, 55), subtitle=subtitle),
                    refresh=True,
                )

                key = _read_key()
                if key in ("up", "k"):
                    selected = max(0, selected - 1)
                elif key in ("down", "j"):
                    selected = min(len(tools) - 1, selected + 1)
                elif key == "enter":
                    return tools[selected][1]
                elif key in ("q", "esc", "ctrl-c"):
                    return None
                elif key.isdigit():
                    n = int(key) - 1
                    if 0 <= n < len(tools):
                        return tools[n][1]
    except Exception:
        # Fallback for non-interactive terminals
        console.print(f"  [bold]{title}[/]")
        for i, (name, _, desc) in enumerate(tools, 1):
            console.print(f"  [bold cyan]{i}[/]  {name} [dim]— {desc}[/]")
        try:
            pick = input("\n  > ").strip()
            if pick.isdigit():
                idx = int(pick) - 1
                if 0 <= idx < len(tools):
                    return tools[idx][1]
        except (EOFError, KeyboardInterrupt):
            pass
        return None


def _launch_tool_with_endpoint(tool, api_base, model_name="localmodel"):
    """Launch a tool connected to a specific API endpoint (local or remote). No model picker."""
    import subprocess, webbrowser

    console.print(f"  Connecting to: [cyan]{api_base}[/]")

    if tool in ("webui", "open-webui", "openwebui", "chat"):
        webui_port = 8080
        webui_dir = os.path.expanduser("~/.localfit/open-webui")
        os.makedirs(webui_dir, exist_ok=True)
        env = os.environ.copy()
        env["OPENAI_API_BASE_URL"] = api_base
        env["OPENAI_API_KEY"] = "no-key-required"
        env["ENABLE_OPENAI_API"] = "True"
        env["ENABLE_OLLAMA_API"] = "False"
        env["DATA_DIR"] = webui_dir
        env["DEFAULT_MODELS"] = model_name
        db_path = os.path.join(webui_dir, "webui.db")
        if not os.path.exists(db_path):
            env["WEBUI_AUTH"] = "False"
            console.print(f"  [dim]First run — no login required[/]")
        else:
            # Check existing users
            try:
                import sqlite3
                _db = sqlite3.connect(db_path)
                _users = _db.execute("SELECT email FROM user LIMIT 1").fetchall()
                _db.close()
                if _users:
                    console.print(f"  [dim]Login: {_users[0][0]}[/]")
                    console.print(f"  [dim]Forgot password? Run: localfit --launch webui --reset-auth[/]")
                    try:
                        _ans = input("  Skip auth for this session? (y/n): ").strip().lower()
                        if _ans == "y":
                            env["WEBUI_AUTH"] = "False"
                    except (EOFError, KeyboardInterrupt):
                        pass
            except Exception:
                pass

        try:
            subprocess.Popen(
                ["uv", "run", "--python", "3.11", "--with", "open-webui", "--", "open-webui", "serve", "--port", str(webui_port)],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            console.print(f"  [green]✓ Open WebUI starting on http://localhost:{webui_port}[/]")
            import time; time.sleep(3)
            webbrowser.open(f"http://localhost:{webui_port}")
        except FileNotFoundError:
            try:
                subprocess.Popen(
                    ["open-webui", "serve", "--port", str(webui_port)],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                console.print(f"  [green]✓ Open WebUI starting on http://localhost:{webui_port}[/]")
                import time; time.sleep(3)
                webbrowser.open(f"http://localhost:{webui_port}")
            except FileNotFoundError:
                console.print(f"  [red]Open WebUI not installed.[/]")
                console.print(f"  [dim]Install: pip install open-webui[/]")
                console.print(f"  [dim]Or: uv tool install open-webui[/]")

    elif tool in ("claude",):
        from localfit.proxy import PROXY_PORT, ensure_proxy_process
        from localfit.safe_config import get_claude_launch_env
        proxy_ready = ensure_proxy_process(llama_url=f"{api_base}/chat/completions", port=PROXY_PORT)
        if proxy_ready:
            subprocess.Popen(["claude", "--bare", "--model", model_name],
                env={**os.environ, **get_claude_launch_env(api_base=f"http://127.0.0.1:{PROXY_PORT}")})

    elif tool in ("opencode",):
        # Use env vars only — don't touch user's opencode config
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = api_base
        env["OPENAI_API_KEY"] = "no-key-required"
        console.print(f"  [green]✓[/] OpenCode → {api_base}")
        subprocess.Popen(["opencode"], env=env)

    elif tool in ("codex",):
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = api_base
        env["OPENAI_API_KEY"] = "no-key-required"
        subprocess.Popen(["codex", "--model", model_name, "-c", "model_provider=openai", "-c", "features.use_responses_api=false"], env=env)

    elif tool in ("aider",):
        env = os.environ.copy()
        env["OPENAI_API_BASE"] = api_base
        env["OPENAI_API_KEY"] = "no-key-required"
        subprocess.Popen(["aider", "--model", f"openai/{model_name}"], env=env)

    else:
        console.print(f"  [yellow]Tool '{tool}' not supported for direct endpoint launch[/]")


def _print_local_ready_hints(port=8089, api_model=None):
    """Show the home menu with model ACTIVE + TOOLS section. Same UI as `localfit`."""
    from localfit.home_menu import show_home_menu

    api_model = api_model or _detect_local_api_model(port)
    api_url = f"http://127.0.0.1:{port}/v1"

    from localfit.backends import get_machine_specs, get_metal_gpu_stats
    specs = get_machine_specs()
    metal = get_metal_gpu_stats()
    gpu_total = specs["gpu_total_mb"] if specs.get("cpu_only") else (metal.get("total_mb") or specs["gpu_total_mb"])

    system = {
        "subtitle": f"{specs.get('chip', 'GPU')}  {gpu_total // 1024}GB",
        "verdict": "SERVING",
        "color": "green",
        "gpu": f"{gpu_total // 1024}GB · {api_model or 'model'} loaded",
        "swap": "",
        "disk": "",
        "model": f"{api_model or '?'} on :{port}",
        "machine": f"{specs.get('chip', '?')} · {specs.get('ram_gb', '?')}GB",
    }

    items = [
        {
            "index": 1, "section": "ACTIVE", "label": api_model or "model",
            "meta": "local", "detail": f"Running on :{port}",
            "repo": f"local:{port}", "source": "local", "accent": "green",
            "badge": "●", "action": "noop", "selectable": False,
        },
    ]

    tools = [
        ("Open WebUI", "webui", "ChatGPT-style browser UI"),
        ("Claude Code", "claude", "AI coding assistant"),
        ("OpenCode", "opencode", "Terminal coding tool"),
        ("Codex", "codex", "OpenAI Codex CLI"),
        ("aider", "aider", "AI pair programming"),
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
                _launch_tool(tool_id, api_model)
                return


def _serve_model(model_query, background=False):
    """Download + start llama-server for a model.

    Args:
        background: If True, start server and return immediately (used by --launch).

    """
    if not background:
        _show_logo_intro()
    from localfit.backends import (
        MODELS,
        find_model_file,
        start_llama_server,
        fetch_hf_model,
        _download_gguf,
        get_machine_specs,
        resolve_model_family,
    )

    specs = get_machine_specs()
    gpu_total = specs["gpu_total_mb"]
    original_query = model_query
    model_query = _normalize_model_query_for_discovery(model_query)
    if model_query != original_query:
        console.print(f"  [dim]{original_query} → {model_query}[/]")

    # Resolve family alias (e.g. "gemma4" → "gemma4-e4b" based on GPU)
    from localfit.backends import MODEL_FAMILIES

    resolved = resolve_model_family(model_query, gpu_total)
    if resolved:
        m = MODELS[resolved]
        family_key = (
            model_query.lower().split(":")[0] if ":" not in model_query else None
        )

        # Check if the resolved model (or explicitly requested one) is too big
        if m["size_gb"] * 1024 > gpu_total:
            gpu_gb = gpu_total // 1024
            console.print(
                f"\n  [red bold]✗ {m['name']} ({m['size_gb']}GB) won't fit on your {gpu_gb}GB GPU[/]"
            )

            # Suggest a variant that does fit locally
            local_alt = None
            if family_key and family_key in MODEL_FAMILIES:
                for alt_mid in MODEL_FAMILIES[family_key]:
                    alt = MODELS[alt_mid]
                    if alt["size_gb"] * 1024 <= gpu_total:
                        local_alt = alt_mid
                        console.print(
                            f"  [green]→ Smaller variant: {alt_mid}[/] ({alt['size_gb']}GB, fits locally)"
                        )
                        break

            # Auto-offer Kaggle/RunPod — Ollama-style, just ask
            _offer_remote(model_query, m["size_gb"])
            return

        if family_key and family_key in MODEL_FAMILIES:
            # Show what was picked and why
            console.print(
                f"\n  [bold]{model_query}[/] → [green]{resolved}[/] ({m['size_gb']}GB)"
            )
            console.print(f"  [dim]{m['description']}[/]")
            headroom = gpu_total // 1024 - int(m["size_gb"])
            console.print(
                f"  [dim]Fits your {gpu_total // 1024}GB GPU ({headroom}GB headroom)[/]"
            )
        else:
            console.print(f"  [dim]{model_query} → {resolved}[/]")
        model_query = resolved

    # Check known models first
    for mid, m in MODELS.items():
        if (
            model_query.lower() in mid.lower()
            or model_query.lower() in m["name"].lower()
        ):
            console.print(f"\n  [bold]Serving {m['name']}...[/]")
            model_file = find_model_file(mid)
            if not model_file:
                console.print(f"  [yellow]Not downloaded. Downloading...[/]")
                from localfit.backends import download_model_hf

                download_model_hf(mid)
                model_file = find_model_file(mid)
            if model_file:
                proc = start_llama_server(mid)
                if proc:
                    console.print(f"  [green]✓ Running on :8089[/]")
                    console.print(f"  [dim]API: http://127.0.0.1:8089/v1[/]")
                    if not background:
                        _print_local_ready_hints(port=8089)
                    if not background:
                        try:
                            proc.wait()
                        except KeyboardInterrupt:
                            proc.kill()
            return

    # ── Unified model picker: MLX + GGUF + Remote in one menu ──
    from localfit.run_menu import collect_options, show_run_menu
    from localfit.backends import (
        check_mlx_available, start_mlx_server,
        stop_conflicting_backends, convert_to_mlx,
    )

    console.print(f"\n  [dim]Searching backends for {model_query}...[/]")
    local_opts, remote_opts, recommended, menu_meta = collect_options(model_query, specs)

    if local_opts or remote_opts:
        # Cross-platform: Mac uses "chip" (Apple M4 Pro), Linux uses nvidia-smi GPU name
        chip = specs.get("chip", "")
        gpu_gb = round(specs.get("gpu_total_mb", 0) / 1024, 1)
        if chip and chip != "Unknown":
            hw_label = f"{chip}  {gpu_gb}GB"
        elif specs.get("gpu_name"):
            hw_label = f"{specs['gpu_name']}  {gpu_gb}GB VRAM"
        else:
            hw_label = f"{gpu_gb}GB GPU" if gpu_gb > 0 else "CPU only"

        choice = show_run_menu(model_query, hw_label, local_opts, remote_opts, recommended)
        if choice in (None, "back"):
            return
        if choice == "quit":
            return "quit"

        all_opts = local_opts + remote_opts
        if choice < 1 or choice > len(all_opts):
            console.print(f"  [red]Invalid choice[/]")
            return

        opt = all_opts[choice - 1]
        action = opt.get("action")

        if action == "mlx":
            stop_conflicting_backends("mlx")
            proc = start_mlx_server(opt["repo"], port=8080)
            if proc:
                if not background:
                    try:
                        proc.wait()
                    except KeyboardInterrupt:
                        proc.kill()
                return
            console.print(f"  [yellow]MLX failed — try a GGUF option[/]")
            return

        elif action == "gguf":
            # Download and serve via llama-server
            data = opt.get("hf_data") or menu_meta.get("hf_data")
            if data:
                path = _download_gguf(opt["repo"], opt["filename"])
                if path:
                    from localfit.prerequisites import ensure_llama_server
                    binary = ensure_llama_server()
                    if not binary:
                        return
                    is_cpu_only = specs.get("cpu_only", False)
                    ngl = "0" if is_cpu_only else "99"
                    # Dynamic context
                    model_mb = int(opt["size_gb"] * 1024)
                    gpu_free = specs.get("gpu_free_mb", gpu_total)
                    free_for_kv = max(0, gpu_free - model_mb - 512)
                    kv_per_1k = 60
                    max_ctx = int(free_for_kv / kv_per_1k * 1024)
                    max_ctx = max(4096, min(max_ctx, 131072))
                    for nice in [131072, 65536, 32768, 16384, 8192, 4096]:
                        if max_ctx >= nice:
                            max_ctx = nice
                            break
                    cmd = [binary, "-m", path, "--port", "8089", "-ngl", ngl, "-c", str(max_ctx), "--jinja"]
                    # VLM mmproj
                    if data.get("is_vlm") and data.get("mmproj_files"):
                        mmproj_name = data["mmproj_files"][0]["filename"]
                        mmproj_path = _download_gguf(opt["repo"], mmproj_name)
                        if mmproj_path:
                            cmd += ["--mmproj", mmproj_path]
                    import tempfile, subprocess as _sp
                    stderr_log = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, prefix="llama-")
                    env = os.environ.copy()
                    bin_dir = os.path.dirname(binary)
                    ld = env.get("LD_LIBRARY_PATH", "")
                    if bin_dir not in ld:
                        env["LD_LIBRARY_PATH"] = bin_dir + (":" + ld if ld else "")
                    console.print(f"  [dim]Starting llama-server...[/]")
                    proc = _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=stderr_log, env=env)
                    import time
                    for i in range(90):
                        if proc.poll() is not None:
                            break
                        try:
                            import urllib.request as _ur
                            _ur.urlopen(f"http://127.0.0.1:8089/health", timeout=1)
                            console.print(f"  [green]✓ Running on :8089[/]")
                            console.print(f"  [dim]API: http://127.0.0.1:8089/v1[/]")
                            if not background:
                                _print_local_ready_hints(port=8089)
                            if not background:
                                try:
                                    proc.wait()
                                except KeyboardInterrupt:
                                    proc.kill()
                            return
                        except Exception:
                            time.sleep(1)
                    console.print(f"  [red]Server did not start[/]")
                    return

        elif action == "kaggle":
            console.print(f"\n  [cyan]localfit run {model_query} --remote kaggle[/]")
            return

        elif action == "runpod":
            console.print(f"\n  [cyan]localfit run {model_query} --cloud[/]")
            return

        return

    # Fallback: original HuggingFace search path (for when menu found nothing)
    console.print(f"\n  [bold]Searching HuggingFace for {model_query}...[/]")
    data = fetch_hf_model(model_query)
    if data and data["gguf_files"]:
        is_cpu_only = specs.get("cpu_only", False)
        is_vlm = data.get("is_vlm", False)
        mmproj_files = data.get("mmproj_files", [])
        mmproj_overhead = mmproj_files[0]["size_gb"] if (is_vlm and mmproj_files) else 0

        # VLM shortcut: check if there's a pre-built Ollama model with vision baked in
        # This avoids the GGUF + mmproj hassle — Ollama handles everything natively
        if is_vlm:
            from localfit.backends import _check_ollama_registry
            import subprocess as _sp

            repo_name = data.get("name", data["repo_id"].split("/")[-1])
            console.print(
                f"  [magenta]Vision model detected — checking Ollama for pre-built VLM...[/]"
            )
            # Search common Ollama tag patterns for this model
            ollama_result = None
            ollama_tag = None
            search_terms = [
                repo_name.lower().replace("_", "-"),
                repo_name.lower().replace("-", "").replace("_", ""),
            ]
            for term in search_terms:
                ollama_result = _check_ollama_registry(term)
                if ollama_result:
                    ollama_tag = ollama_result["tag"]
                    break

            # Also check if already installed in Ollama
            if not ollama_tag:
                try:
                    r = _sp.run(
                        ["ollama", "list"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    for line in r.stdout.strip().split("\n"):
                        name_lower = line.lower()
                        for term in search_terms:
                            if term in name_lower:
                                ollama_tag = line.split()[0]
                                break
                        if ollama_tag:
                            break
                except Exception:
                    pass

            if ollama_tag:
                size_info = ollama_result["size_gb"] if ollama_result else "?"
                console.print(
                    f"  [green]Found Ollama model:[/] [bold]{ollama_tag}[/] ({size_info}GB, vision built-in)"
                )
                console.print(
                    f"  [dim]No mmproj needed — Ollama handles vision natively[/]"
                )

                # Pull and serve
                console.print(f"\n  [bold]Pulling {ollama_tag}...[/]")
                pull_r = _sp.run(["ollama", "pull", ollama_tag], timeout=1800)
                if pull_r.returncode == 0:
                    console.print(f"  [green]✓ Pulled {ollama_tag}[/]")
                    console.print(f"  [dim]Loading model into GPU...[/]")
                    try:
                        _sp.run(
                            ["ollama", "run", ollama_tag, "hi"],
                            capture_output=True,
                            timeout=120,
                        )
                    except Exception:
                        pass
                    console.print(f"  [green]✓ Model loaded and ready[/]")
                    console.print(f"\n  [bold]Serving via Ollama[/]")
                    console.print(f"  API:   http://127.0.0.1:11434/v1")
                    console.print(f"  Model: {ollama_tag}")
                    return
                else:
                    console.print(
                        f"  [yellow]Ollama pull failed — falling back to GGUF + llama-server[/]"
                    )

        # Safety: use FREE gpu memory (not total) and reserve headroom for KV cache + OS
        # On macOS unified memory, OOM = full system crash, so be conservative
        gpu_free = specs.get("gpu_free_mb", gpu_total)
        safety_margin_mb = 2048  # 2GB for KV cache + OS headroom
        usable_mb = gpu_free - safety_margin_mb

        fits = [
            f
            for f in data["gguf_files"]
            if (f["size_gb"] + mmproj_overhead) * 1024 < usable_mb
        ]
        best = fits[-1] if fits else None
        if best:
            total_size = best["size_gb"] + mmproj_overhead
            speed_note = " (CPU-only, ~3-8 tok/s)" if is_cpu_only else ""
            vlm_note = f" + mmproj {mmproj_overhead}GB" if mmproj_overhead else ""
            headroom_gb = round((usable_mb / 1024) - total_size, 1)
            console.print(
                f"  Best fit: {best['quant']} ({best['size_gb']}GB{vlm_note}){speed_note}"
            )
            console.print(
                f"  [dim]GPU free: {gpu_free // 1024}GB · Model: {total_size}GB · Headroom: {headroom_gb}GB[/]"
            )
            path = _download_gguf(data["repo_id"], best["filename"])
            if path:
                # Download mmproj for VLM models
                mmproj_path = None
                if is_vlm and mmproj_files:
                    mmproj_name = mmproj_files[0]["filename"]
                    console.print(
                        f"  [dim]Downloading vision projector {mmproj_name}...[/]"
                    )
                    mmproj_path = _download_gguf(data["repo_id"], mmproj_name)
                    if not mmproj_path:
                        console.print(
                            f"  [yellow]mmproj download failed — model will be text-only[/]"
                        )

                # Ensure llama-server is installed
                from localfit.prerequisites import ensure_llama_server

                binary = ensure_llama_server()
                if not binary:
                    return

                console.print(f"  [dim]Starting server...[/]")
                ngl = "0" if is_cpu_only else "99"
                if is_cpu_only:
                    ctx = "8192"
                else:
                    # Dynamic context: fit KV cache into remaining VRAM
                    model_mb = int(total_size * 1024)
                    free_for_kv = max(0, gpu_free - model_mb - 512)
                    kv_per_1k = 60  # MB per 1K tokens with Q4_0 KV
                    max_ctx = int(free_for_kv / kv_per_1k * 1024)
                    max_ctx = max(4096, min(max_ctx, 131072))
                    for nice in [131072, 65536, 32768, 16384, 8192, 4096]:
                        if max_ctx >= nice:
                            max_ctx = nice
                            break
                    ctx = str(max_ctx)
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
                ]
                if mmproj_path:
                    cmd += ["--mmproj", mmproj_path]

                # Pre-launch safety: re-check free memory right before Popen
                # On macOS unified memory, OOM = full system crash
                fresh_specs = get_machine_specs()
                fresh_free_mb = fresh_specs.get("gpu_free_mb", 0)
                model_total_mb = int(total_size * 1024)
                min_needed_mb = model_total_mb + 1024  # model + 1GB minimum headroom
                if fresh_free_mb > 0 and fresh_free_mb < min_needed_mb:
                    console.print(
                        f"\n  [red bold]✗ Not enough free memory to launch safely[/]"
                    )
                    console.print(
                        f"  [red]Need {min_needed_mb // 1024}GB free, only {fresh_free_mb // 1024}GB available[/]"
                    )
                    console.print(
                        f"  [dim]Close other apps or use a smaller quant to avoid crashing your system[/]"
                    )
                    return

                # Set LD_LIBRARY_PATH for pre-built binaries
                env = os.environ.copy()
                lib_dir = os.path.join(os.path.dirname(binary), "lib")
                if os.path.isdir(lib_dir):
                    ld = env.get("LD_LIBRARY_PATH", "")
                    env["LD_LIBRARY_PATH"] = lib_dir + (":" + ld if ld else "")

                import subprocess

                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
                )
                console.print(f"  [green]✓ Running on :8089[/]")
                if mmproj_path:
                    console.print(f"  [magenta]✓ Vision projector loaded[/]")
                console.print(f"  [dim]API: http://127.0.0.1:8089/v1[/]")
                if not background:
                    _print_local_ready_hints(port=8089)
                if not background:
                    try:
                        proc.wait()
                    except KeyboardInterrupt:
                        proc.kill()
        else:
            # Can't run locally — offer remote
            console.print(
                f"\n  [red bold]✗ Can't run locally[/] — no quant fits your "
                f"{specs['ram_gb']}GB {'RAM' if is_cpu_only else 'GPU'}"
            )
            console.print(f"  [dim]Let's try remotely.[/]\n")
            _offer_remote(model_query, hf_data=data)
    else:
        console.print(f"  [red]Model not found[/]")


def _config_tool(tool_name):
    """Auto-configure a coding tool for local llama-server."""
    tool = tool_name.lower().strip()

    configs = {
        "opencode": {
            "path": os.path.expanduser("~/.config/opencode/opencode.json"),
            "config": {
                "$schema": "https://opencode.ai/config.json",
                "model": "llamacpp/local",
                "provider": {
                    "llamacpp": {
                        "name": "llama.cpp (local via localfit)",
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": "http://127.0.0.1:8089/v1"},
                        "models": {
                            "local": {
                                "name": "Local Model (localfit)",
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
            },
        },
        "aider": {
            "path": os.path.expanduser("~/.aider.conf.yml"),
            "content": "openai-api-base: http://127.0.0.1:8089/v1\nopenai-api-key: local\nmodel: openai/local\n",
        },
        "openclaw": {
            "path": os.path.expanduser("~/.openclaw/openclaw.json"),
            "config": {
                "provider": {
                    "llamacpp": {
                        "name": "llama.cpp (local via localfit)",
                        "apiBase": "http://127.0.0.1:8089/v1",
                    }
                },
            },
        },
        "cursor": {
            "path": None,
            "instructions": "In Cursor Settings → Models → OpenAI API:\n  Base URL: http://127.0.0.1:8089/v1\n  API Key: local\n  Model: local",
        },
        "claude": {
            "type": "claude",
        },
        "codex": {
            "type": "codex",
        },
        "localcoder": {
            "type": "skip",  # localcoder auto-configures via --api flag
        },
        "hermes": {
            "type": "hermes",
        },
    }

    if tool not in configs:
        console.print(f"\n  [red]Unknown tool: {tool}[/]")
        console.print(f"  [dim]Supported: {', '.join(configs.keys())}[/]\n")
        return

    cfg = configs[tool]
    if cfg.get("instructions"):
        console.print(f"\n  [bold]Configure {tool}:[/]")
        console.print(f"  {cfg['instructions']}\n")
        return

    # Claude Code — special handling (Unsloth guide)
    if cfg.get("type") == "claude":
        _config_claude()
        return

    # OpenAI Codex — special handling
    if cfg.get("type") == "codex":
        _config_codex()
        return

    # Hermes Agent — special handling
    if cfg.get("type") == "hermes":
        _config_hermes()
        return

    # Tools that configure via CLI flags (no config file needed)
    if cfg.get("type") == "skip":
        return

    path = cfg["path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if "content" in cfg:
        with open(path, "w") as f:
            f.write(cfg["content"])
    else:
        with open(path, "w") as f:
            json.dump(cfg["config"], f, indent=2)

    console.print(f"\n  [green]✓[/] Configured {tool}")
    console.print(f"  [dim]Written: {path}[/]")
    console.print(f"  [dim]API: http://127.0.0.1:8089/v1[/]")
    console.print(f"\n  [bold]Start model:[/] localfit --serve gemma4-26b")
    console.print(f"  [bold]Then run:[/] {tool}\n")


def _launch_tool(tool_name, model_name=None, tunnel=False):
    """Start model + launch coding tool in one command.

    Like: ollama launch claude --model qwen3.5
    Ours: localfit --launch claude --serve gemma4-26b
    """
    import subprocess, shutil

    tool = tool_name.lower().strip()
    console.print(f"\n  [bold #e07a5f]localfit launch {tool}[/]")

    # 1. Ensure llama-server is running
    from localfit.backends import (
        check_backend_running,
        get_llama_server_config,
        get_disk_info,
        MODELS,
    )

    srv = get_llama_server_config()

    # If user requested a specific model but a different one is running, switch
    if srv["running"] and model_name:
        from localfit.backends import (
            _detect_model_info,
            resolve_model_family,
            get_machine_specs,
        )

        current = _detect_model_info(srv, None)
        current_name = (current.get("name") or "").lower().replace(" ", "-")
        specs = get_machine_specs()
        requested = (
            resolve_model_family(model_name, specs["gpu_total_mb"]) or model_name
        )
        # Check if it's a different model
        if (
            requested.lower() not in current_name
            and model_name.lower().replace(":", "-") not in current_name
        ):
            console.print(
                f"  [yellow]Switching model: {current_name} → {model_name}[/]"
            )
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
            import time

            time.sleep(2)
            srv = {"running": False}

    if not srv["running"]:
        if model_name:
            console.print(f"  [dim]Starting model: {model_name}...[/]")
            _serve_model(model_name, background=True)
        else:
            # Show installed models for quick pick
            di = get_disk_info()
            installed = di.get("models", [])
            known = list(MODELS.items())

            from localfit.backends import (
                fetch_unsloth_top_models,
                fetch_hf_model,
                _download_gguf,
                get_machine_specs,
            )
            from rich.markup import escape as _esc

            specs = get_machine_specs()
            gpu_total = specs["gpu_total_mb"]

            console.print(
                f"\n  [bold]Pick a model:[/]  [dim]{specs['chip']} · {gpu_total // 1024}GB GPU[/]\n"
            )
            all_choices = []

            # Installed models first
            if installed:
                console.print(f"  [dim]── Installed ──[/]")
                for i, m in enumerate(installed[:5], 1):
                    name = m["name"].replace(".gguf", "")
                    console.print(
                        f"  [green]{i}[/]  {_esc(name):<30} [dim]{m['size_gb']}GB  ready[/]"
                    )
                    all_choices.append(("installed", m))

            # Top HuggingFace models (not installed)
            try:
                hf_models = fetch_unsloth_top_models(limit=10)
                installed_bases = {
                    m["name"].lower().replace(".gguf", "").replace("-", "")
                    for m in installed
                }

                available = []
                for hm in hf_models:
                    base = hm["label"].lower().replace("-", "")
                    if not any(base in ib for ib in installed_bases):
                        available.append(hm)

                if available:
                    console.print(f"\n  [dim]── Available (auto-download) ──[/]")
                    for j, hm in enumerate(available[:6], len(all_choices) + 1):
                        dl = hm["downloads"]
                        dl_str = (
                            f"{dl // 1000}K"
                            if dl < 1_000_000
                            else f"{dl / 1_000_000:.1f}M"
                        )
                        est = hm.get("est_smallest_gb")
                        fit_tag = ""
                        if est:
                            fit_tag = (
                                f" [green]fits[/]"
                                if est * 1024 < gpu_total
                                else f" [red]won't fit[/]"
                            )
                        caps = " ".join(
                            f"[{'magenta' if c == 'vision' else 'cyan' if c == 'code' else 'green'}]{c}[/]"
                            for c in hm.get("caps", [])
                        )
                        console.print(
                            f"  [cyan]{j}[/]  {_esc(hm['label']):<30} [dim]{dl_str} dl[/] {caps}{fit_tag}"
                        )
                        all_choices.append(("hf", hm))
            except Exception:
                pass

            console.print(f"\n  [dim]Or type a HuggingFace repo: unsloth/MODEL-GGUF[/]")
            console.print()
            try:
                ans = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                return

            try:
                idx = int(ans) - 1
                if 0 <= idx < len(all_choices):
                    kind, data = all_choices[idx]
                    if kind == "installed":
                        model_name = data["path"]
                    elif kind == "hf":
                        # Download best quant then serve
                        repo = data["repo_id"]
                        console.print(f"\n  [bold]Fetching {data['label']}...[/]")
                        hf_data = fetch_hf_model(repo)
                        if hf_data and hf_data["gguf_files"]:
                            fits = [
                                f
                                for f in hf_data["gguf_files"]
                                if f["size_gb"] * 1024 < gpu_total
                            ]
                            best = fits[-1] if fits else None
                            if best:
                                console.print(
                                    f"  Best quant: {best['quant']} ({best['size_gb']}GB)"
                                )
                                console.print(f"  [dim]Downloading...[/]")
                                path = _download_gguf(repo, best["filename"])
                                if path:
                                    model_name = path
                                else:
                                    console.print(f"  [red]Download failed[/]")
                                    return
                            else:
                                console.print(
                                    f"  [red]No quant fits your {gpu_total // 1024}GB GPU[/]"
                                )
                                return
                        else:
                            console.print(f"  [red]No GGUF files found[/]")
                            return
                    else:
                        model_name = data
            except (ValueError, IndexError):
                # Typed a repo name or model name
                if "/" in ans:
                    # HuggingFace repo — download best quant
                    console.print(f"\n  [bold]Fetching {ans}...[/]")
                    hf_data = fetch_hf_model(ans)
                    if hf_data and hf_data["gguf_files"]:
                        fits = [
                            f
                            for f in hf_data["gguf_files"]
                            if f["size_gb"] * 1024 < gpu_total
                        ]
                        best = fits[-1] if fits else hf_data["gguf_files"][0]
                        console.print(f"  Best: {best['quant']} ({best['size_gb']}GB)")
                        path = _download_gguf(hf_data["repo_id"], best["filename"])
                        if path:
                            model_name = path
                        else:
                            return
                    else:
                        console.print(f"  [red]Not found[/]")
                        return
                else:
                    model_name = ans

            if model_name:
                console.print(f"  [dim]Starting model...[/]")
                if model_name.startswith("/"):
                    binary = os.path.expanduser("~/.unsloth/llama.cpp/llama-server")
                    import subprocess as _sp

                    _sp.Popen(
                        [
                            binary,
                            "-m",
                            model_name,
                            "--port",
                            "8089",
                            "-ngl",
                            "99",
                            "-c",
                            "8192",
                            "-fa",
                            "on",
                            "-ctk",
                            "q4_0",
                            "-ctv",
                            "q4_0",
                            "--jinja",
                        ],
                        stdout=_sp.DEVNULL,
                        stderr=_sp.DEVNULL,
                    )
                    import time as _t

                    for _ in range(30):
                        try:
                            import urllib.request

                            urllib.request.urlopen(
                                "http://127.0.0.1:8089/health", timeout=1
                            )
                            break
                        except:
                            _t.sleep(1)
                else:
                    _serve_model(model_name, background=True)
            else:
                return
    else:
        console.print(f"  [green]✓[/] llama-server already running")

    # Get server port
    srv = get_llama_server_config()
    port = 8089  # default

    # 2. Configure the tool silently
    _config_tool(tool)

    # 3. Launch the tool
    console.print(f"\n  [bold]Launching {tool}...[/]\n")

    # Detect model name from running server
    model_alias = "local"
    srv = get_llama_server_config()
    if srv.get("running"):
        from localfit.backends import _detect_model_info

        mi = _detect_model_info(srv, None)
        if mi.get("name"):
            # Build a clean Unsloth-style alias: "unsloth/Gemma-4-26B"
            name = mi["name"].replace(" ", "-")
            quant = mi.get("quant", "")
            model_alias = f"unsloth/{name}"
            console.print(
                f"  [cyan]Model: {name} {quant}[/]  →  [bold]{model_alias}[/]"
            )
        else:
            # Fallback: use the model filename
            import urllib.request, json as _json

            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=3
                ) as resp:
                    data = _json.loads(resp.read())
                if data.get("data"):
                    model_alias = data["data"][0].get("id", "local")
                    console.print(f"  [cyan]Model: {model_alias}[/]")
            except Exception:
                pass

    env = os.environ.copy()

    if tool == "claude":
        from localfit.proxy import PROXY_PORT, ensure_proxy_process
        from localfit.safe_config import get_claude_launch_cmd, get_claude_launch_env

        proxy_ready = ensure_proxy_process(
            llama_url=f"http://127.0.0.1:{port}/v1/chat/completions",
            port=PROXY_PORT,
        )
        if not proxy_ready:
            console.print(f"  [red]Claude proxy failed to start on :{PROXY_PORT}.[/]")
            console.print(
                f"  [dim]Claude needs the Anthropic Messages proxy, not llama-server's raw OpenAI endpoint.[/]"
            )
            return

        env.update(get_claude_launch_env(api_base=f"http://127.0.0.1:{PROXY_PORT}"))
        # --bare skips OAuth entirely — no need to touch any Claude config files
        cmd = get_claude_launch_cmd(model_alias)

        claude_bin = shutil.which("claude")
        if not claude_bin:
            console.print(f"  [red]Claude Code not installed.[/]")
            console.print(
                f"  [dim]Install: curl -fsSL https://claude.ai/install.sh | bash[/]\n"
            )
            return

        console.print(f"  [bold]Launching:[/] claude --bare --model {model_alias}")
        console.print(f"  [dim]Using local model (your Claude login is untouched)[/]")
        console.print(
            f"  [dim]Anthropic proxy: http://127.0.0.1:{PROXY_PORT} → llama-server :{port}[/]"
        )
        console.print()
        os.execvpe(cmd[0], cmd, env)

    elif tool == "codex":
        env["OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"
        env["OPENAI_API_KEY"] = "sk-no-key-required"
        # Use chat completions endpoint — streaming on /v1/responses hangs with llama-server
        cmd = [
            "codex",
            "--model",
            model_alias,
            "-c",
            "model_provider=openai",
            "-c",
            "features.use_responses_api=false",
        ]

        codex_bin = shutil.which("codex")
        if not codex_bin:
            console.print(
                f"  [red]Codex not installed. Run: npm install -g @openai/codex[/]\n"
            )
            return

        console.print(f"  [bold]Launching:[/] codex --model {model_alias}")
        console.print()
        os.execvpe(cmd[0], cmd, env)

    elif tool == "opencode":
        cmd = ["opencode"]
        opencode_bin = shutil.which("opencode")
        if not opencode_bin:
            console.print(f"  [red]OpenCode not installed.[/]\n")
            return
        os.execvpe(cmd[0], cmd, env)

    elif tool == "aider":
        cmd = [
            "aider",
            "--openai-api-base",
            f"http://localhost:{port}/v1",
            "--openai-api-key",
            "local",
            "--model",
            "openai/local",
        ]
        aider_bin = shutil.which("aider")
        if not aider_bin:
            console.print(
                f"  [red]aider not installed. Run: pip install aider-chat[/]\n"
            )
            return
        os.execvpe(cmd[0], cmd, env)

    elif tool == "localcoder":
        # Use the original model name (e.g. "gemma4-e4b") not the GGUF filename
        # Resolve colon syntax: gemma4:e4b → gemma4-e4b
        lc_model = (model_name or model_alias).replace(":", "-")
        cmd = ["localcoder", "--api", f"http://localhost:{port}", "-m", lc_model]

        localcoder_bin = shutil.which("localcoder")
        if not localcoder_bin:
            console.print(f"  [red]localcoder not installed.[/]")
            console.print(f"  [dim]Install: pipx install localcoder[/]\n")
            return

        console.print(
            f"  [bold]Launching:[/] localcoder --api http://localhost:{port} -m {lc_model}"
        )
        console.print()
        os.execvpe(cmd[0], cmd, env)

    elif tool == "hermes":
        env["OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"
        env["OPENAI_API_KEY"] = "no-key-required"
        _config_hermes()

        hermes_bin = shutil.which("hermes")
        if not hermes_bin:
            console.print(f"  [red]Hermes Agent not installed.[/]")
            console.print(
                f"  [dim]Install: curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash[/]\n"
            )
            return

        console.print(f"  [bold]Launching:[/] hermes (model: {model_alias})")
        console.print()
        os.execvpe("hermes", ["hermes"], env)

    elif tool in ("webui", "open-webui", "openwebui", "chat"):
        # Smart endpoint picker: detect local + remote models
        webui_port = 8080
        api_base = None
        import urllib.request as _wr

        endpoints = []

        # Check local llama-server
        try:
            _wr.urlopen(f"http://localhost:8089/health", timeout=1)
            endpoints.append(("Local llama-server (:8089)", "http://localhost:8089/v1"))
        except Exception:
            pass

        # Check local MLX server
        try:
            _wr.urlopen(f"http://localhost:8080/v1/models", timeout=1)
            endpoints.append(("Local MLX server (:8080)", "http://localhost:8080/v1"))
        except Exception:
            pass

        # Check local Ollama
        try:
            _wr.urlopen(f"http://localhost:11434/api/tags", timeout=1)
            endpoints.append(("Local Ollama (:11434)", "http://localhost:11434/v1"))
        except Exception:
            pass

        # Check remote Kaggle session
        from pathlib import Path as _Path
        kaggle_state = _Path.home() / ".localfit" / "active_kaggle.json"
        if kaggle_state.exists():
            try:
                state = json.loads(kaggle_state.read_text())
                ep = state.get("endpoint")
                if ep:
                    endpoints.append((f"Kaggle remote ({state.get('model', '?')})", f"{ep}/v1"))
            except Exception:
                pass

        if not endpoints:
            console.print(f"\n  [yellow]No running models found.[/]")
            console.print(f"  Start one first: [cyan]localfit run MODEL[/]")
            console.print(f"  Or remote:       [cyan]localfit run MODEL --remote kaggle[/]\n")
            return

        if len(endpoints) == 1:
            api_base = endpoints[0][1]
            console.print(f"  [green]✓[/] Using: {endpoints[0][0]}")
        else:
            console.print(f"\n  [bold]Connect Open WebUI to:[/]\n")
            for i, (label, url) in enumerate(endpoints, 1):
                console.print(f"  [bold cyan]{i}[/]  {label}")
                console.print(f"     [dim]{url}[/]")
            console.print()
            try:
                pick = input(f"  Pick [1-{len(endpoints)}]: ").strip()
                idx = int(pick) - 1
                api_base = endpoints[idx][1]
            except (ValueError, IndexError, EOFError, KeyboardInterrupt):
                api_base = endpoints[0][1]
            console.print(f"  [green]✓[/] Using: {api_base}")

        # Check if Open WebUI already running
        try:
            _wr.urlopen(f"http://localhost:{webui_port}", timeout=2)
            console.print(f"  [green]✓[/] Open WebUI already running on http://localhost:{webui_port}")
            console.print(f"  [dim]Opening browser...[/]")
            import webbrowser
            webbrowser.open(f"http://localhost:{webui_port}")
            return
        except Exception:
            pass

        console.print(f"  [dim]Starting Open WebUI...[/]")

        webui_dir = os.path.expanduser("~/.localfit/open-webui")
        os.makedirs(webui_dir, exist_ok=True)

        webui_env = env.copy()
        webui_env["OPENAI_API_BASE_URL"] = api_base
        webui_env["OPENAI_API_KEY"] = "no-key-required"
        webui_env["ENABLE_OPENAI_API"] = "True"
        webui_env["ENABLE_OLLAMA_API"] = "False"
        webui_env["DATA_DIR"] = webui_dir
        webui_env["DEFAULT_MODELS"] = model_alias
        # Enable web search via DuckDuckGo (no API key needed)
        webui_env["ENABLE_RAG_WEB_SEARCH"] = "True"
        webui_env["RAG_WEB_SEARCH_ENGINE"] = "duckduckgo"
        webui_env["ENABLE_SEARCH_QUERY_GENERATION"] = "True"
        webui_env["ENABLE_WEB_SEARCH"] = "True"
        webui_env["WEB_SEARCH_ENGINE"] = "duckduckgo"
        # Only disable auth on fresh installs (no existing users)
        db_path = os.path.join(webui_dir, "webui.db")
        if not os.path.exists(db_path):
            webui_env["WEBUI_AUTH"] = "False"

        # Check if uv + python 3.11 available
        try:
            r = subprocess.run(
                [
                    "uv",
                    "run",
                    "--python",
                    "3.11",
                    "--with",
                    "open-webui",
                    "--",
                    "open-webui",
                    "--help",
                ],
                capture_output=True,
                timeout=30,
                env=webui_env,
            )
            has_uv = r.returncode == 0
        except Exception:
            has_uv = False

        if not has_uv:
            console.print(f"  [red]Open WebUI requires uv + Python 3.11[/]")
            console.print(
                f"  [dim]Install: curl -LsSf https://astral.sh/uv/install.sh | sh[/]"
            )
            console.print(f"  [dim]Then: uv python install 3.11[/]\n")
            return

        console.print(
            f"  [green]✓[/] Open WebUI starting on http://localhost:{webui_port}"
        )
        console.print(f"  [dim]Model: {model_alias}[/]")
        console.print(f"  [dim]API: http://localhost:{port}/v1[/]")
        console.print()

        # Start tunnel if requested
        tunnel_proc = None
        if tunnel:
            from localfit.prerequisites import check_cloudflared, ensure_cloudflared

            cf = check_cloudflared()
            cf_bin = cf.get("path") if cf["found"] else ensure_cloudflared()
            if cf_bin:
                tunnel_proc = subprocess.Popen(
                    [cf_bin, "tunnel", "--url", f"http://localhost:{webui_port}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                # Read tunnel URL from output
                import threading

                def _read_tunnel():
                    for line in tunnel_proc.stdout:
                        line = line.decode().strip()
                        if "trycloudflare.com" in line or "cfargotunnel.com" in line:
                            url = [w for w in line.split() if "http" in w]
                            if url:
                                console.print(
                                    f"\n  [bold green]🌐 Public URL: {url[0]}[/]"
                                )
                                console.print(
                                    f"  [dim]Share this link — anyone can access your AI chat[/]\n"
                                )

                threading.Thread(target=_read_tunnel, daemon=True).start()
                console.print(f"  [dim]Starting Cloudflare Tunnel...[/]")
            else:
                console.print(
                    f"  [yellow]Skipping tunnel (cloudflared not available)[/]"
                )

        # Open browser after short delay
        import webbrowser, threading

        threading.Timer(
            5, lambda: webbrowser.open(f"http://localhost:{webui_port}")
        ).start()

        # Run (blocks)
        try:
            subprocess.run(
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
                    "--port",
                    str(webui_port),
                ],
                env=webui_env,
            )
        except KeyboardInterrupt:
            console.print(f"\n  [dim]Open WebUI stopped[/]")
        finally:
            if tunnel_proc:
                tunnel_proc.kill()
                console.print(f"  [dim]Tunnel closed[/]")

    else:
        console.print(f"  [red]Unknown tool: {tool}[/]")
        console.print(
            f"  [dim]Supported: claude, codex, opencode, aider, localcoder, hermes, webui[/]\n"
        )


def _config_hermes():
    """Auto-configure Hermes Agent for local llama-server."""
    console.print(f"\n  [bold]Configuring Hermes Agent for local models[/]")
    console.print(f"  [dim]Based on: github.com/NousResearch/hermes-agent[/]\n")

    from localfit.prerequisites import check_hermes_agent

    ha = check_hermes_agent()
    if ha["found"]:
        console.print(f"  [green]✓[/] Hermes Agent found: {ha['path']}")
    else:
        console.print(f"  [yellow]![/] Hermes Agent not installed")
        console.print(
            f"  [dim]Install: curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash[/]\n"
        )

    from localfit.backends import get_llama_server_config

    srv = get_llama_server_config()
    port = 8089
    if srv["running"]:
        console.print(f"  [green]✓[/] llama-server running on :{port}")
    else:
        console.print(f"  [yellow]![/] llama-server not running")
        console.print(f"  [dim]Start it: localfit --serve gemma4-26b[/]\n")

    # Detect model name from /v1/models endpoint
    model_name = "local"
    if srv["running"]:
        try:
            import urllib.request as _req

            with _req.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as resp:
                data = json.loads(resp.read())
            if data.get("data"):
                model_name = data["data"][0].get("id", "local")
        except Exception:
            from localfit.backends import _detect_model_info

            mi = _detect_model_info(srv, None)
            if mi.get("name"):
                model_name = mi["name"].lower().replace(" ", "-")

    # Write config.yaml — provider must be "custom" (hermes maps "llamacpp" at CLI
    # level but the config resolver needs literal "custom" for base_url to take effect)
    hermes_dir = os.path.expanduser("~/.hermes")
    os.makedirs(hermes_dir, exist_ok=True)
    config_path = os.path.join(hermes_dir, "config.yaml")

    existing = {}
    if os.path.exists(config_path):
        try:
            import yaml

            with open(config_path) as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            pass

    if existing and isinstance(existing, dict):
        import yaml

        existing.setdefault("model", {})
        existing["model"]["default"] = model_name
        existing["model"]["provider"] = "custom"
        existing["model"]["base_url"] = f"http://127.0.0.1:{port}/v1"
        with open(config_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)
    else:
        with open(config_path, "w") as f:
            f.write(
                f"# Hermes Agent config — auto-generated by localfit\n"
                f"model:\n"
                f'  default: "{model_name}"\n'
                f'  provider: "custom"\n'
                f'  base_url: "http://127.0.0.1:{port}/v1"\n'
                f"providers: {{}}\nfallback_providers: []\n"
                f"toolsets:\n- hermes-cli\nagent:\n  max_turns: 90\n"
                f"  tool_use_enforcement: auto\n  verbose: false\n"
            )

    console.print(f"  [green]✓[/] Config written: {config_path}")
    console.print(f"    [dim]provider: custom  model: {model_name}[/]")
    console.print(f"    [dim]base_url: http://127.0.0.1:{port}/v1[/]")

    # Write .env (preserve existing entries)
    env_path = os.path.join(hermes_dir, ".env")
    env_lines = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_lines[k.strip()] = v.strip()
    env_lines["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env_lines["OPENAI_API_KEY"] = "no-key-required"
    with open(env_path, "w") as f:
        for k, v in env_lines.items():
            f.write(f"{k}={v}\n")
    os.chmod(env_path, 0o600)
    console.print(f"  [green]✓[/] Env written: {env_path}")

    console.print(f"\n  [bold]Ready! Launch:[/]")
    console.print(f"  [cyan]localfit --launch hermes[/]")
    console.print()


def _config_claude():
    """Show how to use Claude Code with local models. Never modifies Claude config files."""
    import shutil
    from localfit.proxy import PROXY_PORT

    console.print(f"\n  [bold]Claude Code + Local Models[/]")
    console.print(f"  [dim]Safe: does NOT modify any Claude settings files[/]\n")

    claude_bin = shutil.which("claude")
    if not claude_bin:
        console.print(f"  [red]Claude Code not installed.[/]")
        console.print(
            f"  [dim]Install: curl -fsSL https://claude.ai/install.sh | bash[/]\n"
        )
        return

    console.print(f"  [green]✓[/] Claude Code found: {claude_bin}")

    from localfit.backends import get_llama_server_config

    srv = get_llama_server_config()
    port = 8089
    if srv["running"]:
        console.print(f"  [green]✓[/] llama-server running on :{port}")
    else:
        console.print(f"  [yellow]![/] llama-server not running")
        console.print(f"  [dim]Start it: localfit --serve gemma4-26b[/]\n")

    # 5. Show how to run
    model_alias = "local"
    if srv["running"] and srv.get("model_path"):
        model_alias = os.path.basename(srv["model_path"]).replace(".gguf", "")

    # Detect model name for display
    model_name = "local"
    if srv["running"] and srv.get("model_path"):
        from localfit.backends import _detect_model_info

        mi = _detect_model_info(srv, None)
        if mi.get("name"):
            model_name = f"unsloth/{mi['name'].replace(' ', '-')}"

    console.print(f"\n  [bold]Ready! One-command launch:[/]")
    console.print(f"  [cyan]localfit --launch claude[/]")
    console.print(
        f"  [dim]This auto-starts a local Anthropic proxy on :{PROXY_PORT} for Claude Code.[/]"
    )
    console.print(f"\n  [bold]Or manually (env vars scoped to this command only):[/]")
    console.print(
        f"  [cyan]python -m localfit.proxy --port {PROXY_PORT} --llama-url http://127.0.0.1:{port}/v1/chat/completions &[/]"
    )
    console.print(f"  [cyan]ANTHROPIC_AUTH_TOKEN=localfit \\[/]")
    console.print(f"  [cyan]ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT} \\[/]")
    console.print(f"  [cyan]ANTHROPIC_API_KEY= \\[/]")
    console.print(f"  [cyan]claude --bare --model {model_name}[/]")
    console.print(
        f"\n  [dim]No files modified. Your normal Claude login is untouched.[/]"
    )
    console.print(
        f"  [dim]--bare skips OAuth. Env vars only apply to this one command.[/]\n"
    )


def _config_codex():
    """Auto-configure OpenAI Codex CLI for local llama-server."""
    import shutil

    console.print(f"\n  [bold]Configuring OpenAI Codex for local models[/]")
    console.print(f"  [dim]Based on: unsloth.ai/docs/basics/codex[/]\n")

    codex_bin = shutil.which("codex")
    if not codex_bin:
        console.print(f"  [red]Codex CLI not installed.[/]")
        console.print(f"  [dim]Install: npm install -g @openai/codex[/]\n")
        return

    console.print(f"  [green]✓[/] Codex found: {codex_bin}")

    console.print(f"\n  [bold]Ready! One-command launch:[/]")
    console.print(f"  [cyan]localfit --launch codex[/]")
    console.print(f"\n  [bold]Or manually (env vars scoped to this command only):[/]")
    console.print(f"  [cyan]OPENAI_BASE_URL=http://localhost:8089/v1 \\[/]")
    console.print(f"  [cyan]OPENAI_API_KEY=sk-no-key-required \\[/]")
    console.print(f"  [cyan]codex --model local[/]")
    console.print(
        f"\n  [dim]No files modified. Your normal Codex/OpenAI setup is untouched.[/]\n"
    )


def _trending_gallery():
    """Browse trending models from HuggingFace with fit/cloud info."""
    from localfit.backends import (
        get_machine_specs,
        _fetch_all_hf_models,
        fetch_hf_model,
        simulate_hf_model,
    )
    from localfit.matcher import get_quant_quality, get_quality_label, GPUS
    from rich.table import Table
    from rich.markup import escape as _esc

    specs = get_machine_specs()
    gpu_total = specs["gpu_total_mb"]
    gpu_gb = gpu_total // 1024

    os.system("clear" if os.name != "nt" else "cls")
    console.print(
        f"\n  [bold #e07a5f]localfit[/] trending models  [dim]·  {specs['chip']}  ·  {gpu_gb}GB GPU[/]\n"
    )

    with console.status("[dim]Fetching from HuggingFace...[/]", spinner="dots"):
        models = _fetch_all_hf_models()

    table = Table(
        show_header=True,
        header_style="bold",
        border_style="dim",
        padding=(0, 1),
        width=95,
    )
    table.add_column("#", width=3, style="bold cyan")
    table.add_column("Model", width=30)
    table.add_column("Downloads", width=8, justify="right")
    table.add_column("Caps", width=12)
    table.add_column("Local", width=12)
    table.add_column("Cloud", width=15)

    all_models = []
    for i, m in enumerate(
        sorted(models, key=lambda x: x["downloads"], reverse=True)[:20], 1
    ):
        dl = m["downloads"]
        dl_str = f"{dl // 1000}K" if dl < 1_000_000 else f"{dl / 1_000_000:.1f}M"
        label = _esc(m["label"])

        # Caps
        caps = []
        for c in m.get("caps", []):
            if c == "vision":
                caps.append("[magenta]img[/]")
            elif c == "code":
                caps.append("[cyan]code[/]")
            elif c == "MoE":
                caps.append("[green]MoE[/]")
            elif c == "audio":
                caps.append("[yellow]audio[/]")
        cap_str = " ".join(caps) if caps else "[dim]text[/]"

        # Local fit
        est = m.get("est_smallest_gb")
        if est and est * 1024 < gpu_total:
            local_str = f"[green]✓ fits ~{est}GB[/]"
        elif est:
            local_str = f"[red]✗ ~{est}GB[/]"
        else:
            local_str = "[dim]?[/]"

        # Cloud: cheapest option
        if est and est * 1024 >= gpu_total:
            # Find cheapest cloud GPU that fits
            for gpu in GPUS:
                if est and est < gpu["vram"]:
                    cloud_str = f"[yellow]☁ ${gpu['price']:.2f}/hr[/]"
                    break
            else:
                cloud_str = "[red]too big[/]"
        else:
            cloud_str = "[dim]local OK[/]"

        table.add_row(str(i), label, dl_str, cap_str, local_str, cloud_str)
        all_models.append(m)

    console.print(table)
    console.print(
        f"\n  [bold]enter[/] # to check quants  [bold]s[/] search  [bold]q[/] quit"
    )

    try:
        ans = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if ans == "q" or not ans:
        return
    elif ans == "s":
        try:
            q = input("  Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if q:
            simulate_hf_model(q)
    elif ans.isdigit():
        idx = int(ans) - 1
        if 0 <= idx < len(all_models):
            simulate_hf_model(all_models[idx]["repo_id"])


if __name__ == "__main__":
    main()
