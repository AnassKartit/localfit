"""Unified model picker — arrow-key interactive menu with Rich formatting."""

import json
import os
import sys
import urllib.parse
import urllib.request

from rich.columns import Columns
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)


def _read_key():
    """Read a single keypress. Returns 'up', 'down', 'enter', 'q', or the char."""
    import tty, termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return "up"
                if ch3 == "B":
                    return "down"
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "ctrl-c"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _option_label(opt):
    if "backend" in opt:
        name = opt.get("name", "")
        return name.split("/")[-1] if "/" in name else name
    return opt.get("gpu", "")


def _option_source(opt):
    repo = opt.get("repo")
    if repo and "/" in repo:
        return repo.split("/", 1)[0]

    name = opt.get("name", "")
    if "/" in name:
        return name.split("/", 1)[0]

    return ""


def _option_label_text(opt, selected=False):
    label = Text("", no_wrap=True, overflow="ellipsis", style="bold" if selected else "")
    label.append(_option_label(opt))
    source = _option_source(opt)
    if source:
        label.append(" · ", style="dim")
        label.append(source, style="dim")
    return label


def _option_badge(opt):
    return opt.get("backend") or opt.get("provider") or "?"


def _option_meta(opt):
    return opt.get("size") or opt.get("cost") or ""


def _build_selected_panel(all_opts, selected, recommended, width):
    """Render the selected item as a compact details card."""
    if not all_opts:
        return Panel("No options available", title="Selected", border_style="red", width=width)

    opt = all_opts[selected]
    is_local = "backend" in opt
    accent = "green" if is_local else "yellow"

    info = Table.grid(padding=(0, 1))
    info.add_column(style="dim", width=9, no_wrap=True)
    info.add_column(ratio=1)

    info.add_row("Choice", Text(str(selected + 1), style=f"bold {accent}"))
    info.add_row(
        "Scope",
        Text("LOCAL" if is_local else "REMOTE", style=f"bold {accent}"),
    )
    info.add_row("Engine", Text(_option_badge(opt), style=f"bold {accent}"))
    info.add_row("Target", Text(_option_label(opt), style="bold"))
    source = _option_source(opt)
    if source:
        info.add_row("Source", Text(source))
    info.add_row("Size" if is_local else "Cost", Text(_option_meta(opt), style="bold"))
    if is_local and (selected + 1) == recommended:
        info.add_row("Pick", Text("Recommended", style="yellow"))
    info.add_row("Notes", Text(opt.get("note", ""), style="dim"))

    return Panel(
        info,
        title=Text(_option_label(opt), style="bold"),
        subtitle="Selected",
        border_style=accent,
        padding=(0, 1),
        width=width,
    )


def _build_section(title, opts, start_idx, selected, recommended, accent):
    """Render a LOCAL or REMOTE options table with stable column sizing."""
    if not opts:
        return None

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=2, no_wrap=True)
    table.add_column(width=3, justify="right", no_wrap=True)
    table.add_column(width=8, no_wrap=True)
    table.add_column(ratio=1, no_wrap=True)
    table.add_column(width=12, justify="right", no_wrap=True)

    for offset, opt in enumerate(opts):
        absolute_idx = start_idx + offset
        is_selected = absolute_idx == selected
        row_style = "bold black on grey85" if is_selected else ""
        pointer = Text("›" if is_selected else " ", style=f"bold {accent}")
        index = Text(
            str(absolute_idx + 1),
            style="bold bright_white" if is_selected else "dim",
        )
        badge = Text(_option_badge(opt), style=f"bold {accent}" if not is_selected else "bold")
        label = _option_label_text(opt, selected=is_selected)
        meta = Text(_option_meta(opt), style="bold bright_white" if is_selected else "dim")
        if absolute_idx + 1 == recommended:
            meta.append(" ★", style="yellow")
        table.add_row(pointer, index, badge, label, meta, style=row_style)

    return Group(Text(title, style=f"bold {accent}"), table)


def _render_menu(local_opts, remote_opts, selected, recommended, width):
    """Build the menu with Rich tables instead of padded text."""
    all_opts = local_opts + remote_opts
    detail_width = min(36, max(26, width // 3))
    options_width = max(26, width - detail_width - 7)

    detail_panel = _build_selected_panel(all_opts, selected, recommended, detail_width)

    sections = []
    local_section = _build_section("LOCAL", local_opts, 0, selected, recommended, "green")
    if local_section is not None:
        sections.append(local_section)
    remote_section = _build_section(
        "REMOTE", remote_opts, len(local_opts), selected, recommended, "yellow"
    )
    if remote_section is not None:
        if sections:
            sections.append(Text(""))
        sections.append(remote_section)

    options_panel = Panel(
        Group(*sections),
        title=Text("Choices", style="bold bright_white"),
        border_style="bright_white",
        padding=(0, 1),
        width=options_width,
    )

    if width < 72:
        return Group(detail_panel, Text(""), options_panel)

    return Columns([detail_panel, options_panel], expand=False, equal=False, padding=(0, 2))


def show_run_menu(model, hw, local_opts, remote_opts, recommended):
    """Interactive arrow-key menu. Returns 1-indexed choice or a navigation action."""
    from rich.live import Live

    total = len(local_opts) + len(remote_opts)
    if total == 0:
        console.print(f"  [red]No options available[/]")
        return None

    selected = max(0, recommended - 1)
    width = max(20, min(console.width - 2, 110))
    confirm_quit = False

    try:
        with Live(console=console, auto_refresh=False, transient=True) as live:
            while True:
                body = _render_menu(local_opts, remote_opts, selected, recommended, width)
                subtitle = (
                    f"{hw}  [dim]↑↓/jk move · enter select · 1-9 jump · b back · q quit[/]"
                )
                if confirm_quit:
                    subtitle = (
                        f"{hw}  [bold yellow]Quit localfit?[/] [dim]y yes · n no · b back[/]"
                    )
                panel = Panel(
                    body,
                    title=f"[bold bright_white]{escape(model)}[/]",
                    subtitle=subtitle,
                    border_style="bright_white",
                    width=width,
                    padding=(0, 1),
                )
                live.update(panel, refresh=True)

                key = _read_key()
                if confirm_quit:
                    if key in ("y", "enter"):
                        return "quit"
                    if key in ("n",):
                        confirm_quit = False
                    elif key in ("b", "esc"):
                        return "back"
                    continue
                if key == "up":
                    selected = max(0, selected - 1)
                elif key == "down":
                    selected = min(total - 1, selected + 1)
                elif key == "enter":
                    return selected + 1
                elif key in ("b", "esc"):
                    return "back"
                elif key == "q":
                    confirm_quit = True
                elif key == "ctrl-c":
                    return "back"
                elif key.isdigit():
                    num = int(key)
                    if 1 <= num <= min(total, 9):
                        return num
    except Exception:
        # Fallback for non-interactive terminals (piped input, CI, etc.)
        body = _render_menu(local_opts, remote_opts, recommended - 1, recommended, width)
        console.print(
            Panel(
                body,
                title=f"[bold bright_white]{escape(model)}[/]",
                subtitle=hw,
                border_style="bright_white",
                width=width,
                padding=(0, 1),
            )
        )
        try:
            from rich.prompt import Prompt
            choice = Prompt.ask(
                f"[bold]Pick [1-{total}] · b back · q quit[/]",
                default=str(recommended),
            ).strip().lower()
            if choice == "b":
                return "back"
            if choice == "q":
                return "quit"
            return int(choice)
        except (ValueError, EOFError, KeyboardInterrupt):
            return None


def collect_options(model_query, specs):
    """Collect MLX + GGUF + Remote options for a model."""
    from localfit.backends import IS_MAC, check_mlx_available, get_machine_specs, MODELS

    if specs is None:
        specs = get_machine_specs()

    # Resolve Ollama-style tags (gemma4:e4b) to HF search terms
    # Check known models first, then strip colons for HF search
    hf_query = model_query
    resolved = model_query.replace(":", "-").replace("/", "-")
    for mid, m in MODELS.items():
        if m.get("ollama_tag") == model_query or mid == resolved:
            hf_query = m.get("hf_repo", "").split("/")[-1] if m.get("hf_repo") else resolved
            break
    # Also try without colon for HF search (gemma:e2b → gemma-e2b → gemma e2b)
    if ":" in model_query and hf_query == model_query:
        hf_query = model_query.replace(":", " ")

    gpu_total_mb = specs.get("gpu_total_mb", 0)
    usable_mb = gpu_total_mb - 2048

    local_opts = []
    remote_opts = []
    metadata = {}

    # ── MLX options ──
    if IS_MAC and check_mlx_available():
        for repo, info in _find_all_mlx_variants(hf_query):
            size_gb = info.get("size_gb", 0)
            fits = size_gb * 1024 < usable_mb if size_gb else True
            tight = not fits and size_gb and size_gb * 1024 < gpu_total_mb
            if fits:
                tag = "✓ fits"
            elif tight:
                tag = "~ tight"
            else:
                tag = "✗ too large"
            local_opts.append({
                "backend": "MLX", "name": repo,
                "note": f"Native Metal · {tag} · {info.get('quant','?')}",
                "size": f"{size_gb:.1f}GB" if size_gb else "?",
                "size_gb": size_gb, "fits": fits or tight,
                "action": "mlx", "repo": repo,
            })

    # ── GGUF options ──
    from localfit.backends import fetch_hf_model
    data = fetch_hf_model(hf_query, silent=True)
    if data and data.get("gguf_files"):
        metadata["hf_data"] = data
        mmproj = data["mmproj_files"][0]["size_gb"] if data.get("is_vlm") and data.get("mmproj_files") else 0
        for gguf in _pick_interesting_quants(data["gguf_files"], usable_mb, mmproj):
            total_gb = gguf["size_gb"] + mmproj
            fits = total_gb * 1024 < usable_mb
            tight = not fits and total_gb * 1024 < gpu_total_mb
            if fits:
                tag = "✓ fits"
            elif tight:
                tag = "~ tight"
            else:
                tag = "✗ too large"
            local_opts.append({
                "backend": "GGUF", "name": f"{data['repo_id']} {gguf['quant']}",
                "note": f"llama.cpp Metal · {tag}",
                "size": f"{total_gb:.1f}GB", "size_gb": total_gb, "fits": fits or tight,
                "action": "gguf", "repo": data["repo_id"],
                "filename": gguf["filename"], "quant": gguf["quant"], "hf_data": data,
            })

    # ── Remote ──
    remote_opts.append({
        "provider": "Kaggle", "gpu": "T4 16GB", "quant": "best fit",
        "note": "Free 30h/week · no credit card", "cost": "free", "action": "kaggle",
    })
    try:
        from localfit.cloud import get_runpod_key, fetch_gpu_options
        rk = get_runpod_key()
        if rk:
            gpus = fetch_gpu_options(rk)
            if gpus:
                g = gpus[0]
                remote_opts.append({
                    "provider": "RunPod", "gpu": f"{g['name']} {g['vram']}GB",
                    "quant": "best fit", "note": "Paid · auto-stop",
                    "cost": f"${g['price']:.2f}/hr", "action": "runpod",
                })
    except Exception:
        pass

    return local_opts, remote_opts, _pick_recommended(local_opts, usable_mb), metadata


def _find_all_mlx_variants(model_name):
    """Find mlx-community variants with estimated sizes."""
    base = model_name.split("/")[-1]
    base_clean = base.lower().replace("_", "-")
    results = []
    try:
        url = f"https://huggingface.co/api/models?author=mlx-community&search={urllib.parse.quote(base_clean)}&limit=10"
        req = urllib.request.Request(url, headers={"User-Agent": "localfit"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            for h in json.loads(resp.read()):
                rid = h.get("id", "")
                if not rid.startswith("mlx-community/") or base_clean[:6] not in rid.lower():
                    continue
                quant, size_gb = "fp16", 0
                for q, mult in [("3bit", 0.375), ("4bit", 0.5), ("6bit", 0.75), ("8bit", 1.0), ("bf16", 2.0)]:
                    if q in rid.lower():
                        quant = q
                        for ps, pg in [("405b",405),("72b",72),("70b",70),("32b",32),("27b",27),
                                       ("14b",14),("13b",13),("9b",9),("8b",8),("7b",7),
                                       ("4.7",30),("4b",4),("3b",3),("1.5b",1.5),("1b",1),("0.5b",0.5)]:
                            if ps in base_clean:
                                size_gb = round(pg * mult, 1)
                                break
                        break
                results.append((rid, {"quant": quant, "size_gb": size_gb}))
    except Exception:
        pass
    order = {"4bit": 0, "3bit": 1, "6bit": 2, "8bit": 3, "bf16": 4, "fp16": 5}
    results.sort(key=lambda r: order.get(r[1]["quant"], 9))
    return results[:4]


def _pick_interesting_quants(gguf_files, usable_mb, mmproj):
    has_unsloth_ud = any(
        "UD-" in f.get("filename", "").upper() or f.get("quant", "").upper().endswith("_XL")
        for f in gguf_files
    )
    targets = (
        ["Q4_K_XL", "Q3_K_XL", "Q2_K_XL", "Q5_K_M", "Q6_K", "Q8_0"]
        if has_unsloth_ud
        else ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
    )
    picked = []

    def _add(candidate):
        if candidate and candidate not in picked:
            picked.append(candidate)

    fits = [f for f in gguf_files if (f["size_gb"] + mmproj) * 1024 < usable_mb]

    if fits:
        _add(max(fits, key=lambda f: f.get("size_gb", 0)))

    for t in targets:
        for f in gguf_files:
            if f["quant"].upper() == t:
                _add(f)
                break

    if has_unsloth_ud:
        for f in fits:
            _add(f)

    return picked[:4]


def _pick_recommended(local_opts, usable_mb):
    unsloth_gguf = [
        (i, opt)
        for i, opt in enumerate(local_opts, 1)
        if opt.get("fits")
        and opt.get("backend") == "GGUF"
        and _option_source(opt) == "unsloth"
    ]
    if unsloth_gguf:
        return max(unsloth_gguf, key=lambda item: item[1].get("size_gb", 0))[0]

    best_idx, best_score = 1, -1
    for i, opt in enumerate(local_opts, 1):
        if not opt.get("fits"):
            continue
        score = (10 if opt["backend"] == "MLX" else 5) + opt.get("size_gb", 0)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx
