"""localfit dashboard — Textual TUI for managing local AI models."""

import json
import os
import urllib.request
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    ProgressBar,
    Input,
    Button,
)
from textual.timer import Timer


# ── Data helpers ──


def _get_specs():
    try:
        from localfit.backends import (
            get_machine_specs,
            get_metal_gpu_stats,
            get_swap_usage_mb,
            get_disk_info,
        )

        specs = get_machine_specs()
        metal = get_metal_gpu_stats()
        swap = get_swap_usage_mb()
        disk = get_disk_info()
        return {
            "chip": specs.get("chip", "?"),
            "ram_gb": specs.get("ram_gb", 0),
            "gpu_total_mb": metal.get("total_mb", 0) or specs.get("gpu_total_mb", 0),
            "gpu_alloc_mb": metal.get("alloc_mb", 0),
            "swap_mb": swap,
            "disk_free_gb": disk.get("disk_free_gb", 0),
            "hf_cache_gb": disk.get("hf_cache_gb", 0),
            "ollama_cache_gb": disk.get("ollama_cache_gb", 0),
            "total_cache_gb": disk.get("total_cache_gb", 0),
        }
    except Exception:
        return {
            "chip": "?",
            "ram_gb": 0,
            "gpu_total_mb": 0,
            "gpu_alloc_mb": 0,
            "swap_mb": 0,
            "disk_free_gb": 0,
            "hf_cache_gb": 0,
            "ollama_cache_gb": 0,
            "total_cache_gb": 0,
        }


def _get_running():
    """Get all running models across all backends."""
    models = []
    for port, backend in [(8089, "llama.cpp"), (11434, "Ollama")]:
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            d = json.loads(r.read())
            for m in d.get("data", []):
                models.append(
                    {"name": m["id"], "backend": backend, "port": port, "type": "LLM"}
                )
        except Exception:
            pass
    # Ollama loaded in GPU
    try:
        r = urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=1)
        d = json.loads(r.read())
        for m in d.get("models", []):
            name = m.get("name", "")
            size_gb = m.get("size", 0) / (1024**3)
            vram_gb = m.get("size_vram", 0) / (1024**3)
            # Don't duplicate if already in list
            if not any(x["name"] == name for x in models):
                models.append(
                    {
                        "name": name,
                        "backend": "Ollama",
                        "port": 11434,
                        "type": "LLM",
                        "size_gb": round(size_gb, 1),
                        "vram_gb": round(vram_gb, 1),
                    }
                )
    except Exception:
        pass
    # Image server
    try:
        r = urllib.request.urlopen("http://127.0.0.1:8189/health", timeout=1)
        d = json.loads(r.read())
        if d.get("status") == "ok" and d.get("model", "not loaded") != "not loaded":
            models.append(
                {"name": d["model"], "backend": "mflux", "port": 8189, "type": "Image"}
            )
    except Exception:
        pass
    return models


def _get_installed():
    """Get installed Ollama models."""
    models = []
    try:
        r = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        d = json.loads(r.read())
        for m in d.get("models", []):
            models.append(
                {"name": m["name"], "size_gb": round(m.get("size", 0) / (1024**3), 1)}
            )
    except Exception:
        pass
    return sorted(models, key=lambda x: x["size_gb"], reverse=True)


def _get_compatible():
    """Get recommended models for this hardware with fit analysis."""
    try:
        from localfit.backends import (
            MODELS,
            recommend_model,
            get_system_ram_gb,
            get_machine_specs,
            get_metal_gpu_stats,
            can_run_simultaneously,
            IS_MAC,
            check_mlx_available,
        )

        specs = get_machine_specs()
        gpu_mb = get_metal_gpu_stats().get("total_mb", 0) or specs.get(
            "gpu_total_mb", 0
        )
        gpu_gb = gpu_mb / 1024
        usable_gb = gpu_gb - 1.5
        ram = get_system_ram_gb()
        best_id, best_reason = recommend_model(ram)

        has_mlx = False
        if IS_MAC:
            try:
                has_mlx = check_mlx_available()
            except Exception:
                pass

        result = []
        for mid, m in MODELS.items():
            size = m.get("size_gb", 99)
            size_q2 = m.get("size_q2_gb", size)
            # Strict: only "fits" if the shown size actually fits
            fits = size <= usable_gb
            # "tight" only if model itself fits in total GPU (no reserve)
            fits_tight = size <= gpu_gb and size > usable_gb
            # Cloud if even the model Q4 size exceeds total GPU
            cloud_only = m.get("cloud_only", False) or size > gpu_gb
            can_pair_img = can_run_simultaneously(ram, size, 8)

            # Show Q2 size instead if Q4 doesn't fit but Q2 does
            display_size = size
            if cloud_only and size_q2 <= gpu_gb:
                display_size = size_q2
                cloud_only = False
                fits = size_q2 <= usable_gb
                fits_tight = size_q2 <= gpu_gb and size_q2 > usable_gb

            where = "cloud" if cloud_only else "local"
            if where == "local" and IS_MAC:
                where = (
                    "local (MLX)"
                    if has_mlx and m.get("backend") != "llamacpp"
                    else "local (Metal)"
                )

            # Name suffix if showing Q2 fallback
            name_suffix = ""
            if display_size != size and display_size == size_q2:
                name_suffix = " (Q2)"

            result.append(
                {
                    "name": m["name"] + name_suffix,
                    "id": mid,
                    "size_gb": display_size,
                    "recommended": mid == best_id,
                    "reason": best_reason
                    if mid == best_id
                    else m.get("description", ""),
                    "fits": "yes" if fits else "tight" if fits_tight else "cloud",
                    "can_pair_image": can_pair_img,
                    "where": where,
                    "source": m.get("source", ""),
                    "hf_repo": m.get("hf_repo", ""),
                }
            )

        # Sort: recommended first, then fits, then by size desc
        result.sort(
            key=lambda x: (
                0 if x["recommended"] else 1,
                0 if x["fits"] == "yes" else 1 if x["fits"] == "tight" else 2,
                -x["size_gb"],
            )
        )
        return result
    except Exception:
        return []


# ── Widgets ──


class SystemPanel(Static):
    """Shows GPU/RAM/swap/disk status."""

    def compose(self) -> ComposeResult:
        yield Label("Loading...", id="sys-info")

    def update_stats(self, specs: dict):
        gpu_total = specs["gpu_total_mb"] // 1024
        gpu_alloc = specs["gpu_alloc_mb"] // 1024
        gpu_free = gpu_total - gpu_alloc
        swap_gb = specs["swap_mb"] // 1024
        disk_free = specs["disk_free_gb"]
        cache = specs["total_cache_gb"]

        swap_color = "red" if swap_gb > 8 else "yellow" if swap_gb > 2 else "green"
        gpu_color = "red" if gpu_free < 2 else "yellow" if gpu_free < 4 else "green"

        text = (
            f"[bold]{specs['chip']}[/]  {specs['ram_gb']}GB RAM\n\n"
            f"  GPU   [{gpu_color}]{gpu_alloc}/{gpu_total}GB[/] ({gpu_free}GB free)\n"
            f"  Swap  [{swap_color}]{swap_gb}GB[/]"
            + (
                f" [red bold]CRITICAL[/]"
                if swap_gb > 8
                else " [yellow]high[/]"
                if swap_gb > 2
                else ""
            )
            + "\n"
            f"  Disk  {disk_free}GB free · {cache}GB cache\n"
        )
        self.query_one("#sys-info").update(text)


class RunningPanel(Static):
    """Shows currently running models."""

    def compose(self) -> ComposeResult:
        yield Label("Scanning...", id="running-info")

    def update_models(self, models: list):
        if not models:
            self.query_one("#running-info").update(
                "[dim]No models running[/]\n\n[dim]Press [bold]r[/] to run a model[/]"
            )
            return
        lines = []
        for m in models:
            icon = "[green]●[/]" if m["type"] == "LLM" else "[cyan]●[/]"
            extra = f" ({m.get('vram_gb', '?')}GB GPU)" if m.get("vram_gb") else ""
            lines.append(
                f"  {icon} {m['name']}  [dim]{m['backend']} :{m['port']}[/]{extra}"
            )
        self.query_one("#running-info").update("\n".join(lines))


class ModelsTable(DataTable):
    """Browsable table of installed + compatible models."""

    pass


# ── Main App ──


class LocalfitDashboard(App):
    """localfit dashboard — manage local AI models."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 1fr 2fr;
        grid-rows: auto 1fr;
        padding: 0;
    }
    #system-panel {
        border: solid $primary;
        padding: 1 2;
        height: auto;
        max-height: 12;
    }
    #running-panel {
        border: solid $success;
        padding: 1 2;
        height: auto;
        max-height: 12;
    }
    #models-panel {
        border: solid $secondary;
        padding: 0;
        column-span: 2;
    }
    #search-box {
        dock: top;
        margin: 0 1;
    }
    DataTable {
        height: 1fr;
    }
    .section-title {
        text-style: bold;
        margin: 0 0 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "run_model", "Run"),
        Binding("enter", "run_model", "Run/Launch", show=False),
        Binding("s", "search", "Search"),
        Binding("c", "cleanup", "Cleanup"),
        Binding("l", "launch_tool", "Launch"),
        Binding("x", "stop_model", "Stop"),
        Binding("d", "delete_model", "Delete"),
        Binding("m", "makeitfit", "MakeItFit"),
        Binding("w", "will_it_fit", "Will it fit?"),
        Binding("f5", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="system-panel"):
            yield Label("[bold]System[/]", classes="section-title")
            yield SystemPanel()
        with Vertical(id="running-panel"):
            yield Label("[bold]Running[/]", classes="section-title")
            yield RunningPanel()
        with Vertical(id="models-panel"):
            yield Label(
                "[bold]Models[/]  "
                "[dim]Enter[/]=run  [dim]s[/]=search  [dim]w[/]=will it fit?  "
                "[dim]m[/]=make it fit  [dim]x[/]=stop  [dim]d[/]=delete  [dim]c[/]=cleanup",
                classes="section-title",
            )
            yield Input(placeholder="Search models...", id="search-box")
            yield ModelsTable(id="models-table")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "localfit"
        self.sub_title = "local AI model manager"
        # Cache data to avoid re-scanning on every keystroke
        self._cached_specs = _get_specs()
        self._cached_running = _get_running()
        self._cached_installed = _get_installed()
        self._cached_compatible = _get_compatible()
        self._refresh_all()
        self.set_interval(10, self._refresh_status)

    def _refresh_all(self) -> None:
        self._refresh_status()
        self._refresh_models()

    def _refresh_status(self) -> None:
        self._cached_specs = _get_specs()
        self._cached_running = _get_running()
        self.query_one(SystemPanel).update_stats(self._cached_specs)
        self.query_one(RunningPanel).update_models(self._cached_running)

    def _refresh_models(self, search: str = "") -> None:
        table = self.query_one("#models-table", ModelsTable)
        table.clear(columns=True)
        table.add_columns("", "Model", "Size", "Fits", "Where", "+Image", "Source")

        specs = self._cached_specs
        gpu_gb = specs["gpu_total_mb"] // 1024
        usable_gb = gpu_gb - 3  # reserve 3GB for OS + KV cache + apps
        running = self._cached_running
        running_names = {m["name"] for m in running}
        installed = self._cached_installed
        installed_names = {m["name"] for m in installed}
        compatible = self._cached_compatible

        search_lower = search.lower()

        # Running models
        for m in running:
            if search_lower and search_lower not in m["name"].lower():
                continue
            table.add_row(
                "●",
                m["name"],
                "",
                "[green]running[/]",
                f"{m['backend']} :{m['port']}",
                "",
                "",
            )

        # Installed (not running)
        for m in installed:
            if m["name"] in running_names:
                continue
            if search_lower and search_lower not in m["name"].lower():
                continue
            fits = m["size_gb"] <= usable_gb
            fits_gpu = m["size_gb"] <= gpu_gb
            tag = (
                "[green]✓ fits[/]"
                if fits
                else "[yellow]~tight[/]"
                if fits_gpu
                else "[red]no[/]"
            )
            img_ok = "[green]✓[/]" if m["size_gb"] + 8 <= usable_gb else "[dim]-[/]"
            where = "local" if fits_gpu else "[red]cloud[/]"
            table.add_row(
                "✓", m["name"], f"{m['size_gb']}GB", tag, where, img_ok, "Ollama"
            )

        # Catalog models (recommended + compatible + cloud)
        for m in compatible:
            if m["name"] in installed_names or any(
                m["name"] in r["name"] for r in running
            ):
                continue
            if (
                search_lower
                and search_lower not in m["name"].lower()
                and search_lower not in (m.get("id") or "").lower()
                and search_lower not in (m.get("reason") or "").lower()
                and search_lower not in (m.get("hf_repo") or "").lower()
            ):
                continue

            badge = (
                "★" if m.get("recommended") else "→" if m["fits"] != "cloud" else "☁"
            )
            fit_tag = {
                "yes": "[green]✓ fits[/]",
                "tight": "[yellow]~tight[/]",
                "cloud": "[red]cloud[/]",
            }.get(m["fits"], "[dim]?[/]")
            img_ok = "[green]✓[/]" if m.get("can_pair_image") else "[dim]-[/]"
            where = m.get("where", "")
            source = m.get("source", "unsloth")

            notes = m.get("reason", "")[:35]
            table.add_row(
                badge,
                f"{m['name']}",
                f"~{m['size_gb']}GB",
                fit_tag,
                where,
                img_ok,
                source,
            )

        # Image / diffusion models
        try:
            from localfit.image_models import IMAGE_MODELS

            for k, v in IMAGE_MODELS.items():
                if (
                    search_lower
                    and search_lower not in k.lower()
                    and search_lower not in v.get("pipeline", "").lower()
                ):
                    continue
                vram = v.get("vram_gb", 0)
                fits = vram <= usable_gb
                # Check if already running as image server
                is_running = any(
                    k.replace("flux2-", "").replace("flux1-", "") in r.get("name", "")
                    for r in running
                    if r.get("type") == "Image"
                )
                if is_running:
                    table.add_row(
                        "●",
                        f"{k} (image)",
                        f"{vram}GB",
                        "[green]running[/]",
                        "local",
                        "",
                        "image",
                    )
                else:
                    fit_tag = "[green]✓ fits[/]" if fits else "[red]cloud[/]"
                    table.add_row(
                        "◐",
                        f"{k} (image)",
                        f"{vram}GB",
                        fit_tag,
                        "local" if fits else "cloud",
                        "",
                        "image",
                    )
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-box":
            query = event.value or ""
            # Auto-detect URL paste → instant analysis
            if "huggingface.co/" in query:
                repo = (
                    query.replace("https://huggingface.co/", "")
                    .replace("http://huggingface.co/", "")
                    .strip("/")
                )
                if "/" in repo and len(repo) > 5:
                    self.exit(result=("willitfit", repo))
                    return
            # Repo ID paste (org/model format, no spaces)
            if "/" in query and " " not in query and len(query) > 10:
                return  # wait for Enter, might still be typing
            self._refresh_models(query)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """When user presses Enter in search, search HF or analyze URL."""
        if event.input.id == "search-box" and event.value.strip():
            query = event.value.strip()
            if "huggingface.co/" in query or "/" in query and not " " in query:
                # Looks like a HF URL or repo ID — exit to simulate
                repo = query.replace("https://huggingface.co/", "").strip("/")
                self.exit(result=("simulate", repo))
            else:
                self._search_hf(query)

    def _search_hf(self, query: str) -> None:
        """Search HuggingFace for GGUF models and add to table."""
        table = self.query_one("#models-table", ModelsTable)
        try:
            url = f"https://huggingface.co/api/models?search={query}+GGUF&filter=gguf&sort=downloads&direction=-1&limit=10"
            req = urllib.request.Request(url, headers={"User-Agent": "localfit"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                models = json.loads(resp.read())
            for m in models:
                repo = m.get("id", "")
                downloads = m.get("downloads", 0)
                dl_str = (
                    f"{downloads // 1000}K dl"
                    if downloads > 1000
                    else f"{downloads} dl"
                )
                # Skip if already in table
                table.add_row("🔍", repo, "", "[dim]HuggingFace[/]", "", dl_str, "HF")
        except Exception:
            pass

    def action_refresh(self) -> None:
        self._refresh_all()

    def action_run_model(self) -> None:
        table = self.query_one("#models-table", ModelsTable)
        row = table.cursor_row
        if row is not None:
            cells = table.get_row_at(row)
            badge = str(cells[0]).strip()
            model_name = str(cells[1]).strip()
            status = str(cells[3]).strip() if len(cells) > 3 else ""

            if "running" in status:
                # Running model -> launch a tool with it
                self.exit(result=("launch_with", model_name))
            elif badge == "✓":
                # Installed -> run it
                self.exit(result=("run", model_name))
            else:
                # Catalog -> download + run
                self.exit(result=("run", model_name))

    def action_search(self) -> None:
        self.query_one("#search-box", Input).focus()

    def action_cleanup(self) -> None:
        self.exit(result=("cleanup",))

    def action_launch_tool(self) -> None:
        self.exit(result=("launch",))

    def action_stop_model(self) -> None:
        """Stop/unload selected running model."""
        table = self.query_one("#models-table", ModelsTable)
        row = table.cursor_row
        if row is not None:
            cells = table.get_row_at(row)
            model_name = str(cells[1]).strip()
            status = str(cells[3]).strip() if len(cells) > 3 else ""
            where = str(cells[4]).strip() if len(cells) > 4 else ""
            if "running" not in status:
                self.notify("Not a running model", severity="warning")
                return

            stopped = False
            # Ollama model
            if "Ollama" in where or "11434" in where:
                try:
                    payload = json.dumps(
                        {"model": model_name, "keep_alive": 0}
                    ).encode()
                    req = urllib.request.Request(
                        "http://127.0.0.1:11434/api/generate",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    urllib.request.urlopen(req, timeout=5)
                    stopped = True
                except Exception:
                    pass

            # llama-server model
            if "llama" in where or "8089" in where:
                import subprocess

                subprocess.run(
                    ["pkill", "-f", "llama-server"], capture_output=True, timeout=5
                )
                stopped = True

            # Image server
            if "(image)" in model_name or "8189" in where:
                import subprocess

                subprocess.run(
                    ["pkill", "-f", "localfit.image_server"],
                    capture_output=True,
                    timeout=5,
                )
                stopped = True

            if stopped:
                self.notify(f"Stopped {model_name}")
            else:
                self.notify(f"Could not stop {model_name}", severity="error")

            # Refresh
            import time

            time.sleep(1)
            self._cached_running = _get_running()
            self._cached_installed = _get_installed()
            self._refresh_all()
            self._refresh_models()

    def action_delete_model(self) -> None:
        """Delete selected model from disk to free space."""
        table = self.query_one("#models-table", ModelsTable)
        row = table.cursor_row
        if row is not None:
            cells = table.get_row_at(row)
            model_name = str(cells[1]).strip()
            size = str(cells[2]).strip() if len(cells) > 2 else ""
            badge = str(cells[0]).strip()
            source = str(cells[6]).strip() if len(cells) > 6 else ""

            # Stop it first if running
            status = str(cells[3]).strip() if len(cells) > 3 else ""
            if "running" in status:
                self.action_stop_model()
                import time

                time.sleep(1)

            deleted = False
            # Ollama model
            if source == "Ollama" or badge in ("✓", "●"):
                try:
                    payload = json.dumps({"name": model_name}).encode()
                    req = urllib.request.Request(
                        "http://127.0.0.1:11434/api/delete",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="DELETE",
                    )
                    urllib.request.urlopen(req, timeout=10)
                    deleted = True
                except Exception as e:
                    self.notify(f"Delete failed: {e}", severity="error")

            if deleted:
                self.notify(f"Deleted {model_name} ({size})")
            elif not deleted and badge == "✓":
                self.notify(f"Could not delete {model_name}", severity="error")
            else:
                self.notify(f"Only installed models can be deleted", severity="warning")

            # Refresh
            self._cached_running = _get_running()
            self._cached_installed = _get_installed()
            self._refresh_all()
            self._refresh_models()

    def action_delete_model(self) -> None:
        """Delete selected model from disk."""
        table = self.query_one("#models-table", ModelsTable)
        row = table.cursor_row
        if row is not None:
            cells = table.get_row_at(row)
            model_name = str(cells[1]).strip()
            badge = str(cells[0]).strip()
            if badge in ("✓", "●"):
                # Ollama model - delete via API
                try:
                    payload = json.dumps({"name": model_name}).encode()
                    req = urllib.request.Request(
                        "http://127.0.0.1:11434/api/delete",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="DELETE",
                    )
                    urllib.request.urlopen(req, timeout=10)
                    self.notify(f"Deleted {model_name}")
                except Exception as e:
                    self.notify(f"Delete failed: {e}", severity="error")
                # Refresh caches
                self._cached_running = _get_running()
                self._cached_installed = _get_installed()
                self._refresh_all()
                self._refresh_models()

    def action_makeitfit(self) -> None:
        table = self.query_one("#models-table", ModelsTable)
        row = table.cursor_row
        if row is not None:
            cells = table.get_row_at(row)
            model_name = str(cells[1])
            self.exit(result=("makeitfit", model_name))

    def action_will_it_fit(self) -> None:
        """Check if selected model fits this hardware."""
        table = self.query_one("#models-table", ModelsTable)
        row = table.cursor_row
        if row is not None:
            cells = table.get_row_at(row)
            model_name = str(cells[1]).strip()
            self.exit(result=("willitfit", model_name))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_run_model()


def run_dashboard():
    """Entry point for the dashboard."""
    app = LocalfitDashboard()
    result = app.run()
    if result:
        action = result[0]
        if action == "run" and len(result) > 1:
            model = result[1]
            print(f"\n  Running {model}...")
            from localfit.wizard import run_wizard

            run_wizard(model=model)
        elif action == "cleanup":
            from localfit.backends import cleanup_gpu_memory

            cleanup_gpu_memory(force=True)
        elif action == "makeitfit" and len(result) > 1:
            model = result[1]
            print(f"\n  MakeItFit: {model}...")
            from localfit.makeitfit import makeitfit_wizard

            makeitfit_wizard(model)
        elif action in ("simulate", "willitfit") and len(result) > 1:
            model = result[1]
            model = model.replace(" (image)", "").strip()
            print(f"\n  Checking: {model}...")
            from localfit.backends import will_it_fit

            will_it_fit(model)
            input("\n  Press Enter to go back...")
            run_dashboard()
        elif action == "launch":
            from localfit.wizard import _pick_tool_and_launch

            running = _get_running()
            if running:
                m = running[0]
                _pick_tool_and_launch(
                    m["name"], f"http://127.0.0.1:{m['port']}/v1", _get_specs()
                )
            else:
                print("\n  No model running. Run one first.")
        elif action == "launch_with" and len(result) > 1:
            model_name = result[1]
            from localfit.wizard import _pick_tool_and_launch

            running = _get_running()
            # Find the matching running model
            port = 11434
            for m in running:
                if m["name"] == model_name:
                    port = m["port"]
                    break
            _pick_tool_and_launch(
                model_name, f"http://127.0.0.1:{port}/v1", _get_specs()
            )
