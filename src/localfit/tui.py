"""localcoder TUI — Textual-based fixed-layout GPU health dashboard."""
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Footer, Header, DataTable, LoadingIndicator
from textual.reactive import reactive
from textual import work


class StatusIndicator(Static):
    """Top status bar: HEALTHY / DEGRADED / CRITICAL."""
    status = reactive("scanning")

    def render(self):
        colors = {
            "healthy": "green", "degraded": "yellow",
            "critical": "red", "scanning": "cyan",
        }
        c = colors.get(self.status, "dim")
        return f"[bold {c}] {self.status.upper()} [/]"


class GpuBar(Static):
    """Visual VRAM usage bar."""
    gpu_alloc = reactive(0)
    gpu_total = reactive(16384)

    def render(self):
        pct = min(1.0, self.gpu_alloc / max(1, self.gpu_total))
        w = 40
        filled = int(pct * w)
        color = "green" if pct < 0.7 else "yellow" if pct < 0.9 else "red"
        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (w - filled)}[/dim]"
        return f"  VRAM {bar} {self.gpu_alloc // 1024}/{self.gpu_total // 1024}GB"


class InfoCard(Static):
    """A status card (Compute / KV Cache / Memory)."""
    DEFAULT_CSS = """
    InfoCard {
        width: 1fr;
        height: auto;
        min-height: 7;
        border: solid $primary;
        padding: 0 1;
    }
    """


class BottomBar(Static):
    """Pinned bottom status bar with GPU stats + shortcuts."""
    DEFAULT_CSS = """
    BottomBar {
        dock: bottom;
        height: 1;
        background: $surface;
    }
    """
    gpu_text = reactive("")

    def render(self):
        return self.gpu_text or " GPU --/--  SWAP --  MEM --"


class HealthDashboard(App):
    """GPU Health Dashboard — fixed layout, no scrolling."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #header-bar {
        height: 3;
        padding: 0 1;
    }
    #cards-row {
        height: auto;
        min-height: 8;
        max-height: 10;
    }
    #vram-bar {
        height: 2;
        padding: 0 1;
    }
    #procs-panel {
        height: 1fr;
        padding: 0 1;
    }
    #fixes-panel {
        height: auto;
        max-height: 8;
        padding: 0 1;
        border: solid $warning;
    }
    InfoCard {
        border: solid $primary;
        padding: 0 1;
        width: 1fr;
    }
    #status-line {
        height: 1;
        dock: top;
    }
    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "cleanup", "Cleanup"),
        ("d", "debloat", "Debloat"),
        ("s", "simulate", "Simulate"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield StatusIndicator(id="status-line")
        yield Static(id="header-bar")

        with Horizontal(id="cards-row"):
            yield InfoCard(id="card-compute")
            yield InfoCard(id="card-kv")
            yield InfoCard(id="card-mem")

        yield GpuBar(id="vram-bar")

        yield Container(
            DataTable(id="proc-table"),
            id="procs-panel",
        )

        yield Static(id="fixes-panel")
        yield BottomBar(id="bottom-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Start loading data."""
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        """Load GPU data in background thread."""
        from localfit.backends import (
            get_machine_specs, diagnose_gpu_health, get_metal_gpu_stats,
            get_top_memory_processes, get_swap_usage_mb, _detect_model_info,
        )

        specs = get_machine_specs()
        diag = diagnose_gpu_health()
        metal = get_metal_gpu_stats()
        procs = get_top_memory_processes(min_mb=80, limit=8)
        swap_mb = get_swap_usage_mb()
        model_info = _detect_model_info(diag["server_config"], None)

        # Update UI from worker thread
        self.call_from_thread(self._update_ui, specs, diag, metal, procs, swap_mb, model_info)

    def _update_ui(self, specs, diag, metal, procs, swap_mb, model_info):
        """Update all widgets with loaded data."""
        # Status
        status = self.query_one("#status-line", StatusIndicator)
        status.status = diag["status"]

        # Header
        header = self.query_one("#header-bar", Static)
        model_str = ""
        if model_info["name"]:
            parts = [model_info["name"]]
            if model_info["quant"]:
                parts.append(model_info["quant"])
            if model_info["size_gb"]:
                parts.append(f"{model_info['size_gb']}GB")
            model_str = f"\n  [cyan]{'  ·  '.join(parts)}[/cyan]"
        header.update(
            f"  [bold]{specs['chip']}[/bold]  ·  {specs['ram_gb']}GB RAM  ·  "
            f"{specs.get('gpu_cores', '?')} GPU cores{model_str}"
        )

        # Compute card
        srv = diag["server_config"]
        if srv.get("running"):
            gpu_icon = "[green]●[/] GPU (Metal)" if diag["on_gpu"] else "[red]●[/] CPU — SLOW"
            compute = (
                f"[bold]Compute[/bold]\n"
                f"{gpu_icon}\n"
                f"Layers: {diag['gpu_layers']}/99\n"
                f"Util: {diag['gpu_util_pct']}%\n"
                f"Model: {srv.get('footprint_mb', 0)} MB"
            )
        else:
            compute = "[bold]Compute[/bold]\n[dim]Server not running[/dim]"
        self.query_one("#card-compute", InfoCard).update(compute)

        # KV Cache card
        kv_icon = "[green]●[/]" if diag["kv_quantized"] else "[red]●[/]"
        kv_type = f"Type: {diag['kv_type']}\n" if diag["kv_type"] else ""
        fa_icon = "[green]●[/]" if diag["flash_attn"] else "[yellow]●[/]"
        self.query_one("#card-kv", InfoCard).update(
            f"[bold]KV Cache[/bold]\n"
            f"{kv_icon} {'Quantized' if diag['kv_quantized'] else 'Full (2x mem!)'}\n"
            f"{kv_type}"
            f"Size: ~{diag['kv_cache_est_mb']} MB\n"
            f"Ctx: {diag['context_size'] // 1024}K\n"
            f"{fa_icon} FlashAttn: {'on' if diag['flash_attn'] else 'off'}"
        )

        # Memory card
        pc = {"normal": "green", "warn": "yellow", "critical": "red"}.get(diag["mem_pressure"], "dim")
        sc = "red" if swap_mb > 4000 else "green"
        headroom = diag["gpu_total_mb"] - diag["gpu_alloc_mb"]
        hc = "green" if headroom > 2048 else "yellow" if headroom > 0 else "red"
        self.query_one("#card-mem", InfoCard).update(
            f"[bold]Memory[/bold]\n"
            f"Pressure: [{pc}]{diag['mem_pressure']}[/{pc}]\n"
            f"Swap: [{sc}]{swap_mb // 1024}GB[/{sc}]\n"
            f"GPU: {diag['gpu_alloc_mb'] // 1024}/{diag['gpu_total_mb'] // 1024}GB\n"
            f"Free: [{hc}]{headroom // 1024}GB[/{hc}]"
        )

        # VRAM bar
        vram = self.query_one("#vram-bar", GpuBar)
        vram.gpu_alloc = diag["gpu_alloc_mb"]
        vram.gpu_total = diag["gpu_total_mb"]

        # Process table
        table = self.query_one("#proc-table", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Process", "Memory", "Type")
        for i, p in enumerate(procs, 1):
            mb = p["mb"]
            size = f"{mb / 1024:.1f}G" if mb >= 1024 else f"{mb}M"
            cat = {"ml": "ML", "app": "app", "system": "sys", "bloat": "bloat"}.get(p["category"], "?")
            name = p["name"] + (f" ×{p['count']}" if p.get("count", 1) > 1 else "")
            table.add_row(str(i), name, size, cat)

        # Fixes
        fixes = self.query_one("#fixes-panel", Static)
        if diag["issues"]:
            lines = []
            for issue in diag["issues"]:
                lines.append(f"[red]●[/] {issue}")
            for fix in diag.get("fixes", []):
                lines.append(f"[green]→[/] {fix}")
            fixes.update("\n".join(lines))
        else:
            fixes.update("[green]All good — no issues detected[/]")

        # Bottom bar
        bar = self.query_one("#bottom-bar", BottomBar)
        gc = "red" if diag["gpu_alloc_mb"] > diag["gpu_total_mb"] else "green"
        bar.gpu_text = (
            f" GPU [{gc}]{diag['gpu_alloc_mb'] // 1024}/{diag['gpu_total_mb'] // 1024}GB[/{gc}]"
            f"  SWAP [{sc}]{swap_mb // 1024}GB[/{sc}]"
            f"  MEM [{pc}]{diag['mem_pressure']}[/{pc}]"
        )

    def action_refresh(self) -> None:
        status = self.query_one("#status-line", StatusIndicator)
        status.status = "scanning"
        self.load_data()

    def action_cleanup(self) -> None:
        self.exit(return_code=10)  # Signal to CLI to run cleanup

    def action_debloat(self) -> None:
        self.exit(return_code=11)

    def action_simulate(self) -> None:
        self.exit(return_code=12)


def run_tui_dashboard():
    """Launch the Textual TUI dashboard."""
    app = HealthDashboard()
    result = app.run()
    return result
