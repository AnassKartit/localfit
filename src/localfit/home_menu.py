"""Interactive Rich home screen for the default `localfit` command."""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from localfit.run_menu import _read_key

console = Console(highlight=False)


def _caps_text(caps):
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


def _label_text(item, selected=False):
    text = Text("", no_wrap=True, overflow="ellipsis", style="bold" if selected else "")
    text.append(item["label"])
    caps = _caps_text(item.get("caps"))
    if caps.plain:
        text.append("  ")
        text.append_text(caps)
    return text


def _detail_panel(system, item, width):
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=8, no_wrap=True)
    table.add_column(ratio=1)

    table.add_row("Status", Text(system["verdict"], style=f"bold {system['color']}"))
    table.add_row("GPU", Text(system["gpu"]))
    table.add_row("Swap", Text(system["swap"]))
    table.add_row("Disk", Text(system["disk"]))
    table.add_row("Model", Text(system["model"]))
    table.add_row("Machine", Text(system["machine"]))

    if item:
        table.add_row("", Text(""))
        table.add_row("Choice", Text(str(item.get("index", "")), style="bold bright_cyan"))
        table.add_row("Section", Text(item["section"], style=f"bold {item['accent']}"))
        table.add_row("Target", Text(item["label"], style="bold"))
        if item.get("source"):
            table.add_row("Source", Text(item["source"]))
        if item.get("repo"):
            table.add_row("Repo", Text(item["repo"], style="dim", no_wrap=True, overflow="ellipsis"))
        table.add_row("Notes", Text(item["detail"], style="dim"))

    return Panel(
        table,
        title=Text(item["label"] if item else "localfit", style="bold bright_white"),
        subtitle="Selected",
        border_style=item["accent"] if item else system["color"],
        padding=(0, 1),
        width=width,
    )


def _section_block(title, accent, rows, selected_pos):
    if not rows:
        return None

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=3, justify="right", no_wrap=True)
    table.add_column(width=8, no_wrap=True)
    table.add_column(ratio=1, no_wrap=True)
    table.add_column(width=22, justify="right", no_wrap=True)

    for pos, item in rows:
        is_selected = pos == selected_pos
        row_style = "bold black on grey85" if is_selected else ""
        index = Text(
            str(item.get("index", "")) if item.get("index") else "",
            style="bold bright_white" if is_selected else "dim",
        )
        badge = Text(item.get("badge", ""), style=f"bold {item['accent']}" if item.get("badge") else "")
        label = _label_text(item, selected=is_selected)
        meta = Text(item.get("meta", ""), style="bold bright_white" if is_selected else "dim")
        table.add_row(index, badge, label, meta, style=row_style)

    return Group(Text(title, style=f"bold {accent}"), table)


def _render_layout(system, items, selected_pos, width):
    detail_width = min(40, max(30, width // 3))
    options_width = max(30, width - detail_width - 7)
    current = items[selected_pos] if selected_pos is not None else None
    left = _detail_panel(system, current, detail_width)

    sections = []
    for section_title in ("ACTIVE", "INSTALLED", "RECOMMENDED", "COMPATIBLE", "TRENDING", "CLOUD"):
        rows = [(i, item) for i, item in enumerate(items) if item["section"] == section_title]
        accent = rows[0][1]["accent"] if rows else "bright_white"
        block = _section_block(section_title, accent, rows, selected_pos)
        if block is not None:
            if sections:
                sections.append(Text(""))
            sections.append(block)

    right = Panel(
        Group(*sections) if sections else Text("Nothing to show", style="dim"),
        title=Text("Choices", style="bold bright_white"),
        border_style="bright_white",
        padding=(0, 1),
        width=options_width,
    )

    if width < 92:
        return Group(left, Text(""), right)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=detail_width)
    grid.add_column(width=options_width)
    grid.add_row(left, right)
    return grid


def show_home_menu(system, items):
    selectable = [i for i, item in enumerate(items) if item.get("selectable")]
    selected_pos = selectable[0] if selectable else None
    width = max(20, min(console.width - 2, 120))
    confirm_quit = False

    from rich.live import Live

    with Live(console=console, auto_refresh=False, transient=True) as live:
        while True:
            body = _render_layout(system, items, selected_pos, width)
            subtitle = (
                f"{system['subtitle']}  "
                "[dim]↑↓/jk move · enter open · 1-9 jump · s simulate · b bench · h health · q quit[/]"
            )
            if confirm_quit:
                subtitle = (
                    f"{system['subtitle']}  "
                    "[bold yellow]Quit localfit?[/] [dim]y yes · n no[/]"
                )
            live.update(
                Panel(
                    body,
                    title=Text("localfit", style="bold bright_white"),
                    subtitle=subtitle,
                    border_style="bright_white",
                    width=width,
                    padding=(0, 1),
                ),
                refresh=True,
            )

            key = _read_key()
            if confirm_quit:
                if key in ("y", "enter", "ctrl-c"):
                    return {"action": "quit"}
                if key in ("n", "esc"):
                    confirm_quit = False
                continue
            if key in ("q", "esc", "ctrl-c"):
                confirm_quit = True
                continue
            if key == "s":
                return {"action": "simulate"}
            if key == "b":
                return {"action": "bench"}
            if key == "h":
                return {"action": "health"}
            if key in ("down", "j") and selectable and selected_pos is not None:
                idx = selectable.index(selected_pos)
                selected_pos = selectable[min(len(selectable) - 1, idx + 1)]
            elif key in ("up", "k") and selectable and selected_pos is not None:
                idx = selectable.index(selected_pos)
                selected_pos = selectable[max(0, idx - 1)]
            elif key == "enter" and selected_pos is not None:
                item = items[selected_pos]
                return {"action": item["action"], "repo": item.get("repo")}
            elif key.isdigit():
                digit = int(key)
                for item in items:
                    if item.get("index") == digit and item.get("selectable"):
                        return {"action": item["action"], "repo": item.get("repo")}
    return None
