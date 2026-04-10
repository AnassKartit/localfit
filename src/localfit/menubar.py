"""localfit menu bar — Ollama-style macOS menu bar app for local LLMs."""
import json, os, subprocess, threading, time, urllib.request
from pathlib import Path

import rumps

CONFIG_DIR = Path.home() / ".localfit"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
HEALTH_URL = "http://127.0.0.1:8089/health"
MODELS_URL = "http://127.0.0.1:8089/v1/models"


def _detect_terminal():
    """Detect which terminal app is available. Prefers iTerm2."""
    for app in ["iTerm", "iTerm2", "Ghostty", "Alacritty", "kitty", "Warp", "Terminal"]:
        r = subprocess.run(
            ["osascript", "-e", f'tell application "System Events" to (name of processes) contains "{app}"'],
            capture_output=True, text=True, timeout=3,
        )
        # Just check if the .app exists
        app_path = f"/Applications/{app}.app"
        if os.path.isdir(app_path):
            return app
    return "Terminal"


def _open_in_terminal(cmd):
    """Open a command in the user's preferred terminal."""
    settings = _load_settings()
    terminal = settings.get("terminal", _detect_terminal())

    if terminal in ("iTerm", "iTerm2"):
        script = f'''tell application "iTerm"
            activate
            create window with default profile command "/bin/zsh -l -c '{cmd}'"
        end tell'''
    elif terminal == "Ghostty":
        subprocess.Popen(["open", "-a", "Ghostty", "--args", "-e", f"/bin/zsh -l -c '{cmd}'"])
        return
    elif terminal == "kitty":
        subprocess.Popen(["kitty", "--", "zsh", "-l", "-c", cmd])
        return
    elif terminal == "Warp":
        script = f'tell application "Warp" to activate'
        subprocess.Popen(["osascript", "-e", script])
        # Warp doesn't support AppleScript do script well, use open
        subprocess.Popen(["open", "-a", "Warp"])
        return
    else:
        script = f'tell application "Terminal" to do script "{cmd}"'

    subprocess.Popen(["osascript", "-e", script])


def _load_settings():
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {"start_on_login": False, "auto_start_model": False,
            "default_model": "gemma4-26b", "default_tool": "localcoder"}


def _save_settings(s):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))


def _server_status():
    """Returns (healthy, model_name) tuple."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
            healthy = b'"ok"' in r.read()
    except Exception:
        return False, None
    if not healthy:
        return False, None
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=2) as r:
            data = json.loads(r.read())
        model = data.get("data", [{}])[0].get("id", "unknown")
        return True, model
    except Exception:
        return True, "unknown"


def _gpu_stats():
    """Get GPU usage. Returns (used_mb, total_mb) or (0, 0)."""
    try:
        from localfit.backends import get_metal_gpu_stats, get_machine_specs
        metal = get_metal_gpu_stats()
        specs = get_machine_specs()
        total = metal.get("total_mb") or specs.get("gpu_total_mb", 0)
        alloc = metal.get("alloc_mb", 0)
        return alloc, total
    except Exception:
        return 0, 0


def _installed_models():
    """Get list of installed model names and IDs."""
    try:
        from localfit.backends import get_disk_info, MODELS
        di = get_disk_info()
        installed = []
        for m in di.get("models", []):
            installed.append({"name": m["name"].replace(".gguf", ""), "path": m["path"],
                              "size_gb": m["size_gb"]})
        # Also include known models
        for mid, m in MODELS.items():
            from localfit.backends import find_model_file
            if find_model_file(mid):
                if not any(mid in i["name"].lower().replace("-", "") for i in installed):
                    installed.append({"name": m["name"], "model_id": mid,
                                      "size_gb": m["size_gb"]})
        return installed
    except Exception:
        return []


def _available_tools():
    """Check which coding tools are installed."""
    import shutil
    tools = []
    checks = [
        ("Claude Code", "claude"),
        ("Hermes Agent", "hermes"),
        ("localcoder", "localcoder"),
        ("OpenCode", "opencode"),
        ("Codex", "codex"),
        ("aider", "aider"),
    ]
    for label, binary in checks:
        if shutil.which(binary):
            tools.append({"label": label, "command": binary})
    return tools


class LocalFitMenuBar(rumps.App):
    def __init__(self):
        super().__init__("LocalFit", quit_button=None)
        self.settings = _load_settings()
        self._healthy = False
        self._model_name = None
        self._gpu_used = 0
        self._gpu_total = 0

        # Initial state
        self.icon = None  # Will use title as indicator
        self._update_title()

        # Build menu
        self._rebuild_menu()

        # Start background polling
        self._poll_timer = rumps.Timer(self._poll, 5)
        self._poll_timer.start()

        # Auto-start model if configured
        if self.settings.get("auto_start_model") and not self._healthy:
            threading.Thread(target=self._auto_start, daemon=True).start()

    def _update_title(self):
        if self._healthy:
            pct = int(self._gpu_used / max(1, self._gpu_total) * 100) if self._gpu_total else 0
            self.title = f"⚡ {self._gpu_used // 1024}/{self._gpu_total // 1024}GB"
        else:
            self.title = "⚡ off"

    def _rebuild_menu(self):
        self.menu.clear()

        # Status line
        if self._healthy and self._model_name:
            name = self._model_name.replace(".gguf", "")
            if len(name) > 30:
                name = name[:27] + "..."
            status = rumps.MenuItem(f"● {name}", callback=None)
            status.set_callback(None)
        else:
            status = rumps.MenuItem("○ No model running", callback=None)
            status.set_callback(None)
        self.menu.add(status)
        self.menu.add(rumps.separator)

        # Models section
        models_header = rumps.MenuItem("── Models ──")
        models_header.set_callback(None)
        self.menu.add(models_header)

        if self._healthy:
            stop_item = rumps.MenuItem("Stop Model", callback=self._stop_model)
            self.menu.add(stop_item)
        else:
            # Show installed models to start
            from localfit.backends import MODELS
            for mid, m in MODELS.items():
                from localfit.backends import find_model_file
                if find_model_file(mid):
                    item = rumps.MenuItem(
                        f"Start {m['name']} ({m['size_gb']}GB)",
                        callback=lambda _, mid=mid: self._start_model(mid),
                    )
                    self.menu.add(item)

        self.menu.add(rumps.separator)

        # Launch tools section
        tools = _available_tools()
        if tools:
            tools_header = rumps.MenuItem("── Launch ──")
            tools_header.set_callback(None)
            self.menu.add(tools_header)
            for t in tools:
                item = rumps.MenuItem(
                    t["label"],
                    callback=lambda _, cmd=t["command"]: self._launch_tool(cmd),
                )
                self.menu.add(item)
            terminal = rumps.MenuItem(
                "Terminal (localfit CLI)",
                callback=self._open_terminal,
            )
            self.menu.add(terminal)
            self.menu.add(rumps.separator)

        # System section
        sys_header = rumps.MenuItem("── System ──")
        sys_header.set_callback(None)
        self.menu.add(sys_header)

        if self._gpu_total:
            pct = int(self._gpu_used / max(1, self._gpu_total) * 100)
            gpu_item = rumps.MenuItem(
                f"GPU: {self._gpu_used // 1024}/{self._gpu_total // 1024}GB ({pct}%)")
            gpu_item.set_callback(None)
            self.menu.add(gpu_item)

        cleanup = rumps.MenuItem("Cleanup GPU Memory", callback=self._cleanup)
        self.menu.add(cleanup)
        self.menu.add(rumps.separator)

        # Settings
        settings_menu = rumps.MenuItem("Settings")

        login_item = rumps.MenuItem(
            "Start on Login",
            callback=self._toggle_login,
        )
        login_item.state = 1 if self.settings.get("start_on_login") else 0
        settings_menu.add(login_item)

        auto_model = rumps.MenuItem(
            "Auto-start Model",
            callback=self._toggle_auto_model,
        )
        auto_model.state = 1 if self.settings.get("auto_start_model") else 0
        settings_menu.add(auto_model)

        # Terminal picker submenu
        terminal_menu = rumps.MenuItem("Terminal")
        current_term = self.settings.get("terminal", _detect_terminal())
        for term in ["iTerm2", "Ghostty", "Warp", "kitty", "Alacritty", "Terminal"]:
            app_path = f"/Applications/{term}.app"
            if term == "Terminal" or os.path.isdir(app_path):
                item = rumps.MenuItem(
                    term,
                    callback=lambda _, t=term: self._set_terminal(t),
                )
                item.state = 1 if term == current_term or (term == "iTerm2" and current_term == "iTerm") else 0
                terminal_menu.add(item)
        settings_menu.add(terminal_menu)

        self.menu.add(settings_menu)
        self.menu.add(rumps.separator)

        # Quit
        self.menu.add(rumps.MenuItem("Quit LocalFit", callback=self._quit))

    def _poll(self, _):
        """Background poll for server status and GPU."""
        old_healthy = self._healthy
        self._healthy, self._model_name = _server_status()
        self._gpu_used, self._gpu_total = _gpu_stats()
        self._update_title()
        if self._healthy != old_healthy:
            self._rebuild_menu()

    def _auto_start(self):
        """Auto-start default model on app launch."""
        from localfit.daemon import start
        model = self.settings.get("default_model", "gemma4-26b")
        start(model)

    def _start_model(self, model_id):
        def _do():
            from localfit.daemon import start
            start(model_id)
        threading.Thread(target=_do, daemon=True).start()
        rumps.notification("LocalFit", "Starting model...", f"Loading {model_id}")

    def _stop_model(self, _):
        from localfit.daemon import stop
        stop()
        self._healthy = False
        self._model_name = None
        self._update_title()
        self._rebuild_menu()

    def _launch_tool(self, tool_cmd):
        """Launch a coding tool in the user's preferred terminal."""
        cmd = f"localfit --launch {tool_cmd}"
        _open_in_terminal(cmd)

    def _open_terminal(self, _):
        _open_in_terminal("localfit")

    def _cleanup(self, _):
        def _do():
            try:
                from localfit.backends import cleanup_gpu_memory
                result = cleanup_gpu_memory(force=True)
                count = len(result.get("ollama_unloaded", [])) + len(result.get("processes_killed", []))
                rumps.notification("LocalFit", "GPU Cleanup", f"Freed {count} items")
            except Exception as e:
                rumps.notification("LocalFit", "Cleanup Error", str(e))
        threading.Thread(target=_do, daemon=True).start()

    def _set_terminal(self, terminal_name):
        self.settings["terminal"] = terminal_name
        _save_settings(self.settings)
        rumps.notification("LocalFit", "Terminal", f"Set to {terminal_name}")
        self._rebuild_menu()

    def _toggle_login(self, sender):
        from localfit.launchagent import install, uninstall, is_installed
        if is_installed():
            uninstall()
            sender.state = 0
            self.settings["start_on_login"] = False
        else:
            install()
            sender.state = 1
            self.settings["start_on_login"] = True
        _save_settings(self.settings)

    def _toggle_auto_model(self, sender):
        self.settings["auto_start_model"] = not self.settings.get("auto_start_model", False)
        sender.state = 1 if self.settings["auto_start_model"] else 0
        _save_settings(self.settings)

    def _quit(self, _):
        rumps.quit_application()


def main():
    LocalFitMenuBar().run()


if __name__ == "__main__":
    main()
