"""localfit LaunchAgent — auto-start menu bar app on macOS login."""
import os, plistlib, subprocess, sys
from pathlib import Path

LABEL = "com.localfit.menubar"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG_DIR = Path.home() / ".localfit"


def _get_python():
    """Get the Python executable that runs localfit."""
    return sys.executable


def install():
    """Register the LaunchAgent to start menu bar on login."""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    plist = {
        "Label": LABEL,
        "ProgramArguments": [_get_python(), "-m", "localfit.menubar"],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(LOG_DIR / "menubar.log"),
        "StandardErrorPath": str(LOG_DIR / "menubar.err"),
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
        },
    }

    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)

    subprocess.run(["launchctl", "load", str(PLIST_PATH)], capture_output=True)
    return True


def uninstall():
    """Remove the LaunchAgent."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], capture_output=True)
        PLIST_PATH.unlink(missing_ok=True)
    return True


def is_installed():
    """Check if the LaunchAgent is registered."""
    return PLIST_PATH.exists()


def status():
    """Check if the LaunchAgent is loaded and running."""
    if not is_installed():
        return {"installed": False, "running": False}
    result = subprocess.run(
        ["launchctl", "list", LABEL],
        capture_output=True, text=True,
    )
    running = result.returncode == 0
    return {"installed": True, "running": running}
