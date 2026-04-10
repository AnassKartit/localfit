"""localfit daemon — manages llama-server lifecycle in the background."""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

PID_FILE = Path.home() / ".localfit" / "daemon.pid"
LOG_FILE = Path.home() / ".localfit" / "daemon.log"
HEALTH_URL = "http://127.0.0.1:8089/health"
MAX_RESTARTS = 3


def _write_pid(pid):
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid))


def _read_pid():
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def _clear_pid():
    PID_FILE.unlink(missing_ok=True)


def is_running():
    """Check if the daemon-managed llama-server is running."""
    pid = _read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        _clear_pid()
        return False


def server_healthy():
    """Check if llama-server is responding."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
            return b'"ok"' in r.read()
    except Exception:
        return False


def start(model_id=None):
    """Start llama-server for a model. Returns True if started successfully."""
    if is_running():
        return True

    from localfit.backends import MODELS, find_model_file

    model_id = model_id or _load_default_model()
    if not model_id:
        return False

    model = MODELS.get(model_id)
    if not model:
        return False

    model_file = find_model_file(model_id)
    if not model_file:
        return False

    binary = os.path.expanduser("~/.unsloth/llama.cpp/llama-server")
    if not os.path.exists(binary):
        import shutil
        binary = shutil.which("llama-server")
    if not binary:
        return False

    flags = model.get("server_flags", "-ngl 99 -c 32768 --jinja")
    cmd = f"{binary} -m {model_file} --port 8089 {flags}"

    log = open(LOG_FILE, "a")
    proc = subprocess.Popen(cmd.split(), stdout=log, stderr=log)
    _write_pid(proc.pid)

    # Wait for health
    for _ in range(60):
        if server_healthy():
            return True
        time.sleep(2)

    proc.kill()
    _clear_pid()
    return False


def stop():
    """Stop the daemon-managed llama-server."""
    pid = _read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait up to 5s for graceful shutdown
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break
        except OSError:
            pass
    _clear_pid()


def restart(model_id=None):
    """Restart the server."""
    stop()
    time.sleep(1)
    return start(model_id)


def watch(model_id=None):
    """Run a watch loop — restart server if it dies. For use by LaunchAgent."""
    restarts = 0
    model_id = model_id or _load_default_model()

    while restarts < MAX_RESTARTS:
        if not start(model_id):
            restarts += 1
            time.sleep(5)
            continue

        restarts = 0  # reset on successful start

        # Monitor loop
        while True:
            time.sleep(10)
            if not server_healthy() and not is_running():
                break  # server died, restart

    _clear_pid()


def _load_default_model():
    settings_file = Path.home() / ".localfit" / "settings.json"
    if settings_file.exists():
        try:
            return json.loads(settings_file.read_text()).get("default_model", "gemma4-26b")
        except Exception:
            pass
    return "gemma4-26b"
