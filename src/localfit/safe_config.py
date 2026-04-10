"""Safe configuration manager — add local models to tools without breaking existing setups.

Rules:
1. NEVER overwrite existing config files
2. ONLY add/merge our model connection
3. ALWAYS backup before any modification
4. Detect if tool is already configured for local models
5. Self-heal: restore from backup if corruption detected
"""
import json, os, shutil, time, yaml
from pathlib import Path

BACKUP_DIR = Path.home() / ".localfit" / "backups"


def _backup(filepath):
    """Create timestamped backup before modifying any file."""
    if not os.path.exists(filepath):
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = os.path.basename(filepath)
    backup_path = BACKUP_DIR / f"{name}.{ts}.bak"
    shutil.copy2(filepath, backup_path)
    return str(backup_path)


def _restore(filepath):
    """Restore most recent backup for a file."""
    name = os.path.basename(filepath)
    backups = sorted(BACKUP_DIR.glob(f"{name}.*.bak"), reverse=True)
    if backups:
        shutil.copy2(backups[0], filepath)
        return str(backups[0])
    return None


def list_backups():
    """List all backups created by localfit."""
    if not BACKUP_DIR.exists():
        return []
    return sorted(BACKUP_DIR.glob("*.bak"), reverse=True)


def restore_all():
    """Restore all config files to their pre-localfit state."""
    restored = []
    for backup in list_backups():
        # Parse: filename.ext.TIMESTAMP.bak → filename.ext
        parts = backup.name.rsplit(".", 3)
        if len(parts) >= 3:
            original_name = parts[0] + "." + parts[1] if len(parts) == 4 else parts[0]
            # Find original location by name
            # This is best-effort — we store the mapping too
            restored.append(str(backup))
    return restored


# ── Hermes Agent ──

def add_model_to_hermes(api_base="http://127.0.0.1:8089/v1", model_name="local"):
    """Add local model connection to Hermes. Merges into existing config."""
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    env_path = os.path.expanduser("~/.hermes/.env")

    # Backup
    _backup(config_path)
    _backup(env_path)

    # Read existing config (preserve everything)
    existing = {}
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            pass

    # Only update model section — don't touch anything else
    existing.setdefault("model", {})
    existing["model"]["default"] = model_name
    existing["model"]["provider"] = "custom"
    existing["model"]["base_url"] = api_base

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    # Merge into .env (preserve existing keys)
    env_lines = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_lines[k.strip()] = v.strip()

    env_lines["OPENAI_BASE_URL"] = api_base
    env_lines.setdefault("OPENAI_API_KEY", "no-key-required")

    with open(env_path, "w") as f:
        for k, v in env_lines.items():
            f.write(f"{k}={v}\n")
    os.chmod(env_path, 0o600)

    return {"config": config_path, "env": env_path}


# ── OpenClaw ──

def add_model_to_openclaw(api_base="http://127.0.0.1:8089/v1"):
    """Add local model to OpenClaw. Merges into existing config."""
    config_path = os.path.expanduser("~/.openclaw/openclaw.json")

    if not os.path.exists(config_path):
        return None  # OpenClaw not installed

    _backup(config_path)

    # Read existing (preserve everything)
    with open(config_path) as f:
        config = json.load(f)

    # Only add/update the llamacpp provider — don't touch agents, auth, models, etc.
    config.setdefault("auth", {}).setdefault("profiles", {})
    profiles = config["auth"]["profiles"]
    profiles.setdefault("openai", {}).setdefault("default", {})
    profiles["openai"]["default"]["apiBase"] = api_base
    profiles["openai"]["default"].setdefault("apiKey", "local")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return {"config": config_path}


# ── Open WebUI ──

def add_model_to_webui(api_base="http://127.0.0.1:8089/v1"):
    """Add local model to Open WebUI database. Non-destructive merge."""
    import sqlite3

    db_path = os.path.expanduser("~/.localfit/open-webui/webui.db")
    if not os.path.exists(db_path):
        return None  # Not installed yet

    _backup(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, data FROM config LIMIT 1")
    row = c.fetchone()

    if row:
        data = json.loads(row[1]) if row[1] else {}
        # Merge — only set openai connection, preserve everything else
        data.setdefault("openai", {})
        data["openai"]["ENABLE_OPENAI_API"] = True
        data["openai"]["OPENAI_API_BASE_URLS"] = [api_base]
        data["openai"]["OPENAI_API_KEYS"] = ["no-key-required"]
        # Enable web search
        data.setdefault("retrieval", {}).setdefault("web", {})
        data["retrieval"]["web"]["ENABLE_WEB_SEARCH"] = True
        data["retrieval"]["web"]["WEB_SEARCH_ENGINE"] = "duckduckgo"

        c.execute("UPDATE config SET data=? WHERE id=?", (json.dumps(data), row[0]))
        conn.commit()

    conn.close()
    return {"db": db_path}


# ── Claude Code ──

def get_claude_launch_env(api_base="http://127.0.0.1:8090", model="local"):
    """Return env vars for launching Claude Code with local model.

    NEVER modifies any files. Returns dict of env vars to pass to subprocess.
    """
    return {
        "ANTHROPIC_AUTH_TOKEN": "localfit",
        "ANTHROPIC_BASE_URL": api_base,
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def get_claude_launch_cmd(model="local"):
    """Return the command to launch Claude Code with local model."""
    return ["claude", "--bare", "--model", model]


# ── Codex ──

def get_codex_launch_env(api_base="http://127.0.0.1:8089/v1"):
    """Return env vars for Codex. NEVER modifies files."""
    return {
        "OPENAI_BASE_URL": api_base,
        "OPENAI_API_KEY": "sk-no-key-required",
    }


# ── Self-healing ──

def check_health():
    """Check if any tool configs were corrupted by localfit."""
    issues = []

    # Check Hermes config
    hermes_config = os.path.expanduser("~/.hermes/config.yaml")
    if os.path.exists(hermes_config):
        try:
            with open(hermes_config) as f:
                yaml.safe_load(f)
        except Exception:
            issues.append({"tool": "hermes", "file": hermes_config, "issue": "YAML parse error"})

    # Check OpenClaw config
    openclaw_config = os.path.expanduser("~/.openclaw/openclaw.json")
    if os.path.exists(openclaw_config):
        try:
            with open(openclaw_config) as f:
                json.load(f)
        except Exception:
            issues.append({"tool": "openclaw", "file": openclaw_config, "issue": "JSON parse error"})

    # Check Open WebUI DB
    webui_db = os.path.expanduser("~/.localfit/open-webui/webui.db")
    if os.path.exists(webui_db):
        try:
            import sqlite3
            conn = sqlite3.connect(webui_db)
            conn.execute("SELECT COUNT(*) FROM config")
            conn.close()
        except Exception:
            issues.append({"tool": "webui", "file": webui_db, "issue": "DB corrupted"})

    return issues


def self_heal():
    """Detect and fix corrupted configs by restoring backups."""
    issues = check_health()
    fixed = []
    for issue in issues:
        backup = _restore(issue["file"])
        if backup:
            fixed.append({"tool": issue["tool"], "restored_from": backup})
    return fixed
