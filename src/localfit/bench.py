"""localfit bench — local model arena. Download, test, rank models automatically."""
import json, os, time, subprocess, urllib.request, urllib.parse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape

console = Console()
BENCH_FILE = Path.home() / ".localfit" / "benchmarks.json"
LLAMA_SERVER = Path.home() / ".unsloth/llama.cpp/llama-server"
PORT = 8099  # dedicated bench port

# ── Test Suite ──
TESTS = [
    {
        "id": "code_function",
        "name": "Write Function",
        "category": "coding",
        "prompt": "Write a Python function that finds the longest palindromic substring. Include type hints and docstring.",
        "check": lambda r: ("def " in r) + ("->" in r or ": str" in r) + ('"""' in r or "'''" in r),
        "max_score": 3,
    },
    {
        "id": "code_debug",
        "name": "Debug Code",
        "category": "coding",
        "prompt": "This Python code has a bug. Fix it and explain:\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1",
        "check": lambda r: ("left = mid + 1" in r or "left = mid+1" in r) + ("infinite" in r.lower() or "loop" in r.lower() or "bug" in r.lower()) + ("def " in r or "fix" in r.lower()),
        "max_score": 3,
    },
    {
        "id": "reasoning",
        "name": "Logic Puzzle",
        "category": "reasoning",
        "prompt": "A farmer has a fox, a chicken, and a sack of grain. He must cross a river in a boat that can only carry him and one item. If left alone, the fox will eat the chicken and the chicken will eat the grain. How does he get everything across?",
        "check": lambda r: ("chicken" in r.lower()) + ("fox" in r.lower()) + ("grain" in r.lower()) + (r.lower().count("cross") >= 2 or r.lower().count("take") >= 2 or r.lower().count("bring") >= 2),
        "max_score": 4,
    },
    {
        "id": "tool_json",
        "name": "Tool Call JSON",
        "category": "tool_use",
        "prompt": 'You have a tool called "search" with parameter "query" (string). The user says: "find the weather in Paris". Respond with a JSON tool call.',
        "check": lambda r: ('"search"' in r or "'search'" in r) + ('"query"' in r or "'query'" in r) + ("paris" in r.lower()) + ("{" in r),
        "max_score": 4,
    },
    {
        "id": "follow_instructions",
        "name": "Follow Instructions",
        "category": "instruction",
        "prompt": "List exactly 5 programming languages that start with the letter P. Output only the list, nothing else.",
        "check": lambda r: (r.lower().count("python") >= 1) + (sum(1 for line in r.strip().splitlines() if line.strip()) <= 7) + ("perl" in r.lower() or "php" in r.lower() or "prolog" in r.lower()),
        "max_score": 3,
    },
]


def _load_results():
    """Load previous benchmark results."""
    if BENCH_FILE.exists():
        try:
            return json.loads(BENCH_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_results(results):
    """Save benchmark results."""
    BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    BENCH_FILE.write_text(json.dumps(results, indent=2))


def _find_gguf(name_pattern):
    """Find a GGUF file matching a pattern in HF cache."""
    cache = Path.home() / ".cache/huggingface/hub"
    for f in cache.rglob("*.gguf"):
        if "mmproj" in f.name.lower():
            continue
        real = f.resolve()
        if name_pattern.lower().replace("-", "") in f.name.lower().replace("-", ""):
            return str(real)
    return None


def _find_mmproj(model_dir_pattern):
    """Find mmproj file for vision models."""
    cache = Path.home() / ".cache/huggingface/hub"
    for d in cache.iterdir():
        if model_dir_pattern.lower().replace("-","") in d.name.lower().replace("-",""):
            for f in d.rglob("*mmproj*"):
                return str(f.resolve())
    return None


def _start_server(gguf_path, extra_flags=""):
    """Start llama-server with a model, return process."""
    binary = str(LLAMA_SERVER)
    if not os.path.exists(binary):
        import shutil
        binary = shutil.which("llama-server")
    if not binary:
        return None

    cmd = f"{binary} -m {gguf_path} --port {PORT} -ngl 99 -c 8192 -fa on -ctk q4_0 -ctv q4_0 --jinja {extra_flags}"
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(60):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
            with urllib.request.urlopen(req, timeout=1):
                return proc
        except Exception:
            time.sleep(1)

    proc.kill()
    return None


def _run_test(test, max_tokens=512):
    """Run a single test against the server. Returns score, tok/s, response."""
    payload = json.dumps({
        "model": "bench",
        "messages": [{"role": "user", "content": test["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()

    try:
        t0 = time.time()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - t0

        content = data["choices"][0]["message"].get("content", "")
        tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))
        tps = tokens / elapsed if elapsed > 0 else 0
        score = test["check"](content)

        return {
            "score": score,
            "max_score": test["max_score"],
            "tps": round(tps, 1),
            "tokens": tokens,
            "time": round(elapsed, 1),
            "ttft": round(elapsed - (tokens / max(1, tps)), 2) if tps > 0 else 0,
        }
    except Exception as e:
        return {"score": 0, "max_score": test["max_score"], "tps": 0, "tokens": 0, "time": 0, "error": str(e)}


def _download_model(repo, filename):
    """Download a GGUF from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=filename)
        return path
    except ImportError:
        console.print("  [red]pip install huggingface_hub[/]")
        return None
    except Exception as e:
        console.print(f"  [red]{e}[/]")
        return None


def bench_model(name, gguf_path, extra_flags="", skip_if_cached=True):
    """Benchmark a single model. Returns results dict."""
    results = _load_results()

    if skip_if_cached and name in results:
        console.print(f"  [dim]{name}: cached results (run --bench --force to re-test)[/]")
        return results[name]

    console.print(f"\n  [bold cyan]{name}[/]")
    console.print(f"  [dim]Loading model...[/]")

    proc = _start_server(gguf_path, extra_flags)
    if not proc:
        console.print(f"  [red]Failed to start server[/]")
        return None

    console.print(f"  [green]Server ready[/] — running {len(TESTS)} tests")

    model_results = {"name": name, "path": gguf_path, "tests": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M")}
    total_score = 0
    total_max = 0
    total_tps = []

    for test in TESTS:
        console.print(f"    {test['name']:<22}", end="")
        r = _run_test(test)
        stars = r["score"]
        max_s = r["max_score"]
        total_score += stars
        total_max += max_s
        if r["tps"] > 0:
            total_tps.append(r["tps"])

        bar = "[green]" + "★" * stars + "[/][dim]" + "☆" * (max_s - stars) + "[/]"
        console.print(f"  {bar}  {r['tps']:>5.1f} tok/s  {r['time']:>5.1f}s")

        model_results["tests"][test["id"]] = r

    avg_tps = sum(total_tps) / len(total_tps) if total_tps else 0
    model_results["total_score"] = total_score
    model_results["total_max"] = total_max
    model_results["avg_tps"] = round(avg_tps, 1)
    model_results["pct"] = round(total_score / total_max * 100) if total_max > 0 else 0

    console.print(f"    {'─' * 50}")
    console.print(f"    [bold]Total: {total_score}/{total_max} ({model_results['pct']}%)  ·  {avg_tps:.1f} tok/s avg[/]")

    proc.kill()
    proc.wait()
    time.sleep(2)

    # Save
    results[name] = model_results
    _save_results(results)

    return model_results


def show_leaderboard():
    """Show the benchmark leaderboard like LM Arena."""
    results = _load_results()
    if not results:
        console.print("\n  [dim]No benchmarks yet. Run: localfit --bench[/]\n")
        return

    console.print()

    table = Table(
        title="[bold]localfit Arena — Model Leaderboard[/]",
        title_style="bold #e07a5f",
        show_header=True, header_style="bold",
        border_style="dim", padding=(0, 1),
    )
    table.add_column("#", style="bold", width=3)
    table.add_column("Model", width=28)
    table.add_column("Score", justify="center", width=8)
    table.add_column("tok/s", justify="right", width=7)
    table.add_column("Code", justify="center", width=6)
    table.add_column("Reason", justify="center", width=6)
    table.add_column("Tools", justify="center", width=6)
    table.add_column("Instruct", justify="center", width=6)
    table.add_column("", width=16)

    # Sort by score, then speed
    ranked = sorted(results.values(), key=lambda x: (x.get("pct", 0), x.get("avg_tps", 0)), reverse=True)

    for i, r in enumerate(ranked, 1):
        pct = r.get("pct", 0)
        tps = r.get("avg_tps", 0)
        tests = r.get("tests", {})

        # Category scores
        code_score = sum(tests.get(t, {}).get("score", 0) for t in ["code_function", "code_debug"])
        code_max = sum(tests.get(t, {}).get("max_score", 3) for t in ["code_function", "code_debug"])
        reason_score = tests.get("reasoning", {}).get("score", 0)
        reason_max = tests.get("reasoning", {}).get("max_score", 4)
        tool_score = tests.get("tool_json", {}).get("score", 0)
        tool_max = tests.get("tool_json", {}).get("max_score", 4)
        inst_score = tests.get("follow_instructions", {}).get("score", 0)
        inst_max = tests.get("follow_instructions", {}).get("max_score", 3)

        # Visual bar
        bar_w = 14
        filled = int(pct / 100 * bar_w)
        bc = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
        bar = f"[{bc}]{'█' * filled}[/{bc}][dim]{'░' * (bar_w - filled)}[/]"

        # Medal
        medal = {1: "[bold yellow]🥇[/]", 2: "[white]🥈[/]", 3: "[#cd7f32]🥉[/]"}.get(i, f"  ")

        score_str = f"[bold]{pct}%[/]"
        tps_str = f"{tps:.0f}"

        table.add_row(
            medal,
            escape(r.get("name", "?")),
            score_str,
            tps_str,
            f"{code_score}/{code_max}",
            f"{reason_score}/{reason_max}",
            f"{tool_score}/{tool_max}",
            f"{inst_score}/{inst_max}",
            bar,
        )

    console.print(table)
    console.print(f"\n  [dim]Tested on {ranked[0].get('timestamp', '?') if ranked else '?'}  ·  M4 Pro 24GB  ·  llama.cpp[/]")
    console.print(f"  [dim]Run [bold]localfit --bench[/bold] to test more models  ·  Results in ~/.localfit/benchmarks.json[/]\n")


def bench_hermes(port=8089):
    """Benchmark Hermes Agent with the running local model.

    Tests that Hermes Agent can connect to the local llama-server,
    send a prompt, and get a valid response via OpenAI-compatible API.
    """
    from localfit.prerequisites import check_hermes_agent

    ha = check_hermes_agent()
    if not ha["found"]:
        console.print("  [yellow]Hermes Agent not installed — skipping[/]")
        return None

    console.print(f"\n  [bold cyan]Hermes Agent Integration Test[/]")

    results = {"name": "hermes-agent", "tests": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M")}
    total_pass = 0
    total_tests = 0

    # Test 1: API connectivity — can we reach the model via OpenAI-compatible endpoint?
    total_tests += 1
    console.print(f"    API connectivity        ", end="")
    try:
        payload = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 128,
            "temperature": 0.1,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - t0
        msg = data["choices"][0]["message"]
        # Some models put output in reasoning_content before content
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        if content.strip():
            console.print(f"  [green]PASS[/]  {elapsed:.1f}s")
            total_pass += 1
            results["tests"]["api_connect"] = {"pass": True, "time": round(elapsed, 1)}
        else:
            console.print(f"  [red]FAIL[/]  empty response")
            results["tests"]["api_connect"] = {"pass": False, "error": "empty response"}
    except Exception as e:
        console.print(f"  [red]FAIL[/]  {e}")
        results["tests"]["api_connect"] = {"pass": False, "error": str(e)}

    # Test 2: Tool calling format — does the model produce valid tool call JSON?
    total_tests += 1
    console.print(f"    Tool call format        ", end="")
    try:
        payload = json.dumps({
            "model": "local",
            "messages": [{"role": "user", "content": "Use the search tool to find weather in Tokyo. Respond with a JSON tool call: {\"name\": \"search\", \"arguments\": {\"query\": \"...\"}}"}],
            "max_tokens": 512,
            "temperature": 0.1,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - t0
        msg = data["choices"][0]["message"]
        content = (msg.get("content", "") or "") + (msg.get("reasoning_content", "") or "")
        has_json = "{" in content and "search" in content.lower()
        if has_json:
            console.print(f"  [green]PASS[/]  {elapsed:.1f}s")
            total_pass += 1
            results["tests"]["tool_call"] = {"pass": True, "time": round(elapsed, 1)}
        else:
            console.print(f"  [red]FAIL[/]  no tool JSON in response")
            results["tests"]["tool_call"] = {"pass": False, "error": "no tool JSON"}
    except Exception as e:
        console.print(f"  [red]FAIL[/]  {e}")
        results["tests"]["tool_call"] = {"pass": False, "error": str(e)}

    # Test 3: Multi-turn conversation
    total_tests += 1
    console.print(f"    Multi-turn conversation ", end="")
    try:
        payload = json.dumps({
            "model": "local",
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! How can I help you?"},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 128,
            "temperature": 0.1,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - t0
        msg = data["choices"][0]["message"]
        content = (msg.get("content", "") or "") + (msg.get("reasoning_content", "") or "")
        if "alice" in content.lower():
            console.print(f"  [green]PASS[/]  {elapsed:.1f}s")
            total_pass += 1
            results["tests"]["multi_turn"] = {"pass": True, "time": round(elapsed, 1)}
        else:
            console.print(f"  [red]FAIL[/]  didn't recall name")
            results["tests"]["multi_turn"] = {"pass": False, "error": "no name recall"}
    except Exception as e:
        console.print(f"  [red]FAIL[/]  {e}")
        results["tests"]["multi_turn"] = {"pass": False, "error": str(e)}

    # Test 4: Hermes config file exists
    total_tests += 1
    console.print(f"    Config file             ", end="")
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if os.path.exists(config_path):
        console.print(f"  [green]PASS[/]  {config_path}")
        total_pass += 1
        results["tests"]["config"] = {"pass": True}
    else:
        console.print(f"  [yellow]SKIP[/]  run 'localfit --config hermes' first")
        results["tests"]["config"] = {"pass": False, "error": "no config"}

    results["total_pass"] = total_pass
    results["total_tests"] = total_tests
    results["pct"] = round(total_pass / total_tests * 100) if total_tests > 0 else 0

    console.print(f"    {'─' * 50}")
    color = "green" if total_pass == total_tests else "yellow" if total_pass > 0 else "red"
    console.print(f"    [bold {color}]{total_pass}/{total_tests} passed ({results['pct']}%)[/]")

    return results


def run_full_bench(force=False):
    """Run benchmarks on all installed models."""
    from localfit.backends import get_disk_info

    console.print("\n  [bold #e07a5f]localfit Arena[/] — benchmarking all installed models\n")

    di = get_disk_info()
    models_to_test = []

    for m in di.get("models", []):
        name = m["name"].replace(".gguf", "")
        size = m["size_gb"]
        path = m["path"]

        if size * 1024 > 16384:
            console.print(f"  [dim]Skip {name} ({size}GB) — won't fit GPU[/]")
            continue

        # Determine extra flags based on model name
        extra = ""
        if "gemma" in name.lower():
            extra = "--reasoning off --no-mmproj"
        elif "qwen" in name.lower():
            extra = "--reasoning-budget 0"

        models_to_test.append((name, path, extra))

    console.print(f"  Testing {len(models_to_test)} models...\n")

    for name, path, extra in models_to_test:
        bench_model(name, path, extra, skip_if_cached=not force)

    # Run Hermes Agent integration test if installed
    from localfit.prerequisites import check_hermes_agent
    ha = check_hermes_agent()
    if ha["found"]:
        bench_hermes()

    console.print()
    show_leaderboard()
