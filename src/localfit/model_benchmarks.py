"""Model benchmark data from Unsloth documentation.

Source: unsloth.ai/docs/models/
Updated: 2026-04-06

These are published benchmark scores, not our own tests.
localfit shows these alongside fit checks so users know quality, not just size.
"""

# Format: model_base_name → {benchmarks, notes, source_url}
UNSLOTH_BENCHMARKS = {
    "qwen3-coder-next": {
        "params": "80B MoE (3B active)",
        "context": "256K",
        "min_ram": "46GB",
        "benchmarks": {
            "SWE-Bench Verified": 70.6,
            "SWE-Bench Multilingual": 62.8,
            "SWE-Bench Pro": 44.3,
            "Aider": 66.2,
            "Terminal-Bench 2.0": 36.2,
        },
        "quant_note": "IQ3_XXS comes close to BF16 performance",
        "source": "unsloth.ai/docs/models/qwen3-coder-next",
    },
    "qwen3.5-35b-a3b": {
        "params": "35B MoE (3B active)",
        "context": "128K",
        "min_ram": "12GB",
        "benchmarks": {
            "Aider": 55.0,  # approximate from community testing
        },
        "quant_note": "Q2_K_XL (11.3GB) fits 24GB Mac at 46 tok/s",
        "source": "unsloth.ai/docs/models/qwen3.5",
    },
    "gemma-4-26b-a4b": {
        "params": "26B MoE (4B active)",
        "context": "128K (sliding window)",
        "min_ram": "12GB",
        "benchmarks": {
            "MMLU": 74.4,
            "HumanEval": 72.0,
        },
        "quant_note": "Q3_K_XL (12GB) best for 24GB Mac. Needs --reasoning off",
        "source": "unsloth.ai/docs/models/gemma4",
    },
    "glm-4.7-flash": {
        "params": "92B MoE",
        "context": "128K",
        "min_ram": "16GB",
        "benchmarks": {
            "SWE-Bench Verified": 74.2,
            "Aider": 52.1,
        },
        "quant_note": "Q3_K_XL (12.8GB) fits 24GB. Disable thinking for speed.",
        "source": "unsloth.ai/docs/models/glm-4.7-flash",
    },
    "minimax-m2.1": {
        "params": "MoE",
        "context": "128K",
        "min_ram": "96GB",
        "benchmarks": {
            "SWE-Bench Verified": 74.8,
            "SWE-Bench Multilingual": 66.2,
            "Aider": 61.0,
        },
        "quant_note": "Needs 96GB+. Best agentic coding model.",
        "source": "unsloth.ai/docs/models",
    },
    "deepseek-v3.2": {
        "params": "671B MoE",
        "context": "128K",
        "min_ram": "200GB+",
        "benchmarks": {
            "SWE-Bench Verified": 70.2,
            "Aider": 69.9,
        },
        "quant_note": "Too big for consumer hardware. Use API.",
        "source": "unsloth.ai/docs/models",
    },
    "gpt-oss-20b": {
        "params": "20B Dense",
        "context": "32K",
        "min_ram": "12GB",
        "benchmarks": {},
        "quant_note": "F16 fits in 12.8GB. Best dense model for 24GB.",
        "source": "huggingface.co/unsloth/gpt-oss-20b-GGUF",
    },
    "qwen3.5-9b": {
        "params": "9B Dense",
        "context": "128K",
        "min_ram": "6GB",
        "benchmarks": {},
        "quant_note": "Sweet spot for 16GB Macs. Vision support.",
        "source": "unsloth.ai/docs/models/qwen3.5",
    },
    "qwen3.5-4b": {
        "params": "4B Dense",
        "context": "128K",
        "min_ram": "3GB",
        "benchmarks": {},
        "quant_note": "Ultrafast. Good tool calling. Fits any Mac.",
        "source": "unsloth.ai/docs/models/qwen3.5",
    },
}


def get_benchmark_info(model_name):
    """Look up benchmark data for a model by fuzzy name match."""
    name_lower = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")

    for key, data in UNSLOTH_BENCHMARKS.items():
        key_clean = key.replace("-", "").replace("_", "").replace(" ", "")
        if key_clean in name_lower or name_lower in key_clean:
            return data

    return None


def format_benchmark_line(model_name):
    """Format a one-line benchmark summary for display."""
    info = get_benchmark_info(model_name)
    if not info:
        return ""

    parts = []
    if info.get("params"):
        parts.append(info["params"])
    if info.get("context"):
        parts.append(f"ctx {info['context']}")

    benchmarks = info.get("benchmarks", {})
    if benchmarks:
        scores = [f"{k}: {v}" for k, v in list(benchmarks.items())[:3]]
        parts.append(" · ".join(scores))

    return "  ".join(parts) if parts else ""
