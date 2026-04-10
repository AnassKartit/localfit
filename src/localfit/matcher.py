"""Smart model-GPU matching algorithm.

Given a budget and a model, finds the best GPU + quant combination
that maximizes quality while staying within budget.

Quality priority:
  1. Quant quality (Q5 > Q4 > Q3 > Q2 > IQ > Q1)
  2. Speed (tok/s)
  3. Runtime (more hours = better)
"""

# Quant quality scores (higher = better)
QUANT_QUALITY = {
    "F16": 100, "BF16": 100,
    "Q8_0": 95, "Q8_K_XL": 95,
    "Q6_K": 90, "Q6_K_XL": 90,
    "Q5_K_M": 85, "Q5_K_XL": 85, "Q5_K_S": 83,
    "Q4_K_XL": 80, "Q4_K_M": 78, "Q4_K_S": 76, "Q4_0": 74, "Q4_1": 74,
    "IQ4_NL": 73, "IQ4_XS": 72,
    "MXFP4": 70,
    "Q3_K_XL": 65, "Q3_K_M": 63, "Q3_K_S": 60, "Q3_K_L": 62,
    "IQ3_XXS": 55, "IQ3_S": 57, "IQ3_M": 58, "IQ3_XS": 56,
    "Q2_K_XL": 45, "Q2_K_L": 43, "Q2_K": 40, "Q2_K_S": 38,
    "IQ2_M": 35, "IQ2_XS": 30, "IQ2_XXS": 25, "IQ2_S": 28,
    "IQ1_M": 15, "IQ1_S": 10,
    "Q1_0": 5,
}

QUALITY_LABELS = {
    (90, 100): ("excellent", "green"),
    (75, 89):  ("great", "green"),
    (60, 74):  ("good", "yellow"),
    (40, 59):  ("fair", "yellow"),
    (20, 39):  ("poor", "red"),
    (0, 19):   ("bad", "red"),
}

# GPU options — populated at runtime from RunPod API via cloud.py
# cloud_serve() calls GPUS.clear() + GPUS.extend(live_gpus) before matching
GPUS = []


def get_quant_quality(quant_name):
    """Get quality score for a quant (0-100)."""
    return QUANT_QUALITY.get(quant_name, 50)


def get_quality_label(score):
    """Get human-readable quality label."""
    for (lo, hi), (label, color) in QUALITY_LABELS.items():
        if lo <= score <= hi:
            return label, color
    return "unknown", "dim"


def find_best_match(gguf_files, max_spend, min_hours=0.5):
    """Find the best GPU + quant combination for a budget.

    Returns sorted list of options, best first:
    [
        {
            "gpu": {...},
            "quant": {...},
            "quality_score": 80,
            "quality_label": "great",
            "hours": 3.5,
            "total_cost": 1.75,
            "score": 85.5,  # combined ranking score
        },
        ...
    ]
    """
    options = []

    for gpu in GPUS:
        # Find all quants that fit this GPU
        fits = [f for f in gguf_files if f["size_gb"] * 1024 < gpu["vram"] * 1024]
        if not fits:
            continue

        # For each fitting quant, calculate the option
        for quant in fits:
            hours = max_spend / gpu["price"]
            if hours < min_hours:
                continue

            quality = get_quant_quality(quant["quant"])
            label, color = get_quality_label(quality)

            # Combined score: quality matters most, then speed, then runtime
            # Quality: 0-100 (weight 60%)
            # Speed: normalize to 0-100 (weight 25%)
            # Runtime: normalize to 0-100 based on 8h max (weight 15%)
            speed_score = min(100, (gpu["tok_s"] / 90) * 100)
            runtime_score = min(100, (hours / 8) * 100)
            combined = quality * 0.60 + speed_score * 0.25 + runtime_score * 0.15

            options.append({
                "gpu": gpu,
                "quant": quant,
                "quality_score": quality,
                "quality_label": label,
                "quality_color": color,
                "hours": round(hours, 1),
                "total_cost": round(max_spend, 2),
                "tok_s": gpu["tok_s"],
                "score": round(combined, 1),
            })

    # Sort by combined score (best first), deduplicate by GPU (keep best quant per GPU)
    options.sort(key=lambda x: x["score"], reverse=True)

    # Keep only the best quant per GPU
    seen_gpus = set()
    best_options = []
    for opt in options:
        gpu_name = opt["gpu"]["name"]
        if gpu_name not in seen_gpus:
            seen_gpus.add(gpu_name)
            best_options.append(opt)

    return best_options


def find_recommended(gguf_files, max_spend):
    """Find the single best recommendation.

    Prioritizes quality > speed > runtime.
    Returns the option with highest combined score, or None.
    """
    options = find_best_match(gguf_files, max_spend)
    if not options:
        return None

    best = options[0]

    # If the best option has poor quality (<40), warn
    if best["quality_score"] < 40:
        # Check if spending more would help
        for gpu in GPUS:
            fits = [f for f in gguf_files if f["size_gb"] * 1024 < gpu["vram"] * 1024]
            if fits:
                best_quant = max(fits, key=lambda f: get_quant_quality(f["quant"]))
                q = get_quant_quality(best_quant["quant"])
                if q >= 60:
                    min_cost = gpu["price"] * 0.5  # 30 min minimum
                    best["upgrade_hint"] = {
                        "gpu": gpu["name"],
                        "quant": best_quant["quant"],
                        "quality": q,
                        "min_cost": min_cost,
                    }
                    break

    return best
