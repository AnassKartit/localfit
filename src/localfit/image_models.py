"""Image model catalog — pipeline classes, defaults, and GPU requirements.

Each model has its correct diffusers pipeline class, default parameters,
VRAM requirements, and recommended GPUs for Kaggle/RunPod.
"""

# Top image models with their correct pipeline classes and defaults
# Pipeline class is auto-detected by DiffusionPipeline.from_pretrained()
# but we store it for display/documentation purposes
IMAGE_MODELS = {
    # === Text-to-Image ===
    "flux2-klein-4b": {
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline": "Flux2KleinPipeline",
        "task": "text-to-image",
        "params_b": 4,
        "vram_gb": 8,
        "defaults": {"steps": 4, "guidance_scale": 4.0, "size": "1024x1024"},
        "gated": False,
        "kaggle_gpu": "T4",
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA L4"],
    },
    "flux2-klein-base-4b": {
        "repo": "black-forest-labs/FLUX.2-klein-base-4B",
        "pipeline": "Flux2KleinPipeline",
        "task": "text-to-image",
        "params_b": 4,
        "vram_gb": 8,
        "defaults": {"steps": 50, "guidance_scale": 4.0, "size": "1024x1024"},
        "gated": False,
        "kaggle_gpu": "T4",
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA L4"],
    },
    "flux2-klein-9b": {
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "pipeline": "Flux2KleinPipeline",
        "task": "text-to-image",
        "params_b": 9,
        "vram_gb": 18,
        "defaults": {"steps": 4, "guidance_scale": 4.0, "size": "1024x1024"},
        "gated": True,
        "kaggle_gpu": "T4",  # fits with cpu_offload
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"],
    },
    "flux2-dev": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "pipeline": "Flux2Pipeline",
        "task": "text-to-image",
        "params_b": 32,
        "vram_gb": 64,
        "defaults": {"steps": 50, "guidance_scale": 2.5, "size": "1024x1024"},
        "gated": True,
        "kaggle_gpu": None,  # too big for Kaggle
        "runpod_gpus": ["NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"],
    },
    "flux1-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "FluxPipeline",
        "task": "text-to-image",
        "params_b": 12,
        "vram_gb": 24,
        "defaults": {"steps": 4, "guidance_scale": 0.0, "size": "1024x1024"},
        "gated": True,
        "kaggle_gpu": "T4",  # fits with cpu_offload
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"],
    },
    "flux1-dev": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "pipeline": "FluxPipeline",
        "task": "text-to-image",
        "params_b": 12,
        "vram_gb": 24,
        "defaults": {"steps": 50, "guidance_scale": 3.5, "size": "1024x1024"},
        "gated": True,
        "kaggle_gpu": "T4",  # fits with cpu_offload
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"],
    },
    "z-image-turbo": {
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "pipeline": "ZImagePipeline",
        "task": "text-to-image",
        "params_b": 6,
        "vram_gb": 12,
        "defaults": {"steps": 9, "guidance_scale": 0.0, "size": "1024x1024"},
        "gated": False,
        "kaggle_gpu": "T4",
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA L4"],
    },
    "z-image": {
        "repo": "Tongyi-MAI/Z-Image",
        "pipeline": "ZImagePipeline",
        "task": "text-to-image",
        "params_b": 6,
        "vram_gb": 12,
        "defaults": {"steps": 50, "guidance_scale": 4.0, "size": "1024x1024"},
        "gated": False,
        "kaggle_gpu": "T4",
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA L4"],
    },
    "qwen-image": {
        "repo": "Qwen/Qwen-Image",
        "pipeline": "QwenImagePipeline",
        "task": "text-to-image",
        "params_b": 20,
        "vram_gb": 40,
        "defaults": {"steps": 50, "true_cfg_scale": 3.5, "size": "1024x1024"},
        "gated": False,
        "kaggle_gpu": None,  # too big for T4
        "runpod_gpus": ["NVIDIA RTX A6000", "NVIDIA A100-SXM4-80GB"],
    },
    "sdxl": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLPipeline",
        "task": "text-to-image",
        "params_b": 6.6,
        "vram_gb": 7,
        "defaults": {"steps": 30, "guidance_scale": 7.5, "size": "1024x1024"},
        "gated": False,
        "kaggle_gpu": "T4",
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA L4"],
    },
    "sdxl-turbo": {
        "repo": "stabilityai/sdxl-turbo",
        "pipeline": "AutoPipelineForText2Image",
        "task": "text-to-image",
        "params_b": 6.6,
        "vram_gb": 7,
        "defaults": {"steps": 1, "guidance_scale": 0.0, "size": "512x512"},
        "gated": False,
        "kaggle_gpu": "T4",
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA L4"],
    },
    "sd3.5-large": {
        "repo": "stabilityai/stable-diffusion-3.5-large",
        "pipeline": "StableDiffusion3Pipeline",
        "task": "text-to-image",
        "params_b": 8,
        "vram_gb": 16,
        "defaults": {"steps": 28, "guidance_scale": 3.5, "size": "1024x1024"},
        "gated": True,
        "kaggle_gpu": "T4",
        "runpod_gpus": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA L4"],
    },
    # === Image-to-Image (Editing) ===
    "qwen-image-edit": {
        "repo": "Qwen/Qwen-Image-Edit-2511",
        "pipeline": "QwenImageEditPlusPipeline",
        "task": "image-to-image",
        "params_b": 20,
        "vram_gb": 57,
        "defaults": {"steps": 40, "guidance_scale": 1.0, "true_cfg_scale": 4.0, "size": "1024x1024"},
        "gated": False,
        "kaggle_gpu": None,  # 57GB, needs big GPU
        "runpod_gpus": ["NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"],
    },
    "qwen-image-edit-2509": {
        "repo": "Qwen/Qwen-Image-Edit-2509",
        "pipeline": "QwenImageEditPlusPipeline",
        "task": "image-to-image",
        "params_b": 8,
        "vram_gb": 40,
        "defaults": {"steps": 40, "guidance_scale": 1.0, "true_cfg_scale": 4.0, "size": "1024x1024"},
        "gated": False,
        "kaggle_gpu": None,
        "runpod_gpus": ["NVIDIA RTX A6000", "NVIDIA A100-SXM4-80GB"],
    },
}

# Aliases for user-friendly names
ALIASES = {
    "klein4b": "flux2-klein-4b",
    "klein-4b": "flux2-klein-4b",
    "klein9b": "flux2-klein-9b",
    "klein-9b": "flux2-klein-9b",
    "schnell": "flux1-schnell",
    "dev": "flux1-dev",
    "flux-schnell": "flux1-schnell",
    "flux-dev": "flux1-dev",
    "turbo": "z-image-turbo",
    "zimage": "z-image-turbo",
    "sdxl-turbo": "sdxl-turbo",
    "sd3": "sd3.5-large",
    "qwen-edit": "qwen-image-edit",
}


def resolve_image_model(query):
    """Resolve a user query to an image model entry."""
    q = query.lower().strip().replace(" ", "-")
    # Direct match
    if q in IMAGE_MODELS:
        return IMAGE_MODELS[q]
    # Alias match
    if q in ALIASES:
        return IMAGE_MODELS[ALIASES[q]]
    # Fuzzy: check if query is substring of any key or repo
    for key, model in IMAGE_MODELS.items():
        if q in key or q in model["repo"].lower():
            return model
    return None


def get_gpu_recommendation(model_entry, budget=None):
    """Get GPU recommendation for a model, considering budget."""
    vram = model_entry["vram_gb"]

    # RunPod GPU tiers with approximate costs
    gpu_tiers = [
        {"name": "NVIDIA GeForce RTX 3090", "vram": 24, "price": 0.22},
        {"name": "NVIDIA RTX A5000", "vram": 24, "price": 0.16},
        {"name": "NVIDIA L4", "vram": 24, "price": 0.24},
        {"name": "NVIDIA GeForce RTX 4090", "vram": 24, "price": 0.44},
        {"name": "NVIDIA RTX A6000", "vram": 48, "price": 0.79},
        {"name": "NVIDIA L40S", "vram": 48, "price": 0.99},
        {"name": "NVIDIA A100-SXM4-80GB", "vram": 80, "price": 1.64},
        {"name": "NVIDIA H100 80GB HBM3", "vram": 80, "price": 2.49},
    ]

    # With cpu_offload, models need roughly 60% of their full VRAM
    offload_vram = vram * 0.6

    fits = []
    for gpu in gpu_tiers:
        if gpu["vram"] >= offload_vram:
            fits.append(gpu)

    if budget:
        fits = [g for g in fits if g["price"] <= budget]

    return fits


def list_image_models():
    """List all supported image models with details."""
    lines = []
    for key, m in IMAGE_MODELS.items():
        kaggle = m["kaggle_gpu"] or "too big"
        gated = " (gated)" if m["gated"] else ""
        lines.append(
            f"  {key:25s} {m['pipeline']:35s} {m['params_b']:>5.1f}B  "
            f"{m['vram_gb']:>3d}GB  kaggle={kaggle:>4s}{gated}"
        )
    return "\n".join(lines)
