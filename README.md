# localfit

```
██╗      ██████╗  ██████╗ █████╗ ██╗     ███████╗██╗████████╗
██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██╔════╝██║╚══██╔══╝
██║     ██║   ██║██║     ███████║██║     █████╗  ██║   ██║
██║     ██║   ██║██║     ██╔══██║██║     ██╔══╝  ██║   ██║
███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║     ██║   ██║
╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝   ╚═╝
```

**Will it fit?** Say what model you want — localfit figures out the rest.

Text and image generation. Fits locally? Run it. Doesn't fit? Free Kaggle GPU. Still too big? RunPod cloud. One command does everything: downloads models, starts servers, configures tools, launches your UI.

```bash
pip install localfit
```

---

## One Command — Everything Works

```bash
# Chat + image gen in Open WebUI (gemma4 LLM + Flux Klein 4B images)
localfit launch openwebui --model gemma4:e4b --img klein-4b

# Code with image gen in localcoder
localfit launch localcoder --model gemma4:e4b --img klein-4b

# Claude Code with image MCP tools
localfit launch claude --model gemma4:e4b --img klein-4b

# Model doesn't fit locally? Run on free Kaggle GPU
localfit launch openwebui --model gemma4:26b --remote kaggle

# Need more power? RunPod cloud ($0.22/hr)
localfit launch openwebui --model gemma4:26b --img klein-9b --remote runpod --budget $2
```

localfit auto-detects your hardware, downloads models, installs Ollama if needed, starts the image server, configures Open WebUI with image generation enabled, and opens your browser. Zero manual config.

## The Run Menu

```bash
localfit run gemma4:e4b
```

Interactive menu with **arrow key navigation** — pick your backend before anything downloads:

```
╭──────────────────────── gemma-4-E4B ────────────────────────╮
│   LOCAL                                                     │
│   › MLX   gemma-4-E4B-4bit                      3.5GB      │
│     GGUF  Q4_K_M                                4.4GB      │
│     GGUF  Q8_0                                  7.5GB      │
│                                                             │
│   REMOTE                                                    │
│     Kaggle  T4 16GB                               free      │
│     RunPod  RTX 3090 24GB                     $0.22/hr      │
╰──────────────────────── Apple Silicon 24GB ─────────────────╯
```

## Image Generation

One server, two APIs — works with Open WebUI, Claude Code, and any OpenAI-compatible client.

```bash
# Starts automatically with --img flag, or manually:
python -m localfit.image_server 8189 klein-4b 4
```

**OpenAI API:** `POST /v1/images/generations`
**AUTOMATIC1111 API:** `POST /sdapi/v1/txt2img` (Open WebUI native)

### Supported Image Models

| Model | Pipeline | Params | Local Mac | Kaggle T4 | RunPod 3090 |
|-------|----------|--------|:---------:|:---------:|:-----------:|
| FLUX.2 Klein 4B | Flux2KleinPipeline | 4B | **24s** | 65s | **2s** |
| FLUX.2 Klein 9B | Flux2KleinPipeline | 9B | mflux | cpu_offload | 3090+ |
| FLUX.1 Schnell | FluxPipeline | 12B | 79s | cpu_offload | 3090 |
| FLUX.1 Dev | FluxPipeline | 12B | mflux | cpu_offload | 3090+ |
| Z-Image-Turbo | ZImagePipeline | 6B | 90s | T4 | 3090 |
| Qwen-Image | QwenImagePipeline | 20B | — | — | A6000+ |
| Qwen-Image-Edit | QwenImageEditPlusPipeline | 20B | — | — | A100 |
| SDXL | StableDiffusionXLPipeline | 6.6B | diffusers | T4 | 3090 |
| SD 3.5 Large | StableDiffusion3Pipeline | 8B | diffusers | T4 | 3090 |

All models auto-detected via `DiffusionPipeline.from_pretrained()`.

### MCP Server for Claude Code

```bash
claude mcp add localfit-image --transport stdio -- python3 -m localfit.mcp_image
```

6 tools available:

| Tool | Description |
|------|-------------|
| `check_resources` | GPU/VRAM info, loaded model, ETA estimates |
| `generate_image` | Text-to-image with timing + Image object |
| `edit_image` | Image-to-image editing |
| `show_image` | Display image in terminal via timg |
| `list_image_models` | Available models |
| `image_server_status` | Server health |

### Benchmarks (Verified)

| Platform | GPU | Model | Size | Steps | Time |
|----------|-----|-------|------|-------|------|
| **Mac** | M4 Pro | Klein 4B | 1024x1024 | 4 | **24s** |
| **Mac** | M4 Pro | Schnell | 512x512 | 4 | **79s** |
| **Kaggle** | T4 16GB | Klein 4B | 512x512 | 4 | **65s** |
| **RunPod** | RTX 3090 | Klein 4B | 512x512 | 4 | **2s** |

## LLM Backends

### Ollama (Default)

```bash
localfit run gemma4:e4b               # auto-installs Ollama, pulls model, serves
```

### MLX (Apple Silicon)

```bash
pip install mlx-lm
localfit run gemma-3-4b-it            # auto-picks mlx-community model
```

### GGUF (llama.cpp)

```bash
localfit run gemma4:26b               # downloads best GGUF quant for your GPU
```

## Remote Serving

### Kaggle (Free — 30h/week GPU)

```bash
localfit login kaggle
localfit run gemma4:e4b --remote kaggle
```

### RunPod (Paid — any GPU)

```bash
localfit login runpod
localfit run gemma4:26b --cloud --budget $2
```

### Combined Remote (LLM + Image on same GPU)

```bash
localfit launch openwebui --model gemma4:e4b --img klein-4b --remote runpod
```

Deploys both models on one RunPod GPU, tunnels via Cloudflare, launches Open WebUI connected to both endpoints. Tested: gemma4:e4b + Klein 4B on RTX 3090 — LLM 0.18s + image 2s.

## Launch Any Tool

| Tool | Command |
|------|---------|
| Open WebUI | `localfit launch openwebui --model MODEL --img IMAGE` |
| Claude Code | `localfit launch claude --model MODEL --img IMAGE` |
| localcoder | `localfit launch localcoder --model MODEL --img IMAGE` |
| OpenAI Codex | `localfit launch codex --model MODEL` |
| OpenCode | `localfit launch opencode --model MODEL` |
| aider | `localfit launch aider --model MODEL` |

All tools are auto-configured with the right endpoints. Env vars scoped to subprocess — your normal setup is never touched.

## Make It Fit — Remote Quantization

```bash
localfit makeitfit Qwen2.5-7B-Instruct
# → Quantizes on Kaggle (free) or RunPod
# → Uploads to your HuggingFace repo
# → Run it: localfit run yourname/model-Q4_K_M-GGUF-localfit
```

## All Commands

```bash
# Model management
localfit run MODEL                    # interactive menu → serve
localfit run MODEL --remote kaggle    # serve on free Kaggle GPU
localfit run MODEL --cloud            # serve on RunPod
localfit run MODEL --img IMAGE_MODEL  # serve LLM + image model
localfit show MODEL                   # quants + fit analysis + pricing
localfit list                         # installed models
localfit stop                         # stop servers

# Image generation
localfit launch openwebui --img klein-4b    # with Open WebUI
localfit launch claude --img klein-4b       # with Claude Code MCP

# Tools
localfit launch TOOL --model MODEL    # serve + launch tool
localfit doctor                       # check all tool configs

# Hardware
localfit                              # GPU dashboard + trending models
localfit health                       # VRAM, temp, processes
localfit bench                        # benchmark models
localfit simulate                     # "will this model fit?"

# Cloud
localfit login kaggle                 # save credentials
localfit login runpod                 # save API key
localfit login huggingface            # save HF token
localfit --remote-status              # check Kaggle session
localfit --remote-stop                # stop remote session

# System
localfit check                        # check prerequisites
localfit cleanup                      # free GPU memory
localfit makeitfit MODEL              # quantize remotely
```

## Supported Platforms

| Platform | GPU | LLM | Image Gen |
|----------|-----|-----|-----------|
| macOS Apple Silicon | Metal | MLX + llama.cpp + Ollama | mflux (MLX native) |
| Linux NVIDIA | CUDA | llama.cpp + Ollama | diffusers (CUDA) |
| Linux AMD | ROCm | llama.cpp + Ollama | diffusers (ROCm) |
| Windows (WSL2) | CUDA | llama.cpp + Ollama | diffusers (CUDA) |
| Kaggle (free) | T4 16GB | Ollama via tunnel | diffusers via tunnel |
| RunPod (paid) | Any GPU | Ollama via tunnel | diffusers via tunnel |

## Requirements

- Python 3.10+
- Optional: [mflux](https://github.com/filipstrand/mflux) for Mac image gen
- Optional: [Ollama](https://ollama.com) (auto-installed by localfit)

```bash
pip install localfit                  # core
pip install mflux                     # + image generation (Mac)
```

## License

Apache-2.0
