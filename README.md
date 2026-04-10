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

Fits locally? Run it via MLX or llama.cpp. Doesn't fit? Kaggle free GPU. Still too big? RunPod cloud. Need a custom quant? Quantize remotely and upload to HuggingFace. You never think about hardware.

```bash
pip install localfit
```

---

## Quick Start

```bash
localfit                              # GPU dashboard + trending models
localfit run gemma4:e4b               # interactive menu: pick MLX, GGUF, or Cloud
localfit run qwen3:14b                # doesn't fit? menu shows Kaggle/RunPod options
localfit launch openwebui --model gemma4:e4b                    # serve + launch tool
localfit launch openwebui --model gemma4:e4b --remote kaggle    # serve on free Kaggle GPU + launch
localfit launch claude --model gemma4:26b --remote runpod --budget $2
localfit makeitfit llama-4-scout      # quantize remotely → upload to your HuggingFace
```

## The Run Menu

When you `localfit run MODEL`, you get an interactive menu with **arrow key navigation** — pick your backend before anything downloads:

```
╭──────────────────────── Qwen2.5-7B-Instruct ─────────────────────────╮
│   LOCAL                                                              │
│   › MLX   Qwen2.5-7B-Instruct-4bit             3.5GB                │
│     MLX   Qwen2.5-7B-Instruct-8bit             7.0GB  ⭐             │
│     GGUF  Q4_K_M                                4.4GB                │
│     GGUF  Q8_0                                  7.5GB                │
│                                                                      │
│   REMOTE                                                             │
│     Kaggle  T4 16GB                               free               │
│     RunPod  RTX A5000 24GB                    $0.16/hr               │
╰──────────────────────────── Apple Silicon 16GB ──────────────────────╯
```

- **MLX** — Native Apple Silicon, fastest on Mac. Auto-discovers mlx-community models.
- **GGUF** — llama.cpp with Metal/CUDA. Works everywhere.
- **Kaggle** — Free 30h/week GPU. One click to deploy.
- **RunPod** — Paid cloud GPU. Auto-picks cheapest that fits.

## Backends

### MLX (Apple Silicon)

localfit auto-detects if you have `mlx-lm` installed and finds MLX models on HuggingFace:

```bash
pip install mlx-lm                    # one-time setup
localfit run gemma-3-4b-it            # auto-picks mlx-community/gemma-3-4b-it-8bit
```

If no mlx-community model exists, localfit can **convert any HuggingFace model to MLX locally**:

```bash
localfit run bytedance-research/UI-TARS-7B-DPO
# → No mlx-community model found
# → "Convert locally? mlx_lm needs ~14GB RAM (you have 24GB)"
# → Converts to MLX 4-bit → serves immediately
```

### GGUF (llama.cpp)

The default for all platforms. localfit downloads the best GGUF quant for your GPU and serves via llama-server:

```bash
localfit run gemma4:26b               # MoE, 12GB, best quality on 24GB Mac
localfit show gemma4:26b              # show all quants + fit analysis + cloud pricing
```

### Remote Kaggle (Free)

30 hours/week of free T4 GPU. localfit auto-deploys via Cloudflare tunnel:

```bash
localfit run qwen3:14b --remote kaggle
localfit --remote-status              # check active session
localfit --remote-stop                # stop + free quota
```

### Remote RunPod (Paid)

Any GPU size. Live pricing from the API. Auto-stop when budget runs out:

```bash
localfit login runpod                 # save API key
localfit run gemma4:27b --cloud       # auto-provision + tunnel
localfit --stop                       # kill pod + stop billing
```

## Make It Fit — Remote Quantization

Can't find the right quant? Create your own and upload to HuggingFace:

```bash
localfit makeitfit Qwen2.5-7B-Instruct
```

```
  Your GPU: Apple Silicon 16GB
  Model: Qwen/Qwen2.5-7B-Instruct (14GB BF16)

  1  Quantize on Kaggle (free) → Q4_K_M GGUF     ~7 min
  2  Quantize on RunPod         → Q5_K_M GGUF     ~$0.10
  3  Serve remotely (no quant)

  Pick option:
```

How it works:
1. Picks Kaggle GPU (free) or RunPod (cheapest available)
2. Downloads model from HuggingFace
3. Converts to F16 GGUF via llama.cpp
4. Quantizes to your chosen method (Q4_K_M, Q5_K_M, Q8_0, etc.)
5. Uploads to **your HuggingFace repo**
6. Run it: `localfit run yourname/model-Q4_K_M-GGUF-localfit`

Uses llama.cpp native tools — no Unsloth dependency, works reliably on Kaggle and RunPod.

## Launch Any Tool — Local or Remote

One command: pick a model, serve it (locally or cloud), launch your tool connected to it.

```bash
# Local (model fits your GPU)
localfit launch openwebui --model gemma4:e4b
localfit launch claude --model gemma4:26b
localfit launch codex --model qwen3:8b
localfit launch opencode --model gemma4:e4b
localfit launch aider --model gemma4:26b

# Remote Kaggle (free 30h/week GPU)
localfit launch openwebui --model gemma4:e4b --remote kaggle --budget 1h
localfit launch claude --model gemma4:31b --remote kaggle --budget 2h

# Remote RunPod (paid, any GPU)
localfit launch openwebui --model gemma4:31b --remote runpod --budget $2
localfit launch claude --model llama3:70b --remote runpod --budget $5
```

Budget: `30m`, `1h`, `2h` (time) or `$1`, `$2`, `$5` (money → auto-calculates time on cheapest GPU).

Shows remaining quota/balance before launch:
```
  Kaggle GPU quota: 17h remaining (of 30h/week)
  Duration: 60min
  
  ✓ Endpoint ready: https://xxx.trycloudflare.com
  ✓ Open WebUI launched: http://localhost:8080
```

### Supported Tools

| Tool | Command |
|------|---------|
| Open WebUI | `localfit launch openwebui` |
| Claude Code | `localfit launch claude` |
| OpenAI Codex | `localfit launch codex` |
| OpenCode | `localfit launch opencode` |
| aider | `localfit launch aider` |
| Open WebUI + tunnel | `localfit launch webui --tunnel` |

Works with both local and remote models. Env vars are **scoped to the subprocess only** — your normal tool setup is never touched.

## All Commands

### Model Management

```bash
localfit run MODEL                    # interactive menu → pick backend → serve
localfit run MODEL --remote kaggle    # serve on free Kaggle GPU
localfit run MODEL --cloud            # serve on RunPod (paid)
localfit pull MODEL                   # download only
localfit list                         # installed models
localfit ps                           # running models
localfit stop                         # stop local server
localfit show MODEL                   # all quants + fit analysis + pricing
```

### Quantization

```bash
localfit makeitfit MODEL              # quantize remotely → upload to HuggingFace
localfit login huggingface            # save HF write token (for uploads)
```

### GPU & Hardware

```bash
localfit                              # GPU dashboard + trending models
localfit health                       # GPU VRAM, temp, processes
localfit specs                        # full machine specs
localfit simulate                     # interactive "will this model fit?"
localfit bench                        # benchmark installed models
localfit arena                        # leaderboard on YOUR hardware
localfit trending                     # top models with fit/cloud tags
```

### Tool Integration

```bash
localfit --launch TOOL                # start model + launch tool
localfit --config TOOL                # show safe launch command
localfit doctor                       # check all tool configs
localfit restore                      # restore configs from backup
```

### Cloud & Remote

```bash
localfit login kaggle                 # save Kaggle credentials
localfit login runpod                 # save RunPod API key
localfit login huggingface            # save HF token
localfit --remote-status              # check active Kaggle session
localfit --remote-stop                # stop Kaggle session
localfit --stop                       # stop RunPod pod
```

### System

```bash
localfit check                        # check prerequisites (llama-server, CUDA, etc.)
localfit cleanup                      # free GPU memory
localfit debloat                      # disable macOS services stealing GPU
```

## Supported Platforms

| Platform | GPU Detection | Backends |
|----------|--------------|----------|
| macOS Apple Silicon | Metal | MLX + llama.cpp + Ollama |
| Linux NVIDIA | CUDA (nvidia-smi) | llama.cpp + Ollama |
| Linux AMD | ROCm (rocm-smi) | llama.cpp + Ollama |
| Windows (WSL2) | CUDA (nvidia-smi) | llama.cpp + Ollama |

## Dynamic VRAM Context Sizing

localfit auto-calculates the optimal context window:

| Machine | Model | Context |
|---------|-------|---------|
| M4 Pro 24GB | Gemma 4 26B (12GB) | 32K |
| M4 Pro 24GB | Gemma 4 E4B (4.6GB) | 128K |
| M4 Max 64GB | Gemma 4 26B (12GB) | 128K |

## Cloud Setup

### Kaggle (Free)

```bash
# 1. Get your Legacy API Key at https://www.kaggle.com/settings
#    → "Legacy API Credentials" → "Create Legacy API Key" → downloads kaggle.json
# 2. Save it:
localfit login kaggle
# 3. Run any model:
localfit run gemma4:e4b --remote kaggle
```

### RunPod (Paid)

```bash
# 1. Get API key at https://www.runpod.io/console/user/settings
# 2. Save it:
localfit login runpod
# 3. Run any model:
localfit run gemma4:27b --cloud
```

### HuggingFace (For Uploads)

```bash
# 1. Create a write token at https://huggingface.co/settings/tokens
# 2. Save it:
localfit login huggingface
# 3. Quantize + upload:
localfit makeitfit Qwen2.5-7B-Instruct
```

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) or [Ollama](https://ollama.com) (auto-installed)
- Optional: [mlx-lm](https://github.com/ml-explore/mlx-lm) for Apple Silicon MLX backend

```bash
pip install localfit                  # core
pip install 'localfit[all]'           # + TUI dashboard + HF downloads
pip install mlx-lm                    # + MLX backend (Mac only)
```

## License

Apache-2.0
