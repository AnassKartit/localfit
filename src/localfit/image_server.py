"""OpenAI-compatible image generation API using mflux (Mac) or diffusers (Linux/Windows).

Serves /v1/images/generations — same API as OpenAI DALL-E.
Works with Open WebUI, Claude Code, any OpenAI-compatible client.

Mac: uses mflux (MLX native, fastest on Apple Silicon)
Linux: uses diffusers with CUDA (fallback)
"""

import json
import os
import sys
import base64
import time
import io
from http.server import HTTPServer, BaseHTTPRequestHandler

DEFAULT_PORT = 8189

# Global model instance (loaded once, reused)
_model = None
_model_name = None


def _load_model(model_name=None, quantize=4):
    """Load image model — mflux on Mac, diffusers on Linux."""
    global _model, _model_name

    # If model already loaded and no specific model requested, reuse it
    if _model and (model_name is None or _model_name == model_name):
        return _model

    if model_name is None:
        model_name = "schnell"

    print(f"  Loading {model_name} (quantize={quantize})...", flush=True)

    # Try mflux first (Mac)
    try:
        if "schnell" in model_name:
            from mflux.models.flux.variants.txt2img.flux import Flux1
            from mflux.models.common.config import ModelConfig

            _model = Flux1(model_config=ModelConfig.schnell(), quantize=quantize)
        elif "dev" in model_name and "dev" == model_name.split("-")[-1]:
            from mflux.models.flux.variants.txt2img.flux import Flux1
            from mflux.models.common.config import ModelConfig

            _model = Flux1(model_config=ModelConfig.dev(), quantize=quantize)
        elif "z-image-turbo" in model_name:
            from mflux.models.z_image import ZImageTurbo

            _model = ZImageTurbo(quantize=quantize)
        elif "z-image" in model_name:
            from mflux.models.z_image import ZImage

            _model = ZImage(quantize=quantize)
        elif "klein-4b" in model_name or "flux2-klein-4b" in model_name:
            from mflux.models.flux2.variants import Flux2Klein
            from mflux.models.common.config import ModelConfig

            _model = Flux2Klein(
                model_config=ModelConfig.flux2_klein_4b(), quantize=quantize
            )
        elif "klein-9b" in model_name or "flux2-klein-9b" in model_name:
            from mflux.models.flux2.variants import Flux2Klein
            from mflux.models.common.config import ModelConfig

            _model = Flux2Klein(
                model_config=ModelConfig.flux2_klein_9b(), quantize=quantize
            )
        elif "fibo" in model_name:
            from mflux.models.fibo.variants.txt2img.fibo import FIBO
            from mflux.models.common.config import ModelConfig

            _model = FIBO(model_config=ModelConfig.fibo(), quantize=quantize)
        else:
            from mflux.models.flux.variants.txt2img.flux import Flux1
            from mflux.models.common.config import ModelConfig

            _model = Flux1(model_config=ModelConfig.schnell(), quantize=quantize)
            model_name = "schnell"

        _model_name = model_name
        print(f"  ✓ {model_name} loaded (mflux, {quantize}-bit)", flush=True)
        return _model

    except ImportError:
        pass

    # Fallback: diffusers (Linux/Windows with CUDA)
    try:
        import torch
        from diffusers import FluxPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        repo = {
            "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
            "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",
        }.get(model_name, "black-forest-labs/FLUX.2-klein-4B")

        _model = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to(
            device
        )
        _model_name = model_name
        print(f"  ✓ {model_name} loaded (diffusers, {device})", flush=True)
        return _model

    except ImportError:
        print("  ✗ Neither mflux nor diffusers available", flush=True)
        print(
            "  Install: pip install mflux (Mac) or pip install diffusers torch (Linux)",
            flush=True,
        )
        return None


def _generate(model, prompt, width=1024, height=1024, steps=4, seed=None):
    """Generate image, returns PIL Image."""
    import random

    seed = seed or random.randint(0, 2**32)

    # mflux API
    if hasattr(model, "generate_image"):
        img = model.generate_image(
            prompt=prompt,
            seed=seed,
            num_inference_steps=steps,
            width=width,
            height=height,
        )
        return img

    # diffusers API
    if hasattr(model, "__call__"):
        result = model(
            prompt=prompt, width=width, height=height, num_inference_steps=steps
        )
        return result.images[0]

    return None


class ImageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/images/generations":
            self._handle_generate()
        elif self.path == "/v1/images/edits":
            self._handle_edit()
        elif self.path == "/sdapi/v1/txt2img":
            self._handle_sdapi_txt2img()
        elif self.path == "/sdapi/v1/options":
            # POST to set options (Open WebUI sends this)
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)  # consume body
            self._json(200, {"sd_model_checkpoint": _model_name or "image-gen"})
        else:
            self._json(404, {"error": {"message": "Not found"}})

    def _handle_generate(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        prompt = body.get("prompt", "a beautiful landscape")
        size = body.get("size", "1024x1024")
        steps = body.get("steps", 4)
        seed = body.get("seed")

        t0 = time.time()
        print(f"  [openai] {size} steps={steps} prompt={prompt[:50]}...", flush=True)

        w, h = (int(x) for x in size.split("x")) if "x" in size else (1024, 1024)

        model = _load_model()
        if not model:
            self._json(500, {"error": {"message": "No image model loaded"}})
            return

        try:
            img = _generate(model, prompt, w, h, steps, seed)
            if img is None:
                self._json(500, {"error": {"message": "Generation returned None"}})
                return

            pil_img = img.image if hasattr(img, "image") else img
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            self._json(
                200,
                {
                    "created": int(time.time()),
                    "data": [{"b64_json": img_b64}],
                },
            )
        except Exception as e:
            self._json(500, {"error": {"message": str(e)[:300]}})

    def _handle_edit(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        prompt = body.get("prompt", "")
        src_b64 = body.get("image", "")
        strength = body.get("strength", 0.7)
        steps = body.get("steps", 8)
        seed = body.get("seed")
        size = body.get("size", "")

        if not src_b64:
            self._json(400, {"error": {"message": "Missing 'image' (base64 PNG)"}})
            return

        # Decode source image
        from PIL import Image

        src_bytes = base64.b64decode(src_b64)
        src_img = Image.open(io.BytesIO(src_bytes))
        w, h = src_img.size
        if size and "x" in size:
            w, h = (int(x) for x in size.split("x"))

        model = _load_model()
        if not model:
            self._json(500, {"error": {"message": "No image model loaded"}})
            return

        try:
            # mflux image-to-image: save source to temp, pass as image_path
            if hasattr(model, "generate_image"):
                import tempfile, random

                seed = seed or random.randint(0, 2**32)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                src_img.save(tmp.name)
                try:
                    img = model.generate_image(
                        prompt=prompt,
                        seed=seed,
                        num_inference_steps=steps,
                        width=w,
                        height=h,
                        image_path=tmp.name,
                        image_strength=strength,
                    )
                finally:
                    os.unlink(tmp.name)
            # diffusers img2img
            elif hasattr(model, "__call__"):
                img = model(
                    prompt=prompt,
                    image=src_img,
                    width=w,
                    height=h,
                    num_inference_steps=steps,
                    strength=strength,
                )
                img = img.images[0]
            else:
                self._json(
                    500, {"error": {"message": "Model does not support editing"}}
                )
                return

            if img is None:
                self._json(500, {"error": {"message": "Edit returned None"}})
                return

            pil_img = img.image if hasattr(img, "image") else img
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            self._json(
                200,
                {
                    "created": int(time.time()),
                    "data": [{"b64_json": img_b64}],
                },
            )
        except Exception as e:
            self._json(500, {"error": {"message": str(e)[:300]}})

    def _handle_sdapi_txt2img(self):
        """AUTOMATIC1111 /sdapi/v1/txt2img — Open WebUI native image integration."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        prompt = body.get("prompt", "a beautiful landscape")
        w = body.get("width", 512)
        h = body.get("height", 512)
        steps = body.get("steps", 4)
        seed = body.get("seed", -1)

        t0 = time.time()
        print(f"  [sdapi] {w}x{h} steps={steps} prompt={prompt[:50]}...", flush=True)
        if seed == -1:
            import random

            seed = random.randint(0, 2**32)

        model = _load_model()
        if not model:
            self._json(500, {"error": "No image model loaded"})
            return

        try:
            img = _generate(model, prompt, w, h, steps, seed)
            if img is None:
                self._json(500, {"error": "Generation returned None"})
                return

            pil_img = img.image if hasattr(img, "image") else img
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            elapsed = time.time() - t0
            print(
                f"  [sdapi] done in {elapsed:.1f}s ({len(img_b64) // 1024}KB)",
                flush=True,
            )
            self._json(
                200,
                {
                    "images": [f"data:image/png;base64,{img_b64}"],
                    "parameters": body,
                    "info": json.dumps(
                        {"seed": seed, "prompt": prompt, "time": f"{elapsed:.1f}s"}
                    ),
                },
            )
        except Exception as e:
            print(f"  [sdapi] error: {e}", flush=True)
            self._json(500, {"error": str(e)[:300]})

    def do_GET(self):
        if self.path in ("/health", "/"):
            if _model is not None and _model_name:
                self._json(200, {"status": "ok", "model": _model_name})
            else:
                self._json(503, {"status": "loading", "model": "not loaded"})
        elif self.path == "/v1/models":
            self._json(
                200, {"data": [{"id": _model_name or "image-gen", "object": "model"}]}
            )
        elif self.path == "/sdapi/v1/sd-models":
            self._json(
                200,
                [
                    {
                        "title": _model_name or "image-gen",
                        "model_name": _model_name or "image-gen",
                    }
                ],
            )
        elif self.path == "/sdapi/v1/options":
            self._json(200, {"sd_model_checkpoint": _model_name or "image-gen"})
        elif self.path == "/sdapi/v1/progress":
            self._json(200, {"progress": 0, "eta_relative": 0, "state": {"job": ""}})
        else:
            self._json(404, {"error": {"message": "Not found"}})

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # quiet


def start_image_server(port=DEFAULT_PORT, model="z-image-turbo", quantize=4):
    """Start the image generation API server."""
    print(f"\n  localfit image server", flush=True)
    result = _load_model(model, quantize)
    if result is None:
        print(f"\n  ERROR: Failed to load model '{model}'.", flush=True)
        print(f"  Install mflux (Mac): pipx inject localfit mflux", flush=True)
        print(
            f"  Install diffusers (Linux): pipx inject localfit diffusers torch",
            flush=True,
        )
        sys.exit(1)
    server = HTTPServer(("127.0.0.1", port), ImageHandler)
    print(f"  API: http://127.0.0.1:{port}/v1/images/generations", flush=True)
    print(f"  Health: http://127.0.0.1:{port}/health", flush=True)
    print(
        f"\n  Configure Open WebUI: Settings → Images → OpenAI URL → http://127.0.0.1:{port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    model = sys.argv[2] if len(sys.argv) > 2 else "z-image-turbo"
    quantize = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    start_image_server(port, model, quantize)
