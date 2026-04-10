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


def _load_model(model_name="z-image-turbo", quantize=4):
    """Load image model — mflux on Mac, diffusers on Linux."""
    global _model, _model_name

    if _model and _model_name == model_name:
        return _model

    print(f"  Loading {model_name} (quantize={quantize})...", flush=True)

    # Try mflux first (Mac)
    try:
        if "z-image-turbo" in model_name:
            from mflux.models.z_image import ZImageTurbo
            _model = ZImageTurbo(quantize=quantize)
        elif "z-image" in model_name:
            from mflux.models.z_image import ZImage
            _model = ZImage(quantize=quantize)
        elif "klein-4b" in model_name or "flux2-klein-4b" in model_name:
            from mflux.models.flux2.variants import Flux2Klein
            from mflux.models.common.config import ModelConfig
            _model = Flux2Klein(model_config=ModelConfig.flux2_klein_4b(), quantize=quantize)
        elif "klein-9b" in model_name or "flux2-klein-9b" in model_name:
            from mflux.models.flux2.variants import Flux2Klein
            from mflux.models.common.config import ModelConfig
            _model = Flux2Klein(model_config=ModelConfig.flux2_klein_9b(), quantize=quantize)
        elif "fibo" in model_name:
            from mflux.models.fibo import Fibo
            _model = Fibo(quantize=quantize)
        else:
            from mflux.models.z_image import ZImageTurbo
            _model = ZImageTurbo(quantize=quantize)
            model_name = "z-image-turbo"

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

        _model = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device)
        _model_name = model_name
        print(f"  ✓ {model_name} loaded (diffusers, {device})", flush=True)
        return _model

    except ImportError:
        print("  ✗ Neither mflux nor diffusers available", flush=True)
        print("  Install: pip install mflux (Mac) or pip install diffusers torch (Linux)", flush=True)
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
        result = model(prompt=prompt, width=width, height=height, num_inference_steps=steps)
        return result.images[0]

    return None


class ImageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/images/generations":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            prompt = body.get("prompt", "a beautiful landscape")
            size = body.get("size", "1024x1024")
            steps = body.get("steps", 4)
            seed = body.get("seed")

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

                # Get PIL image from mflux GeneratedImage or use directly
                pil_img = img.image if hasattr(img, "image") else img
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()

                self._json(200, {
                    "created": int(time.time()),
                    "data": [{"b64_json": img_b64}],
                })
            except Exception as e:
                self._json(500, {"error": {"message": str(e)[:300]}})
        else:
            self._json(404, {"error": {"message": "Not found"}})

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._json(200, {"status": "ok", "model": _model_name or "not loaded"})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": _model_name or "image-gen", "object": "model"}]})
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
    _load_model(model, quantize)
    server = HTTPServer(("127.0.0.1", port), ImageHandler)
    print(f"  API: http://127.0.0.1:{port}/v1/images/generations", flush=True)
    print(f"  Health: http://127.0.0.1:{port}/health", flush=True)
    print(f"\n  Configure Open WebUI: Settings → Images → OpenAI URL → http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    model = sys.argv[2] if len(sys.argv) > 2 else "z-image-turbo"
    quantize = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    start_image_server(port, model, quantize)
