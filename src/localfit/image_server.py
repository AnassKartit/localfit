"""Tiny OpenAI-compatible image generation API wrapping mflux (Flux on Apple Silicon)."""

import json
import os
import subprocess
import sys
import tempfile
import time
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

DEFAULT_PORT = 8189
MODEL = "flux2-klein-4b"


class ImageHandler(BaseHTTPRequestHandler):
    """Serves /v1/images/generations compatible with Open WebUI."""

    def do_POST(self):
        if self.path == "/v1/images/generations":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            prompt = body.get("prompt", "a beautiful sunset")
            size = body.get("size", "1024x1024")
            n = body.get("n", 1)
            model = body.get("model", MODEL)
            quality = body.get("quality", "standard")

            w, h = size.split("x") if "x" in size else ("1024", "1024")
            steps = 4 if "schnell" in model or "klein" in model else 20

            # Generate with mflux
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                out_path = f.name

            cmd = [
                "mflux-generate-flux2",
                "--base-model", model.replace("flux2-", "").replace("flux-", ""),
                "--prompt", prompt,
                "--width", w,
                "--height", h,
                "--steps", str(steps),
                "--quantize", "4",
                "--output", out_path,
            ]

            # Fallback to mflux-generate if flux2 command doesn't exist
            if not os.path.exists("/opt/homebrew/bin/mflux-generate-flux2"):
                cmd[0] = "mflux-generate"
                cmd[1:3] = ["--base-model", "flux2-klein-4b"]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    self._error(500, f"mflux failed: {result.stderr[:300]}")
                    return
            except subprocess.TimeoutExpired:
                self._error(500, "Generation timed out (120s)")
                return
            except FileNotFoundError:
                self._error(500, "mflux not installed: pip install mflux")
                return

            # Read image and encode
            with open(out_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            os.unlink(out_path)

            response = {
                "created": int(time.time()),
                "data": [{"b64_json": img_data}],
            }
            self._json(200, response)

        elif self.path == "/v1/models":
            self._json(200, {
                "data": [{"id": MODEL, "object": "model", "owned_by": "localfit"}]
            })
        else:
            self._error(404, "Not found")

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._json(200, {"status": "ok", "model": MODEL})
        elif self.path == "/v1/models":
            self._json(200, {
                "data": [{"id": MODEL, "object": "model", "owned_by": "localfit"}]
            })
        else:
            self._error(404, "Not found")

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg):
        self._json(code, {"error": {"message": msg, "type": "server_error"}})

    def log_message(self, format, *args):
        # Quiet logging
        pass


def start_image_server(port=DEFAULT_PORT, model=None):
    """Start the image generation API server."""
    global MODEL
    if model:
        MODEL = model

    server = HTTPServer(("127.0.0.1", port), ImageHandler)
    print(f"  Image API: http://127.0.0.1:{port}/v1/images/generations")
    print(f"  Model: {MODEL} (mflux, 4-bit quantized)")
    server.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    model = sys.argv[2] if len(sys.argv) > 2 else None
    start_image_server(port, model)
