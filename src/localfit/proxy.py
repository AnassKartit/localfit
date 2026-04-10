"""Anthropic API proxy for llama-server.

Translates Anthropic Messages API → OpenAI Chat Completions API.
This lets Claude Code, Gemini CLI, and other tools use local llama-server.

Usage:
    localfit --serve gemma4-26b --proxy
    # Then:
    ANTHROPIC_BASE_URL=http://localhost:8090 claude --model gemma4-26b

Architecture:
    Claude Code → Anthropic API (port 8090) → this proxy → OpenAI API (port 8089) → llama-server
"""
import json
import argparse
import http.server
import subprocess
import sys
import time
import urllib.request
import urllib.error
import socketserver
import threading

LLAMA_URL = "http://127.0.0.1:8089/v1/chat/completions"
PROXY_PORT = 8090


class ReusableTCPServer(socketserver.TCPServer):
    """TCPServer variant that can be restarted without waiting on TIME_WAIT."""

    allow_reuse_address = True


class AnthropicProxyHandler(http.server.BaseHTTPRequestHandler):
    """Translates Anthropic Messages API requests to OpenAI format."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_POST(self):
        if self.path == "/v1/messages":
            self._handle_messages()
        else:
            self.send_error(404, f"Unknown endpoint: {self.path}")

    def do_GET(self):
        if self.path == "/v1/models" or self.path == "/models":
            self._handle_models()
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "proxy": "localfit"}).encode())
        else:
            self.send_error(404)

    def _handle_models(self):
        """Forward model list from llama-server."""
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8089/v1/models",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            # Convert to Anthropic format
            models = []
            for m in data.get("data", []):
                models.append({
                    "id": m.get("id", "local"),
                    "type": "model",
                    "display_name": m.get("id", "local"),
                })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": models}).encode())
        except Exception as e:
            self.send_error(502, f"llama-server unavailable: {e}")

    def _handle_messages(self):
        """Translate Anthropic Messages API → OpenAI Chat Completions."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            anthropic_req = json.loads(body)

            # ── Translate request ──
            openai_req = self._anthropic_to_openai(anthropic_req)

            # ── Forward to llama-server ──
            payload = json.dumps(openai_req).encode()
            req = urllib.request.Request(
                LLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=300) as resp:
                openai_resp = json.loads(resp.read())

            # ── Translate response ──
            anthropic_resp = self._openai_to_anthropic(openai_resp, anthropic_req)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(anthropic_resp).encode())

        except urllib.error.URLError as e:
            self.send_error(502, f"llama-server unavailable: {e}")
        except Exception as e:
            self.send_error(500, f"Proxy error: {e}")

    def _anthropic_to_openai(self, req):
        """Convert Anthropic Messages API request to OpenAI Chat Completions."""
        messages = []

        # System message
        system = req.get("system", "")
        if isinstance(system, list):
            system = " ".join(s.get("text", str(s)) for s in system)
        if system:
            messages.append({"role": "system", "content": system})

        # Convert messages
        for msg in req.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Anthropic uses content blocks, OpenAI uses strings
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        # Tool call in assistant message
                        pass
                    elif block.get("type") == "tool_result":
                        text_parts.append(f"Tool result: {block.get('content', '')}")
                content = "\n".join(text_parts)

            # Map Anthropic role to OpenAI
            if role == "assistant":
                # Check for tool_use blocks
                tool_calls = []
                if isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            })
                if tool_calls:
                    messages.append({"role": "assistant", "content": content or None, "tool_calls": tool_calls})
                else:
                    messages.append({"role": "assistant", "content": content})
            elif role == "user":
                # Check for tool_result blocks
                if isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                tool_content = " ".join(
                                    t.get("text", str(t)) for t in tool_content
                                )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(tool_content),
                            })
                    # Also add any text content
                    text_parts = [
                        b.get("text", "") for b in msg["content"]
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if text_parts:
                        messages.append({"role": "user", "content": "\n".join(text_parts)})
                else:
                    messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": role, "content": content})

        # Build OpenAI request
        openai_req = {
            "messages": messages,
            "max_tokens": req.get("max_tokens", 4096),
            "temperature": req.get("temperature", 0.7),
            "stream": False,  # TODO: streaming support
        }

        # Convert Anthropic tools to OpenAI function format
        tools = req.get("tools", [])
        if tools:
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                })
            openai_req["tools"] = openai_tools

        return openai_req

    def _openai_to_anthropic(self, resp, original_req):
        """Convert OpenAI Chat Completions response to Anthropic Messages format."""
        choice = resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = resp.get("usage", {})

        # Build content blocks
        content = []

        # Text content
        text = message.get("content", "")
        if text:
            content.append({"type": "text", "text": text})

        # Tool calls
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {"raw": func.get("arguments", "")}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{hash(func.get('name', '')) % 10**12:012x}"),
                "name": func.get("name", ""),
                "input": args,
            })

        # Stop reason mapping
        finish = choice.get("finish_reason", "stop")
        stop_reason = "tool_use" if finish == "tool_calls" else "end_turn"

        return {
            "id": f"msg_{resp.get('id', 'local')}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_req.get("model", "local"),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }


def start_proxy(port=PROXY_PORT, llama_url=LLAMA_URL):
    """Start the Anthropic-to-OpenAI proxy server."""
    global LLAMA_URL
    LLAMA_URL = llama_url

    with ReusableTCPServer(("", port), AnthropicProxyHandler) as httpd:
        httpd.serve_forever()


def start_proxy_background(port=PROXY_PORT, llama_url=LLAMA_URL):
    """Start proxy in background thread."""
    t = threading.Thread(target=start_proxy, args=(port, llama_url), daemon=True)
    t.start()
    return t


def proxy_healthcheck(port=PROXY_PORT, timeout=1.0):
    """Return True if the local Anthropic compatibility proxy is responding."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_proxy(port=PROXY_PORT, timeout=5.0):
    """Wait for the proxy to accept requests."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proxy_healthcheck(port=port, timeout=1.0):
            return True
        time.sleep(0.2)
    return False


def ensure_proxy_process(llama_url=LLAMA_URL, port=PROXY_PORT):
    """Start a detached proxy process if one is not already running."""
    if wait_for_proxy(port=port, timeout=0.4):
        return True

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "localfit.proxy",
            "--port",
            str(port),
            "--llama-url",
            llama_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return wait_for_proxy(port=port, timeout=5.0)


def main(argv=None):
    """Run the Anthropic compatibility proxy as a standalone process."""
    parser = argparse.ArgumentParser(description="localfit Anthropic compatibility proxy")
    parser.add_argument("--port", type=int, default=PROXY_PORT)
    parser.add_argument("--llama-url", default=LLAMA_URL)
    args = parser.parse_args(argv)
    start_proxy(port=args.port, llama_url=args.llama_url)


if __name__ == "__main__":
    main()
