import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# 1. Add 'src' to module search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 2. Environment defaults for Hugging Face Spaces
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("RAG_DEV_IDENTITY_ENABLED", "true")
os.environ.setdefault("RAG_DATA_DIR", "/tmp/data")
os.environ["GRADIO_SSR_MODE"] = "False"

import gradio as gr
import uvicorn

try:
    import spaces

    @spaces.GPU(duration=60)
    def gpu_ready():
        return "ZeroGPU Engine Active"
except Exception:
    def gpu_ready():
        return "CPU Engine Active"

from production_rag.api import app as fastapi_app

# 3. Mount Gradio interface with demo.queue() so ZeroGPU registers the GPU worker
with gr.Blocks() as demo:
    status_text = gr.Textbox(label="Engine Status", value="Online")
    check_button = gr.Button("Ping Engine")
    check_button.click(fn=gpu_ready, outputs=status_text)

demo.queue()

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio", ssr_mode=False)


def initialize_zerogpu():
    """Explicitly trigger ZeroGPU startup broker registration for FastAPI/ASGI servers."""
    try:
        from spaces import zero as _spaces_zero

        if hasattr(_spaces_zero, "startup"):
            _spaces_zero.startup()
            print("[ZeroGPU] Startup broker registration sent successfully.")
    except Exception as error:
        print(f"[ZeroGPU] Startup registration skipped: {error!r}")


def create_listening_socket(host: str, port: int, max_retries: int = 15):
    """Bind socket with SO_REUSEADDR and SO_REUSEPORT, retrying if container socket is transitioning."""
    print(f"[Startup] Attempting to bind socket on {host}:{port} (PID {os.getpid()})...")
    for attempt in range(1, max_retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind((host, port))
            sock.listen(128)
            print(f"[Startup] Successfully bound to {host}:{port} on attempt {attempt}.")
            return sock
        except OSError as err:
            print(f"[Startup] Attempt {attempt}/{max_retries} failed to bind {host}:{port}: {err}")
            time.sleep(2)
    raise RuntimeError(f"Could not bind to {host}:{port} after {max_retries} attempts")


if __name__ == "__main__":
    initialize_zerogpu()
    port_str = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7860"
    target_port = int(port_str)
    bound_socket = create_listening_socket("0.0.0.0", target_port)
    config = uvicorn.Config(app=app, log_level="info")
    server = uvicorn.Server(config=config)
    server.run(sockets=[bound_socket])
