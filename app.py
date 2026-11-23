import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("RAG_DEV_IDENTITY_ENABLED", "true")
os.environ.setdefault("RAG_DATA_DIR", "/tmp/data")


import gradio as gr
import uvicorn

try:
    import spaces

    @spaces.GPU
    def gpu_ready():
        return "ZeroGPU Engine Active"
except Exception:
    def gpu_ready():
        return "CPU Engine Active"

from production_rag.api import app as fastapi_app

# 3. Mount Gradio interface with the @spaces.GPU hook so ZeroGPU allocates hardware
with gr.Blocks() as demo:
    status_text = gr.Textbox(label="Engine Status", value="Online")
    check_button = gr.Button("Ping Engine")
    check_button.click(fn=gpu_ready, outputs=status_text)

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
