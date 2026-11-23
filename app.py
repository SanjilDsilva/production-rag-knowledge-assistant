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
from production_rag.api import app as fastapi_app


demo = gr.Blocks()
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
