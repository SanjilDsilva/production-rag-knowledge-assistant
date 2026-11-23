import gradio as gr
import uvicorn
from production_rag.api import app as fastapi_app


demo = gr.Blocks()
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
