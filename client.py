import os

import gradio as gr
from langserve import RemoteRunnable

runner = RemoteRunnable("http://localhost:8000/rag-chroma-private")

def gradio_interface(input_description):
    result = runner.invoke(input=input_description)
    print(f"LLM Output: {type(result)}")
    return result
    

interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=5, label="Input Description"),
    outputs="text",
    title="Patent Advisor",
    description="Enter a description related to patents to get the appropriate response.",
    allow_flagging='never'
)

interface.launch()
