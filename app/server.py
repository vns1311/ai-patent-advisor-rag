from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_chroma_private import chain as rag_chroma_private_chain
from dotenv import load_dotenv, find_dotenv
from langchain_core.pydantic_v1 import BaseModel
from typing import List, Union, Dict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(find_dotenv())

app = FastAPI()

# Add typing for input
class Question(BaseModel):
    __root__: str

class Output(BaseModel):
    # chat_history: List[Union[HumanMessage, AIMessage]]
    input: str
    context: List[Document]
    answer: str

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(
    app,
    rag_chroma_private_chain,
    output_type=Output,
    path="/rag-chroma-private",
    playground_type="default",
    include_callback_events=False,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
