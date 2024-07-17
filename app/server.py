from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_chroma_private import chain as rag_chroma_private_chain
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(
    app,
    rag_chroma_private_chain,
    path="/rag-chroma-private",
    playground_type="default",
    include_callback_events=False,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
