# Load
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from typing import List, Union, Dict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

loader = CSVLoader(file_path="/home/vns1311/ai-patent/ai-patent-advisor/packages/rag-chroma-private/docs/sample.csv", source_column="id", metadata_columns=["date","codes","sections"])
data = loader.load()

# Add to vectorDB
model_name = "AI-Growth-Lab/PatentSBERTa"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

vectorstore = Chroma.from_documents(
    documents=data,
    collection_name="rag-private",
    embedding=embedding_function,
)
retriever = vectorstore.as_retriever()

# Prompt
# Optionally, pull from the Hub
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")
# Or, define your own:
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
# Select the LLM that you downloaded
ollama_llm = "llama3"
model = ChatOllama(model=ollama_llm, temperature=0)

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str

class Output(BaseModel):
    chat_history: List[Union[HumanMessage, AIMessage]]
    input: str
    context: List[Document]
    answer: str


chain = chain.with_types(input_type=Question, output_type=Output)
