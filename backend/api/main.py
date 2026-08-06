import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))


@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, world!"}


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    response = llm.invoke(request.message)
    return ChatResponse(reply=response.content)
