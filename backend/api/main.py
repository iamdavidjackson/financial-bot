import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.predict import predict_stock_return
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain_core.tools import tool
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


@tool
def get_stock_return_prediction(ticker: str) -> dict:
    try:
        return predict_stock_return(ticker)
    except ValueError as exc:
        return {"error": str(exc)}


agent = create_agent(
    llm,
    tools=[get_stock_return_prediction],
    system_prompt=(
        "You are a helpful financial advisor assistant. When asked about a stock's near-term "
        "outlook, use the get_stock_return_prediction tool rather than guessing, then "
        "explain the result in plain, non-technical language. Always make clear this "
        "is a model prediction, not financial advice."
    ),
)


@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, world!"}


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    result = agent.invoke({"messages": [{"role": "user", "content": request.message}]})
    return ChatResponse(reply=result["messages"][-1].content)
