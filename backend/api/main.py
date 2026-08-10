import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fetch_data import get_latest_close
from core.portfolio_store import get_portfolio, record_trade
from core.predict import predict_stock_return
from core.tickers import TICKERS
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
    """Predict a stock's return over the next 5 trading days using the trained LSTM model."""
    try:
        return predict_stock_return(ticker)
    except ValueError as exc:
        return {"error": str(exc)}


@tool
def get_portfolio_positions() -> dict:
    """Look up the user's current cash balance and share holdings."""
    return get_portfolio()


@tool
def record_portfolio_trade(ticker: str, action: str, shares: float) -> dict:
    """Record a buy or sell the user says they already made, e.g. "I bought 20 shares of AAPL".

    action must be "buy" or "sell". Priced at today's closing price, not
    whatever price the user may have actually paid.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        return {"error": f"{ticker} is not a ticker this app tracks."}

    try:
        price = get_latest_close(ticker)
        return record_trade(ticker, action, shares, price)
    except ValueError as exc:
        return {"error": str(exc)}


agent = create_agent(
    llm,
    tools=[get_stock_return_prediction, get_portfolio_positions, record_portfolio_trade],
    system_prompt=(
        "You are a helpful financial advisor assistant. When asked about a stock's near-term "
        "outlook, use the get_stock_return_prediction tool rather than guessing, then "
        "explain the result in plain, non-technical language. Always make clear this "
        "is a model prediction, not financial advice.\n\n"
        "Use get_portfolio_positions whenever you need to know the user's current cash or "
        "holdings, including before answering questions about their portfolio. When the user "
        "says they bought or sold shares (e.g. 'I bought 20 shares of AAPL'), call "
        "record_portfolio_trade to log it, then confirm what was recorded and the resulting "
        "cash balance. If a trade fails (e.g. not enough cash or shares), explain why in plain "
        "language instead of retrying with different numbers."
    ),
)


@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, world!"}


@app.get("/portfolio")
def portfolio() -> dict:
    return get_portfolio()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    result = agent.invoke({"messages": [{"role": "user", "content": request.message}]})
    return ChatResponse(reply=result["messages"][-1].content)
