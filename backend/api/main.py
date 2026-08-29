import contextvars
import os
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fetch_data import get_latest_close
from core.portfolio_agent import PORTFOLIO_TICKERS, recommend_trades
from core.portfolio_store import get_portfolio, record_trade, reset_portfolio
from core.predict import predict_stock_return
from core.tickers import TICKERS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel


def seed_random_portfolio() -> None:
    # Resets the portfolio and seeds fresh random positions on every server start.
    reset_portfolio()

    tickers = random.sample(PORTFOLIO_TICKERS, k=random.randint(1, len(PORTFOLIO_TICKERS)))

    for ticker in tickers:
        try:
            price = get_latest_close(ticker)
            record_trade(ticker, "buy", shares=float(random.randint(5, 50)), price=price)
        except ValueError as exc:
            print(f"Could not seed a random {ticker} position: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_random_portfolio()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))

# Save widgets emitted during a chat request in a context variable so they can be returned in the response.
_widgets: contextvars.ContextVar[list | None] = contextvars.ContextVar("widgets", default=None)


def _emit_widget(widget_type: str, data: dict) -> None:
    """Attach a structured payload to the in-flight chat response, if one is collecting."""
    collected = _widgets.get()
    if collected is not None:
        collected.append({"type": widget_type, "data": data})


@tool
def get_stock_return_prediction(ticker: str) -> dict:
    """Predict a stock's return over the next 5 trading days using the trained LSTM model."""
    try:
        result = predict_stock_return(ticker)
    except ValueError as exc:
        return {"error": str(exc)}
    _emit_widget("stock_prediction", result)
    return result


@tool
def get_portfolio_positions() -> dict:
    """Look up the user's current cash balance and share holdings."""
    result = get_portfolio()
    _emit_widget("portfolio", result)
    return result


@tool
def get_portfolio_trade_recommendation() -> dict:
    """Ask the PPO agent what action to take with each tracked ticker."""
    try:
        result = recommend_trades()
    except RuntimeError as exc:
        return {"error": str(exc)}
    _emit_widget("trade_recommendations", result)
    return result


@tool
def record_portfolio_trade(ticker: str, action: str, shares: float) -> dict:
    """Record a buy or sell trade."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        return {"error": f"{ticker} is not a ticker this app tracks."}

    try:
        price = get_latest_close(ticker)
        result = record_trade(ticker, action, shares, price)
    except ValueError as exc:
        return {"error": str(exc)}
    _emit_widget(
        "trade_confirmation",
        {"ticker": ticker, "action": action.lower(), "shares": shares, "price": price, **result},
    )
    return result


agent = create_agent(
    llm,
    tools=[
        get_stock_return_prediction,
        get_portfolio_positions,
        get_portfolio_trade_recommendation,
        record_portfolio_trade,
    ],
    system_prompt=(
        "You are a helpful financial advisor assistant. When asked about a stock's near-term "
        "outlook, use the get_stock_return_prediction tool rather than guessing, then "
        "explain the result in plain, non-technical language. Always make clear this "
        "is a model prediction, not financial advice.\n\n"
        "Use get_portfolio_positions whenever you need to know the user's current cash or "
        "holdings, including before answering questions about their portfolio. When the user "
        "asks what they should buy, sell, or hold, or otherwise asks for trading advice on their "
        "portfolio, use get_portfolio_trade_recommendation rather than guessing, then explain "
        "each recommended action in plain language along with the reasoning available (current "
        "price, holding size, signal strength). Always make clear this is a trained model's "
        "suggestion, not financial advice, and that it only covers the tickers it was trained on. "
        "When the user says they bought or sold shares (e.g. 'I bought 20 shares of AAPL'), call "
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


class Widget(BaseModel):
    type: str
    data: dict


class ChatResponse(BaseModel):
    reply: str
    widgets: list[Widget] = []


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    collected: list[dict] = []
    token = _widgets.set(collected)
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": request.message}]})
    finally:
        _widgets.reset(token)
    return ChatResponse(reply=result["messages"][-1].content, widgets=collected)
