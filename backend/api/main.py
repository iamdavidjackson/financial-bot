import contextvars
import json
import os
import random
import re
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fetch_data import get_latest_close, get_recent_closes
from core.portfolio_agent import (
    PORTFOLIO_TICKER_NAMES,
    PORTFOLIO_TICKERS,
    recommend_trades,
)
from core.portfolio_store import get_portfolio, record_trade, reset_portfolio
from core.predict import predict_stock_return
from core.tickers import TICKERS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel


def seed_random_portfolio() -> None:
    # Resets the portfolio and seeds a random-sized position in every tracked ticker on each server start.
    reset_portfolio()

    for ticker in PORTFOLIO_TICKERS:
        try:
            price = get_latest_close(ticker)
            record_trade(
                ticker, "buy", shares=float(random.randint(5, 50)), price=price
            )
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

# Temperature 0 makes small models less likely to write a tool call as plain text.
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"), temperature=0)

# Save widgets emitted during a chat request in a context variable so they can be returned in the response.
_widgets: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "widgets", default=None
)


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
        {
            "ticker": ticker,
            "action": action.lower(),
            "shares": shares,
            "price": price,
            **result,
        },
    )
    return result


TOOLS = [
    get_stock_return_prediction,
    get_portfolio_positions,
    get_portfolio_trade_recommendation,
    record_portfolio_trade,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

agent = create_agent(
    llm,
    tools=TOOLS,
    system_prompt=(
        "You are a helpful financial advisor assistant. When asked about a stock's near-term "
        "outlook, use the get_stock_return_prediction tool rather than guessing, then "
        "explain the result in plain, non-technical language. Always make clear this "
        "is a model prediction, not financial advice.\n\n"
        "get_stock_return_prediction only works for these tickers, so use the exact symbol "
        "from this list and do not guess or invent one:\n"
        f"{', '.join(TICKERS)}.\n"
        "If the user names a company that is not in this list, tell them it is not supported "
        "instead of substituting a different ticker.\n\n"
        "Use get_portfolio_positions whenever you need to know the user's current cash or "
        "holdings, including before answering questions about their portfolio. When the user "
        "asks what they should buy, sell, or hold, or otherwise asks for trading advice on their "
        "portfolio, use get_portfolio_trade_recommendation rather than guessing, then explain "
        "each recommended action in plain language along with the data available (current "
        "price, holding size, signal rank) and, for each buy or sell, how many shares the "
        "agent would trade (recommended_shares), the rough dollar value, and the resulting "
        "holding (holding_after_trade_shares, use this number rather than calculating it). "
        "signal_rank is where the stock's predicted 5-day return ranks against the other "
        "tracked stocks today: rank 1 means the highest predicted return, not a confident "
        "or strong prediction, so describe it as e.g. 'ranked 1 of 5'. The trading agent "
        "weighs this rank together with cash, holdings and recent prices, so its action can "
        "go against the rank; it gives no reasons, so do not invent one, and if the action "
        "seems to conflict with the rank, say so plainly. Always make clear this is a trained model's "
        "suggestion, not financial advice, and that it only covers the tickers it was trained on, "
        "which are: "
        f"{', '.join(f'{ticker} ({PORTFOLIO_TICKER_NAMES.get(ticker, ticker)})' for ticker in PORTFOLIO_TICKERS)}. "
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


DEFAULT_QUOTE_TICKERS = PORTFOLIO_TICKERS[:3]


@app.get("/quotes")
def quotes(tickers: str | None = None) -> list[dict]:
    symbols = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else DEFAULT_QUOTE_TICKERS
    )

    results = []
    for ticker in symbols:
        price, previous_close = get_recent_closes(ticker)
        change = price - previous_close
        results.append(
            {
                "ticker": ticker,
                "name": PORTFOLIO_TICKER_NAMES.get(ticker, ticker),
                "price": price,
                "previous_close": previous_close,
                "change": change,
                "change_percent": change / previous_close * 100,
            }
        )
    return results


class ChatRequest(BaseModel):
    message: str


class Widget(BaseModel):
    type: str
    data: dict


class ChatResponse(BaseModel):
    reply: str
    widgets: list[Widget] = []


def _parse_text_tool_call(text: str) -> tuple[str, dict] | None:
    """Recognise a tool call the model wrote as plain text instead of calling the tool."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0).replace('\\"', '"'))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("name") not in TOOLS_BY_NAME:
        return None
    args = payload.get("parameters", payload.get("arguments", {}))
    return payload["name"], args if isinstance(args, dict) else {}


def _recover_text_tool_call(messages: list) -> list:
    """Run a tool call written as text, then let the agent answer with the real result."""
    parsed = _parse_text_tool_call(messages[-1].content)
    if parsed is None:
        return messages

    name, args = parsed
    call_id = f"call_{uuid.uuid4().hex}"
    try:
        output = TOOLS_BY_NAME[name].invoke(args)
    except Exception as exc:  # e.g. missing or wrongly named arguments
        output = {"error": str(exc)}

    recovered = [
        *messages[:-1],
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}]),
        ToolMessage(content=json.dumps(output, default=str), tool_call_id=call_id),
    ]
    return agent.invoke({"messages": recovered})["messages"]


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    collected: list[dict] = []
    token = _widgets.set(collected)
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]}
        )
        messages = _recover_text_tool_call(result["messages"])
    finally:
        _widgets.reset(token)
    return ChatResponse(reply=messages[-1].content, widgets=collected)
