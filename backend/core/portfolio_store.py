import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "portfolio.db"
DEFAULT_INITIAL_CASH = 100_000.0


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the db tables if they don't exist yet with starting cash."""
    with _connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS cash (id INTEGER PRIMARY KEY CHECK (id = 1), balance REAL NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS holdings (ticker TEXT PRIMARY KEY, shares REAL NOT NULL)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO cash (id, balance) VALUES (1, ?)", (DEFAULT_INITIAL_CASH,)
        )


def get_portfolio(db_path: Path = DB_PATH) -> dict:
    """Return the current cash balance and every ticker with a position."""
    init_db(db_path)

    with _connect(db_path) as connection:
        cash = connection.execute("SELECT balance FROM cash WHERE id = 1").fetchone()[0]
        holdings = dict(connection.execute("SELECT ticker, shares FROM holdings ORDER BY ticker").fetchall())

    return {"cash": cash, "holdings": holdings}


def record_trade(ticker: str, action: str, shares: float, price: float, db_path: Path = DB_PATH) -> dict:
    """Apply a buy or sell to cash and holdings, and return the updated portfolio."""
    ticker = ticker.upper()
    action = action.lower()

    if action not in ("buy", "sell"):
        raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")
    if shares <= 0:
        raise ValueError("shares must be positive")
    if price <= 0:
        raise ValueError("price must be positive")

    init_db(db_path)

    with _connect(db_path) as connection:
        cash = connection.execute("SELECT balance FROM cash WHERE id = 1").fetchone()[0]
        current_row = connection.execute(
            "SELECT shares FROM holdings WHERE ticker = ?", (ticker,)
        ).fetchone()
        current_shares = current_row[0] if current_row else 0.0

        trade_value = shares * price

        if action == "buy":
            # Check that the user has enough cash to make the purchase.
            if trade_value > cash:
                raise ValueError(
                    f"Not enough cash: buying {shares} {ticker} at ${price:.2f} costs "
                    f"${trade_value:,.2f}, but only ${cash:,.2f} is available."
                )
            new_cash = cash - trade_value
            new_shares = current_shares + shares
        else:
            # Check that the user has enough shares to sell.
            if shares > current_shares:
                raise ValueError(
                    f"Not enough shares: trying to sell {shares} {ticker}, but only "
                    f"{current_shares} are held."
                )
            new_cash = cash + trade_value
            new_shares = current_shares - shares

        connection.execute("UPDATE cash SET balance = ? WHERE id = 1", (new_cash,))

        if new_shares > 0:
            connection.execute(
                "INSERT INTO holdings (ticker, shares) VALUES (?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET shares = excluded.shares",
                (ticker, new_shares),
            )
        else:
            # Drop the row instead of leaving a zero-share position sitting around.
            connection.execute("DELETE FROM holdings WHERE ticker = ?", (ticker,))

    return get_portfolio(db_path)
