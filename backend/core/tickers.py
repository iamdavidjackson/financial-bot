TECHNOLOGY_AND_COMMUNICATION_SERVICES_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL',
    'CRM', 'ADBE', 'AMD', 'INTC', 'CSCO', 'IBM', 'QCOM', 'TXN', 'NOW', 'INTU', 'ACN',
    'DIS', 'NFLX', 'CMCSA', 'TMUS', 'VZ', 'T',
]

FINANCIALS_TICKERS = [
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA',
    'SPGI', 'CB', 'PGR',
]

HEALTHCARE_TICKERS = [
    'UNH', 'JNJ', 'LLY', 'MRK', 'ABBV', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
    'GILD', 'BMY', 'ISRG', 'MDT',
]

CONSUMER_STAPLES_AND_DISCRETIONARY_TICKERS = [
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW',
    'TGT', 'CL', 'EL',
]

INDUSTRIALS_TICKERS = [
    'CAT', 'DE', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'BA', 'MMM', 'ETN', 'EMR',
]

ENERGY_TICKERS = [
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX',
]

UTILITIES_TICKERS = [
    'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'SRE',
]

MATERIALS_TICKERS = [
    'LIN', 'APD', 'SHW', 'FCX', 'NEM',
]

REAL_ESTATE_TICKERS = [
    'AMT', 'PLD', 'EQIX',
]

TICKER_GROUPS = {
    'Technology and communication services': TECHNOLOGY_AND_COMMUNICATION_SERVICES_TICKERS,
    'Financials': FINANCIALS_TICKERS,
    'Healthcare': HEALTHCARE_TICKERS,
    'Consumer staples and discretionary': CONSUMER_STAPLES_AND_DISCRETIONARY_TICKERS,
    'Industrials': INDUSTRIALS_TICKERS,
    'Energy': ENERGY_TICKERS,
    'Utilities': UTILITIES_TICKERS,
    'Materials': MATERIALS_TICKERS,
    'Real estate': REAL_ESTATE_TICKERS,
}

TICKERS = [ticker for group in TICKER_GROUPS.values() for ticker in group]

def get_ticker_groups_and_tickers():
    """Return the ticker groups mapping and the flat ticker list."""
    return TICKER_GROUPS, TICKERS
