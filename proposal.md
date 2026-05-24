# Project Proposal: Jackson Investments - Financial Advisor Bot

**Student:** David Jackson
**Module:** CM3070 Computer Science Final Project

---

## Project Overview

For my final project, I plan to build a financial advisor bot that can analyse stock market data and give a user a simple recommendation about whether a stock looks like a buy, sell, or hold.

The application will consider a portfolio of stocks rather than looking at each stock completely on its own. This means the buy, sell, or hold recommendation for one ticker should also take into account the user's positions in other tickers, so the system can make decisions based on the overall portfolio state.

The first version will focus on equities listed on the TSX, NYSE, and NASDAQ exchanges. I plan to use publicly available historical price data through the `yfinance` library.

---

## AI/ML Approach

The main AI part of the project will use two stages. I am using this approach because predicting a signal and deciding what to do with that signal are related, but they are not exactly the same problem.

As a reference point, Danelfin describes a similar kind of stock analysis workflow on its "How it Works" page. It says that it collects and generates daily technical, fundamental, and sentiment indicators, turns those into stock features, and then uses decision trees to estimate whether a stock is likely to beat the market over the next three months: https://danelfin.com/how-it-works/infographic

My project will take a slightly different approach. Instead of using decision trees as the main prediction method, I plan to use a Long Short-Term Memory model to analyse the time-series stock data and predict what the stock may do next. I will then use a reinforcement learning model to decide how that prediction should affect the state of the portfolio.

**Stage 1 - LSTM Signal Generator**

The first stage will be an LSTM neural network trained on historical OHLCV price data. I will also add common technical indicators such as RSI, MACD, Bollinger Bands, ATR, and volume ratios.

For each stock, the model will look at a rolling window of recent trading days and output a directional signal, such as the probability that the price will rise over the next day. I chose an LSTM because stock data is time-based, and I want the model to be able to learn patterns across several days rather than treating each day in isolation.

**Stage 2 - Reinforcement Learning Agent**

The second stage will be a reinforcement learning agent. It will use the LSTM signal, along with information about the current portfolio state, to decide whether to buy, sell, or hold.

This part is important because a good signal does not always mean the system should immediately trade. The agent needs to consider context such as whether the stock is already held, whether the portfolio has available cash, and how much risk is already being taken. In this design, the LSTM focuses on identifying patterns in the data, while the reinforcement learning agent focuses on the trading decision.

---

## System Components

- **Data pipeline:** Downloads historical market data from `yfinance` and creates the features needed for the models.
- **Training pipeline:** Trains the LSTM first, then trains the reinforcement learning agent separately from the user-facing application.
- **Advisor bot:** Uses the trained models to return a recommendation, confidence level, and a short explanation.
- **Backtesting module:** Tests the strategy on historical data that was not used during training.
- **Web interface:** Provides a chat-style interface built with Next.js. The user will be able to ask about a stock and receive a plain-language answer. I may use a local LLM through Ollama as the presentation layer, but the LLM will not make the investment decision itself. It will only explain the structured output from the models in a more readable way.
