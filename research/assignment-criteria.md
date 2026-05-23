# 4.2 Project Idea 2: Financial Advisor Bot

## What problem is this project solving, or what is the project idea?

This project involves the creation of a financial advisor bot. The bot analyses financial data in order to make recommendations for a dynamic investment strategy, which you might also call 'active portfolio management'.

## What is the background and context to the question or project idea above?

You will need to identify which kind of financial systems the bot will advise about. For example, the stock market, currency exchange market, crypto-currencies, classic car trading and so on.

You will need to decide what kind of AI/ML techniques you will use to implement the bot. We recommend that you use techniques that you have seen in the course, for example, modelling the problem as 'decision making under uncertainty' would lend itself to reinforcement learning. Or you might attempt to evolve an investment strategy.

It should be possible for a non-technical user to interact with the bot to receive advice, for example, via a web interface. The bot should present its analysis and recommendations to the user with explanations. For example, the bot might say 'NVIDIA stock is likely to rise by 25% so we recommend taking up a position on that now and exiting after 20% growth'. It is up to you to decide how the bot presents this advice. You could consider using local language models to format the recommendations into prose.

## Here are some recommended sources for you to begin your research.

You can find stock price data feeds:
- https://github.com/ranaroussi/yfinance

Or crypto data feeds:
- https://github.com/ccxt/ccxt

You might need a backtesting system to test your advisor's trading strategies:
- https://github.com/topics/backtesting

You might consider using local language models to help the user interact with the advisor bot. For example, you might conceive of the bot as a tool-using agent. The following systems might prove useful:
- https://ollama.com/
- https://docs.openwebui.com/

## What would the final product or final outcome look like?

The final product would be an integrated and tested system designed for non-technical users to use. You will probably need separate systems for training models and gathering data, if you choose to do that. We would not expect the non-technical user to be able to interact with the model training system. But they should be able to interact with the advisor bot once trained. You should present evidence that you have evaluated the advice that the advisor bot generates.

## What would a prototype look like?

A prototype would need to have the main parts functioning but it would not need to have the full user interface. So you would need a way to interact with the advisor bot, even if it is command line based. You would need to identify appropriate data sources and to gather and ingest that data. You would need a way for the advisor bot to analyse and/or learn from data.

## What kinds of techniques/processes/CS fundamentals are relevant to this project?

You can choose how to implement the advisor bot, but reinforcement learning, evolutionary strategies, neural networks and so on would be the typical techniques. You might want to use some language model techniques as well.

You will need to do some significant software engineering and testing here so you would need the techniques from the degree relating to those.

There is also a strong data science component around gathering and analysing data.

Web programming will be needed to create the web interface.

## What would the output of these techniques/processes/CS fundamentals look like?

- We expect to see well tested software
- We expect to see a solid data analysis and machine learning/AI workflow implemented
- We expect to see a working web user interface

## How will this project be evaluated and assessed by the student (i.e. during iteration of the project)? What criteria are important?

We imagine that you will work iteratively on designing, implementing and testing the components of the system. The components include the data gathering, training, advisor bot interaction, web interface.

## For this brief, what might a minimum pass (e.g. 3rd) student project look like?

A basic project should present a complete working system but with limited complexity of components. We would need to see some design, implementation and evaluation work.

## For this brief, what might a good (e.g. 2:2 – 2:1) student project look like?

A 2:2–2:1 project should present a complete working system with some significant development and testing work evident in some or all of the components. You should present some detailed evaluation of the components of the system and clear planning. We would want to see some thought put into the design of the advisor bot and its algorithm in particular.

## For this brief, what might an outstanding (e.g. 1st) student project look like?

An outstanding project should present a complete working system with some significant development and testing work evident in all of the components. You should present detailed evaluation and testing of the components of the system and clear planning. We would expect the advisor bot to implement some sort of interesting and advanced algorithm with significant effort put into it.
