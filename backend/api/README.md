# API

FastAPI backend with a LangChain-powered `/chat` endpoint.

## Prerequisites

- [Ollama](https://ollama.com) must be installed and running locally, with the `llama3.2:3b` model pulled:

  ```bash
  brew install ollama
  brew services start ollama
  ollama pull llama3.2:3b
  ```

  The `/chat` endpoint uses Ollama on `localhost:11434`, so it will fail if Ollama isn't running. `dev.sh` in the project root will start the Ollama service for you if it isn't already running.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

