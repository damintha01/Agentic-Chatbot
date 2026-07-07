# Agentic Chatbot with LangGraph

A production-ready, agentic conversational assistant built with LangGraph and Streamlit. This repository provides a tool-aware chatbot that supports RAG (PDF ingestion), web searches, a calculator, simulated stock purchases with human-in-the-loop (HITL) approvals, and persistent conversation checkpoints.

Key features
- Topic-based conversation threads (auto-generated titles)
- Newest-first conversation history in the sidebar
- Upload and index PDF content for RAG-powered answers
- Tool integrations: web search, calculator, stock lookup, weather, RAG
- Human-in-the-loop approvals for sensitive actions
- Dark green/black production-style Streamlit UI

Repository layout

- `app.py` — Streamlit frontend and UI wiring
- `backend.py` — LangGraph graph, tools, and checkpoint helpers
- `Dockerfile` — image for containerized deployment
- `requirements.txt` — Python dependencies
- `chatbot.db` — LangGraph SQLite checkpoint DB (created at runtime)
- `faiss_db/` — optional local FAISS vector index after ingesting docs

Quick start (development)

Prerequisites

- Python 3.11 (the project ships a local environment in `lang/`)
- `pip` or equivalent environment manager
- Optional: Docker (for containerized deployment)

Install

1. Create / activate your Python environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. (Optional) If you're using the included Conda environment, use the `lang/` Python binary as configured in this repo.

Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` by default. Use the left sidebar to view conversation threads and the main page to interact with the assistant.

Usage notes

- Starting a new conversation: use the "New Chat" button in the sidebar. New conversations appear at the top.
- Topic titles: the app will auto-generate a short topic title from the first user message; titles are cached per thread.
- Composer: use the visible prompt composer to type messages and optionally attach a PDF to index.
- RAG (PDF ingestion): upload a PDF from the composer or call the RAG tool; indexed vectors are persisted in `faiss_db/`.
- Human-in-the-loop: certain tool actions (e.g., `purchase_stock`) will pause and show an approval banner; approve or reject using the displayed buttons.

Tools included

- `rag_tool(query)` — retrieves document text from indexed PDFs
- `calculator(expression)` — safe math evaluation for simple expressions
- `get_stock_price(symbol)` — fetches a stock quote (requires external API keys if enabled)
- `purchase_stock(symbol, quantity)` — simulated purchase; triggers HITL approval
- `get_current_weather(location)` — current weather (requires `OPENWEATHER_API_KEY`)

Configuration and environment variables

- `OPENWEATHER_API_KEY` — required if you want the weather tool to work
- Any other API keys (e.g., for search or embeddings) should be configured via `.env` or your environment.

Persistence and checkpoints

The app uses LangGraph checkpointing backed by a local SQLite database (`chatbot.db`) to store conversation state. Each conversation thread is identified by a UUID and stored as checkpoints; the UI shows the most recently updated threads first.

PDF / FAISS index

Uploading a PDF will split and index the document into a FAISS index stored under the `faiss_db/` folder. This folder is read by the RAG retriever; keep it in your project directory or mount it when deploying containers.

Docker deployment

Build and run the provided image for a simple containerized deployment:

```bash
docker build -t agentic-chatbot:latest .
docker run -p 8501:8501 --env OPENWEATHER_API_KEY="<key>" -v $(pwd)/faiss_db:/app/faiss_db agentic-chatbot:latest
```

Security and production notes

- Do not commit API keys or secrets. Use environment variables or your cloud secret manager.
- Use HTTPS and proper authentication if exposing the app publicly.
- Backup `chatbot.db` and the `faiss_db/` folder to retain conversation history and indexed content.

Troubleshooting

- Import / startup errors: ensure you run the same Python runtime used to install dependencies. The repository provides a `lang/` environment for convenience.
- If conversation titles are missing or generic, first user message may be too short — add a descriptive prompt.
- If the UI shows a stale import error after code changes, restart the Streamlit server to clear module cache.

Contributing

Contributions, bug reports and feature requests are welcome. Please open issues or pull requests on the repository. When contributing:

- Follow the existing code style in `app.py` and `backend.py`.
- Keep changes small and test locally with `streamlit run app.py`.



