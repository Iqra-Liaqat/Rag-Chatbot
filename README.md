# RAG Chatbot

A local Retrieval-Augmented Generation chatbot — upload PDF/DOCX/TXT/MD documents and ask questions about them. Uses **FAISS** for vector search, **sentence-transformers** for embeddings, and **Groq** (free, fast LLM API) for answers.

---

## Quick Start

### 1. Get a free Groq API key

Sign up at **https://console.groq.com** → API Keys → Create Key (it's free).

### 2. Set up the backend

```bash
cd backend

# Copy the env file and add your key
cp .env.example .env
# Edit .env and replace "your_groq_api_key_here" with your actual key

# Install dependencies (use a virtualenv if you prefer)
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

You should see:
```
[RAG] Fresh index created.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Open the frontend

Open `frontend/index.html` directly in your browser — **no build step needed**.

Or access it via the backend at: http://localhost:8000/app

---

## Bugs Fixed (from original)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `main.py` | `python-dotenv` imported but `load_dotenv()` never called → `GROQ_API_KEY` always missing | Added `load_dotenv()` at top of file |
| 2 | `rag_engine.py` | `os.environ["GROQ_API_KEY"]` raises cryptic `KeyError` if key not set | Changed to `os.environ.get()` with a clear human-readable error message |
| 3 | `rag_engine.py` | Model `llama3-8b-8192` was deprecated on Groq | Updated to `llama-3.1-8b-instant` (current, fast, free) |
| 4 | `main.py` | CORS `allow_credentials=True` is incompatible with `allow_origins=["*"]` (browser blocks it) | Set `allow_credentials=False` |
| 5 | `main.py` | No `/health` endpoint for connection checking | Added `GET /health` |
| 6 | `frontend/index.html` | `API = 'http://localhost:8000'` hardcoded — no way to change it | Added sidebar input to set backend URL at runtime (persisted in localStorage) |
| 7 | `frontend/index.html` | No connection status feedback — users didn't know if backend was reachable | Added live connection status indicator with `/health` polling |
| 8 | `frontend/index.html` | Error message always said "localhost:8000" even if URL was changed | Error now shows the actual configured API URL |

---

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py           # FastAPI app, routes, CORS
│   ├── rag_engine.py     # FAISS index, embeddings, Groq LLM
│   ├── requirements.txt
│   ├── .env.example      # Copy to .env and add GROQ_API_KEY
│   └── vector_store/     # Created automatically after first ingest
└── frontend/
    └── index.html        # Single-file UI (no build step)
```

---

## Troubleshooting

**"Cannot reach backend"** in the UI
- Make sure `uvicorn main:app --reload --port 8000` is running
- Check the Backend URL in the sidebar matches where your server is running

**"GROQ_API_KEY is not set"** on startup
- Make sure you copied `.env.example` to `.env` and filled in your key
- The `.env` file must be in the `backend/` directory

**"groq.AuthenticationError"**
- Your Groq API key is invalid or expired — get a new one at https://console.groq.com

**"No documents yet" after uploading**
- Refresh stats with the auto-refresh (every 10s), or re-open the page
