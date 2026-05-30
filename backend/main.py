"""
FastAPI Backend — RAG Chatbot API
Run:  uvicorn main:app --reload --port 8000
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv          # FIX 1: load .env before anything else
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from rag_engine import RAGEngine


# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Looks up one level from 'backend' folder to find index.html in the root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    html_path = os.path.join(root_dir, "index.html")
    
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,      # FIX 2: must be False when allow_origins=["*"]
)

rag = RAGEngine(index_path="vector_store")

# ── Pydantic models ────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str    # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[list[ChatMessage]] = []

class TextIngestRequest(BaseModel):
    text: str
    source_name: str = "manual-input"


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Chatbot API is running"}


@app.get("/health")                     # FIX 3: dedicated health-check endpoint
def health():
    return {"status": "ok"}


@app.get("/stats")
def get_stats():
    """Return current knowledge base statistics."""
    return rag.stats()


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a PDF, DOCX, TXT, or MD file."""
    allowed = {".pdf", ".docx", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    # Save to a temp file so pdfplumber / docx can open by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        chunk_count = rag.ingest_file(tmp_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

    return {
        "status":   "success",
        "filename": file.filename,
        "chunks":   chunk_count,
        "message":  f"Successfully ingested '{file.filename}' into {chunk_count} chunks.",
    }


@app.post("/ingest/text")
def ingest_text(req: TextIngestRequest):
    """Ingest raw text directly (useful for demos)."""
    chunk_count = rag.ingest_text(req.text, req.source_name)
    return {
        "status": "success",
        "chunks": chunk_count,
        "source": req.source_name,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Retrieves relevant chunks and generates an LLM answer.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    history = [{"role": m.role, "content": m.content} for m in req.history]
    result  = rag.answer(req.query, chat_history=history)
    return result


@app.post("/clear")
def clear_index():
    """Wipe the entire vector store. Use with caution."""
    rag.clear()
    return {"status": "success", "message": "Knowledge base cleared."}


# ── Serve frontend (optional) ──────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    print(f"[WARNING] Frontend directory not found at {frontend_dir}")

    @app.get("/ui", include_in_schema=False)
    def serve_ui():
        return FileResponse(str(frontend_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
