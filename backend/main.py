"""
FastAPI Backend — RAG Chatbot API
Run:  uvicorn main:app --reload --port 8000
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_engine import RAGEngine


# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

rag = RAGEngine(index_path="vector_store")

# ── Pydantic models ────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def get_stats():
    return rag.stats()


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {allowed}")

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
    chunk_count = rag.ingest_text(req.text, req.source_name)
    return {"status": "success", "chunks": chunk_count, "source": req.source_name}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    history = [{"role": m.role, "content": m.content} for m in req.history]
    return rag.answer(req.query, chat_history=history)


@app.post("/clear")
def clear_index():
    rag.clear()
    return {"status": "success", "message": "Knowledge base cleared."}


# ── Serve frontend ─────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    print(f"[INFO] Frontend served at /app from {frontend_dir}")
else:
    print(f"[WARNING] Frontend not found at {frontend_dir} — UI will not be available")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)