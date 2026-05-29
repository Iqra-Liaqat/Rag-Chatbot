"""
RAG Engine - Core retrieval and generation logic
"""

import os
import re
from pathlib import Path
from typing import Optional

# ── Vector store & embeddings ──────────────────────────────────────────────
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Document loaders ───────────────────────────────────────────────────────
import pdfplumber
import docx

# ── LLM (Groq – free & fast, can swap to OpenAI) ──────────────────────────
from groq import Groq

# ── Persistence ───────────────────────────────────────────────────────────
import pickle


EMBED_MODEL   = "all-MiniLM-L6-v2"   # 384-dim, runs on CPU
CHUNK_SIZE    = 500                   # characters per chunk
CHUNK_OVERLAP = 100
TOP_K         = 5                     # chunks to retrieve

# FIX 5: Updated to a current, working Groq model (llama3-8b-8192 was deprecated)
GROQ_MODEL    = "llama-3.1-8b-instant"


class RAGEngine:
    """
    Full RAG pipeline:
      1. Ingest documents  →  chunk  →  embed  →  store in FAISS
      2. Query  →  embed query  →  retrieve top-k chunks  →  LLM answer
    """

    def __init__(self, index_path: str = "vector_store"):
        self.index_path  = Path(index_path)
        self.embedder    = SentenceTransformer(EMBED_MODEL)
        self.index       = None          # FAISS index
        self.chunks      = []            # parallel list of text chunks
        self.metadata    = []            # source filename per chunk

        # FIX 6: Graceful error when GROQ_API_KEY is missing
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Copy backend/.env.example to backend/.env and add your key "
                "(free at https://console.groq.com)."
            )
        self.groq_client = Groq(api_key=api_key)
        self._load_or_init()

    # ── Init / persistence ─────────────────────────────────────────────────

    def _load_or_init(self):
        idx_file  = self.index_path / "index.faiss"
        data_file = self.index_path / "chunks.pkl"
        if idx_file.exists() and data_file.exists():
            self.index = faiss.read_index(str(idx_file))
            with open(data_file, "rb") as f:
                saved = pickle.load(f)
            self.chunks   = saved["chunks"]
            self.metadata = saved["metadata"]
            print(f"[RAG] Loaded {len(self.chunks)} chunks from disk.")
        else:
            # 384 = all-MiniLM-L6-v2 dimension
            self.index = faiss.IndexFlatL2(384)
            print("[RAG] Fresh index created.")

    def _save(self):
        self.index_path.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)

    # ── Document ingestion ─────────────────────────────────────────────────

    def ingest_file(self, file_path: str) -> int:
        """Parse a file, chunk it, embed, add to FAISS. Returns chunk count."""
        path = Path(file_path)
        text = self._extract_text(path)
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        embeddings = self.embedder.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.metadata.extend([path.name] * len(chunks))
        self._save()
        print(f"[RAG] Ingested '{path.name}' → {len(chunks)} chunks.")
        return len(chunks)

    def ingest_text(self, text: str, source_name: str = "manual") -> int:
        """Ingest raw text directly."""
        chunks = self._chunk_text(text)
        if not chunks:
            return 0
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.metadata.extend([source_name] * len(chunks))
        self._save()
        return len(chunks)

    # ── Text extraction ────────────────────────────────────────────────────

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        elif suffix == ".docx":
            return self._read_docx(path)
        elif suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _read_pdf(self, path: Path) -> str:
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)

    def _read_docx(self, path: Path) -> str:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    # ── Chunking ───────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """Sliding window character chunker."""
        text = re.sub(r"\s+", " ", text).strip()
        chunks, start = [], 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return [c for c in chunks if len(c.strip()) > 30]

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Embed query, search FAISS, return top-k chunks with metadata."""
        if self.index.ntotal == 0:
            return []
        q_emb = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append({
                    "text":   self.chunks[idx],
                    "source": self.metadata[idx],
                    "score":  float(dist),
                })
        return results

    # ── Generation ─────────────────────────────────────────────────────────

    def answer(self, query: str, chat_history: Optional[list] = None) -> dict:
        """
        Full RAG answer:
          1. Retrieve relevant chunks
          2. Build prompt with context
          3. Call Groq LLM
          4. Return answer + sources
        """
        chunks = self.retrieve(query)
        sources = list({c["source"] for c in chunks})

        if chunks:
            context = "\n\n---\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in chunks
            )
            system_prompt = (
                "You are a helpful assistant. Answer the user's question using ONLY "
                "the context provided below. If the context doesn't contain enough "
                "information to answer, say so honestly. Be concise and accurate.\n\n"
                f"CONTEXT:\n{context}"
            )
        else:
            system_prompt = (
                "You are a helpful assistant. No documents have been uploaded yet. "
                "Let the user know they should upload a document first."
            )

        messages = []
        if chat_history:
            messages.extend(chat_history[-6:])   # keep last 3 turns
        messages.append({"role": "user", "content": query})

        # FIX 7: use current model name; wrap in try/except for clear LLM errors
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}") from e

        answer_text = response.choices[0].message.content
        return {
            "answer":  answer_text,
            "sources": sources,
            "chunks":  len(chunks),
        }

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        unique_sources = list(set(self.metadata))
        return {
            "total_chunks":  len(self.chunks),
            "total_vectors": self.index.ntotal,
            "sources":       unique_sources,
        }

    def clear(self):
        self.index    = faiss.IndexFlatL2(384)
        self.chunks   = []
        self.metadata = []
        import shutil
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        print("[RAG] Index cleared.")
