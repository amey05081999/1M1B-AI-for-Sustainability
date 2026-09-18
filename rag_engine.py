"""
RAG Engine — document loading, chunking, embedding, vector store, and retrieval.

Supports two embedding backends:
  - sentence-transformers (local, no API key needed)   [default]
  - OpenAI text-embedding-ada-002 (requires OPENAI_API_KEY)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

import config


# ── Optional imports (graceful degradation) ─────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


# ── Data structures ──────────────────────────────────────────────────────────
class Document:
    """A single text chunk with metadata."""

    def __init__(self, content: str, source: str, chunk_id: int = 0):
        self.content = content
        self.source = source
        self.chunk_id = chunk_id

    def __repr__(self) -> str:
        return f"Document(source={self.source!r}, chunk_id={self.chunk_id}, len={len(self.content)})"


# ── Text chunking ─────────────────────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split *text* into overlapping chunks of approximately *chunk_size* words."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ── Knowledge-base loader ────────────────────────────────────────────────────
def load_knowledge_base(kb_dir: str = "data/kb") -> List[Document]:
    """Load all .txt files in *kb_dir* and split into overlapping chunks."""
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    documents: List[Document] = []
    txt_files = sorted(kb_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {kb_dir}")

    for filepath in txt_files:
        text = filepath.read_text(encoding="utf-8")
        chunks = _chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            documents.append(Document(content=chunk, source=filepath.name, chunk_id=i))

    return documents


# ── Embedding backend ─────────────────────────────────────────────────────────
class EmbeddingModel:
    """Wraps sentence-transformers or OpenAI embeddings behind a common interface."""

    def __init__(self):
        self._backend: str = "sentence_transformers"
        self._st_model = None
        self._openai_client = None
        self._dim: int | None = None

        if _HAS_ST:
            print(f"[RAG] Loading embedding model: {config.EMBEDDING_MODEL}")
            self._st_model = _ST(config.EMBEDDING_MODEL)
            self._dim = self._st_model.get_embedding_dimension()
            print(f"[RAG] Embedding dimension: {self._dim}")
        elif config.OPENAI_API_KEY:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self._backend = "openai"
            self._dim = 1536  # text-embedding-ada-002
            print("[RAG] Using OpenAI embeddings (text-embedding-ada-002)")
        else:
            raise RuntimeError(
                "No embedding backend available. Install sentence-transformers or set OPENAI_API_KEY."
            )

    @property
    def dimension(self) -> int:
        return self._dim  # type: ignore[return-value]

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._backend == "sentence_transformers":
            return self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            # OpenAI batched
            response = self._openai_client.embeddings.create(
                input=texts, model="text-embedding-ada-002"
            )
            return np.array([e.embedding for e in response.data], dtype=np.float32)


# ── Vector store ──────────────────────────────────────────────────────────────
class VectorStore:
    """
    FAISS-backed vector store with a numpy fallback when faiss-cpu is not installed.
    """

    def __init__(self, dimension: int):
        self._dim = dimension
        self._documents: List[Document] = []
        self._embeddings: np.ndarray | None = None  # (N, D)

        if _HAS_FAISS:
            self._index = _faiss.IndexFlatL2(dimension)
            self._backend = "faiss"
        else:
            self._index = None
            self._backend = "numpy"

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        store_path = Path(path)
        store_path.mkdir(parents=True, exist_ok=True)

        # Save documents
        with open(store_path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        # Save embeddings / FAISS index
        if self._backend == "faiss":
            _faiss.write_index(self._index, str(store_path / "faiss.index"))
        else:
            np.save(str(store_path / "embeddings.npy"), self._embeddings)

        print(f"[RAG] Vector store saved to {path} ({len(self._documents)} chunks)")

    def load(self, path: str) -> bool:
        store_path = Path(path)
        doc_file = store_path / "documents.pkl"
        if not doc_file.exists():
            return False

        with open(doc_file, "rb") as f:
            self._documents = pickle.load(f)

        if self._backend == "faiss":
            index_file = store_path / "faiss.index"
            if index_file.exists():
                self._index = _faiss.read_index(str(index_file))
        else:
            emb_file = store_path / "embeddings.npy"
            if emb_file.exists():
                self._embeddings = np.load(str(emb_file))

        print(f"[RAG] Vector store loaded from {path} ({len(self._documents)} chunks)")
        return True

    # ── indexing ──────────────────────────────────────────────────────────────
    def add(self, documents: List[Document], embeddings: np.ndarray) -> None:
        self._documents.extend(documents)
        embeddings = embeddings.astype(np.float32)

        if self._backend == "faiss":
            self._index.add(embeddings)
        else:
            if self._embeddings is None:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])

    # ── retrieval ─────────────────────────────────────────────────────────────
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Document]:
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        if self._backend == "faiss":
            _, indices = self._index.search(query_embedding, top_k)
            return [self._documents[i] for i in indices[0] if i < len(self._documents)]
        else:
            # Cosine similarity via numpy
            if self._embeddings is None or len(self._embeddings) == 0:
                return []
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
            normed = self._embeddings / norms
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            scores = normed @ q_norm.T  # (N,1)
            indices = np.argsort(scores[:, 0])[::-1][:top_k]
            return [self._documents[i] for i in indices]


# ── RAG pipeline ──────────────────────────────────────────────────────────────
class RAGPipeline:
    """
    Orchestrates loading, indexing, and retrieval.

    Usage:
        rag = RAGPipeline()
        rag.build()          # index knowledge base
        docs = rag.retrieve("How do I reduce plastic?")
    """

    def __init__(self):
        self._embedding_model = EmbeddingModel()
        self._vector_store = VectorStore(self._embedding_model.dimension)

    # ── build / load ──────────────────────────────────────────────────────────
    def build(self, kb_dir: str = "data/kb", force_rebuild: bool = False) -> None:
        """Load or build the vector store."""
        store_path = config.VECTOR_STORE_PATH
        if not force_rebuild and self._vector_store.load(store_path):
            return  # loaded from disk – done

        print("[RAG] Building vector store from knowledge base…")
        documents = load_knowledge_base(kb_dir)
        print(f"[RAG] Loaded {len(documents)} chunks from {kb_dir}")

        texts = [doc.content for doc in documents]
        print("[RAG] Generating embeddings…")
        embeddings = self._embedding_model.encode(texts)

        self._vector_store.add(documents, embeddings)
        self._vector_store.save(store_path)
        print("[RAG] Build complete.")

    # ── retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """Return the *top_k* most relevant document chunks for *query*."""
        k = top_k if top_k is not None else config.TOP_K_RETRIEVAL
        query_emb = self._embedding_model.encode([query])
        return self._vector_store.search(query_emb, top_k=k)

    def retrieve_context(self, query: str, top_k: int | None = None) -> str:
        """Convenience method — returns retrieved chunks as a single formatted string."""
        docs = self.retrieve(query, top_k=top_k)
        parts = []
        for doc in docs:
            parts.append(f"[Source: {doc.source}]\n{doc.content}")
        return "\n\n---\n\n".join(parts)


# ── Singleton accessor ────────────────────────────────────────────────────────
_pipeline: RAGPipeline | None = None


def get_pipeline(force_rebuild: bool = False) -> RAGPipeline:
    """Return the global RAGPipeline, building it on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        _pipeline.build(force_rebuild=force_rebuild)
    return _pipeline
