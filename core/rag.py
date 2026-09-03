"""
Vector store + retrieval. This is what satisfies section 3's requirement to
minimize hallucination: every retrieved chunk carries its source+location,
and the lesson planner / explainer prompts are required to only use
retrieved chunks, citing location, and to say "not covered in the material"
rather than inventing content when nothing relevant is retrieved.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .ingestion import Chunk, ingest_file

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for retrieval, runs fine on CPU


@dataclass
class RetrievedChunk:
    text: str
    source: str
    location: str
    score: float


class VectorStore:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    # ---- building ----
    def build_from_files(self, file_paths: List[str]) -> int:
        all_chunks: List[Chunk] = []
        for path in file_paths:
            all_chunks.extend(ingest_file(path))
        if not all_chunks:
            raise ValueError("No text could be extracted from the uploaded material.")
        self.chunks = all_chunks
        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        index.add(embeddings.astype(np.float32))
        self.index = index
        return len(all_chunks)

    def save(self, name: str):
        path = os.path.join(self.store_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks}, f)
        faiss.write_index(self.index, os.path.join(self.store_dir, f"{name}.faiss"))

    def load(self, name: str):
        with open(os.path.join(self.store_dir, f"{name}.pkl"), "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.index = faiss.read_index(os.path.join(self.store_dir, f"{name}.faiss"))

    # ---- querying ----
    def query(self, text: str, k: int = 5, min_score: float = 0.25) -> List[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError("Vector store is empty. Build or load it first.")
        q_emb = self.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb.astype(np.float32), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or score < min_score:
                continue
            c = self.chunks[idx]
            results.append(RetrievedChunk(c.text, c.source, c.location, float(score)))
        return results


def format_context_for_prompt(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks with explicit source tags so the LLM can cite
    them and so we can tell it to refuse to go beyond them."""
    if not chunks:
        return "(No relevant material found in the uploaded document for this query.)"
    blocks = []
    for c in chunks:
        blocks.append(f"[Source: {c.source}, {c.location}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)
