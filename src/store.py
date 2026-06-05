from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": {**doc.metadata, "doc_id": doc.id},
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored = []
        for record in records:
            score = compute_similarity(query_embedding, record["embedding"])
            scored.append({**record, "score": score})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []

        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)
            ids.append(record["id"])
            documents.append(record["content"])
            metadatas.append(record["metadata"])
            embeddings.append(record["embedding"])

        if self._use_chroma and self._collection is not None:
            try:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            except Exception:
                pass

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma and self._collection is not None:
            try:
                response = self._collection.query(
                    query_embeddings=[self._embedding_fn(query)],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                results: list[dict[str, Any]] = []
                for idx, content in enumerate(response.get("documents", [[]])[0]):
                    metadata = response.get("metadatas", [[]])[0][idx]
                    distance = response.get("distances", [[]])[0][idx]
                    results.append({
                        "content": content,
                        "metadata": metadata,
                        "score": 1.0 - distance,
                    })
                return results
            except Exception:
                pass

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        if self._use_chroma and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:
                return len(self._store)
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if metadata_filter:
            filtered = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]
        else:
            filtered = list(self._store)

        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        original_size = len(self._store)
        self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        deleted = len(self._store) < original_size

        if deleted and self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass

        return deleted
