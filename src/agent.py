from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self.store.search(question, top_k=top_k)
        context_blocks = []
        for index, item in enumerate(retrieved, start=1):
            metadata = item.get("metadata", {})
            source = metadata.get("doc_id") or metadata.get("source")
            heading = f"Chunk {index}"
            if source:
                heading += f" (source={source})"
            context_blocks.append(f"{heading}: {item.get('content', '')}")

        prompt = (
            "You are a helpful assistant. Use the following context to answer the question.\n\n"
            f"{chr(10).join(context_blocks)}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return self.llm_fn(prompt)
