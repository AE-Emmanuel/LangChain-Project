"""Demonstration script for the consolidated research assistant module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict

from research_assistant import (
    QAAgent,
    build_faiss_index,
    embed_chunks,
    load_raw_documents,
    split_documents,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("Datasets")
INDEX_PATH = Path("indexes/faiss_index_all_mini.index")


def ensure_index(dataset_dir: Path, index_path: Path) -> None:
    """Create the FAISS index if it does not already exist."""

    meta_path = index_path.with_suffix(".meta.json")
    if index_path.exists() and meta_path.exists():
        logger.info("Re-using existing FAISS index at %s", index_path)
        return

    logger.info("Index not found. Building a new one from %s", dataset_dir)
    documents = load_raw_documents(str(dataset_dir))
    for doc in documents:
        doc["cleaned_text"] = doc.get("raw_text", "")

    chunks = split_documents(documents, chunk_size=1100, chunk_overlap=220)
    logger.info("Generated %d chunks", len(chunks))

    metadata: List[Dict] = []
    for chunk in chunks:
        metadata.append(
            {
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "source_path": chunk.get("source_path"),
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "text": chunk["text"],
            }
        )

    embeddings, _ = embed_chunks(chunks, model_name="all-MiniLM-L6-v2", batch_size=32)
    build_faiss_index(embeddings, index_path=str(index_path), metadata=metadata)
    logger.info("Finished building FAISS index")


def main() -> None:
    ensure_index(DATASET_DIR, INDEX_PATH)

    agent = QAAgent(index_path=str(INDEX_PATH), top_k=6)
    question = "What are the main goals of software testing?"
    response = agent.answer(question)

    print("Question:\n", question)
    print("\nAnswer:\n", response["answer"])
    print("\nSources:")
    for doc in response.get("source_documents", []):
        print(f"- {doc.metadata.get('source_path')} (chunk {doc.metadata.get('chunk_id')})")


if __name__ == "__main__":
    main()
