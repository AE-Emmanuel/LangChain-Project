from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.faiss import FAISS as LangChainFAISS
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_REFERENCE_HEADINGS = {"references", "bibliography", "external links"}


# ---------------------------------------------------------------------------
# Dataset helpers 
# ---------------------------------------------------------------------------

def load_raw_documents(dataset_dir: str) -> List[Dict]:
    """Load every non-empty .txt document under ``dataset_dir``."""

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    documents: List[Dict] = []
    for file_path in sorted(dataset_path.rglob("*.txt")):
        rel_parts = file_path.relative_to(dataset_path).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if file_path.is_dir():
            continue

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("%s is not UTF-8 encoded; decoding with latin-1.", file_path)
            raw_text = file_path.read_text(encoding="latin-1")

        if not raw_text.strip():
            logger.warning("Skipping empty file %s", file_path)
            continue

        documents.append(
            {
                "doc_id": file_path.stem,
                "source_path": str(file_path.resolve()),
                "raw_text": raw_text,
                "word_count": len(raw_text.split()),
            }
        )

    return documents


def _locate_chunk_span(text: str, chunk: str, cursor: int, window: int = 1000) -> Tuple[int, int]:
    """Robustly locate the span (start, end) of `chunk` in `text`."""

    if not chunk:
        return cursor, cursor

    start = text.find(chunk, cursor)
    if start != -1:
        return start, start + len(chunk)

    trimmed = chunk.strip()
    if trimmed:
        start = text.find(trimmed, cursor)
        if start != -1:
            return start, start + len(trimmed)

    prefix = chunk.strip()[:60]
    if prefix:
        fw_end = min(len(text), cursor + window)
        pos = text.find(prefix, cursor, fw_end)
        if pos != -1:
            approx_end = min(len(text), pos + len(chunk))
            return pos, approx_end

    low = max(0, cursor - window)
    high = min(len(text), cursor + window)
    try:
        occurrences = [m.start() for m in re.finditer(re.escape(trimmed if trimmed else prefix), text[low:high])]
    except re.error:
        occurrences = []

    if occurrences:
        abs_occ = [low + o for o in occurrences]
        abs_occ.sort(key=lambda p: abs(p - cursor))
        pos = abs_occ[0]
        end = min(len(text), pos + len(chunk))
        return pos, end

    logger.warning(
        "Could not reliably locate chunk near cursor=%d; falling back. Chunk preview: %r",
        cursor,
        (chunk[:80] + "...") if len(chunk) > 80 else chunk,
    )
    start = max(0, min(len(text), cursor))
    end = min(len(text), start + len(chunk))
    return start, end


def split_documents(
    documents: List[Dict],
    chunk_size: int = 1100,
    chunk_overlap: int = 220,
) -> List[Dict]:
    """Chunk cleaned/raw docs using LangChain's RecursiveCharacterTextSplitter."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Dict] = []

    for doc in documents:
        doc_id = doc.get("doc_id") or doc.get("id") or "<unknown>"
        text = doc.get("cleaned_text") or doc.get("raw_text") or ""
        if not text:
            logger.warning("Document %s has empty text; skipping", doc_id)
            continue

        doc_chunks: List[Dict] = []
        cursor = 0
        for chunk_text in splitter.split_text(text):
            chunk_text_stripped = chunk_text.strip()
            if not chunk_text_stripped:
                cursor += len(chunk_text)
                continue

            start, end = _locate_chunk_span(text, chunk_text_stripped, cursor)
            cursor = max(cursor, start)

            doc_chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": len(doc_chunks),
                    "text": chunk_text_stripped,
                    "start_char": start,
                    "end_char": end,
                    "source": doc_id,
                }
            )

        for chunk in doc_chunks:
            chunk["source_path"] = doc.get("source_path")

        all_chunks.extend(doc_chunks)

    return all_chunks


def embed_chunks(
    chunks: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Tuple[np.ndarray, int]:
    """Compute embeddings for chunk dicts and return (embeddings, dim)."""

    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(model_name)
    logger.info("Using embedding model: %s", model_name)

    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeds.append(emb)
    embeddings = np.vstack(all_embeds)
    logger.info("Computed embeddings: shape=%s", embeddings.shape)
    return embeddings, embeddings.shape[1]


def build_faiss_index(
    embeddings: np.ndarray,
    index_path: str,
    metadata: List[Dict],
    index_factory: str | None = None,
) -> None:
    """Build a FAISS index and persist both the index and metadata."""

    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    if index_factory:
        logger.info("Creating FAISS index via factory: %s", index_factory)
        index = faiss.index_factory(d, index_factory, faiss.METRIC_INNER_PRODUCT)
    else:
        logger.info("Creating FAISS IndexFlatIP (inner product) with dim=%d", d)
        index = faiss.IndexFlatIP(d)

    logger.info("Adding %d vectors to FAISS index...", embeddings.shape[0])
    index.add(embeddings)

    idx_path = Path(index_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(idx_path))
    logger.info("Saved FAISS index to %s", idx_path)

    meta_path = idx_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.info("Saved metadata sidecar to %s", meta_path)


def load_faiss_index(index_path: str) -> Tuple[faiss.Index, List[Dict]]:
    """Load a FAISS index and its metadata sidecar."""

    idx_path = Path(index_path)
    if not idx_path.exists():
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    index = faiss.read_index(str(idx_path))
    meta_path = idx_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata sidecar not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    logger.info("Loaded FAISS index (%s) and metadata (%d entries)", idx_path, len(metadata))
    return index, metadata


def query_index(
    index: faiss.Index,
    metadata: List[Dict],
    query_embedding: np.ndarray,
    top_k: int = 6,
) -> List[Dict]:
    """Return metadata rows for the top-k closest vectors."""

    q = query_embedding.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)

    D, I = index.search(q, top_k)
    dists = D[0].tolist()
    idxs = I[0].tolist()

    results = []
    for score, idx in zip(dists, idxs):
        if idx < 0:
            continue
        entry = metadata[idx].copy()
        entry["_faiss_id"] = idx
        entry["_score"] = float(score)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# LangChain RetrievalQA integration 
# ---------------------------------------------------------------------------

class LangChainRetrievalQAChain:
    """LangChain RetrievalQA chain built on the project's FAISS index."""

    def __init__(
        self,
        index_path: str,
        embed_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "llama3",
        llm_temperature: float = 0.0,
        top_k: int = 6,
    ) -> None:
        self.embeddings = SentenceTransformerEmbeddings(model_name=embed_model_name)
        self.vectorstore = self._build_vectorstore(index_path)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.llm = self._create_llm(llm_model_name, llm_temperature)
        self.chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
        )

    def _build_vectorstore(self, index_path: str) -> LangChainFAISS:
        index, metadata = load_faiss_index(index_path)
        documents: List[Document] = []
        for entry in metadata:
            doc_metadata = {k: v for k, v in entry.items() if k != "text"}
            page_content = entry.get("text") or ""
            documents.append(Document(page_content=page_content, metadata=doc_metadata))

        docstore = InMemoryDocstore({str(idx): doc for idx, doc in enumerate(documents)})
        index_to_docstore_id = {idx: str(idx) for idx in range(len(documents))}
        return LangChainFAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

    def _create_llm(self, model_name: str, temperature: float) -> BaseLLM:
        try:
            return ChatOllama(model=model_name, temperature=float(temperature))
        except TypeError:
            logger.debug("ChatOllama does not accept temperature; retrying without it")
            return ChatOllama(model=model_name)

    def answer(self, question: str) -> Dict[str, Any]:
        """Run the RetrievalQA chain and return the answer + sources."""
        return self.chain.invoke({"query": question})

    def similarity_search_with_score(self, query: str, k: int = 3):
        """Expose FAISS similarity search with scores for reporting."""
        return self.vectorstore.similarity_search_with_score(query=query, k=k)


class QAAgent:
    """Thin wrapper around `LangChainRetrievalQAChain` for compatibility."""

    def __init__(
        self,
        index_path: str = "indexes/faiss_index_all_mini.index",
        embed_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "llama3",
        llm_temperature: float = 0.0,
        top_k: int = 6,
        chain: LangChainRetrievalQAChain | None = None,
    ) -> None:
        self.chain = chain or LangChainRetrievalQAChain(
            index_path=index_path,
            embed_model_name=embed_model_name,
            llm_model_name=llm_model_name,
            llm_temperature=llm_temperature,
            top_k=top_k,
        )
        self.top_k = top_k
        logger.info("QAAgent initialized with LangChain RetrievalQA backend")

    def answer(self, query: str) -> Dict[str, Any]:
        """Return the chain result plus a simplified source summary."""
        logger.info("Running RetrievalQA for query: %s", query)
        chain_response = self.chain.answer(query)
        sources: List[Document] = chain_response.get("source_documents", [])  # type: ignore[assignment]
        top_sources = self._top_sources(query)
        return {
            "answer": chain_response.get("result", ""),
            "source_documents": sources,
            "retrieved": self._summarize_sources(sources),
            "top_sources": top_sources,
        }

    @staticmethod
    def _summarize_sources(source_documents: List[Document]) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for doc in source_documents:
            meta = dict(doc.metadata)
            meta["text"] = doc.page_content
            summaries.append(meta)
        return summaries

    def _top_sources(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Return top sources with similarity scores for reporting."""
        try:
            hits = self.chain.similarity_search_with_score(query, k=limit)
        except Exception:
            return []

        results: List[Dict[str, Any]] = []
        for doc, score in hits:
            meta = dict(doc.metadata)
            meta["score"] = float(score)
            results.append(
                {
                    "source_path": meta.get("source_path"),
                    "chunk_id": meta.get("chunk_id"),
                    "score": float(score),
                }
            )
        return results


__all__ = [
    "QAAgent",
    "LangChainRetrievalQAChain",
    "load_raw_documents",
    "split_documents",
    "embed_chunks",
    "build_faiss_index",
    "load_faiss_index",
    "query_index",
]
