"""Helpers for loading and cleaning the research dataset."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_REFERENCE_HEADINGS = {"references", "bibliography", "external links"}


def load_raw_documents(dataset_dir: str) -> List[Dict]:
    """Load every non-empty .txt document under ``dataset_dir``."""

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset directory not found: %s" % dataset_path)

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
            logger.warning(
                "%s is not UTF-8 encoded; decoding with latin-1.", file_path
            )
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
    """
    Robustly locate the span (start, end) of `chunk` in `text` searching at/after `cursor`.
    Fallback strategies:
      1. exact chunk via text.find(chunk, cursor)
      2. stripped chunk (chunk.strip())
      3. search for the first N characters of chunk in a bounded forward window
      4. search in a backward/forward window around cursor for best match
      5. fallback to (cursor, cursor + len(chunk)) with a warning
    Returns (start, end).
    """
    if not chunk:
        return cursor, cursor

    # 1) try exact match at/after cursor
    start = text.find(chunk, cursor)
    if start != -1:
        return start, start + len(chunk)

    # 2) try trimmed chunk (remove leading/trailing whitespace)
    trimmed = chunk.strip()
    if trimmed:
        start = text.find(trimmed, cursor)
        if start != -1:
            return start, start + len(trimmed)

    # 3) try searching for prefix of chunk (avoid huge prefix — use up to 60 chars)
    prefix = chunk.strip()[:60]
    if prefix:
        # search only a limited forward window for speed
        fw_end = min(len(text), cursor + window)
        pos = text.find(prefix, cursor, fw_end)
        if pos != -1:
            # attempt to expand to approximate end using length of chunk or until next newline
            approx_end = min(len(text), pos + max(len(trimmed), len(prefix)))
            # best-effort: extend approx_end to include up to chunk length if possible
            approx_end = min(len(text), pos + len(chunk))
            return pos, approx_end

    # 4) search in a symmetric window around cursor for the nearest occurrence
    low = max(0, cursor - window)
    high = min(len(text), cursor + window)
    try:
        # find all occurrences of the trimmed chunk in the window and pick the one closest to cursor
        occurrences = [m.start() for m in re.finditer(re.escape(trimmed if trimmed else prefix), text[low:high])]
    except re.error:
        occurrences = []

    if occurrences:
        # convert to absolute positions
        abs_occ = [low + o for o in occurrences]
        # pick occurrence closest to cursor
        abs_occ.sort(key=lambda p: abs(p - cursor))
        pos = abs_occ[0]
        # end: try to preserve original chunk length but clamp to text length
        end = min(len(text), pos + len(chunk))
        return pos, end

    # 5) final fallback
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
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
    """
    Compute embeddings for a list of chunk dicts (each must have 'text').
    Returns (embeddings_array, dim).
    Embeddings are NOT normalized here; we'll normalize before indexing.
    """
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(model_name)
    logger.info("Using embedding model: %s", model_name)

    # compute embeddings in batches
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
    index_factory: str = None,
) -> None:
    """
    Build a FAISS IndexFlatIP on L2-normalized embeddings and save index + metadata.
    embeddings: np.ndarray (N, D)
    index_path: path to save index (.index will be used)
    metadata: list of dicts length N mapping vector id -> metadata
    index_factory: optional index factory string if you want a different index type (None -> IndexFlatIP)
    """
    # normalize embeddings for cosine similarity
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

    # ensure output path exists
    idx_path = Path(index_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(idx_path))
    logger.info("Saved FAISS index to %s", idx_path)

    # save metadata sidecar
    meta_path = idx_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.info("Saved metadata sidecar to %s", meta_path)


def load_faiss_index(index_path: str) -> Tuple[faiss.Index, List[Dict]]:
    """
    Load a FAISS index and its metadata sidecar.
    Returns (index, metadata_list).
    """
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
    """
    query_embedding: 1D numpy array of shape (D,)
    Returns list of results: each is metadata dict augmented with 'score' (similarity).
    """
    # normalize query embedding to match index normalization
    q = query_embedding.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)

    D, I = index.search(q, top_k)  # D: distances (inner product), I: indices
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
