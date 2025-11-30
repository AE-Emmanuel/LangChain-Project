# build_index.py
from research_assistant import load_raw_documents, split_documents , embed_chunks, build_faiss_index
import json

# 1) load and split (we assume you've already set cleaned_text earlier; adjust if needed)
docs = load_raw_documents("Datasets")
for d in docs:
    d["cleaned_text"] = d["raw_text"]

chunks = split_documents(docs, chunk_size=1100, chunk_overlap=220)
print("Chunks:", len(chunks))

# 2) prepare metadata
metadata = []
for c in chunks:
    metadata.append({
        "doc_id": c["doc_id"],
        "chunk_id": c["chunk_id"],
        "source_path": c.get("source_path"),
        "start_char": c["start_char"],
        "end_char": c["end_char"],
        "text": c["text"],
    })

# 3) embed
embeddings, dim = embed_chunks(chunks, model_name="all-MiniLM-L6-v2", batch_size=32)

# 4) build and save index
build_faiss_index(embeddings, index_path="indexes/faiss_index_all_mini.index", metadata=metadata)
print("Index built and saved.")
