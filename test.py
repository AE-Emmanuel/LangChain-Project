# test_splitter.py
import logging
import json
from collections import defaultdict

from research_assistant import load_raw_documents, split_documents

logging.basicConfig(level=logging.INFO)

# 1) Load docs
docs = load_raw_documents("Datasets/data")
print(f"Loaded {len(docs)} documents.")
for d in docs:
    print(f" - {d['doc_id']}: approx words={d['word_count']} chars={len(d['raw_text'])}")

# For now we use raw_text (you removed cleaning); if you have cleaned_text, assign it:
for d in docs:
    d["cleaned_text"] = d["raw_text"]

# 2) Split
chunks = split_documents(docs, chunk_size=1100, chunk_overlap=220)
print(f"Total chunks produced: {len(chunks)}")

# 3) Per-document counts & sample chunk
counts = defaultdict(int)
for c in chunks:
    counts[c["doc_id"]] += 1
print("Chunks per doc:", dict(counts))

# Show one sample chunk per doc
seen = set()
for c in chunks:
    if c["doc_id"] not in seen:
        seen.add(c["doc_id"])
        print("\nSample:", c["doc_id"], "chunk_id", c["chunk_id"])
        print("start,end:", c["start_char"], c["end_char"], "len:", len(c["text"]))
        print(c["text"][:400].replace("\n", " ") + "...\n")

# 4) Sanity assertions: chunk text matches original slice
mismatches = 0
for c in chunks:
    doc = next(d for d in docs if d["doc_id"] == c["doc_id"])
    src = doc["cleaned_text"]
    start, end = c["start_char"], c["end_char"]
    sliced = src[start:end].strip()
    if sliced != c["text"]:
        mismatches += 1
        # print one example mismatch for debugging
        if mismatches <= 3:
            print("MISMATCH example for", c["doc_id"], c["chunk_id"])
            print("sliced[:120]:", sliced[:120].replace("\n"," "))
            print("chunk_text[:120]:", c["text"][:120].replace("\n"," "))
print("Total provenance mismatches:", mismatches)

# 5) No empty chunks
empty_chunks = [c for c in chunks if not c["text"].strip()]
print("Empty chunks count:", len(empty_chunks))

# 6) Chunk length distribution and max length check
lens = [len(c["text"]) for c in chunks]
print("Chunk len: min, median, max =", min(lens), sorted(lens)[len(lens)//2], max(lens))
over_limit = [l for l in lens if l > 1150]  # chunk_size + 50 buffer
print("Chunks exceeding chunk_size+50:", len(over_limit))

# 7) Overlap sanity: compute overlap between consecutive chunks of same doc
from collections import defaultdict
overlaps = defaultdict(list)
by_doc = defaultdict(list)
for c in sorted(chunks, key=lambda x: (x["doc_id"], x["chunk_id"])):
    by_doc[c["doc_id"]].append(c)
for doc_id, clist in by_doc.items():
    for i in range(1, len(clist)):
        prev, cur = clist[i-1], clist[i]
        overlap = max(0, prev["end_char"] - cur["start_char"])
        overlaps[doc_id].append(overlap)
    if overlaps[doc_id]:
        print(f"{doc_id} overlaps: min/max/avg = {min(overlaps[doc_id])}/{max(overlaps[doc_id])}/{sum(overlaps[doc_id])/len(overlaps[doc_id]):.1f}")

# 8) Coverage ratio: fraction of characters covered by union of chunk spans
import numpy as np
for d in docs:
    src = d["cleaned_text"]
    L = len(src)
    mask = np.zeros(L, dtype=bool)
    for c in [c for c in chunks if c["doc_id"] == d["doc_id"]]:
        s, e = c["start_char"], c["end_char"]
        s = max(0, min(L-1, s))
        e = max(0, min(L, e))
        mask[s:e] = True
    coverage = mask.sum() / max(1, L)
    print(f"{d['doc_id']} coverage ratio: {coverage:.3f}")

