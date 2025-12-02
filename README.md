<div align="center">

# 🔍 Research Assistant

**A privacy-first RAG system for intelligent Q&A over your documents**

[![Python](https://img.shields.io/badge/Python-3.13+-3776ab?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.1+-1c3c3c?logo=chainlink&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3-f97316?logo=meta&logoColor=white)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467df)](https://github.com/facebookresearch/faiss)

*Ask questions. Get answers. With sources.*

</div>

---

## 💡 What is this System about in short?

A **Retrieval-Augmented Generation (RAG)** pipeline that combines semantic search with local LLM generation. ask questions in natural language, and get accurate answers backed by source citations—all running locally on your machine.


---

## 🏗️ How It Works

```
┌────────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   📄 Documents  →  ✂️ Chunking  →  🔢 Embeddings  →  💾 FAISS      │ 
│      (.txt)        (1100 chars)   (MiniLM-L6-v2)      Index        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ❓ Question  →  🔍 Retrieve  →  🧠 Generate  →  ✅ Answer        │
│                    Top-K          (Llama3)        + Sources        │
│                    Chunks                                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
.
├── QA_agent.py                   #  QA Agent wrapper class
├── langchain_retrieval_chain.py  #  LangChain RetrievalQA chain
├── research_assistant.py         #  Document loading, chunking & indexing
├── build_index.py                #  Index builder script
├── pyproject.toml                #  Project dependencies
└── indexes/                      #  FAISS index files (generated)
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `chunk_size` | `1100` | Characters per chunk |
| `chunk_overlap` | `220` | Overlap between chunks |
| `embed_model` | `all-MiniLM-L6-v2` | Embedding model |
| `llm_model` | `llama3` | Generation model |
| `top_k` | `6` | Retrieved chunks per query |


---

## 🔧 Tech Stack

| Layer | Technology |
|:------|:-----------|
| **Embeddings** | [Sentence Transformers](https://sbert.net) (MiniLM-L6-v2) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) |
| **LLM** | [Ollama](https://ollama.com) (Llama3) |
| **Orchestration** | [LangChain](https://langchain.com) |

---

<div align="center">

*Built for Deep Learning coursework* 

</div>
