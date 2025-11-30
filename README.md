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

## 💡 What is this?

A **Retrieval-Augmented Generation (RAG)** pipeline that combines semantic search with local LLM generation. Upload your documents, ask questions in natural language, and get accurate answers backed by source citations—all running locally on your machine.

\`\`\`
"What is Agile development?"
    ↓
🔍 Semantic search finds relevant chunks
    ↓
🧠 Llama3 generates contextual answer
    ↓
📄 Returns answer + source references
\`\`\`

---

## ✨ Key Features

| | |
|---|---|
| 🔒 **Privacy First** | Everything runs locally—no data leaves your machine |
| ⚡ **Fast Retrieval** | FAISS vector similarity search in milliseconds |
| 📚 **Source Citations** | Every answer includes document references |
| 🔧 **Extensible** | Add your own documents and rebuild the index |

---

## 🚀 Quick Start

### Prerequisites

\`\`\`bash
# Install Ollama (macOS)
brew install ollama

# Pull the Llama3 model
ollama pull llama3
\`\`\`

### Setup

\`\`\`bash
# Clone & enter project
git clone https://github.com/AE-Emmanuel/LangChain-Project.git
cd LangChain-Project

# Create environment & install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Build the search index
python build_index.py
\`\`\`

### Run

\`\`\`bash
python V2.0/run_research_assistant.py
\`\`\`

---

## 🏗️ How It Works

\`\`\`
┌────────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   📄 Documents  →  ✂️ Chunking  →  🔢 Embeddings  →  💾 FAISS     │
│      (.txt)        (1100 chars)   (MiniLM-L6-v2)      Index       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ❓ Question  →  🔍 Retrieve  →  🧠 Generate  →  ✅ Answer        │
│                    Top-K          (Llama3)        + Sources       │
│                    Chunks                                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
\`\`\`

---

## 📁 Project Structure

\`\`\`
.
├── V2.0/
│   ├── research_assistant.py     # 🎯 Core RAG pipeline
│   └── run_research_assistant.py # Demo script
├── Datasets/
│   ├── data/                     # Source documents
│   └── export_wiki.py            # Wikipedia fetcher
├── indexes/                      # FAISS index files
├── build_index.py                # Index builder
└── pyproject.toml                # Dependencies
\`\`\`

---

## ⚙️ Configuration

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| \`chunk_size\` | \`1100\` | Characters per chunk |
| \`chunk_overlap\` | \`220\` | Overlap between chunks |
| \`embed_model\` | \`all-MiniLM-L6-v2\` | Embedding model |
| \`llm_model\` | \`llama3\` | Generation model |
| \`top_k\` | \`6\` | Retrieved chunks per query |

---

## 📖 Usage Examples

### Basic Usage

\`\`\`python
from V2.0.research_assistant import QAAgent

agent = QAAgent(index_path="indexes/faiss_index_all_mini.index")
response = agent.answer("What is software testing?")

print(response["answer"])
# Software testing is the process of evaluating and verifying 
# that a software product or application does what it is supposed to do...

print(response["top_sources"])
# [{'source_path': 'Datasets/data/Software_testing.txt', 'score': 0.8234}]
\`\`\`

### Add Your Own Documents

\`\`\`bash
# 1. Add .txt files to Datasets/data/
cp my_document.txt Datasets/data/

# 2. Rebuild the index
python build_index.py

# 3. Query your new content
python V2.0/run_research_assistant.py
\`\`\`

---

## 🔧 Tech Stack

| Layer | Technology |
|:------|:-----------|
| **Embeddings** | [Sentence Transformers](https://sbert.net) (MiniLM-L6-v2) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) |
| **LLM** | [Ollama](https://ollama.com) (Llama3) |
| **Orchestration** | [LangChain](https://langchain.com) |

---

## 📄 License

MIT © [AE-Emmanuel](https://github.com/AE-Emmanuel)

---

<div align="center">

*Built for Deep Learning coursework* 🎓

</div>
