"""LangChain-style RetrievalQA chain backed by the project FAISS index."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import ChatOllama

from research_assistant import load_faiss_index


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LangChainRetrievalQAChain:
    """Expose a properly-instantiated LangChain RetrievalQA chain."""

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

    def _build_vectorstore(self, index_path: str) -> FAISS:
        index, metadata = load_faiss_index(index_path)
        documents: List[Document] = []
        for entry in metadata:
            doc_metadata = {k: v for k, v in entry.items() if k != "text"}
            page_content = entry.get("text") or ""
            documents.append(Document(page_content=page_content, metadata=doc_metadata))

        docstore = InMemoryDocstore({str(idx): doc for idx, doc in enumerate(documents)})
        index_to_docstore_id = {idx: str(idx) for idx in range(len(documents))}
        return FAISS(
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
        return self.chain({"query": question})


if __name__ == "__main__":
    chain = LangChainRetrievalQAChain(index_path="indexes/faiss_index_all_mini.index", top_k=6)
    response = chain.answer("What are the main activities in software engineering?")
    print("Answer:\n", response["result"])
    print("Sources:")
    for doc in response.get("source_documents", []):
        print(f"- {doc.metadata.get('source_path')} (chunk {doc.metadata.get('chunk_id')})")
