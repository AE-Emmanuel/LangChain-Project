"""QAAgent wrapper that delegates to the LangChain RetrievalQA chain."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from langchain_retrieval_chain import LangChainRetrievalQAChain


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QAAgent:
    """Thin wrapper around `LangChainRetrievalQAChain` for backward compat."""

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
        logger.info("QAAgent initialized with LangChain RetrievalQA backend")

    def answer(self, query: str) -> Dict[str, Any]:
        """Return the chain result plus a simplified source summary."""
        logger.info("Running RetrievalQA for query: %s", query)
        chain_response = self.chain.answer(query)
        sources: List[Document] = chain_response.get("source_documents", [])  # type: ignore[assignment]
        return {
            "answer": chain_response.get("result", ""),
            "source_documents": sources,
            "retrieved": self._summarize_sources(sources),
        }

    @staticmethod
    def _summarize_sources(source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Convert LangChain documents into serializable metadata blocks."""
        summaries: List[Dict[str, Any]] = []
        for doc in source_documents:
            meta = dict(doc.metadata)
            meta["text"] = doc.page_content
            summaries.append(meta)
        return summaries


if __name__ == "__main__":
    question = "What are the main goals of software testing?"
    agent = QAAgent(index_path="indexes/faiss_index_all_mini.index", top_k=6)
    response = agent.answer(question)
    print(response["answer"])
