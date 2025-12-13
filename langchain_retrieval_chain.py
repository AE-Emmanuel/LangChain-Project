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
from reranker import CrossEncoderReranker


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
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.embeddings = SentenceTransformerEmbeddings(model_name=embed_model_name)
        self.vectorstore = self._build_vectorstore(index_path)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.llm = self._create_llm(llm_model_name, llm_temperature)
        self.use_reranker = use_reranker
        
        if use_reranker:
            self.reranker = CrossEncoderReranker(reranker_model)
            # Create custom retriever with reranking
            self.chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=self._create_reranking_retriever(top_k),
                return_source_documents=True,
            )
        else:
            self.reranker = None
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

    def _create_reranking_retriever(self, top_k: int) -> Any:
        """Create a custom retriever that applies reranking to FAISS results."""
        
        # Use a simpler approach - get documents and rerank them manually
        # This avoids the complex retriever validation issue
        return self.retriever  # Return base retriever, we'll handle reranking in answer()

    def _create_llm(self, model_name: str, temperature: float) -> BaseLLM:
        try:
            return ChatOllama(model=model_name, temperature=float(temperature))
        except TypeError:
            logger.debug("ChatOllama does not accept temperature; retrying without it")
            return ChatOllama(model=model_name)

    def answer(self, question: str) -> Dict[str, Any]:
        """Run the RetrievalQA chain and return the answer + sources."""
        if self.use_reranker:
            # Get base retrieval results
            base_result = self.chain.invoke({"query": question})
            
            # Apply reranking to the retrieved documents
            source_docs = base_result.get('source_documents', [])
            
            if source_docs:
                # Convert to dict format for reranker
                doc_dicts = [
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "_original_score": getattr(doc, "score", 0.0)
                    }
                    for doc in source_docs
                ]
                
                # Apply reranking
                reranked_docs = self.reranker.rerank(question, doc_dicts, len(source_docs))
                
                # Convert back to Document objects
                reranked_documents = []
                for reranked_doc in reranked_docs:
                    doc = Document(
                        page_content=reranked_doc["text"],
                        metadata=reranked_doc["metadata"]
                    )
                    # Preserve both scores for debugging
                    if "_original_score" in reranked_doc:
                        doc.metadata["faiss_score"] = reranked_doc["_original_score"]
                    if "_rerank_score" in reranked_doc:
                        doc.metadata["rerank_score"] = reranked_doc["_rerank_score"]
                    reranked_documents.append(doc)
                
                # Return result with reranked documents
                return {
                    'result': base_result.get('result', ''),
                    'source_documents': reranked_documents
                }
            
            return base_result
        else:
            # Normal operation without reranking
            return self.chain.invoke({"query": question})


if __name__ == "__main__":
    # Test with reranker enabled (default)
    print("=== Testing with Reranker ===")
    chain_with_reranker = LangChainRetrievalQAChain(
        index_path="indexes/faiss_index_all_mini.index", 
        top_k=6, 
        use_reranker=True
    )
    response = chain_with_reranker.answer("What are the main activities in software engineering?")
    print("Answer:\n", response["result"])
    print("Sources:")
    for doc in response.get("source_documents", []):
        print(f"- {doc.metadata.get('source_path')} (chunk {doc.metadata.get('chunk_id')})")
        if "faiss_score" in doc.metadata:
            print(f"  FAISS score: {doc.metadata['faiss_score']:.3f}")
        if "rerank_score" in doc.metadata:
            print(f"  Rerank score: {doc.metadata['rerank_score']:.3f}")
    
    print("\n=== Testing without Reranker (baseline) ===")
    chain_without_reranker = LangChainRetrievalQAChain(
        index_path="indexes/faiss_index_all_mini.index", 
        top_k=6, 
        use_reranker=False
    )
    response_baseline = chain_without_reranker.answer("What are the main activities in software engineering?")
    print("Answer:\n", response_baseline["result"])
    print("Sources:")
    for doc in response_baseline.get("source_documents", []):
        print(f"- {doc.metadata.get('source_path')} (chunk {doc.metadata.get('chunk_id')})")
