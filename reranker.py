"""Reranking utilities for the RAG system."""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CrossEncoderReranker:
    """Lightweight cross-encoder reranker using Sentence Transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            logger.info("Loading cross-encoder model: %s", self.model_name)
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error("Failed to load cross-encoder model: %s", e)
            raise

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to the query.
        
        Args:
            query: The search query
            documents: List of document dicts (must have 'text' key)
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with updated scores
        """
        if not self.model:
            raise RuntimeError("Cross-encoder model not loaded")
        
        if not documents:
            return []
        
        # Prepare document texts
        doc_texts = [doc["text"] for doc in documents]
        
        # Create query-document pairs
        pairs = [[query, text] for text in doc_texts]
        
        # Score pairs using cross-encoder
        scores = self.model.predict(pairs)
        
        # Combine documents with their scores
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_doc = doc.copy()
            scored_doc["_rerank_score"] = float(score)
            scored_docs.append(scored_doc)
        
        # Sort by score (descending) and return top_k
        scored_docs.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return scored_docs[:top_k]


def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """Factory function to create a reranker instance."""
    return CrossEncoderReranker(model_name)