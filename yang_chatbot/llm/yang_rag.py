"""RAG system with hybrid search for YANG model queries."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from yang_chatbot.llm.confidence_scorer import ConfidenceScorer
from yang_chatbot.llm.prompt_templates import QUERY_TEMPLATE, SYSTEM_PROMPT
from yang_chatbot.models.schema_models import QueryContext, YANGChunk
from yang_chatbot.storage.keyword_search import KeywordSearch
from yang_chatbot.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class YANGRagSystem:
    """RAG system combining vector and keyword search for YANG model queries.

    Implements hybrid retrieval (vector + BM25) following IETF framework patterns
    and NetLLMBench best practices.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_search: KeywordSearch,
        llm_client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.llm_client = llm_client
        self.config = config or {}
        self.confidence_scorer = ConfidenceScorer(
            confidence_threshold=self.config.get("confidence_threshold", 0.90)
        )
        self.hybrid_alpha = self.config.get("hybrid_alpha", 0.7)
        self.top_k = self.config.get("top_k", 5)

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> Tuple[List[YANGChunk], List[float]]:
        """Retrieve relevant YANG chunks using hybrid search.

        Combines vector similarity (semantic) with BM25 (keyword) search.
        """
        k = top_k or self.top_k

        # Vector search
        vector_results = self.vector_store.similarity_search(query, top_k=k)

        # Keyword search
        keyword_results = self.keyword_search.search(query, top_k=k)

        # Hybrid merge and re-rank
        chunks, scores = self._hybrid_rerank(vector_results, keyword_results, k)

        logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
        return chunks, scores

    def _hybrid_rerank(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int,
    ) -> Tuple[List[YANGChunk], List[float]]:
        """Merge and re-rank results from vector and keyword search."""
        scored_chunks: Dict[str, Tuple[YANGChunk, float]] = {}
        alpha = self.hybrid_alpha

        # Score vector results
        for i, result in enumerate(vector_results):
            chunk_id = result.get("id", "")
            # Convert distance to similarity score (lower distance = better)
            distance = result.get("distance", 1.0)
            vector_score = 1.0 / (1.0 + distance)

            # Reconstruct chunk from metadata if needed
            chunk = self._result_to_chunk(result)
            if chunk:
                scored_chunks[chunk_id or chunk.chunk_id] = (
                    chunk,
                    alpha * vector_score,
                )

        # Score keyword results
        for i, result in enumerate(keyword_results):
            chunk = result.get("chunk")
            if not chunk:
                continue

            keyword_score = result.get("score", 0.0)
            # Normalize keyword score
            max_kw = max((r.get("score", 0) for r in keyword_results), default=1)
            if max_kw > 0:
                keyword_score = keyword_score / max_kw

            cid = chunk.chunk_id
            if cid in scored_chunks:
                existing_chunk, existing_score = scored_chunks[cid]
                scored_chunks[cid] = (existing_chunk, existing_score + (1 - alpha) * keyword_score)
            else:
                scored_chunks[cid] = (chunk, (1 - alpha) * keyword_score)

        # Sort by combined score
        ranked = sorted(scored_chunks.values(), key=lambda x: x[1], reverse=True)[:top_k]

        chunks = [item[0] for item in ranked]
        scores = [item[1] for item in ranked]
        return chunks, scores

    def _result_to_chunk(self, result: Dict[str, Any]) -> Optional[YANGChunk]:
        """Convert a vector search result back to a YANGChunk."""
        metadata = result.get("metadata", {})
        document = result.get("document", "")

        if not metadata:
            return None

        from yang_chatbot.models.schema_models import ChunkType

        chunk_type_str = metadata.get("chunk_type", "module")
        try:
            chunk_type = ChunkType(chunk_type_str)
        except ValueError:
            chunk_type = ChunkType.MODULE

        return YANGChunk(
            content=document,
            xpath=metadata.get("xpath", ""),
            module=metadata.get("module", ""),
            description=metadata.get("description", ""),
            chunk_type=chunk_type,
        )

    def generate_response(
        self,
        query: str,
        context_chunks: List[YANGChunk],
        session_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a response using the LLM with retrieved context."""
        context_text = self._format_context(context_chunks)
        prompt = QUERY_TEMPLATE.format(context=context_text, query=query)

        if self.llm_client:
            return self._call_llm(prompt, session_history)

        # Fallback: construct response from context without LLM
        return self._fallback_response(query, context_chunks)

    def _call_llm(
        self, prompt: str, session_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Call the LLM API to generate a response."""
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            if session_history:
                messages.extend(session_history[-6:])  # Last 3 exchanges

            messages.append({"role": "user", "content": prompt})

            response = self.llm_client.chat.completions.create(
                model=self.config.get("model", "gpt-4-turbo"),
                messages=messages,
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.1),
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error generating response: {e}"

    def _fallback_response(self, query: str, chunks: List[YANGChunk]) -> str:
        """Generate a response without LLM using retrieved context."""
        if not chunks:
            return (
                "I couldn't find relevant information in the loaded YANG models. "
                "Try rephrasing your question or check that the relevant models are loaded."
            )

        response_parts = [
            "Based on the loaded YANG models, here is the relevant information:\n"
        ]

        for i, chunk in enumerate(chunks[:3], 1):
            response_parts.append(f"\n--- Source {i}: {chunk.module} ({chunk.chunk_type.value}) ---")
            response_parts.append(f"Path: {chunk.xpath}")
            if chunk.description:
                response_parts.append(f"Description: {chunk.description}")
            if chunk.constraints:
                response_parts.append(f"Constraints: {', '.join(chunk.constraints)}")

            # Show a truncated version of content
            content_lines = chunk.content.split("\n")
            if len(content_lines) > 15:
                response_parts.append("Definition (truncated):")
                response_parts.extend(content_lines[:15])
                response_parts.append("  ...")
            else:
                response_parts.append("Definition:")
                response_parts.append(chunk.content)

        response_parts.append(
            "\nNote: For more detailed analysis, configure an LLM provider in config.yaml."
        )

        return "\n".join(response_parts)

    def _format_context(self, chunks: List[YANGChunk]) -> str:
        """Format chunks into context text for the LLM prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[{i}] Module: {chunk.module} | Type: {chunk.chunk_type.value}")
            parts.append(f"    XPath: {chunk.xpath}")
            if chunk.description:
                parts.append(f"    Description: {chunk.description}")
            if chunk.constraints:
                parts.append(f"    Constraints: {'; '.join(chunk.constraints)}")
            # Limit content size per chunk
            content = chunk.content
            if len(content) > 1500:
                content = content[:1500] + "\n    ... (truncated)"
            parts.append(f"    Definition:\n{content}")
            parts.append("")
        return "\n".join(parts)

    def answer_query(
        self,
        query: str,
        session_history: Optional[List[Dict[str, str]]] = None,
    ) -> QueryContext:
        """Full RAG pipeline: retrieve, generate, score confidence."""
        # Retrieve
        chunks, retrieval_scores = self.retrieve_context(query)

        # Generate
        response = self.generate_response(query, chunks, session_history)

        # Score confidence
        confidence = self.confidence_scorer.calculate_confidence(
            query, response, chunks, retrieval_scores
        )

        return QueryContext(
            query=query,
            retrieved_chunks=chunks,
            response=response,
            confidence=confidence,
        )
