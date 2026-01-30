"""Confidence scoring for RAG responses."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from yang_chatbot.models.schema_models import YANGChunk

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Calculate confidence scores for RAG responses.

    Uses multiple heuristic signals and optional LLM-based evaluation.
    """

    def __init__(self, confidence_threshold: float = 0.90):
        self.confidence_threshold = confidence_threshold

    def calculate_confidence(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[YANGChunk],
        retrieval_scores: Optional[List[float]] = None,
    ) -> float:
        """Calculate composite confidence score from multiple signals."""
        scores = []

        # Signal 1: Retrieval quality
        retrieval_conf = self._retrieval_confidence(retrieved_chunks, retrieval_scores)
        scores.append(("retrieval", retrieval_conf, 0.35))

        # Signal 2: Context coverage
        coverage_conf = self._context_coverage(query, retrieved_chunks)
        scores.append(("coverage", coverage_conf, 0.25))

        # Signal 3: Response quality heuristics
        response_conf = self._response_quality(query, response, retrieved_chunks)
        scores.append(("response", response_conf, 0.25))

        # Signal 4: Query complexity
        complexity_conf = self._query_complexity_factor(query)
        scores.append(("complexity", complexity_conf, 0.15))

        # Weighted average
        total_weight = sum(w for _, _, w in scores)
        weighted_score = sum(s * w for _, s, w in scores) / total_weight

        logger.debug(
            f"Confidence breakdown: "
            + ", ".join(f"{name}={score:.2f}" for name, score, _ in scores)
            + f" -> {weighted_score:.2f}"
        )

        return round(min(max(weighted_score, 0.0), 1.0), 3)

    def should_escalate(self, confidence: float) -> bool:
        """Determine if the response should be escalated to human review."""
        return confidence < self.confidence_threshold

    def _retrieval_confidence(
        self, chunks: List[YANGChunk], scores: Optional[List[float]]
    ) -> float:
        """Assess retrieval quality."""
        if not chunks:
            return 0.1

        # Number of relevant chunks found
        chunk_score = min(len(chunks) / 3.0, 1.0)

        # Retrieval score quality (if available)
        if scores and len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            # Normalize: lower distance = better match for cosine
            score_quality = max(0, 1.0 - avg_score)
        else:
            score_quality = 0.5

        return 0.6 * chunk_score + 0.4 * score_quality

    def _context_coverage(self, query: str, chunks: List[YANGChunk]) -> float:
        """Assess how well retrieved chunks cover the query terms."""
        query_terms = set(query.lower().split())
        # Remove common words
        stop_words = {"what", "is", "the", "a", "an", "how", "do", "i", "to", "in", "of", "for", "and", "or"}
        query_terms -= stop_words
        query_terms = {t for t in query_terms if len(t) > 2}

        if not query_terms:
            return 0.5

        chunk_text = " ".join(
            f"{c.module} {c.description} {c.content} {c.xpath}".lower()
            for c in chunks
        )

        covered = sum(1 for term in query_terms if term in chunk_text)
        return covered / len(query_terms)

    def _response_quality(
        self, query: str, response: str, chunks: List[YANGChunk]
    ) -> float:
        """Heuristic assessment of response quality."""
        if not response:
            return 0.0

        score = 0.5  # Base score

        # Response length check
        word_count = len(response.split())
        if 20 <= word_count <= 500:
            score += 0.1
        elif word_count < 10:
            score -= 0.2

        # Check if response references YANG elements from chunks
        chunk_terms = set()
        for chunk in chunks:
            chunk_terms.add(chunk.module.lower())
            if chunk.metadata.get("element_name"):
                chunk_terms.add(chunk.metadata["element_name"].lower())

        response_lower = response.lower()
        references = sum(1 for term in chunk_terms if term in response_lower)
        if references > 0:
            score += min(references * 0.05, 0.2)

        # Check for YANG terminology usage
        yang_terms = {"container", "leaf", "list", "grouping", "module", "augment", "typedef", "identity", "xpath"}
        yang_usage = sum(1 for term in yang_terms if term in response_lower)
        if yang_usage > 0:
            score += min(yang_usage * 0.03, 0.15)

        # Penalize hedging language
        hedge_phrases = ["i'm not sure", "i don't know", "unclear", "cannot determine"]
        if any(phrase in response_lower for phrase in hedge_phrases):
            score -= 0.15

        return min(max(score, 0.0), 1.0)

    def _query_complexity_factor(self, query: str) -> float:
        """Adjust confidence based on query complexity.

        More complex queries inherently have lower confidence.
        """
        word_count = len(query.split())

        # Simple queries are easier to answer confidently
        if word_count <= 5:
            return 0.9
        elif word_count <= 10:
            return 0.8
        elif word_count <= 20:
            return 0.7
        else:
            return 0.6
