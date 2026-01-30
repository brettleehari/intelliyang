"""BM25-based keyword search for YANG chunks."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from yang_chatbot.models.schema_models import YANGChunk

logger = logging.getLogger(__name__)


class KeywordSearch:
    """BM25-based keyword search with fallback implementation.

    Attempts to use rank_bm25 library, falls back to a built-in
    BM25 implementation if not available.
    """

    def __init__(self):
        self._documents: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25 = None
        self._use_library = False

    def index_chunks(self, chunks: List[YANGChunk]):
        """Index YANG chunks for keyword search."""
        self._documents = []
        self._tokenized_corpus = []

        for chunk in chunks:
            text = self._chunk_to_searchable_text(chunk)
            tokens = self._tokenize(text)
            self._documents.append({
                "chunk": chunk,
                "text": text,
                "tokens": tokens,
            })
            self._tokenized_corpus.append(tokens)

        # Try to use rank_bm25 library
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            self._use_library = True
            logger.info(f"Indexed {len(chunks)} chunks with rank_bm25")
        except ImportError:
            self._use_library = False
            self._build_bm25_index()
            logger.info(f"Indexed {len(chunks)} chunks with built-in BM25")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks using BM25 keyword matching."""
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)

        if self._use_library and self._bm25:
            scores = self._bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({
                        "chunk": self._documents[idx]["chunk"],
                        "score": float(scores[idx]),
                        "text": self._documents[idx]["text"],
                    })
            return results

        # Fallback BM25
        return self._builtin_bm25_search(query_tokens, top_k)

    def _build_bm25_index(self):
        """Build internal BM25 index data structures."""
        self._doc_count = len(self._tokenized_corpus)
        self._avgdl = (
            sum(len(doc) for doc in self._tokenized_corpus) / self._doc_count
            if self._doc_count > 0
            else 0
        )

        # Document frequency for each term
        self._df: Dict[str, int] = Counter()
        for doc_tokens in self._tokenized_corpus:
            for token in set(doc_tokens):
                self._df[token] += 1

        # Term frequency per document
        self._tf: List[Counter] = [Counter(tokens) for tokens in self._tokenized_corpus]

    def _builtin_bm25_search(self, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
        """BM25 scoring implementation."""
        k1 = 1.5
        b = 0.75
        scores = []

        for doc_idx, doc_tokens in enumerate(self._tokenized_corpus):
            score = 0.0
            doc_len = len(doc_tokens)
            tf_counter = self._tf[doc_idx]

            for token in query_tokens:
                if token not in self._df:
                    continue
                df = self._df[token]
                idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)
                tf = tf_counter.get(token, 0)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self._avgdl))
                score += idf * tf_norm

            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "chunk": self._documents[idx]["chunk"],
                    "score": scores[idx],
                    "text": self._documents[idx]["text"],
                })
        return results

    def _chunk_to_searchable_text(self, chunk: YANGChunk) -> str:
        """Convert chunk to searchable text."""
        parts = [
            chunk.module,
            chunk.chunk_type.value,
            chunk.xpath,
            chunk.description,
            chunk.content,
        ]
        if chunk.constraints:
            parts.extend(chunk.constraints)
        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 processing."""
        text = text.lower()
        # Split on non-alphanumeric characters, keep hyphens in YANG identifiers
        tokens = re.findall(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", text)
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "this", "that", "these", "those", "it", "its",
        }
        return [t for t in tokens if t not in stop_words and len(t) > 1]

    def get_count(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)
