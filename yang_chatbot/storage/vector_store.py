"""ChromaDB integration for vector similarity search."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from yang_chatbot.models.schema_models import YANGChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store for YANG chunk embeddings."""

    def __init__(self, collection_name: str = "yang_chunks", persist_directory: str = "./chroma_data"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB client."""
        if self._client is not None:
            return
        try:
            import chromadb

            self._client = chromadb.Client(chromadb.Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            ))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
        except ImportError:
            logger.warning("ChromaDB not installed. Using in-memory fallback.")
            self._init_fallback()
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}. Using in-memory fallback.")
            self._init_fallback()

    def _init_fallback(self):
        """Initialize simple in-memory fallback store."""
        self._client = "fallback"
        self._fallback_docs: List[Dict[str, Any]] = []

    @property
    def is_fallback(self) -> bool:
        return self._client == "fallback"

    def add_chunks(self, chunks: List[YANGChunk]):
        """Add YANG chunks to the vector store."""
        self._ensure_initialized()

        if not chunks:
            return

        if self.is_fallback:
            for chunk in chunks:
                self._fallback_docs.append({
                    "id": chunk.chunk_id,
                    "document": self._chunk_to_text(chunk),
                    "metadata": {
                        "module": chunk.module,
                        "xpath": chunk.xpath,
                        "chunk_type": chunk.chunk_type.value,
                        "description": chunk.description[:500],
                    },
                })
            logger.info(f"Added {len(chunks)} chunks to fallback store")
            return

        # ChromaDB path
        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            doc_text = self._chunk_to_text(chunk)
            documents.append(doc_text)
            metadatas.append({
                "module": chunk.module,
                "xpath": chunk.xpath,
                "chunk_type": chunk.chunk_type.value,
                "description": chunk.description[:500] if chunk.description else "",
            })
            ids.append(chunk.chunk_id)

        # ChromaDB has batch size limits, add in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self._collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )

        logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        self._ensure_initialized()

        if self.is_fallback:
            return self._fallback_search(query, top_k)

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count() or 1),
            )

            search_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    result = {
                        "document": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "id": results["ids"][0][i] if results["ids"] else "",
                        "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    }
                    search_results.append(result)

            return search_results
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return self._fallback_search(query, top_k) if self.is_fallback else []

    def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based fallback search."""
        query_terms = set(query.lower().split())
        scored = []

        for doc in self._fallback_docs:
            doc_text = doc["document"].lower()
            score = sum(1 for term in query_terms if term in doc_text)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "document": item[1]["document"],
                "metadata": item[1]["metadata"],
                "id": item[1]["id"],
                "distance": 1.0 / (1.0 + item[0]),
            }
            for item in scored[:top_k]
        ]

    def _chunk_to_text(self, chunk: YANGChunk) -> str:
        """Convert a YANG chunk to searchable text with path-based embedding.

        Follows Code2Vec-style path encoding: include the XPath hierarchy
        as a structured prefix so that path relationships are captured
        in the embedding space. This helps the vector search understand
        containment and hierarchy without explicit graph traversal.
        """
        # Path-based prefix: encode hierarchy for embedding
        path_parts = chunk.xpath.strip("/").split("/")
        path_context = " > ".join(path_parts)

        parts = [
            f"Path: {path_context}",
            f"Module: {chunk.module}",
            f"Type: {chunk.chunk_type.value}",
            f"XPath: {chunk.xpath}",
        ]
        if chunk.description:
            parts.append(f"Description: {chunk.description}")
        if chunk.constraints:
            parts.append(f"Constraints: {'; '.join(chunk.constraints)}")

        # Include relationship context for richer embeddings
        relationships = chunk.relationships
        if relationships.get("imports"):
            parts.append(f"Imports: {', '.join(relationships['imports'][:5])}")
        if relationships.get("uses"):
            parts.append(f"Uses groupings: {', '.join(relationships['uses'][:5])}")
        if relationships.get("leafrefs"):
            parts.append(f"Leafref paths: {', '.join(relationships['leafrefs'][:3])}")

        # Include parent context for nested elements
        parent_kw = chunk.metadata.get("parent_keyword", "")
        parent_name = chunk.metadata.get("parent_name", "")
        if parent_kw and parent_name:
            parts.append(f"Parent: {parent_kw} {parent_name}")

        parts.append(f"Definition:\n{chunk.content}")
        return "\n".join(parts)

    def get_count(self) -> int:
        """Return number of stored chunks."""
        self._ensure_initialized()
        if self.is_fallback:
            return len(self._fallback_docs)
        return self._collection.count() if self._collection else 0

    def clear(self):
        """Clear all stored chunks."""
        self._ensure_initialized()
        if self.is_fallback:
            self._fallback_docs.clear()
        elif self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
