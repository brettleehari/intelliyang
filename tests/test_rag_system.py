"""Tests for the RAG system."""

import pytest

from yang_chatbot.llm.confidence_scorer import ConfidenceScorer
from yang_chatbot.llm.yang_rag import YANGRagSystem
from yang_chatbot.models.schema_models import ChunkType, YANGChunk
from yang_chatbot.storage.keyword_search import KeywordSearch
from yang_chatbot.storage.vector_store import VectorStore


def _make_chunk(module: str, xpath: str, description: str, content: str = "", chunk_type: ChunkType = ChunkType.CONTAINER) -> YANGChunk:
    return YANGChunk(
        content=content or f"container {xpath.split('/')[-1]} {{ description \"{description}\"; }}",
        xpath=xpath,
        module=module,
        description=description,
        chunk_type=chunk_type,
    )


@pytest.fixture
def sample_chunks():
    return [
        _make_chunk("org-openroadm-device", "/org-openroadm-device/shelf",
                     "A shelf represents a physical chassis containing slots for optical cards.",
                     "container shelf { leaf shelf-name { type string; } list slots { key slot-name; } }"),
        _make_chunk("org-openroadm-device", "/org-openroadm-device/circuit-pack",
                     "Circuit pack represents a pluggable or fixed card in a slot.",
                     "container circuit-pack { leaf circuit-pack-name { type string; } list ports { key port-name; } }"),
        _make_chunk("org-openroadm-device", "/org-openroadm-device/degree",
                     "A degree represents a direction or fiber pair in a ROADM node.",
                     "container degree { leaf degree-number { type uint16; } }"),
        _make_chunk("org-openroadm-device", "/org-openroadm-device/srg",
                     "Shared Risk Group represents a set of add/drop ports sharing common risk.",
                     "container srg { leaf srg-number { type uint16; } }"),
        _make_chunk("org-openroadm-network", "/org-openroadm-network/node",
                     "A network node representing an optical device in the topology.",
                     chunk_type=ChunkType.CONTAINER),
    ]


@pytest.fixture
def indexed_stores(sample_chunks):
    vs = VectorStore()
    ks = KeywordSearch()
    vs.add_chunks(sample_chunks)
    ks.index_chunks(sample_chunks)
    return vs, ks


class TestKeywordSearch:
    def test_index_and_search(self, sample_chunks):
        ks = KeywordSearch()
        ks.index_chunks(sample_chunks)
        assert ks.get_count() == 5

        results = ks.search("shelf physical chassis")
        assert len(results) > 0
        assert results[0]["chunk"].module == "org-openroadm-device"

    def test_search_empty_index(self):
        ks = KeywordSearch()
        results = ks.search("anything")
        assert results == []

    def test_search_degree(self, sample_chunks):
        ks = KeywordSearch()
        ks.index_chunks(sample_chunks)
        results = ks.search("degree direction fiber ROADM")
        assert len(results) > 0
        assert any("degree" in r["chunk"].xpath for r in results)


class TestVectorStore:
    def test_add_and_search(self, sample_chunks):
        vs = VectorStore()
        vs.add_chunks(sample_chunks)
        assert vs.get_count() == 5

        results = vs.similarity_search("shelf chassis", top_k=3)
        assert len(results) > 0

    def test_search_empty_store(self):
        vs = VectorStore()
        results = vs.similarity_search("test query")
        assert results == []

    def test_clear(self, sample_chunks):
        vs = VectorStore()
        vs.add_chunks(sample_chunks)
        assert vs.get_count() > 0
        vs.clear()
        assert vs.get_count() == 0


class TestYANGRagSystem:
    def test_retrieve_context(self, indexed_stores):
        vs, ks = indexed_stores
        rag = YANGRagSystem(vector_store=vs, keyword_search=ks)

        chunks, scores = rag.retrieve_context("What is a shelf?")
        assert len(chunks) > 0

    def test_fallback_response(self, indexed_stores):
        vs, ks = indexed_stores
        rag = YANGRagSystem(vector_store=vs, keyword_search=ks)

        result = rag.answer_query("What is a shelf in OpenROADM?")
        assert result.response
        assert result.confidence > 0
        assert len(result.retrieved_chunks) > 0

    def test_no_results_response(self):
        vs = VectorStore()
        ks = KeywordSearch()
        rag = YANGRagSystem(vector_store=vs, keyword_search=ks)

        result = rag.answer_query("Explain quantum computing")
        assert "couldn't find" in result.response.lower() or len(result.response) > 0


class TestConfidenceScorer:
    def test_high_confidence(self, sample_chunks):
        scorer = ConfidenceScorer()
        score = scorer.calculate_confidence(
            query="What is a shelf?",
            response="A shelf in OpenROADM represents a physical chassis containing slots for optical cards.",
            retrieved_chunks=sample_chunks[:3],
            retrieval_scores=[0.1, 0.2, 0.3],
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_low_confidence_no_chunks(self):
        scorer = ConfidenceScorer()
        score = scorer.calculate_confidence(
            query="What is a shelf?",
            response="I'm not sure about this.",
            retrieved_chunks=[],
        )
        assert score < 0.5

    def test_escalation_threshold(self):
        scorer = ConfidenceScorer(confidence_threshold=0.90)
        assert scorer.should_escalate(0.85)
        assert not scorer.should_escalate(0.95)
