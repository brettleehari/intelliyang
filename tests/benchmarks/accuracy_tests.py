"""Accuracy benchmark tests for the RAG system."""

import pytest

from tests.benchmarks.test_queries import ACCURACY_TARGETS, BENCHMARK_BASELINES, TEST_QUERIES
from yang_chatbot.llm.yang_rag import YANGRagSystem
from yang_chatbot.models.schema_models import ChunkType, YANGChunk
from yang_chatbot.storage.keyword_search import KeywordSearch
from yang_chatbot.storage.vector_store import VectorStore


def _make_openroadm_chunks():
    """Create a set of representative OpenROADM chunks for testing."""
    return [
        YANGChunk(
            content="container shelf { leaf shelf-name { type string; } list slots { key slot-name; leaf slot-name { type string; } } description \"Physical shelf/chassis in the device\"; }",
            xpath="/org-openroadm-device/shelf",
            module="org-openroadm-device",
            description="A shelf represents a physical chassis containing slots for optical cards and equipment.",
            chunk_type=ChunkType.CONTAINER,
        ),
        YANGChunk(
            content="container circuit-pack { leaf circuit-pack-name { type string; } list ports { key port-name; } }",
            xpath="/org-openroadm-device/circuit-pack",
            module="org-openroadm-device",
            description="Circuit pack represents a pluggable or fixed optical card installed in a slot.",
            chunk_type=ChunkType.CONTAINER,
        ),
        YANGChunk(
            content="container degree { leaf degree-number { type uint16; } leaf max-wavelengths { type uint16; } }",
            xpath="/org-openroadm-device/degree",
            module="org-openroadm-device",
            description="A degree represents a direction or fiber pair in a ROADM node for wavelength routing.",
            chunk_type=ChunkType.CONTAINER,
        ),
        YANGChunk(
            content="container srg { leaf srg-number { type uint16; } leaf max-add-drop-ports { type uint16; } }",
            xpath="/org-openroadm-device/srg",
            module="org-openroadm-device",
            description="Shared Risk Group represents a set of add/drop ports sharing common optical risk.",
            chunk_type=ChunkType.CONTAINER,
        ),
        YANGChunk(
            content="container device-info { leaf node-id { type string; mandatory true; } leaf node-type { type org-openroadm-common-types:node-types; mandatory true; } }",
            xpath="/org-openroadm-device/device-info",
            module="org-openroadm-device",
            description="Top-level container for device identification and configuration.",
            chunk_type=ChunkType.CONTAINER,
        ),
    ]


@pytest.fixture
def rag_system():
    chunks = _make_openroadm_chunks()
    vs = VectorStore()
    ks = KeywordSearch()
    vs.add_chunks(chunks)
    ks.index_chunks(chunks)
    return YANGRagSystem(vector_store=vs, keyword_search=ks)


class TestBasicStructureQueries:
    """Test basic structure queries - target accuracy: 95%."""

    def test_shelf_query_returns_results(self, rag_system):
        result = rag_system.answer_query("What is a shelf in OpenROADM?")
        assert result.response
        assert result.confidence > 0
        assert len(result.retrieved_chunks) > 0

    def test_device_info_query(self, rag_system):
        result = rag_system.answer_query("List the mandatory fields for device-info")
        assert result.response
        assert len(result.retrieved_chunks) > 0

    def test_circuit_pack_query(self, rag_system):
        result = rag_system.answer_query("What is a circuit-pack in OpenROADM?")
        assert result.response
        assert len(result.retrieved_chunks) > 0


class TestRelationshipQueries:
    """Test relationship queries - target accuracy: 80%."""

    def test_srg_degree_relationship(self, rag_system):
        result = rag_system.answer_query("How do SRGs relate to degrees in OpenROADM?")
        assert result.response
        assert len(result.retrieved_chunks) > 0

    def test_circuit_pack_port_relationship(self, rag_system):
        result = rag_system.answer_query("Explain the relationship between circuit-pack and port")
        assert result.response


class TestBenchmarkTargets:
    """Verify benchmark targets are defined correctly."""

    def test_targets_defined(self):
        assert "basic_structure" in ACCURACY_TARGETS
        assert "configuration" in ACCURACY_TARGETS
        assert "relationships" in ACCURACY_TARGETS
        assert "constraints" in ACCURACY_TARGETS

    def test_baselines_defined(self):
        assert BENCHMARK_BASELINES["our_target"] > BENCHMARK_BASELINES["netconfeval_baseline"]
        assert BENCHMARK_BASELINES["production_threshold"] == 0.90

    def test_all_query_categories_have_queries(self):
        for category in ACCURACY_TARGETS:
            assert category in TEST_QUERIES
            assert len(TEST_QUERIES[category]) >= 3
