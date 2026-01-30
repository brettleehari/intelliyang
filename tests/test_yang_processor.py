"""Tests for YANG model processor."""

import os
import tempfile
from pathlib import Path

import pytest

from yang_chatbot.models.schema_models import ChunkType, YANGModule
from yang_chatbot.models.yang_processor import YANGProcessor

SAMPLE_YANG = """
module sample-device {
    namespace "http://example.com/sample-device";
    prefix sd;

    import ietf-inet-types {
        prefix inet;
    }

    revision 2024-01-01 {
        description "Initial revision";
    }

    description
        "Sample OpenROADM device model for testing.";

    typedef device-type {
        type enumeration {
            enum rdm;
            enum xpdr;
        }
        description "Type of optical device.";
    }

    identity card-type {
        description "Base identity for card types.";
    }

    identity transponder {
        base card-type;
        description "Transponder card.";
    }

    grouping shelf-attributes {
        description "Common shelf attributes.";
        leaf shelf-name {
            type string;
            description "Name of the shelf.";
        }
        leaf shelf-type {
            type string;
        }
    }

    container device-info {
        description "Top-level device information container.";
        leaf node-id {
            type string;
            description "Unique node identifier.";
        }
        leaf node-type {
            type device-type;
        }
        list shelves {
            key "shelf-name";
            description "List of physical shelves.";
            uses shelf-attributes;
            list slots {
                key "slot-name";
                leaf slot-name {
                    type string;
                }
                container circuit-pack {
                    leaf circuit-pack-name {
                        type string;
                    }
                    list ports {
                        key "port-name";
                        leaf port-name {
                            type string;
                        }
                        leaf port-direction {
                            type enumeration {
                                enum tx;
                                enum rx;
                                enum bidirectional;
                            }
                        }
                    }
                }
            }
        }
    }

    rpc get-device-info {
        description "Retrieve device information.";
        output {
            uses shelf-attributes;
        }
    }
}
"""


@pytest.fixture
def yang_dir():
    """Create a temporary directory with sample YANG files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yang_file = Path(tmpdir) / "sample-device.yang"
        yang_file.write_text(SAMPLE_YANG)
        yield tmpdir


@pytest.fixture
def processor():
    return YANGProcessor()


class TestYANGProcessor:
    def test_parse_yang_files(self, processor, yang_dir):
        modules = processor.parse_yang_files(yang_dir)
        assert len(modules) == 1
        assert "sample-device" in modules

    def test_module_metadata(self, processor, yang_dir):
        modules = processor.parse_yang_files(yang_dir)
        mod = modules["sample-device"]
        assert mod.name == "sample-device"
        assert mod.namespace == "http://example.com/sample-device"
        assert mod.prefix == "sd"
        assert mod.revision == "2024-01-01"
        assert "ietf-inet-types" in mod.imports

    def test_module_elements(self, processor, yang_dir):
        modules = processor.parse_yang_files(yang_dir)
        mod = modules["sample-device"]
        assert "device-info" in mod.containers
        assert "shelf-attributes" in mod.groupings
        assert "card-type" in mod.identities
        assert "device-type" in mod.typedefs
        assert "get-device-info" in mod.rpcs

    def test_extract_semantic_chunks(self, processor, yang_dir):
        processor.parse_yang_files(yang_dir)
        chunks = processor.extract_semantic_chunks()
        assert len(chunks) > 0

        # Should have a module-level chunk
        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.MODULE]
        assert len(module_chunks) == 1

        # Should have container chunks
        container_chunks = [c for c in chunks if c.chunk_type == ChunkType.CONTAINER]
        assert len(container_chunks) > 0

        # Should have grouping chunks
        grouping_chunks = [c for c in chunks if c.chunk_type == ChunkType.GROUPING]
        assert len(grouping_chunks) > 0

    def test_chunk_has_description(self, processor, yang_dir):
        processor.parse_yang_files(yang_dir)
        chunks = processor.extract_semantic_chunks()
        device_info = [c for c in chunks if "device-info" in c.xpath]
        assert len(device_info) > 0
        assert "device information" in device_info[0].description.lower()

    def test_empty_directory(self, processor):
        with tempfile.TemporaryDirectory() as tmpdir:
            modules = processor.parse_yang_files(tmpdir)
            assert len(modules) == 0

    def test_nonexistent_directory(self, processor):
        modules = processor.parse_yang_files("/nonexistent/path")
        assert len(modules) == 0

    def test_statistics(self, processor, yang_dir):
        processor.parse_yang_files(yang_dir)
        processor.extract_semantic_chunks()
        stats = processor.get_statistics()
        assert stats["total_modules"] == 1
        assert stats["total_chunks"] > 0
        assert stats["total_containers"] > 0

    def test_get_module(self, processor, yang_dir):
        processor.parse_yang_files(yang_dir)
        mod = processor.get_module("sample-device")
        assert mod is not None
        assert mod.name == "sample-device"

        assert processor.get_module("nonexistent") is None

    def test_get_module_names(self, processor, yang_dir):
        processor.parse_yang_files(yang_dir)
        names = processor.get_module_names()
        assert "sample-device" in names
