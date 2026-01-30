"""YANG model processor with semantic chunking based on IETF best practices."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from yang_chatbot.models.schema_models import ChunkType, YANGChunk, YANGModule

logger = logging.getLogger(__name__)


class YANGProcessor:
    """Parse OpenROADM YANG files and extract semantic chunks.

    Uses regex-based parsing as a lightweight alternative to pyang,
    with semantic chunking by containers/groupings following IETF best practices.
    """

    STATEMENT_PATTERNS = {
        ChunkType.MODULE: re.compile(r"^\s*module\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.SUBMODULE: re.compile(r"^\s*submodule\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.CONTAINER: re.compile(r"^\s*container\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.GROUPING: re.compile(r"^\s*grouping\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.LIST: re.compile(r"^\s*list\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.IDENTITY: re.compile(r"^\s*identity\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.TYPEDEF: re.compile(r"^\s*typedef\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.RPC: re.compile(r"^\s*rpc\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.NOTIFICATION: re.compile(r"^\s*notification\s+([\w-]+)\s*\{", re.MULTILINE),
        ChunkType.AUGMENT: re.compile(r"^\s*augment\s+\"?([^\"{]+)\"?\s*\{", re.MULTILINE),
    }

    IMPORT_PATTERN = re.compile(r"^\s*import\s+([\w-]+)\s*\{", re.MULTILINE)
    INCLUDE_PATTERN = re.compile(r"^\s*include\s+([\w-]+)\s*;", re.MULTILINE)
    NAMESPACE_PATTERN = re.compile(r'^\s*namespace\s+"([^"]+)"', re.MULTILINE)
    PREFIX_PATTERN = re.compile(r'^\s*prefix\s+"?(\S+?)"?\s*;', re.MULTILINE)
    REVISION_PATTERN = re.compile(r"^\s*revision\s+([\d-]+)\s*\{", re.MULTILINE)
    DESCRIPTION_PATTERN = re.compile(r'^\s*description\s*\n?\s*"((?:[^"\\]|\\.)*)"', re.MULTILINE | re.DOTALL)
    WHEN_PATTERN = re.compile(r'^\s*when\s+"([^"]+)"', re.MULTILINE)
    MUST_PATTERN = re.compile(r'^\s*must\s+"([^"]+)"', re.MULTILINE)

    def __init__(self):
        self.parsed_modules: Dict[str, YANGModule] = {}
        self.all_chunks: List[YANGChunk] = []

    def parse_yang_files(self, yang_directory: str) -> Dict[str, YANGModule]:
        """Parse all YANG files in the given directory."""
        yang_dir = Path(yang_directory)
        if not yang_dir.exists():
            logger.warning(f"YANG directory not found: {yang_directory}")
            return {}

        yang_files = list(yang_dir.rglob("*.yang"))
        logger.info(f"Found {len(yang_files)} YANG files in {yang_directory}")

        for yang_file in sorted(yang_files):
            try:
                module = self._parse_single_file(yang_file)
                if module:
                    self.parsed_modules[module.name] = module
            except Exception as e:
                logger.error(f"Error parsing {yang_file}: {e}")

        logger.info(f"Successfully parsed {len(self.parsed_modules)} YANG modules")
        return self.parsed_modules

    def _parse_single_file(self, file_path: Path) -> Optional[YANGModule]:
        """Parse a single YANG file and extract module metadata."""
        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Determine module name
        module_match = self.STATEMENT_PATTERNS[ChunkType.MODULE].search(content)
        submodule_match = self.STATEMENT_PATTERNS[ChunkType.SUBMODULE].search(content)

        if not module_match and not submodule_match:
            return None

        name = (module_match or submodule_match).group(1)

        module = YANGModule(
            name=name,
            file_path=str(file_path),
        )

        # Extract metadata
        ns_match = self.NAMESPACE_PATTERN.search(content)
        if ns_match:
            module.namespace = ns_match.group(1)

        prefix_match = self.PREFIX_PATTERN.search(content)
        if prefix_match:
            module.prefix = prefix_match.group(1)

        rev_match = self.REVISION_PATTERN.search(content)
        if rev_match:
            module.revision = rev_match.group(1)

        desc_match = self.DESCRIPTION_PATTERN.search(content)
        if desc_match:
            module.description = desc_match.group(1).strip()

        module.imports = [m.group(1) for m in self.IMPORT_PATTERN.finditer(content)]
        module.includes = [m.group(1) for m in self.INCLUDE_PATTERN.finditer(content)]

        # Extract named elements
        for chunk_type, pattern in self.STATEMENT_PATTERNS.items():
            if chunk_type in (ChunkType.MODULE, ChunkType.SUBMODULE):
                continue
            matches = [m.group(1) for m in pattern.finditer(content)]
            attr_map = {
                ChunkType.CONTAINER: "containers",
                ChunkType.GROUPING: "groupings",
                ChunkType.IDENTITY: "identities",
                ChunkType.TYPEDEF: "typedefs",
                ChunkType.RPC: "rpcs",
                ChunkType.NOTIFICATION: "notifications",
                ChunkType.AUGMENT: "augments",
            }
            attr = attr_map.get(chunk_type)
            if attr and matches:
                setattr(module, attr, matches)

        return module

    def extract_semantic_chunks(self, parsed_modules: Optional[Dict[str, YANGModule]] = None) -> List[YANGChunk]:
        """Extract semantic chunks from parsed YANG modules.

        Chunks by containers/groupings, preserving relationships and context.
        """
        modules = parsed_modules or self.parsed_modules
        self.all_chunks = []

        for module_name, module in modules.items():
            try:
                content = Path(module.file_path).read_text(encoding="utf-8", errors="replace")
                chunks = self._extract_chunks_from_content(content, module)
                self.all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error extracting chunks from {module_name}: {e}")

        logger.info(f"Extracted {len(self.all_chunks)} semantic chunks")
        return self.all_chunks

    def _extract_chunks_from_content(self, content: str, module: YANGModule) -> List[YANGChunk]:
        """Extract chunks from YANG file content."""
        chunks = []

        # Module-level chunk with overview
        chunks.append(YANGChunk(
            content=self._get_module_summary(content, module),
            xpath=f"/{module.name}",
            module=module.name,
            description=module.description,
            constraints=[],
            relationships={
                "imports": module.imports,
                "includes": module.includes,
            },
            chunk_type=ChunkType.MODULE,
            imports=module.imports,
            metadata={
                "namespace": module.namespace,
                "prefix": module.prefix,
                "revision": module.revision,
            },
        ))

        # Extract statement-level chunks
        for chunk_type, pattern in self.STATEMENT_PATTERNS.items():
            if chunk_type in (ChunkType.MODULE, ChunkType.SUBMODULE):
                continue
            for match in pattern.finditer(content):
                element_name = match.group(1)
                block_content = self._extract_block(content, match.start())
                if not block_content:
                    continue

                description = ""
                desc_match = self.DESCRIPTION_PATTERN.search(block_content)
                if desc_match:
                    description = desc_match.group(1).strip()

                constraints = []
                constraints.extend(m.group(1) for m in self.WHEN_PATTERN.finditer(block_content))
                constraints.extend(m.group(1) for m in self.MUST_PATTERN.finditer(block_content))

                # Find child relationships
                children = {}
                for child_type, child_pattern in self.STATEMENT_PATTERNS.items():
                    if child_type in (ChunkType.MODULE, ChunkType.SUBMODULE):
                        continue
                    child_matches = [m.group(1) for m in child_pattern.finditer(block_content)]
                    # Exclude the element itself
                    if element_name in child_matches:
                        child_matches.remove(element_name)
                    if child_matches:
                        children[child_type.value] = child_matches

                xpath = f"/{module.name}/{element_name}"
                if chunk_type == ChunkType.AUGMENT:
                    xpath = f"/{module.name}/augment[{element_name}]"

                chunks.append(YANGChunk(
                    content=block_content,
                    xpath=xpath,
                    module=module.name,
                    description=description,
                    constraints=constraints,
                    relationships=children,
                    chunk_type=chunk_type,
                    imports=module.imports,
                    metadata={"element_name": element_name},
                ))

        return chunks

    def _extract_block(self, content: str, start_pos: int) -> Optional[str]:
        """Extract a brace-delimited block from content."""
        brace_start = content.find("{", start_pos)
        if brace_start == -1:
            return None

        depth = 0
        i = brace_start
        while i < len(content):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    # Include the statement keyword from start_pos
                    return content[start_pos : i + 1]
            i += 1
        return None

    def _get_module_summary(self, content: str, module: YANGModule) -> str:
        """Generate a summary of the module for the module-level chunk."""
        lines = [
            f"module {module.name} {{",
            f'  namespace "{module.namespace}";' if module.namespace else "",
            f'  prefix "{module.prefix}";' if module.prefix else "",
            f'  revision {module.revision};' if module.revision else "",
        ]

        for imp in module.imports[:10]:
            lines.append(f"  import {imp};")

        summary_elements = {
            "containers": module.containers,
            "groupings": module.groupings,
            "identities": module.identities,
            "typedefs": module.typedefs,
            "rpcs": module.rpcs,
            "notifications": module.notifications,
        }
        for elem_type, elements in summary_elements.items():
            if elements:
                lines.append(f"  // {elem_type}: {', '.join(elements[:20])}")

        lines.append("}")
        return "\n".join(line for line in lines if line)

    def get_module_names(self) -> List[str]:
        """Return list of all parsed module names."""
        return list(self.parsed_modules.keys())

    def get_module(self, name: str) -> Optional[YANGModule]:
        """Get a specific parsed module by name."""
        return self.parsed_modules.get(name)

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about parsed YANG models."""
        total_containers = sum(len(m.containers) for m in self.parsed_modules.values())
        total_groupings = sum(len(m.groupings) for m in self.parsed_modules.values())
        total_identities = sum(len(m.identities) for m in self.parsed_modules.values())
        total_typedefs = sum(len(m.typedefs) for m in self.parsed_modules.values())
        total_rpcs = sum(len(m.rpcs) for m in self.parsed_modules.values())
        total_augments = sum(len(m.augments) for m in self.parsed_modules.values())

        return {
            "total_modules": len(self.parsed_modules),
            "total_chunks": len(self.all_chunks),
            "total_containers": total_containers,
            "total_groupings": total_groupings,
            "total_identities": total_identities,
            "total_typedefs": total_typedefs,
            "total_rpcs": total_rpcs,
            "total_augments": total_augments,
        }
