"""YANG model processor with AST-based semantic chunking.

Upgraded from regex-based parsing to tree-sitter AST parsing following
research best practices:
- ASTNN: Statement subtree encoding
- cAST: Chunk at syntactic boundaries (container/grouping)
- Aider: Repository mapping via AST extraction + PageRank
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from yang_chatbot.models.schema_models import ChunkType, YANGChunk, YANGModule
from yang_chatbot.parser import ASTNode, TreeSitterYANGParser

logger = logging.getLogger(__name__)

# Map YANG keywords to ChunkType
_KEYWORD_TO_CHUNK_TYPE = {
    "module": ChunkType.MODULE,
    "submodule": ChunkType.SUBMODULE,
    "container": ChunkType.CONTAINER,
    "list": ChunkType.LIST,
    "leaf": ChunkType.LEAF,
    "leaf-list": ChunkType.LEAF_LIST,
    "grouping": ChunkType.GROUPING,
    "typedef": ChunkType.TYPEDEF,
    "identity": ChunkType.IDENTITY,
    "rpc": ChunkType.RPC,
    "notification": ChunkType.NOTIFICATION,
    "augment": ChunkType.AUGMENT,
}


class YANGProcessor:
    """Parse OpenROADM YANG files and extract semantic chunks.

    Uses tree-sitter AST parsing for structural extraction with
    fallback to regex-based parsing. Chunks at YANG syntactic
    boundaries (containers, groupings, lists) preserving hierarchy.
    """

    def __init__(self):
        self.parsed_modules: Dict[str, YANGModule] = {}
        self.all_chunks: List[YANGChunk] = []
        self._ast_cache: Dict[str, List[ASTNode]] = {}
        self._parser = TreeSitterYANGParser()

    @property
    def ast_parser(self) -> TreeSitterYANGParser:
        """Expose the AST parser for external use (validation, etc.)."""
        return self._parser

    def parse_yang_files(self, yang_directory: str) -> Dict[str, YANGModule]:
        """Parse all YANG files in the given directory using AST parser."""
        yang_dir = Path(yang_directory)
        if not yang_dir.exists():
            logger.warning(f"YANG directory not found: {yang_directory}")
            return {}

        yang_files = list(yang_dir.rglob("*.yang"))
        logger.info(f"Found {len(yang_files)} YANG files in {yang_directory}")

        parser_type = "tree-sitter" if self._parser.available else "fallback"
        logger.info(f"Using {parser_type} parser")

        for yang_file in sorted(yang_files):
            try:
                content = yang_file.read_text(encoding="utf-8", errors="replace")
                ast_nodes = self._parser.parse(content)
                self._ast_cache[str(yang_file)] = ast_nodes

                module = self._module_from_ast(ast_nodes, str(yang_file), content)
                if module:
                    self.parsed_modules[module.name] = module
            except Exception as e:
                logger.error(f"Error parsing {yang_file}: {e}")

        logger.info(f"Successfully parsed {len(self.parsed_modules)} YANG modules")
        return self.parsed_modules

    def _module_from_ast(
        self, ast_nodes: List[ASTNode], file_path: str, content: str
    ) -> Optional[YANGModule]:
        """Build a YANGModule from AST nodes."""
        if not ast_nodes:
            return None

        root = ast_nodes[0]
        if root.yang_keyword not in ("module", "submodule"):
            return None

        module = YANGModule(
            name=root.name,
            file_path=file_path,
            description=root.description,
        )

        # Extract metadata from AST references
        module.imports = root.references.get("imports", [])
        module.includes = root.references.get("includes", [])

        # If tree-sitter didn't extract full metadata, supplement with regex
        if not module.imports:
            module.imports = [
                m.group(1)
                for m in re.finditer(r"import\s+([\w-]+)\s*\{", content)
            ]
        if not module.includes:
            module.includes = [
                m.group(1)
                for m in re.finditer(r"include\s+([\w-]+)\s*;", content)
            ]

        # Extract namespace, prefix, revision via regex (reliable for these)
        ns_match = re.search(r'namespace\s+"([^"]+)"', content)
        if ns_match:
            module.namespace = ns_match.group(1)

        prefix_match = re.search(r'prefix\s+"?(\S+?)"?\s*;', content)
        if prefix_match:
            module.prefix = prefix_match.group(1)

        rev_match = re.search(r"revision\s+([\d-]+)\s*\{", content)
        if rev_match:
            module.revision = rev_match.group(1)

        if not module.description:
            desc_match = re.search(
                r'description\s*\n?\s*"((?:[^"\\]|\\.)*)"',
                content, re.DOTALL,
            )
            if desc_match:
                module.description = desc_match.group(1).strip()

        # Categorize children from AST
        for child in root.children:
            kw = child.yang_keyword
            name = child.name
            if kw == "container":
                module.containers.append(name)
            elif kw == "grouping":
                module.groupings.append(name)
            elif kw == "identity":
                module.identities.append(name)
            elif kw == "typedef":
                module.typedefs.append(name)
            elif kw == "rpc":
                module.rpcs.append(name)
            elif kw == "notification":
                module.notifications.append(name)
            elif kw == "augment":
                module.augments.append(name)
            elif kw == "list":
                module.containers.append(name)  # lists shown with containers

        # Supplement from regex if AST missed elements (e.g. nested ones)
        self._supplement_from_regex(module, content)

        return module

    def _supplement_from_regex(self, module: YANGModule, content: str):
        """Fill in any elements the AST parser missed."""
        patterns = {
            "containers": (r"^\s*container\s+([\w-]+)\s*\{", module.containers),
            "groupings": (r"^\s*grouping\s+([\w-]+)\s*\{", module.groupings),
            "identities": (r"^\s*identity\s+([\w-]+)\s*\{", module.identities),
            "typedefs": (r"^\s*typedef\s+([\w-]+)\s*\{", module.typedefs),
            "rpcs": (r"^\s*rpc\s+([\w-]+)\s*\{", module.rpcs),
            "notifications": (r"^\s*notification\s+([\w-]+)\s*\{", module.notifications),
        }
        for attr_name, (pattern, existing) in patterns.items():
            found = [m.group(1) for m in re.finditer(pattern, content, re.MULTILINE)]
            for name in found:
                if name not in existing:
                    existing.append(name)

    def extract_semantic_chunks(
        self, parsed_modules: Optional[Dict[str, YANGModule]] = None
    ) -> List[YANGChunk]:
        """Extract semantic chunks using AST-based boundaries.

        Following cAST pattern: chunk at syntactic boundaries
        (container, grouping, list, rpc, etc.) preserving
        complete YANG constructs within each chunk.
        """
        modules = parsed_modules or self.parsed_modules
        self.all_chunks = []

        for module_name, module in modules.items():
            try:
                content = Path(module.file_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                ast_nodes = self._ast_cache.get(module.file_path)
                if ast_nodes is None:
                    ast_nodes = self._parser.parse(content)

                chunks = self._chunks_from_ast(ast_nodes, module, content)
                self.all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error extracting chunks from {module_name}: {e}")

        logger.info(f"Extracted {len(self.all_chunks)} semantic chunks")
        return self.all_chunks

    def _chunks_from_ast(
        self, ast_nodes: List[ASTNode], module: YANGModule, content: str
    ) -> List[YANGChunk]:
        """Convert AST nodes to semantic chunks."""
        chunks = []

        if not ast_nodes:
            return chunks

        root = ast_nodes[0]

        # Module-level chunk
        chunks.append(YANGChunk(
            content=self._build_module_summary(module),
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
                "ast_parser": "tree-sitter" if self._parser.available else "fallback",
            },
        ))

        # Recurse into children — chunk at structural boundaries
        for child in root.children:
            child_chunks = self._ast_node_to_chunks(child, module, content)
            chunks.extend(child_chunks)

        return chunks

    def _ast_node_to_chunks(
        self, node: ASTNode, module: YANGModule, content: str,
        parent_xpath: str = "",
    ) -> List[YANGChunk]:
        """Convert an AST node and its children into chunks.

        Follows the cAST pattern: structural boundaries become
        chunk boundaries. Each container, grouping, list, rpc, etc.
        becomes its own chunk with full context.
        """
        chunks = []
        chunk_type = _KEYWORD_TO_CHUNK_TYPE.get(node.yang_keyword)

        if chunk_type is None:
            # Not a chunkable element, but recurse into children
            for child in node.children:
                chunks.extend(
                    self._ast_node_to_chunks(child, module, content, parent_xpath)
                )
            return chunks

        # Build xpath
        if node.yang_keyword == "augment":
            xpath = f"/{module.name}/augment[{node.name}]"
        else:
            base = parent_xpath or f"/{module.name}"
            xpath = f"{base}/{node.name}"

        # Build child relationships
        children_by_type: Dict[str, List[str]] = {}
        for child in node.children:
            child_type = _KEYWORD_TO_CHUNK_TYPE.get(child.yang_keyword)
            if child_type:
                children_by_type.setdefault(child_type.value, []).append(
                    child.name
                )

        # Merge AST references
        relationships = {**children_by_type}
        for ref_type, ref_values in node.references.items():
            relationships[ref_type] = ref_values

        # Collect constraints
        constraints = list(node.constraints)

        # Extract leafref paths from nested children
        leafrefs = node.references.get("leafrefs", [])
        if leafrefs:
            relationships["leafrefs"] = leafrefs

        chunk = YANGChunk(
            content=node.text,
            xpath=xpath,
            module=module.name,
            description=node.description,
            constraints=constraints,
            relationships=relationships,
            chunk_type=chunk_type,
            imports=module.imports,
            metadata={
                "element_name": node.name,
                "parent_keyword": node.parent_keyword,
                "parent_name": node.parent_name,
                "depth": node.depth,
                "has_error": node.has_error,
                "start_line": node.start_line,
                "end_line": node.end_line,
            },
        )
        chunks.append(chunk)

        # Recurse for nested structural elements
        for child in node.children:
            child_chunks = self._ast_node_to_chunks(
                child, module, content, parent_xpath=xpath
            )
            chunks.extend(child_chunks)

        return chunks

    def _build_module_summary(self, module: YANGModule) -> str:
        """Generate a module summary for the module-level chunk."""
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

    def get_ast_for_file(self, file_path: str) -> Optional[List[ASTNode]]:
        """Get cached AST nodes for a file."""
        return self._ast_cache.get(file_path)

    def get_module_names(self) -> List[str]:
        """Return list of all parsed module names."""
        return list(self.parsed_modules.keys())

    def get_module(self, name: str) -> Optional[YANGModule]:
        """Get a specific parsed module by name."""
        return self.parsed_modules.get(name)

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about parsed YANG models."""
        total_containers = sum(
            len(m.containers) for m in self.parsed_modules.values()
        )
        total_groupings = sum(
            len(m.groupings) for m in self.parsed_modules.values()
        )
        total_identities = sum(
            len(m.identities) for m in self.parsed_modules.values()
        )
        total_typedefs = sum(
            len(m.typedefs) for m in self.parsed_modules.values()
        )
        total_rpcs = sum(
            len(m.rpcs) for m in self.parsed_modules.values()
        )
        total_augments = sum(
            len(m.augments) for m in self.parsed_modules.values()
        )

        return {
            "total_modules": len(self.parsed_modules),
            "total_chunks": len(self.all_chunks),
            "total_containers": total_containers,
            "total_groupings": total_groupings,
            "total_identities": total_identities,
            "total_typedefs": total_typedefs,
            "total_rpcs": total_rpcs,
            "total_augments": total_augments,
            "parser_type": "tree-sitter" if self._parser.available else "fallback",
        }
