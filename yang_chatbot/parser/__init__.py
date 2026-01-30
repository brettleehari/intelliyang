"""Tree-sitter based AST parser for YANG models.

Uses the tree-sitter-yang grammar (RFC 7950) for structural parsing,
producing a typed AST that preserves hierarchical relationships,
scope boundaries, and constraint information.

References:
- tree-sitter-yang: github.com/Hubro/tree-sitter-yang (Apache 2.0)
- ASTNN pattern: Split large ASTs into statement subtrees
- cAST pattern: Chunk at syntactic boundaries (container/grouping)
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# AST node types that represent YANG structural elements
YANG_STRUCTURAL_KEYWORDS = frozenset({
    "module", "submodule", "container", "list", "leaf", "leaf-list",
    "grouping", "uses", "augment", "rpc", "notification", "identity",
    "typedef", "choice", "case", "input", "output", "action",
    "extension", "feature", "deviation", "anydata", "anyxml",
})

# Keywords that define constraints
YANG_CONSTRAINT_KEYWORDS = frozenset({
    "when", "must", "if-feature", "mandatory", "min-elements",
    "max-elements", "unique", "ordered-by",
})

# Keywords that define references/relationships
YANG_REFERENCE_KEYWORDS = frozenset({
    "import", "include", "uses", "augment", "base", "type",
    "leafref", "path", "refine", "deviation",
})


@dataclass
class ASTNode:
    """A node in the YANG AST with structural metadata."""

    node_type: str  # tree-sitter node type (e.g., 'statement')
    yang_keyword: str  # YANG keyword (e.g., 'container', 'leaf')
    name: str  # element name/argument
    text: str  # full source text
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    children: List[ASTNode] = field(default_factory=list)
    parent_keyword: str = ""
    parent_name: str = ""
    depth: int = 0
    description: str = ""
    constraints: List[str] = field(default_factory=list)
    references: Dict[str, List[str]] = field(default_factory=dict)
    has_error: bool = False
    error_nodes: List[Tuple[int, int, str]] = field(default_factory=list)

    @property
    def xpath(self) -> str:
        """Compute XPath-like path for this node."""
        parts = []
        if self.parent_name:
            parts.append(self.parent_name)
        parts.append(self.name)
        return "/" + "/".join(parts)

    @property
    def is_structural(self) -> bool:
        return self.yang_keyword in YANG_STRUCTURAL_KEYWORDS

    @property
    def is_constraint(self) -> bool:
        return self.yang_keyword in YANG_CONSTRAINT_KEYWORDS

    @property
    def is_reference(self) -> bool:
        return self.yang_keyword in YANG_REFERENCE_KEYWORDS


class TreeSitterYANGParser:
    """Parse YANG files using tree-sitter for AST extraction.

    Falls back to regex-based parsing if tree-sitter is not available.
    This follows the Aider pattern: use tree-sitter at the edges for
    structural extraction, with the main processing in Python.
    """

    def __init__(self, parser_lib_path: Optional[str] = None):
        self._parser = None
        self._language = None
        self._available = False
        self._init_parser(parser_lib_path)

    def _init_parser(self, lib_path: Optional[str] = None):
        """Initialize tree-sitter parser with YANG grammar."""
        if lib_path is None:
            # Look for the compiled grammar in standard locations
            candidates = [
                Path(__file__).parent / "yang-parser.so",
                Path(__file__).parent.parent / "parser" / "yang-parser.so",
                Path("yang_chatbot/parser/yang-parser.so"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    lib_path = str(candidate)
                    break

        if lib_path is None or not Path(lib_path).exists():
            logger.info("tree-sitter-yang grammar not found. Using fallback parser.")
            return

        try:
            from tree_sitter import Language, Parser

            lib = ctypes.CDLL(lib_path)
            lang_func = lib.tree_sitter_yang
            lang_func.restype = ctypes.c_void_p
            ptr = lang_func()

            self._language = Language(ptr)
            self._parser = Parser(self._language)
            self._available = True
            logger.info("tree-sitter-yang parser initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize tree-sitter-yang: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def parse(self, source: str) -> List[ASTNode]:
        """Parse YANG source into a list of top-level AST nodes.

        Returns structured AST nodes with full hierarchy, constraints,
        and cross-reference information extracted.
        """
        if not self._available:
            return self._fallback_parse(source)

        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        # Check for parse errors (tree-sitter ERROR nodes)
        errors = self._collect_errors(root, source_bytes)
        if errors:
            logger.warning(f"YANG parse errors found: {len(errors)} error nodes")

        return self._extract_nodes(root, source_bytes, errors=errors)

    def parse_file(self, file_path: str) -> List[ASTNode]:
        """Parse a YANG file and return AST nodes."""
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return self.parse(content)

    def get_errors(self, source: str) -> List[Tuple[int, int, str]]:
        """Return parse errors with (line, column, context) tuples.

        Uses tree-sitter ERROR node detection for syntax validation.
        """
        if not self._available:
            return []

        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        return self._collect_errors(tree.root_node, source_bytes)

    def _collect_errors(
        self, node, source_bytes: bytes
    ) -> List[Tuple[int, int, str]]:
        """Recursively collect ERROR nodes from the parse tree."""
        errors = []
        if node.type == "ERROR" or node.is_missing:
            context = source_bytes[
                max(0, node.start_byte - 20) : node.end_byte + 20
            ].decode("utf-8", errors="replace")
            errors.append((
                node.start_point[0] + 1,  # 1-indexed line
                node.start_point[1],
                context.strip(),
            ))
        for child in node.children:
            errors.extend(self._collect_errors(child, source_bytes))
        return errors

    def _extract_nodes(
        self,
        ts_node,
        source_bytes: bytes,
        parent_keyword: str = "",
        parent_name: str = "",
        depth: int = 0,
        errors: Optional[List[Tuple[int, int, str]]] = None,
    ) -> List[ASTNode]:
        """Recursively extract ASTNodes from tree-sitter parse tree."""
        nodes = []
        errors = errors or []

        for child in ts_node.named_children:
            if child.type in ("module", "submodule"):
                node = self._process_module_node(
                    child, source_bytes, depth, errors
                )
                if node:
                    nodes.append(node)
            elif child.type == "statement":
                keyword, name = self._extract_keyword_and_name(
                    child, source_bytes
                )
                if keyword:
                    node = self._process_statement_node(
                        child, source_bytes, keyword, name,
                        parent_keyword, parent_name, depth, errors,
                    )
                    if node:
                        nodes.append(node)

        return nodes

    def _process_module_node(
        self, ts_node, source_bytes: bytes, depth: int,
        errors: List[Tuple[int, int, str]],
    ) -> Optional[ASTNode]:
        """Process a module/submodule node."""
        # Find identifier child
        name = ""
        for child in ts_node.named_children:
            if child.type == "identifier":
                name = source_bytes[child.start_byte:child.end_byte].decode()
                break

        text = source_bytes[ts_node.start_byte:ts_node.end_byte].decode(
            "utf-8", errors="replace"
        )
        keyword = ts_node.type  # 'module' or 'submodule'

        node = ASTNode(
            node_type=ts_node.type,
            yang_keyword=keyword,
            name=name,
            text=text,
            start_line=ts_node.start_point[0] + 1,
            end_line=ts_node.end_point[0] + 1,
            start_byte=ts_node.start_byte,
            end_byte=ts_node.end_byte,
            depth=depth,
        )

        # Extract errors within this node's range
        node.has_error = any(
            ts_node.start_point[0] + 1 <= e[0] <= ts_node.end_point[0] + 1
            for e in errors
        )
        node.error_nodes = [
            e for e in errors
            if ts_node.start_point[0] + 1 <= e[0] <= ts_node.end_point[0] + 1
        ]

        # Process block children
        for child in ts_node.named_children:
            if child.type == "block":
                sub_nodes = self._extract_block_children(
                    child, source_bytes, keyword, name, depth + 1, errors
                )
                node.children.extend(sub_nodes)

                # Extract module-level metadata
                self._extract_metadata(child, source_bytes, node)

        return node

    def _process_statement_node(
        self, ts_node, source_bytes: bytes,
        keyword: str, name: str,
        parent_keyword: str, parent_name: str,
        depth: int, errors: List[Tuple[int, int, str]],
    ) -> Optional[ASTNode]:
        """Process a statement node (container, leaf, list, etc.)."""
        text = source_bytes[ts_node.start_byte:ts_node.end_byte].decode(
            "utf-8", errors="replace"
        )

        node = ASTNode(
            node_type=ts_node.type,
            yang_keyword=keyword,
            name=name,
            text=text,
            start_line=ts_node.start_point[0] + 1,
            end_line=ts_node.end_point[0] + 1,
            start_byte=ts_node.start_byte,
            end_byte=ts_node.end_byte,
            parent_keyword=parent_keyword,
            parent_name=parent_name,
            depth=depth,
        )

        # Extract errors within this node
        node.has_error = any(
            ts_node.start_point[0] + 1 <= e[0] <= ts_node.end_point[0] + 1
            for e in errors
        )
        node.error_nodes = [
            e for e in errors
            if ts_node.start_point[0] + 1 <= e[0] <= ts_node.end_point[0] + 1
        ]

        # Process block (nested statements)
        for child in ts_node.named_children:
            if child.type == "block":
                sub_nodes = self._extract_block_children(
                    child, source_bytes, keyword, name, depth + 1, errors
                )
                node.children.extend(sub_nodes)
                self._extract_metadata(child, source_bytes, node)

        return node

    def _extract_block_children(
        self, block_node, source_bytes: bytes,
        parent_keyword: str, parent_name: str,
        depth: int, errors: List[Tuple[int, int, str]],
    ) -> List[ASTNode]:
        """Extract child nodes from a block."""
        children = []
        for child in block_node.named_children:
            if child.type == "statement":
                keyword, name = self._extract_keyword_and_name(
                    child, source_bytes
                )
                if keyword:
                    node = self._process_statement_node(
                        child, source_bytes, keyword, name,
                        parent_keyword, parent_name, depth, errors,
                    )
                    if node:
                        children.append(node)
        return children

    def _extract_keyword_and_name(
        self, statement_node, source_bytes: bytes
    ) -> Tuple[str, str]:
        """Extract the keyword and argument/name from a statement node."""
        keyword = ""
        name = ""

        for child in statement_node.named_children:
            if child.type == "statement_keyword":
                keyword = source_bytes[
                    child.start_byte:child.end_byte
                ].decode()
            elif child.type == "argument":
                # The argument may contain a string, node_identifier, etc.
                name = source_bytes[
                    child.start_byte:child.end_byte
                ].decode().strip('"').strip("'")

        return keyword, name

    def _extract_metadata(
        self, block_node, source_bytes: bytes, target: ASTNode
    ):
        """Extract description, constraints, and references from a block."""
        for child in block_node.named_children:
            if child.type != "statement":
                continue

            keyword, value = self._extract_keyword_and_name(
                child, source_bytes
            )

            if keyword == "description":
                target.description = value

            elif keyword in YANG_CONSTRAINT_KEYWORDS:
                target.constraints.append(f"{keyword}: {value}")

            elif keyword == "import":
                target.references.setdefault("imports", []).append(value)

            elif keyword == "include":
                target.references.setdefault("includes", []).append(value)

            elif keyword == "uses":
                target.references.setdefault("uses", []).append(value)

            elif keyword == "base":
                target.references.setdefault("bases", []).append(value)

            elif keyword == "type":
                target.references.setdefault("types", []).append(value)
                # Check for leafref path
                for subchild in child.named_children:
                    if subchild.type == "block":
                        for stmt in subchild.named_children:
                            if stmt.type == "statement":
                                kw, val = self._extract_keyword_and_name(
                                    stmt, source_bytes
                                )
                                if kw == "path":
                                    target.references.setdefault(
                                        "leafrefs", []
                                    ).append(val)

    def _fallback_parse(self, source: str) -> List[ASTNode]:
        """Fallback regex-based parser when tree-sitter is unavailable.

        Provides basic structural extraction without full AST fidelity.
        """
        import re

        nodes = []
        # Match top-level module
        module_match = re.search(
            r"^\s*(module|submodule)\s+([\w-]+)\s*\{", source, re.MULTILINE
        )
        if not module_match:
            return nodes

        keyword = module_match.group(1)
        name = module_match.group(2)
        module_block = self._extract_brace_block(source, module_match.start())

        module_node = ASTNode(
            node_type=keyword,
            yang_keyword=keyword,
            name=name,
            text=module_block or source,
            start_line=1,
            end_line=source.count("\n") + 1,
            start_byte=0,
            end_byte=len(source.encode("utf-8")),
        )

        # Extract description
        desc_match = re.search(
            r'description\s*\n?\s*"((?:[^"\\]|\\.)*)"',
            module_block or source, re.DOTALL,
        )
        if desc_match:
            module_node.description = desc_match.group(1).strip()

        # Extract imports
        for imp_match in re.finditer(
            r"import\s+([\w-]+)\s*\{", module_block or source
        ):
            module_node.references.setdefault("imports", []).append(
                imp_match.group(1)
            )

        # Extract structural children
        structural_pattern = re.compile(
            r"^\s*(container|list|leaf-list|leaf|grouping|typedef|identity|rpc|"
            r"notification|augment|choice|case)\s+([\w-]+|\"[^\"]+\")\s*\{",
            re.MULTILINE,
        )
        content = module_block or source
        for match in structural_pattern.finditer(content):
            child_keyword = match.group(1)
            child_name = match.group(2).strip('"')
            child_block = self._extract_brace_block(content, match.start())

            child_desc = ""
            if child_block:
                child_desc_match = re.search(
                    r'description\s*\n?\s*"((?:[^"\\]|\\.)*)"',
                    child_block, re.DOTALL,
                )
                if child_desc_match:
                    child_desc = child_desc_match.group(1).strip()

            # Extract constraints
            constraints = []
            if child_block:
                for c_match in re.finditer(
                    r'(when|must)\s+"([^"]+)"', child_block
                ):
                    constraints.append(
                        f"{c_match.group(1)}: {c_match.group(2)}"
                    )

            child_node = ASTNode(
                node_type="statement",
                yang_keyword=child_keyword,
                name=child_name,
                text=child_block or "",
                start_line=content[:match.start()].count("\n") + 1,
                end_line=content[:match.end()].count("\n") + 1
                if not child_block
                else content[:match.start() + len(child_block)].count("\n") + 1,
                start_byte=match.start(),
                end_byte=match.start() + len(child_block)
                if child_block
                else match.end(),
                parent_keyword=keyword,
                parent_name=name,
                depth=1,
                description=child_desc,
                constraints=constraints,
            )
            module_node.children.append(child_node)

        nodes.append(module_node)
        return nodes

    def _extract_brace_block(self, content: str, start: int) -> Optional[str]:
        """Extract a brace-delimited block."""
        brace_start = content.find("{", start)
        if brace_start == -1:
            return None
        depth = 0
        in_string = False
        i = brace_start
        while i < len(content):
            c = content[i]
            if c == '"' and (i == 0 or content[i - 1] != "\\"):
                in_string = not in_string
            elif not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return content[start:i + 1]
            i += 1
        return None
