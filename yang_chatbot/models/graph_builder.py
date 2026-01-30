"""Code Property Graph construction for YANG model relationships.

Extends basic dependency graphs into Code Property Graphs (CPGs) that
combine AST structure with data flow edges from YANG leafrefs,
when/must constraints, uses/grouping references, and type hierarchies.

References:
- Code Property Graph: Combines AST + CFG + PDG (Yamaguchi et al.)
- FA-AST: Flow-Augmented AST for code clone detection
- GraphCodeBERT: Data flow graph integration with transformers
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import networkx as nx

from yang_chatbot.models.schema_models import YANGChunk, YANGModule

logger = logging.getLogger(__name__)


class YANGGraphBuilder:
    """Build Code Property Graphs from YANG model relationships.

    Constructs a directed graph combining:
    1. AST hierarchy (parent-child containment)
    2. Data flow edges (leafrefs, type references)
    3. Constraint edges (when/must conditions linking nodes)
    4. Uses/grouping reference edges
    5. Import/include module dependencies
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_modules(self, modules: Dict[str, YANGModule]) -> nx.DiGraph:
        """Build a dependency graph from parsed YANG modules."""
        for name, module in modules.items():
            self.graph.add_node(name, **{
                "type": "module",
                "namespace": module.namespace,
                "prefix": module.prefix,
                "revision": module.revision,
                "description": module.description[:200] if module.description else "",
            })

        for name, module in modules.items():
            for imp in module.imports:
                if imp in modules:
                    self.graph.add_edge(name, imp, relationship="imports")
            for inc in module.includes:
                if inc in modules:
                    self.graph.add_edge(name, inc, relationship="includes")

        logger.info(
            f"Built module graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def build_cpg_from_chunks(self, chunks: List[YANGChunk]) -> nx.DiGraph:
        """Build a Code Property Graph from semantic chunks.

        Creates a rich graph with multiple edge types:
        - containment: AST parent-child (hierarchy)
        - leafref: Data flow via leafref paths
        - constraint: when/must condition dependencies
        - uses: Grouping references
        - type: Type hierarchy references
        - import: Module dependencies
        """
        # Add all chunk nodes
        for chunk in chunks:
            node_id = chunk.chunk_id
            self.graph.add_node(node_id, **{
                "type": chunk.chunk_type.value,
                "module": chunk.module,
                "xpath": chunk.xpath,
                "description": chunk.description[:200] if chunk.description else "",
                "depth": chunk.metadata.get("depth", 0),
                "has_error": chunk.metadata.get("has_error", False),
            })

        # Build edges by type
        chunk_by_id = {c.chunk_id: c for c in chunks}
        chunk_by_xpath = {c.xpath: c for c in chunks}
        chunk_by_name: Dict[str, List[YANGChunk]] = {}
        for c in chunks:
            name = c.metadata.get("element_name", c.xpath.split("/")[-1])
            chunk_by_name.setdefault(name, []).append(c)

        for chunk in chunks:
            node_id = chunk.chunk_id
            relationships = chunk.relationships

            # 1. Containment edges (AST hierarchy)
            for child_type, child_names in relationships.items():
                if child_type in ("container", "list", "leaf", "leaf-list",
                                  "grouping", "rpc", "notification", "identity",
                                  "typedef"):
                    for child_name in child_names:
                        child_xpath = f"{chunk.xpath}/{child_name}"
                        child_chunk = chunk_by_xpath.get(child_xpath)
                        if child_chunk:
                            self.graph.add_edge(
                                node_id, child_chunk.chunk_id,
                                relationship="containment",
                                edge_type="ast",
                            )

            # 2. Leafref data flow edges
            for leafref_path in relationships.get("leafrefs", []):
                target = self._resolve_leafref(
                    leafref_path, chunk, chunk_by_xpath, chunk_by_name
                )
                if target:
                    self.graph.add_edge(
                        node_id, target.chunk_id,
                        relationship="leafref",
                        edge_type="data_flow",
                        path=leafref_path,
                    )

            # 3. Uses/grouping reference edges
            for uses_target in relationships.get("uses", []):
                target_chunks = chunk_by_name.get(uses_target, [])
                for target in target_chunks:
                    if target.chunk_type.value == "grouping":
                        self.graph.add_edge(
                            node_id, target.chunk_id,
                            relationship="uses",
                            edge_type="reference",
                        )

            # 4. Type reference edges
            for type_ref in relationships.get("types", []):
                type_name = type_ref.split(":")[-1] if ":" in type_ref else type_ref
                target_chunks = chunk_by_name.get(type_name, [])
                for target in target_chunks:
                    if target.chunk_type.value in ("typedef", "identity"):
                        self.graph.add_edge(
                            node_id, target.chunk_id,
                            relationship="type_ref",
                            edge_type="data_flow",
                        )

            # 5. Base identity edges
            for base_ref in relationships.get("bases", []):
                base_name = base_ref.split(":")[-1] if ":" in base_ref else base_ref
                target_chunks = chunk_by_name.get(base_name, [])
                for target in target_chunks:
                    if target.chunk_type.value == "identity":
                        self.graph.add_edge(
                            node_id, target.chunk_id,
                            relationship="base",
                            edge_type="hierarchy",
                        )

            # 6. Constraint edges (when/must reference other paths)
            for constraint in chunk.constraints:
                referenced = self._extract_paths_from_constraint(constraint)
                for ref_path in referenced:
                    target = self._resolve_constraint_ref(
                        ref_path, chunk, chunk_by_xpath, chunk_by_name
                    )
                    if target:
                        self.graph.add_edge(
                            node_id, target.chunk_id,
                            relationship="constraint",
                            edge_type="constraint",
                            condition=constraint,
                        )

            # 7. Import edges
            for imp in chunk.imports:
                imp_node = f"{imp}:/{imp}"
                if imp_node in self.graph:
                    self.graph.add_edge(
                        node_id, imp_node,
                        relationship="imports",
                        edge_type="dependency",
                    )

        stats = self._compute_edge_stats()
        logger.info(
            f"Built CPG: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges "
            f"(containment={stats.get('containment', 0)}, "
            f"leafref={stats.get('leafref', 0)}, "
            f"uses={stats.get('uses', 0)}, "
            f"constraint={stats.get('constraint', 0)})"
        )
        return self.graph

    def _resolve_leafref(
        self, path: str, source_chunk: YANGChunk,
        by_xpath: Dict[str, YANGChunk],
        by_name: Dict[str, List[YANGChunk]],
    ) -> Optional[YANGChunk]:
        """Resolve a leafref path to a target chunk."""
        if path.startswith("/"):
            for xpath, chunk in by_xpath.items():
                if xpath.endswith(path) or path in xpath:
                    return chunk

        leaf_name = path.rstrip("/").split("/")[-1]
        candidates = by_name.get(leaf_name, [])
        if candidates:
            same_module = [c for c in candidates if c.module == source_chunk.module]
            return same_module[0] if same_module else candidates[0]

        return None

    def _resolve_constraint_ref(
        self, ref_path: str, source_chunk: YANGChunk,
        by_xpath: Dict[str, YANGChunk],
        by_name: Dict[str, List[YANGChunk]],
    ) -> Optional[YANGChunk]:
        """Resolve a path reference from a when/must constraint."""
        parts = ref_path.strip("./").split("/")
        if not parts:
            return None

        target_name = parts[-1]
        candidates = by_name.get(target_name, [])
        if candidates:
            same_module = [c for c in candidates if c.module == source_chunk.module]
            return same_module[0] if same_module else candidates[0]

        return None

    def _extract_paths_from_constraint(self, constraint: str) -> List[str]:
        """Extract XPath-like path references from when/must conditions."""
        paths = re.findall(r"(?:\.\./|/)[a-zA-Z][\w\-/]*", constraint)
        return paths

    def _compute_edge_stats(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        stats: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relationship", "unknown")
            stats[rel] = stats.get(rel, 0) + 1
        return stats

    def get_dependencies(self, module_name: str) -> List[str]:
        """Get all dependencies of a module (transitive)."""
        if module_name not in self.graph:
            return []
        try:
            return list(nx.descendants(self.graph, module_name))
        except nx.NetworkXError:
            return []

    def get_dependents(self, module_name: str) -> List[str]:
        """Get all modules that depend on this module."""
        if module_name not in self.graph:
            return []
        try:
            return list(nx.ancestors(self.graph, module_name))
        except nx.NetworkXError:
            return []

    def get_related_nodes(self, node_id: str, max_depth: int = 2) -> List[str]:
        """Get nodes within max_depth hops of the given node."""
        if node_id not in self.graph:
            return []
        undirected = self.graph.to_undirected()
        related = set()
        for target in undirected.nodes():
            if target == node_id:
                continue
            try:
                path_length = nx.shortest_path_length(undirected, node_id, target)
                if path_length <= max_depth:
                    related.add(target)
            except nx.NetworkXNoPath:
                continue
        return list(related)

    def get_nodes_by_edge_type(
        self, node_id: str, edge_type: str
    ) -> List[str]:
        """Get nodes connected by a specific edge type."""
        if node_id not in self.graph:
            return []
        result = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("edge_type") == edge_type or data.get("relationship") == edge_type:
                result.append(target)
        return result

    def compute_pagerank(self, top_k: int = 20) -> List[str]:
        """Compute PageRank to find most important nodes.

        Following the Aider pattern of using PageRank to identify
        the most relevant code elements for context selection.
        """
        if self.graph.number_of_nodes() == 0:
            return []
        try:
            scores = nx.pagerank(self.graph, alpha=0.85)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in ranked[:top_k]]
        except Exception as e:
            logger.warning(f"PageRank computation failed: {e}")
            return []

    def get_hierarchy(self, module_name: str) -> Dict[str, Any]:
        """Get the hierarchical structure of a module."""
        hierarchy = {"name": module_name, "children": []}
        prefix = f"{module_name}:/{module_name}/"

        for node_id in self.graph.nodes():
            if node_id.startswith(prefix) and node_id != f"{module_name}:/{module_name}":
                node_data = self.graph.nodes[node_id]
                hierarchy["children"].append({
                    "name": node_id.split("/")[-1],
                    "type": node_data.get("type", "unknown"),
                    "description": node_data.get("description", ""),
                })

        return hierarchy

    def build_from_chunks(self, chunks: List[YANGChunk]) -> nx.DiGraph:
        """Build graph from chunks (delegates to CPG builder)."""
        return self.build_cpg_from_chunks(chunks)

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Return statistics about the graph."""
        edge_stats = self._compute_edge_stats()
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "edge_types": edge_stats,
            "connected_components": nx.number_weakly_connected_components(self.graph)
            if self.graph.is_directed()
            else nx.number_connected_components(self.graph),
            "density": nx.density(self.graph),
        }
