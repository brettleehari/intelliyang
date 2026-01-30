"""NetworkX graph construction for YANG model relationships."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from yang_chatbot.models.schema_models import YANGChunk, YANGModule

logger = logging.getLogger(__name__)


class YANGGraphBuilder:
    """Build and query a graph representation of YANG model relationships."""

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

        # Add import/include edges
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

    def build_from_chunks(self, chunks: List[YANGChunk]) -> nx.DiGraph:
        """Build a detailed graph from semantic chunks."""
        for chunk in chunks:
            node_id = chunk.chunk_id
            self.graph.add_node(node_id, **{
                "type": chunk.chunk_type.value,
                "module": chunk.module,
                "xpath": chunk.xpath,
                "description": chunk.description[:200] if chunk.description else "",
            })

        # Add relationship edges
        for chunk in chunks:
            node_id = chunk.chunk_id
            for rel_type, related in chunk.relationships.items():
                for target in related:
                    # Try to find the target node
                    target_id = f"{chunk.module}:/{chunk.module}/{target}"
                    if target_id in self.graph:
                        self.graph.add_edge(node_id, target_id, relationship=rel_type)

            # Add import edges
            for imp in chunk.imports:
                imp_node = f"{imp}:/{imp}"
                if imp_node in self.graph:
                    self.graph.add_edge(node_id, imp_node, relationship="imports")

        logger.info(
            f"Built chunk graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

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

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Return statistics about the graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "connected_components": nx.number_weakly_connected_components(self.graph)
            if self.graph.is_directed()
            else nx.number_connected_components(self.graph),
            "density": nx.density(self.graph),
        }
