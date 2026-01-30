"""Pydantic data models for YANG processing and chatbot operations."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Types of YANG schema elements that can be chunked."""

    CONTAINER = "container"
    GROUPING = "grouping"
    IDENTITY = "identity"
    TYPEDEF = "typedef"
    LIST = "list"
    LEAF = "leaf"
    LEAF_LIST = "leaf-list"
    RPC = "rpc"
    NOTIFICATION = "notification"
    AUGMENT = "augment"
    MODULE = "module"
    SUBMODULE = "submodule"


class YANGChunk(BaseModel):
    """A semantic chunk extracted from a YANG model."""

    content: str = Field(description="Raw YANG definition text")
    xpath: str = Field(description="XPath location in the schema tree")
    module: str = Field(description="Source YANG module name")
    description: str = Field(default="", description="Human-readable description")
    constraints: List[str] = Field(default_factory=list, description="When/must constraints")
    relationships: Dict[str, List[str]] = Field(
        default_factory=dict, description="Related elements by relationship type"
    )
    chunk_type: ChunkType = Field(description="Type of YANG element")
    imports: List[str] = Field(default_factory=list, description="Module dependencies")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def chunk_id(self) -> str:
        return f"{self.module}:{self.xpath}"


class ValidationResult(BaseModel):
    """Result of multi-stage validation pipeline."""

    is_valid: bool = Field(description="Overall validation result")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    escalate: bool = Field(default=False, description="Whether to escalate to human")
    stages_passed: List[str] = Field(default_factory=list, description="Validation stages passed")
    stages_failed: List[str] = Field(default_factory=list, description="Validation stages failed")


class QueryContext(BaseModel):
    """Full context for a processed query."""

    query: str = Field(description="Original user query")
    retrieved_chunks: List[YANGChunk] = Field(default_factory=list)
    response: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    validation: Optional[ValidationResult] = None
    session_id: str = Field(default="")


class YANGModule(BaseModel):
    """Parsed YANG module metadata."""

    name: str
    namespace: str = ""
    prefix: str = ""
    revision: str = ""
    description: str = ""
    imports: List[str] = Field(default_factory=list)
    includes: List[str] = Field(default_factory=list)
    containers: List[str] = Field(default_factory=list)
    groupings: List[str] = Field(default_factory=list)
    identities: List[str] = Field(default_factory=list)
    typedefs: List[str] = Field(default_factory=list)
    rpcs: List[str] = Field(default_factory=list)
    notifications: List[str] = Field(default_factory=list)
    augments: List[str] = Field(default_factory=list)
    file_path: str = ""


class SessionMessage(BaseModel):
    """A single message in the chat session."""

    role: str = Field(description="user or assistant")
    content: str
    context: Optional[QueryContext] = None


class ChatSession(BaseModel):
    """Chat session state."""

    session_id: str
    messages: List[SessionMessage] = Field(default_factory=list)
    loaded_modules: List[str] = Field(default_factory=list)

    def add_message(self, role: str, content: str, context: Optional[QueryContext] = None):
        self.messages.append(SessionMessage(role=role, content=content, context=context))

    @property
    def history_for_llm(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]


class AppConfig(BaseModel):
    """Application configuration."""

    yang_models: Dict[str, Any] = Field(default_factory=lambda: {
        "directory": "./openroadm-yang",
        "versions": ["18.1.0"],
        "modules": ["device", "network", "service", "common"],
    })
    llm: Dict[str, Any] = Field(default_factory=lambda: {
        "provider": "openai",
        "model": "gpt-4-turbo",
        "max_tokens": 2000,
        "temperature": 0.1,
    })
    rag: Dict[str, Any] = Field(default_factory=lambda: {
        "chunk_size": 1000,
        "overlap": 200,
        "top_k": 5,
        "hybrid_alpha": 0.7,
    })
    validation: Dict[str, Any] = Field(default_factory=lambda: {
        "confidence_threshold": 0.90,
        "escalation_enabled": True,
        "syntax_validation": True,
        "semantic_validation": True,
        "optical_constraints": True,
    })
    cli: Dict[str, Any] = Field(default_factory=lambda: {
        "prompt": "OpenROADM> ",
        "max_history": 100,
        "auto_save_session": True,
        "rich_formatting": True,
    })
