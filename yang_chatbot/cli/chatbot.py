"""Main CLI chatbot interface with session management."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from yang_chatbot.cli.rich_display import RichDisplay
from yang_chatbot.llm.yang_rag import YANGRagSystem
from yang_chatbot.models.graph_builder import YANGGraphBuilder
from yang_chatbot.models.schema_models import AppConfig, ChatSession, QueryContext
from yang_chatbot.models.yang_processor import YANGProcessor
from yang_chatbot.storage.keyword_search import KeywordSearch
from yang_chatbot.storage.vector_store import VectorStore
from yang_chatbot.validation.escalation import EscalationManager
from yang_chatbot.validation.yang_validator import YANGValidator

logger = logging.getLogger(__name__)


class YANGChatbot:
    """Interactive CLI chatbot for OpenROADM YANG models.

    Integrates YANG processing, RAG-based Q&A, validation,
    and rich CLI display.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.display = RichDisplay(enabled=self.config.cli.get("rich_formatting", True))
        self.session = ChatSession(session_id=str(uuid.uuid4())[:8])

        # Core components
        self.processor = YANGProcessor()
        self.graph_builder = YANGGraphBuilder()
        self.vector_store = VectorStore()
        self.keyword_search = KeywordSearch()
        self.validator = YANGValidator(config=self.config.validation)
        self.escalation_manager = EscalationManager()

        # RAG system (initialized after loading models)
        self.rag_system: Optional[YANGRagSystem] = None

        # LLM client
        self.llm_client = None

    def _load_config(self, config_path: str) -> AppConfig:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                return AppConfig(**data)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        return AppConfig()

    def initialize(self, yang_directory: Optional[str] = None):
        """Initialize the chatbot: parse YANG models, build indexes."""
        yang_dir = yang_directory or self.config.yang_models.get("directory", "./openroadm-yang")

        self.display.print_info(f"Loading YANG models from: {yang_dir}")

        # Parse YANG files
        modules = self.processor.parse_yang_files(yang_dir)
        if not modules:
            self.display.print_info(
                f"No YANG models found in {yang_dir}. "
                "The chatbot will operate with limited functionality. "
                "Place .yang files in the directory and restart."
            )

        # Extract semantic chunks
        chunks = self.processor.extract_semantic_chunks()

        # Build graph
        self.graph_builder.build_from_modules(modules)

        # Index chunks for search
        if chunks:
            self.vector_store.add_chunks(chunks)
            self.keyword_search.index_chunks(chunks)
            self.display.print_info(f"Indexed {len(chunks)} semantic chunks from {len(modules)} modules")

        # Initialize LLM client
        self._init_llm_client()

        # Initialize RAG system
        self.rag_system = YANGRagSystem(
            vector_store=self.vector_store,
            keyword_search=self.keyword_search,
            llm_client=self.llm_client,
            config={
                **self.config.rag,
                **self.config.llm,
                "confidence_threshold": self.config.validation.get("confidence_threshold", 0.90),
            },
        )

        self.session.loaded_modules = list(modules.keys())

    def _init_llm_client(self):
        """Initialize the LLM client based on configuration."""
        provider = self.config.llm.get("provider", "openai")
        api_key = os.environ.get("OPENAI_API_KEY", "")

        if provider == "openai" and api_key:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=api_key)
                self.display.print_info("LLM client initialized (OpenAI)")
            except ImportError:
                self.display.print_info(
                    "OpenAI package not installed. Using context-based responses."
                )
        else:
            self.display.print_info(
                "No LLM API key configured. Using context-based responses. "
                "Set OPENAI_API_KEY to enable LLM-powered answers."
            )

    def start_interactive_session(self):
        """Main interactive chat loop."""
        self.display.print_welcome()
        self.display.print_info(f"Session: {self.session.session_id}")

        if self.session.loaded_modules:
            self.display.print_info(
                f"Loaded {len(self.session.loaded_modules)} YANG modules"
            )

        prompt = self.config.cli.get("prompt", "OpenROADM> ")

        while True:
            try:
                user_input = input(f"\n{prompt}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self.display.print_info("Goodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if self._handle_command(user_input):
                continue

            # Process query
            self.process_query(user_input)

    def process_query(self, user_input: str):
        """Handle a user query through the full RAG + validation pipeline."""
        if not self.rag_system:
            self.display.print_error("Chatbot not initialized. Call initialize() first.")
            return

        # Add to session
        self.session.add_message("user", user_input)

        # RAG pipeline
        query_context = self.rag_system.answer_query(
            user_input,
            session_history=self.session.history_for_llm[:-1],
        )

        # Validation
        validation_result = self.validator.validate(
            query_context.response, query_context.confidence
        )
        query_context.validation = validation_result

        # Display response
        self.display.print_response(query_context.response, query_context.confidence)

        # Handle escalation
        if validation_result.escalate:
            escalation_record = self.escalation_manager.escalate(query_context)
            escalation_msg = self.escalation_manager.get_escalation_message(
                query_context.confidence, validation_result
            )
            self.display.print_escalation(escalation_msg)

        # Show validation warnings
        if validation_result.warnings and not validation_result.escalate:
            for warning in validation_result.warnings:
                self.display.print_info(f"Warning: {warning}")

        # Add response to session
        self.session.add_message("assistant", query_context.response, query_context)

    def _handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if input was a command."""
        cmd = user_input.lower().strip()

        if cmd in ("quit", "exit", "q"):
            self.display.print_info("Goodbye!")
            raise SystemExit(0)

        if cmd == "help":
            self.display.print_help()
            return True

        if cmd == "stats":
            stats = self.processor.get_statistics()
            stats["vector_store_count"] = self.vector_store.get_count()
            stats["keyword_index_count"] = self.keyword_search.get_count()
            stats["graph_stats"] = str(self.graph_builder.get_graph_statistics())
            self.display.print_statistics(stats)
            return True

        if cmd == "modules":
            modules_info = []
            for name in self.session.loaded_modules:
                mod = self.processor.get_module(name)
                if mod:
                    modules_info.append({
                        "name": mod.name,
                        "revision": mod.revision,
                        "description": mod.description[:80] if mod.description else "",
                    })
            self.display.print_modules(modules_info)
            return True

        if cmd.startswith("validate "):
            content = user_input[9:].strip()
            result = self.validator.validate(content)
            self.display.print_validation(result)
            return True

        if cmd == "history":
            for msg in self.session.messages[-20:]:
                role_label = "You" if msg.role == "user" else "Bot"
                conf_str = ""
                if msg.context and msg.context.confidence:
                    conf_str = f" [{msg.context.confidence:.0%}]"
                self.display.print_info(f"  {role_label}: {msg.content[:100]}{conf_str}")
            return True

        if cmd == "clear":
            self.session.messages.clear()
            self.display.print_info("Conversation history cleared.")
            return True

        return False
