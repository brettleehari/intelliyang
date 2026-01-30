"""Human escalation logic for low-confidence responses."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from yang_chatbot.models.schema_models import QueryContext, ValidationResult

logger = logging.getLogger(__name__)


class EscalationManager:
    """Manage escalation of low-confidence responses to human review.

    Follows the Rakuten Symphony production pattern of only allowing
    autonomous AI decisions above 90% confidence.
    """

    def __init__(self, log_directory: str = "./escalation_logs"):
        self.log_directory = Path(log_directory)
        self.escalation_history: List[Dict[str, Any]] = []

    def escalate(self, query_context: QueryContext) -> Dict[str, Any]:
        """Process an escalation event."""
        escalation_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query_context.query,
            "response": query_context.response,
            "confidence": query_context.confidence,
            "validation": query_context.validation.model_dump() if query_context.validation else None,
            "retrieved_chunks_count": len(query_context.retrieved_chunks),
            "modules_referenced": list({
                c.module for c in query_context.retrieved_chunks
            }),
            "status": "pending_review",
        }

        self.escalation_history.append(escalation_record)
        self._persist_escalation(escalation_record)

        logger.warning(
            f"Escalated query (confidence={query_context.confidence:.2f}): "
            f"{query_context.query[:80]}..."
        )

        return escalation_record

    def get_escalation_message(self, confidence: float, validation: Optional[ValidationResult] = None) -> str:
        """Generate a user-facing escalation message."""
        parts = [
            f"Low confidence ({confidence:.0%}) - this response may need expert review.",
        ]

        if validation:
            if validation.stages_failed:
                parts.append(f"Failed validation stages: {', '.join(validation.stages_failed)}")
            if validation.warnings:
                parts.append("Warnings:")
                for warning in validation.warnings[:3]:
                    parts.append(f"  - {warning}")

        parts.append("Consider consulting OpenROADM documentation or a domain expert.")
        return "\n".join(parts)

    def get_pending_count(self) -> int:
        """Return count of pending escalations."""
        return sum(
            1 for e in self.escalation_history if e.get("status") == "pending_review"
        )

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent escalation history."""
        return self.escalation_history[-limit:]

    def _persist_escalation(self, record: Dict[str, Any]):
        """Save escalation record to disk."""
        try:
            self.log_directory.mkdir(parents=True, exist_ok=True)
            timestamp = record["timestamp"].replace(":", "-")
            filepath = self.log_directory / f"escalation_{timestamp}.json"
            filepath.write_text(json.dumps(record, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to persist escalation: {e}")
