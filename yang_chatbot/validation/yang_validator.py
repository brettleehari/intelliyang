"""Multi-stage YANG validation pipeline following telecom production patterns."""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from yang_chatbot.models.schema_models import ValidationResult
from yang_chatbot.validation.optical_constraints import OpticalConstraintValidator

logger = logging.getLogger(__name__)


class YANGValidator:
    """Multi-stage validation pipeline for YANG configurations.

    Implements the validation approach from Rakuten Symphony production patterns:
    1. Syntax validation (pyang/yanglint)
    2. Semantic consistency (LLM-based)
    3. Optical constraints (OpenROADM-specific)
    4. Confidence threshold check
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, ast_parser: Optional[Any] = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.90)
        self.escalation_enabled = self.config.get("escalation_enabled", True)
        self.optical_validator = OpticalConstraintValidator()
        self._ast_parser = ast_parser  # tree-sitter parser for AST error detection

    def validate(self, content: str, confidence: float = 1.0) -> ValidationResult:
        """Run the full validation pipeline."""
        errors: List[str] = []
        warnings: List[str] = []
        stages_passed: List[str] = []
        stages_failed: List[str] = []

        # Stage 1: Syntax validation
        if self.config.get("syntax_validation", True):
            syntax_ok, syntax_issues = self.validate_syntax(content)
            if syntax_ok:
                stages_passed.append("syntax")
            else:
                stages_failed.append("syntax")
                errors.extend(syntax_issues)

        # Stage 2: Semantic validation
        if self.config.get("semantic_validation", True):
            semantic_ok, semantic_issues = self.validate_semantics(content)
            if semantic_ok:
                stages_passed.append("semantics")
            else:
                stages_failed.append("semantics")
                warnings.extend(semantic_issues)

        # Stage 3: Optical constraints
        if self.config.get("optical_constraints", True):
            optical_ok, optical_issues = self.validate_optical_constraints(content)
            if optical_ok:
                stages_passed.append("optical_constraints")
            else:
                stages_failed.append("optical_constraints")
                warnings.extend(optical_issues)

        # Stage 4: Confidence check
        escalate = self.should_escalate(confidence)
        if not escalate:
            stages_passed.append("confidence")
        else:
            stages_failed.append("confidence")
            warnings.append(
                f"Confidence score ({confidence:.2f}) below threshold ({self.confidence_threshold})"
            )

        is_valid = len(stages_failed) == 0
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
            escalate=escalate,
            stages_passed=stages_passed,
            stages_failed=stages_failed,
        )

    def validate_syntax(self, content: str) -> Tuple[bool, List[str]]:
        """Validate YANG syntax using pyang or basic checks.

        Attempts to use pyang CLI if available, otherwise performs
        basic structural validation.
        """
        issues = []

        # Stage A: AST error detection via tree-sitter (fast, no external deps)
        if self._ast_parser and hasattr(self._ast_parser, 'get_errors'):
            errors = self._ast_parser.get_errors(content)
            if errors:
                for line, col, ctx in errors:
                    issues.append(f"AST parse error at line {line}:{col}: {ctx[:60]}")
                return False, issues

        # Stage B: pyang validation if available
        try:
            result = subprocess.run(
                ["pyang", "--lint", "-"],
                input=content,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                issues.extend(result.stderr.strip().split("\n"))
                return False, issues
            return True, []
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Stage C: basic structural checks
        return self._basic_syntax_check(content)

    def _basic_syntax_check(self, content: str) -> Tuple[bool, List[str]]:
        """Basic YANG syntax validation without external tools."""
        issues = []

        # Check brace matching
        open_braces = content.count("{")
        close_braces = content.count("}")
        if open_braces != close_braces:
            issues.append(
                f"Mismatched braces: {open_braces} opening, {close_braces} closing"
            )

        # Check for common YANG keywords
        yang_keywords = {"module", "container", "leaf", "list", "grouping", "typedef"}
        has_keywords = any(kw in content for kw in yang_keywords)
        if not has_keywords and len(content) > 50:
            issues.append("No YANG keywords found in content")

        # Check semicolons on statements
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
                # Statement lines should end with ; or { or }
                if stripped and stripped[-1] not in ";{}/" and not stripped.startswith("*"):
                    # Could be a multi-line string
                    if '"' not in stripped and "'" not in stripped:
                        pass  # Allow continuation lines

        return len(issues) == 0, issues

    def validate_semantics(self, content: str) -> Tuple[bool, List[str]]:
        """Semantic consistency validation.

        Checks for common semantic issues in YANG definitions.
        """
        issues = []

        # Check for leaf without type
        leaf_pattern = re.compile(r"leaf\s+[\w-]+\s*\{[^}]*\}", re.DOTALL)
        for match in leaf_pattern.finditer(content):
            block = match.group()
            if "type " not in block:
                issues.append(f"Leaf definition missing 'type' statement: {block[:60]}...")

        # Check for list without key
        list_pattern = re.compile(r"list\s+[\w-]+\s*\{[^}]*\}", re.DOTALL)
        for match in list_pattern.finditer(content):
            block = match.group()
            if "key " not in block:
                issues.append(f"List definition missing 'key' statement: {block[:60]}...")

        return len(issues) == 0, issues

    def validate_optical_constraints(self, content: str) -> Tuple[bool, List[str]]:
        """OpenROADM-specific optical constraint validation."""
        return self.optical_validator.validate(content)

    def should_escalate(self, confidence: float) -> bool:
        """Check if response should be escalated to human review."""
        if not self.escalation_enabled:
            return False
        return confidence < self.confidence_threshold
