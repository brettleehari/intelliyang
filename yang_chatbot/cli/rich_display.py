"""Rich formatting utilities for CLI output."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Custom theme for OpenROADM chatbot
THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "yang.module": "blue bold",
    "yang.container": "magenta",
    "yang.leaf": "green",
    "confidence.high": "green bold",
    "confidence.medium": "yellow",
    "confidence.low": "red bold",
}) if RICH_AVAILABLE else None


class RichDisplay:
    """Rich-formatted CLI output for the chatbot."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and RICH_AVAILABLE
        if self.enabled:
            self.console = Console(theme=THEME)
        else:
            self.console = None

    def print_welcome(self):
        """Display welcome banner."""
        if self.enabled:
            self.console.print(Panel(
                "[bold cyan]OpenROADM YANG Chatbot[/bold cyan]\n"
                "AI-powered assistant for OpenROADM YANG models\n\n"
                "Type [bold]help[/bold] for commands, [bold]quit[/bold] to exit",
                title="Welcome",
                border_style="cyan",
            ))
        else:
            print("=" * 60)
            print("  OpenROADM YANG Chatbot")
            print("  AI-powered assistant for OpenROADM YANG models")
            print("  Type 'help' for commands, 'quit' to exit")
            print("=" * 60)

    def print_response(self, response: str, confidence: float):
        """Display a chatbot response with confidence indicator."""
        conf_label, conf_style = self._confidence_style(confidence)

        if self.enabled:
            self.console.print()
            self.console.print(response)
            self.console.print(
                f"\n[{conf_style}][Confidence: {confidence:.0%} {conf_label}][/{conf_style}]"
            )
        else:
            print(f"\n{response}")
            print(f"\n[Confidence: {confidence:.0%} {conf_label}]")

    def print_escalation(self, message: str):
        """Display an escalation warning."""
        if self.enabled:
            self.console.print(Panel(
                message,
                title="[warning]Escalation Notice[/warning]",
                border_style="yellow",
            ))
        else:
            print(f"\n--- Escalation Notice ---")
            print(message)
            print("---")

    def print_error(self, message: str):
        """Display an error message."""
        if self.enabled:
            self.console.print(f"[error]Error:[/error] {message}")
        else:
            print(f"Error: {message}")

    def print_info(self, message: str):
        """Display an info message."""
        if self.enabled:
            self.console.print(f"[info]{message}[/info]")
        else:
            print(message)

    def print_success(self, message: str):
        """Display a success message."""
        if self.enabled:
            self.console.print(f"[success]{message}[/success]")
        else:
            print(message)

    def print_statistics(self, stats: Dict[str, Any]):
        """Display YANG model statistics in a table."""
        if self.enabled:
            table = Table(title="YANG Model Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")
            for key, value in stats.items():
                table.add_row(key.replace("_", " ").title(), str(value))
            self.console.print(table)
        else:
            print("\nYANG Model Statistics:")
            for key, value in stats.items():
                label = key.replace("_", " ").title()
                print(f"  {label}: {value}")

    def print_modules(self, modules: List[Dict[str, str]]):
        """Display list of loaded modules."""
        if self.enabled:
            table = Table(title="Loaded YANG Modules")
            table.add_column("Module", style="yang.module")
            table.add_column("Revision", style="dim")
            table.add_column("Description", max_width=50)
            for mod in modules:
                table.add_row(
                    mod.get("name", ""),
                    mod.get("revision", ""),
                    mod.get("description", "")[:50],
                )
            self.console.print(table)
        else:
            print("\nLoaded YANG Modules:")
            for mod in modules:
                print(f"  {mod.get('name', '')} (rev: {mod.get('revision', '')})")

    def print_help(self):
        """Display help information."""
        help_text = """
Commands:
  help              Show this help message
  stats             Show YANG model statistics
  modules           List loaded YANG modules
  validate <text>   Validate YANG content
  history           Show conversation history
  clear             Clear conversation history
  quit / exit       Exit the chatbot

Query Examples:
  What is a shelf in OpenROADM?
  Show me the device hierarchy
  What are the timing constraints for OLM?
  How do SRGs relate to degrees?
  List the mandatory fields for device-info
"""
        if self.enabled:
            self.console.print(Panel(help_text.strip(), title="Help", border_style="cyan"))
        else:
            print(help_text)

    def print_validation(self, result: Any):
        """Display validation result."""
        if self.enabled:
            status = "[success]VALID[/success]" if result.is_valid else "[error]INVALID[/error]"
            self.console.print(f"Validation: {status}")
            if result.errors:
                for err in result.errors:
                    self.console.print(f"  [error]Error:[/error] {err}")
            if result.warnings:
                for warn in result.warnings:
                    self.console.print(f"  [warning]Warning:[/warning] {warn}")
            if result.stages_passed:
                self.console.print(f"  [success]Passed:[/success] {', '.join(result.stages_passed)}")
        else:
            status = "VALID" if result.is_valid else "INVALID"
            print(f"Validation: {status}")
            for err in result.errors:
                print(f"  Error: {err}")
            for warn in result.warnings:
                print(f"  Warning: {warn}")

    def _confidence_style(self, confidence: float):
        """Get style for confidence level."""
        if confidence >= 0.90:
            return "HIGH", "confidence.high"
        elif confidence >= 0.70:
            return "MEDIUM", "confidence.medium"
        else:
            return "LOW", "confidence.low"
