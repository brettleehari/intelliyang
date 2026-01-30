"""Entry point for the OpenROADM YANG CLI Chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from yang_chatbot.utils.logging_config import setup_logging


@click.command()
@click.option(
    "--config", "-c",
    default="config.yaml",
    help="Path to configuration file",
)
@click.option(
    "--yang-dir", "-d",
    default=None,
    help="Path to YANG models directory (overrides config)",
)
@click.option(
    "--log-level", "-l",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level",
)
@click.option(
    "--log-file",
    default=None,
    help="Log file path",
)
def main(config: str, yang_dir: str, log_level: str, log_file: str):
    """OpenROADM YANG CLI Chatbot - AI-powered YANG model assistant."""
    setup_logging(level=log_level, log_file=log_file)

    from yang_chatbot.cli.chatbot import YANGChatbot

    chatbot = YANGChatbot(config_path=config)
    chatbot.initialize(yang_directory=yang_dir)
    chatbot.start_interactive_session()


if __name__ == "__main__":
    main()
