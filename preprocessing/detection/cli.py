"""Shared argparse plumbing for the ``add_*.py`` detector CLIs.

Provides ``--lang`` and ``--mixed`` flags so every detector script
exposes the same multilingual interface.
"""

from __future__ import annotations

import argparse

from preprocessing.detection.language import SUPPORTED_LANGS


def add_language_args(parser: argparse.ArgumentParser) -> None:
    """Attach ``--lang`` and ``--mixed`` to a CLI parser."""
    parser.add_argument(
        "--lang",
        choices=sorted(SUPPORTED_LANGS),
        default=None,
        help="Explicit language for the corpus. Without --mixed, every "
        "sentence is treated as this language; with --mixed, --lang "
        "acts as a filter (only matching sentences are processed).",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Detect language per sentence instead of per document. "
        "Use for corpora with sentences in different languages.",
    )
