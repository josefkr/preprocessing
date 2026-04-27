"""Recover XMI character offsets from a udapi node's MISC column.

Tokens converted via :mod:`preprocessing.detection.cas_conllu` carry
``t_start`` / ``t_end`` in MISC. These helpers extract them, for both
single-token and multi-token (phrase) spans.
"""

from __future__ import annotations

from typing import Iterable


def token_offsets(node) -> tuple[int, int]:
    """Return ``(begin, end)`` offsets stored in a node's MISC."""
    try:
        return int(node.misc["t_start"]), int(node.misc["t_end"])
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"Node ord={node.ord} form={node.form!r} has no usable "
            "t_start/t_end in MISC; was it produced by view_to_conllu?"
        ) from e


def span_offsets(nodes: Iterable) -> tuple[int, int]:
    """Return ``(min begin, max end)`` across a sequence of nodes."""
    pairs = [token_offsets(n) for n in nodes]
    if not pairs:
        raise ValueError("span_offsets requires at least one node")
    begins, ends = zip(*pairs)
    return min(begins), max(ends)
