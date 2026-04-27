"""Pure verbal-ellipsis detector.

Operates on a udapi document and returns findings; no CAS dependency.

Detection rule:
  - Token has POS=AUX (in either UPOS or XPOS column; see note below).
  - Its dependency relation to its head is NOT one of {aux, aux:pass, cop}.

Note on POS columns: when the document comes from
:mod:`preprocessing.detection.cas_conllu`, ``upos`` is hardcoded by
py_lift's ``cas_to_str`` and the real PosValue lives in ``xpos``.
The detector therefore matches AUX in either column, which also lets
hand-written UD-convention fixtures (AUX in UPOS) work as expected.

This detector has no per-language lexicon, but still respects the
language tagging on each tree so multilingual documents can be
filtered consistently with the other detectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import tree_lang
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

AUX_DEP_TYPES = {"aux", "aux:pass", "cop"}


@dataclass(frozen=True)
class VerbalEllipsisFinding:
    begin: int
    end: int
    text: str
    deprel: str
    lang: str | None


def detect_verbal_ellipsis(
    doc, *, restrict_to_lang: str | None = None
) -> list[VerbalEllipsisFinding]:
    """Find verbal-ellipsis cases in a udapi ``Document``."""
    findings: list[VerbalEllipsisFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue

        for node in tree.descendants:
            if "AUX" not in (node.upos, node.xpos):
                continue
            if node.deprel in AUX_DEP_TYPES:
                continue

            begin, end = token_offsets(node)
            findings.append(
                VerbalEllipsisFinding(
                    begin=begin, end=end, text=node.form, deprel=node.deprel,
                    lang=lang,
                )
            )
            logger.debug(
                f"Verbal ellipsis[{lang}]: {node.form!r} [{begin}:{end}] "
                f"deprel={node.deprel}"
            )
    return findings
