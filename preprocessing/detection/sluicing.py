"""Pure sluicing detector.

Operates on a udapi document and returns findings; no CAS dependency.

Detection rule (Universal Dependencies):
  - X is a wh-word for the sentence's language.
  - X is the dependent of G via ``ccomp``, or via ``advmod`` when X
    follows G linearly.
  - X has no child with a subject relation (nsubj, csubj, nsubj:pass).

Each tree's language is read from its ``# lang =`` comment. Trees
without a language tag, with a tag whose lexicon is unsupported, or
that don't match ``restrict_to_lang`` (when given) are skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.sluicing_wh import wh_words
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

SUBJECT_RELS = {"nsubj", "csubj", "nsubj:pass"}


@dataclass(frozen=True)
class SluicingFinding:
    x_begin: int
    x_end: int
    g_begin: int
    g_end: int
    x_text: str
    g_text: str
    lang: str


def detect_sluicing(
    doc, *, restrict_to_lang: str | None = None
) -> list[SluicingFinding]:
    """Find sluicing cases in a udapi ``Document`` and return findings."""
    findings: list[SluicingFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if lang is None:
            logger.debug(f"sentence {tree.sent_id}: no `# lang =` tag, skipping")
            continue
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue
        try:
            wh = wh_words(lang)
        except UnsupportedLanguage as e:
            logger.warning(str(e))
            continue

        for node in tree.descendants:
            if node.deprel == "ccomp":
                pass
            elif node.deprel == "advmod":
                if node.parent is None or node.ord <= node.parent.ord:
                    continue
            else:
                continue

            if node.form is None or node.form.lower() not in wh:
                continue

            if any(child.deprel in SUBJECT_RELS for child in node.children):
                continue

            g = node.parent
            if g is None or g.is_root():
                continue

            x_begin, x_end = token_offsets(node)
            g_begin, g_end = token_offsets(g)

            findings.append(
                SluicingFinding(
                    x_begin=x_begin, x_end=x_end,
                    g_begin=g_begin, g_end=g_end,
                    x_text=node.form, g_text=g.form,
                    lang=lang,
                )
            )
            logger.debug(f"Sluicing[{lang}]: G={g.form!r} --> X={node.form!r}")
    return findings
