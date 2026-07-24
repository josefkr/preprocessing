"""Pure subject-sharing detector.

Operates on a udapi document and returns findings; no CAS dependency.

Detection rule (Universal Dependencies):
  - X is the dependent of Y via the ``conj`` relation.
  - Y has a child S with a subject relation (nsubj, csubj, nsubj:pass).
  - X has no child with a subject relation.

This detector has no per-language lexicon, but still respects the
language tagging on each tree so multilingual documents can be
filtered consistently with the other detectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import tree_lang
from preprocessing.detection.offsets import span_offsets, token_offsets

logger = logging.getLogger(__name__)

SUBJECT_RELS = {"nsubj", "csubj", "nsubj:pass"}


@dataclass(frozen=True)
class SharedSubject:
    begin: int
    end: int
    text: str


@dataclass(frozen=True)
class SubjectSharingFinding:
    x_begin: int
    x_end: int
    y_begin: int
    y_end: int
    x_text: str
    y_text: str
    shared_subjects: tuple[SharedSubject, ...]
    lang: str | None


def detect_subject_sharing(
    doc, *, restrict_to_lang: str | None = None
) -> list[SubjectSharingFinding]:
    """Find subject-sharing conjuncts in a udapi ``Document``."""
    findings: list[SubjectSharingFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue

        for node in tree.descendants:
            if node.deprel != "conj":
                continue

            x = node
            y = node.parent
            if y is None or y.is_root():
                continue

            y_subjects = [c for c in y.children if c.deprel in SUBJECT_RELS]
            if not y_subjects:
                continue
            if any(c.deprel in SUBJECT_RELS for c in x.children):
                continue

            x_begin, x_end = token_offsets(x)
            y_begin, y_end = token_offsets(y)
            # Mark the whole subject phrase (head + its subtree), so the span
            # matches the material the normalizer copies into the right conjunct.
            shared = tuple(
                SharedSubject(
                    begin=b, end=e,
                    text=" ".join(n.form for n in nodes),
                )
                for s in y_subjects
                for nodes in [list(s.descendants(add_self=True))]
                for (b, e) in [span_offsets(nodes)]
            )

            findings.append(
                SubjectSharingFinding(
                    x_begin=x_begin, x_end=x_end,
                    y_begin=y_begin, y_end=y_end,
                    x_text=x.form, y_text=y.form,
                    shared_subjects=shared,
                    lang=lang,
                )
            )
            logger.debug(
                f"Subject sharing[{lang}]: Y={y.form!r} conj-> X={x.form!r}, "
                f"shared subjects={[s.text for s in shared]}"
            )
    return findings
