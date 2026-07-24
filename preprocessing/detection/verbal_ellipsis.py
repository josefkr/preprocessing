"""Pure verbal-ellipsis detector.

Operates on a udapi document and returns findings; no CAS dependency.

Detection rule:
  - Token has POS=AUX (in either UPOS or XPOS column; see note below).
  - Its dependency relation to its head is NOT one of {aux, aux:pass, cop}.

Plus two lexicon-gated extensions for elements that parsers routinely tag
VERB instead of AUX, which the rule above would then miss: German stranded
modals (:data:`DE_MODAL_LEMMAS`) and English stranded pro-verbs
(:data:`EN_PROVERB_LEMMAS`). Both require no overt verbal complement and no
direct object, so lexical uses ("kann Französisch", "do my homework") are
excluded.

Note on POS columns: the detector matches AUX in either UPOS or XPOS so
both DKPro CAS input (UD ``coarseValue`` -> UPOS, Penn-Treebank
``PosValue`` -> XPOS, where AUX appears in UPOS) and hand-written
fixtures (AUX in either column) work.

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

# German modal lemmas. Used by the "stranded modal" extension below:
# in passage context Stanza often tags an absolutely-used modal
# ("er will nicht", "wir durften nicht") as UPOS=VERB (deprel root/conj)
# rather than AUX, so the AUX-only rule misses it. This lexicon lets the
# detector also flag a modal-lemma VERB that has no overt verbal
# complement and no direct object — i.e. the main VP is elided.
DE_MODAL_LEMMAS = frozenset(
    {"wollen", "können", "dürfen", "müssen", "sollen", "mögen"}
)

# Deprels marking a direct/accusative object. A modal carrying one is a
# *lexical* modal use ("sie kann Französisch" = she knows French), not
# verb-phrase ellipsis. (Oblique/directional dependents like
# "auf die Seitenlage" are NOT excluded — a stranded modal may keep a
# place/direction adverbial: "ich will nicht auf die Seitenlage".)
_OBJECT_DEPRELS = {"obj", "obja"}


# English pro-verb lemmas. Same problem as DE_MODAL_LEMMAS in the other
# direction: a stranded pro-verb ("So do I", "I do as well", "Yes, I have")
# is frequently tagged UPOS=VERB rather than AUX, so the AUX-only rule misses
# it. "be" is deliberately excluded — copular/existential uses tagged VERB
# ("I think, therefore I am") are not ellipsis; genuine stranded "be" is
# reliably tagged AUX and already caught by Rule 1. Modals likewise stay AUX
# in English, unlike German.
EN_PROVERB_LEMMAS = frozenset({"do", "have"})


def _is_stranded_proform(node, lemmas: frozenset[str]) -> bool:
    """True for a lemma from ``lemmas`` tagged VERB/AUX that stands in for a
    missing main verb: not itself an auxiliary of another verb, with no overt
    verbal complement and no direct object.

    Shared by the two lexicon-gated extensions (German modals, English
    pro-verbs); each lexicon self-gates by language, so neither fires on the
    other's data.
    """
    if (node.lemma or "").lower() not in lemmas:
        return False
    if node.upos not in ("VERB", "AUX"):
        return False
    if node.deprel in AUX_DEP_TYPES:
        return False
    for child in node.children:
        if child.upos in ("VERB", "AUX"):
            return False  # overt main verb present — not ellipsis
        if child.deprel in _OBJECT_DEPRELS:
            return False  # lexical use ("kann Französisch", "do my homework")
    return True


def _is_stranded_modal(node) -> bool:
    """German modal standing in for an elided main verb."""
    return _is_stranded_proform(node, DE_MODAL_LEMMAS)


def _is_stranded_proverb(node) -> bool:
    """English pro-verb standing in for an elided main verb."""
    return _is_stranded_proform(node, EN_PROVERB_LEMMAS)


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
            # Rule 1: a token tagged AUX (UPOS or XPOS) whose relation to
            # its head is not aux/aux:pass/cop.
            is_stranded_aux = (
                "AUX" in (node.upos, node.xpos)
                and node.deprel not in AUX_DEP_TYPES
            )
            # Rule 2: a German modal lemma tagged VERB (the AUX-only rule
            # misses these in passage context) standing in for a missing
            # main verb.
            # Rule 3: the English counterpart — a stranded pro-verb
            # ("I do as well") that parsers tag VERB rather than AUX.
            if not (
                is_stranded_aux
                or _is_stranded_modal(node)
                or _is_stranded_proverb(node)
            ):
                continue

            begin, end = token_offsets(node)
            findings.append(
                VerbalEllipsisFinding(
                    begin=begin,
                    end=end,
                    text=node.form,
                    deprel=node.deprel,
                    lang=lang,
                )
            )
            logger.debug(
                f"Verbal ellipsis[{lang}]: {node.form!r} [{begin}:{end}] "
                f"deprel={node.deprel}"
            )
    return findings
