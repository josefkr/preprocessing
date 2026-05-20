"""Pure bare-wh-question detector.

A bare wh-question is an elliptical sentence consisting only of a
wh-phrase — no verb, no embedding governor. The missing predicate is
inferred from the preceding discourse. Examples:

    Why?            What for?          For what?
    What man?       How viable?        Which powers?
    Until when?     About what?        By what?

Distinct from sluicing: a sluice is an embedded question reduced to its
wh-word and lives *inside* a host sentence under a question-embedding
governor (verb or noun). A bare wh-question stands alone as the whole
sentence and has no governor.

Detection rule (Universal Dependencies):
  - The sentence contains a ``?`` token (sentence-final question marker).
  - No descendant is a ``VERB`` or ``AUX``; no descendant has an
    ``nsubj``/``csubj``/``nsubj:pass`` child — i.e. nothing in the tree
    is a clause.
  - Either the tree's root is itself a wh-word, *or* the root has a
    wh-word child whose relation is one of
    ``{det, advmod, amod, case}`` — the modifier slots from which a
    wh-word can head a wh-phrase ("What man?" — ``det``; "How viable?" —
    ``advmod``; "For what?" — ``case`` on the wh-root).
  - This deliberately excludes "Morris who?" / "Bobby who?" echo
    questions, where ``who`` attaches by ``appos``/``parataxis`` — *not*
    a wh-phrase modifier slot.

Each tree's language is read from its ``# lang =`` comment, exactly as
the sluicing detector does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.sluicing_wh import wh_words
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

CLAUSAL_RELS = frozenset({"nsubj", "csubj", "nsubj:pass"})
WH_PHRASE_RELS = frozenset({"det", "advmod", "amod", "case"})
VERBAL_UPOS = frozenset({"VERB", "AUX"})


@dataclass(frozen=True)
class BareQuestionFinding:
    begin: int
    end: int
    text: str
    wh_form: str
    lang: str


def detect_bare_questions(
    doc, *, restrict_to_lang: str | None = None
) -> list[BareQuestionFinding]:
    """Find bare wh-question sentences in a udapi ``Document``."""
    findings: list[BareQuestionFinding] = []
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

        finding = _classify_tree(tree, wh, lang)
        if finding is not None:
            findings.append(finding)
            logger.debug(
                f"BareWhQuestion[{lang}]: {finding.text!r} "
                f"(wh={finding.wh_form!r})"
            )
    return findings


def _tree_root(tree):
    """Return the syntactic root token of ``tree`` (the unique node whose
    head is the technical zero root). ``None`` if the tree has no node
    with that property (defensive)."""
    for node in tree.descendants:
        parent = node.parent
        if parent is not None and parent.is_root():
            return node
    return None


def _classify_tree(tree, wh: frozenset[str], lang: str):
    descendants = list(tree.descendants)
    if not descendants:
        return None

    # rule 1: sentence carries a '?' token
    if not any((d.form or "") == "?" for d in descendants):
        return None

    # rule 2: nothing in the tree is a clause — no finite verb/aux and
    # no clausal-subject child.
    for d in descendants:
        if d.upos in VERBAL_UPOS:
            return None
        if d.deprel in CLAUSAL_RELS:
            return None

    root = _tree_root(tree)
    if root is None:
        return None

    # rule 3: the wh-word is the root, OR is a wh-phrase modifier of it.
    root_form = (root.form or "").lower()
    if root_form in wh:
        wh_node = root
    else:
        wh_node = None
        for child in root.children:
            if child.deprel in WH_PHRASE_RELS and (child.form or "").lower() in wh:
                wh_node = child
                break
        if wh_node is None:
            return None

    # Span = the wh-phrase: every token in the tree except punctuation
    # (quotes, '?', stray commas). Captures the full surface — "What for",
    # "How viable", "What man", "For what" — not just the wh-token.
    phrase_nodes = [d for d in descendants if d.upos != "PUNCT"]
    if not phrase_nodes:
        return None
    begin = min(token_offsets(n)[0] for n in phrase_nodes)
    end = max(token_offsets(n)[1] for n in phrase_nodes)
    text = " ".join(
        n.form for n in sorted(phrase_nodes, key=lambda n: n.ord)
    )

    return BareQuestionFinding(
        begin=begin,
        end=end,
        text=text,
        wh_form=wh_node.form,
        lang=lang,
    )
