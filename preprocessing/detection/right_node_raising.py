"""Right-node-raising (RNR) detector — coordination subset.

Operates on a udapi document and returns findings; no CAS dependency.

RNR shares a constituent at the right edge of a coordination, elided from the
non-final conjunct: "Sam likes but Sue dislikes opera." (shared object "opera").
Only the **coordination** subset is detected here; comparative/subordinate RNR
("more X than Y", "those who voted against … outnumbered those who voted for …")
has no ``conj`` and is left to the LLM normalizer.

Structural signal (from the UD parse of the coordination):

  - a ``conj`` links two predicates V1 (non-final, earlier) and V2 (final);
  - V2 has a core-argument child (obj / obl / iobj / xcomp / ccomp) whose subtree
    reaches the right edge of the sentence — the candidate shared constituent;
  - V1 lacks a core argument of that class (the gap).

To separate genuine RNR from ordinary VP-coordination that merely *looks* like a
gap ("John went and bought a fridge" — ``went`` is intransitive, ``a fridge``
belongs only to ``bought``), one of two conditions must also hold:

  - **clausal** RNR: both conjuncts have their own overt subject (distinct
    subjects → each conjunct is a clause), or
  - **stranded preposition**: V1 carries a stranded preposition — an ``obl``/
    ``obj`` child that is a bare ADP with no nominal of its own ("knew of __",
    "interested in __") — a direct sign its argument was raised out.

Each tree's ``# lang =`` comment is honored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import tree_lang
from preprocessing.detection.offsets import span_offsets, token_offsets

logger = logging.getLogger(__name__)

PRED_UPOS = {"VERB", "AUX", "ADJ"}
SUBJ_RELS = {"nsubj", "csubj"}                       # matched on udeprel
CORE_ARG_RELS = {"obj", "iobj", "obl", "xcomp", "ccomp"}  # matched on udeprel


@dataclass(frozen=True)
class RNRRole:
    begin: int
    end: int
    text: str


@dataclass(frozen=True)
class RNRFinding:
    kind: str  # "coordination" (only subset detected structurally)
    trigger: str  # "distinct_subjects" | "stranded_prep"
    left_predicate: RNRRole   # the non-final predicate (missing the shared arg)
    right_predicate: RNRRole  # the final predicate (supplies the shared arg)
    shared_arg: RNRRole       # the shared constituent (subtree of V2's core arg)
    lang: str | None


def _role(node) -> RNRRole:
    b, e = token_offsets(node)
    return RNRRole(begin=b, end=e, text=node.form)


def _phrase_role(node) -> RNRRole:
    nodes = list(node.descendants(add_self=True))
    b, e = span_offsets(nodes)
    return RNRRole(begin=b, end=e, text=" ".join(n.form for n in nodes))


def _has_subject(pred) -> bool:
    return any(c.udeprel in SUBJ_RELS for c in pred.children)


def _has_stranded_prep(pred) -> bool:
    """A bare preposition attached directly to ``pred`` as obl/obj with no
    nominal of its own — its object was raised out ("knew of __")."""
    for c in pred.children:
        if c.udeprel in ("obl", "obj") and c.upos == "ADP":
            if not any(gc.udeprel in ("obj", "obl", "nmod", "nsubj")
                       for gc in c.children):
                return True
    return False


def _right_edge(tree) -> int | None:
    ends = [token_offsets(t)[1] for t in tree.descendants if t.upos != "PUNCT"]
    return max(ends) if ends else None


def _shared_arg_at_right_edge(final, right_edge):
    """A core-argument child of ``final`` whose subtree reaches the sentence's
    right edge (the raised constituent). Returns the child node or ``None``."""
    best = None
    for c in final.children:
        if c.udeprel not in CORE_ARG_RELS:
            continue
        _, e = span_offsets(list(c.descendants(add_self=True)))
        if right_edge is not None and e >= right_edge:
            best = c
    return best


def _detect_tree(tree, lang) -> list[RNRFinding]:
    findings: list[RNRFinding] = []
    right_edge = _right_edge(tree)
    for c in tree.descendants:
        if c.deprel != "conj":
            continue
        p = c.parent
        if p is None or p.is_root():
            continue
        if c.upos not in PRED_UPOS or p.upos not in PRED_UPOS:
            continue

        # Linear order: non-final = earlier, final = later.
        nonfinal, final = (p, c) if p.ord < c.ord else (c, p)

        shared = _shared_arg_at_right_edge(final, right_edge)
        if shared is None:
            continue
        shared_cls = shared.udeprel

        # V1 must lack a *filled* core arg of the shared class (the gap).
        if any(ch.udeprel == shared_cls and ch is not shared
               for ch in nonfinal.children if ch.upos not in ("ADP",)):
            continue

        distinct_subjects = _has_subject(nonfinal) and _has_subject(final)
        stranded = _has_stranded_prep(nonfinal)
        if not (distinct_subjects or stranded):
            continue  # ordinary VP-coordination, not RNR

        findings.append(
            RNRFinding(
                kind="coordination",
                trigger="distinct_subjects" if distinct_subjects else "stranded_prep",
                left_predicate=_role(nonfinal),
                right_predicate=_role(final),
                shared_arg=_phrase_role(shared),
                lang=lang,
            )
        )
        logger.debug(
            f"RNR[{lang}]: left={nonfinal.form!r} right={final.form!r} "
            f"shared={findings[-1].shared_arg.text!r} ({findings[-1].trigger})"
        )
    return findings


def detect_right_node_raising(
    doc, *, restrict_to_lang: str | None = None
) -> list[RNRFinding]:
    """Find (coordination-subset) right-node-raising in a udapi ``Document``."""
    findings: list[RNRFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue
        findings.extend(_detect_tree(tree, lang))
    return findings
