"""Pure gapped-coordination detector.

Detects coordinated clauses where the second (or later) conjunct lacks
its main predicate — the verb (or, in copular sentences, the copula)
has to be supplied from the antecedent conjunct.

Example: "Paul wanted a milk shake and Mr Leonard a coffee."
                                       └─ gapped clause; the verb
                                          "wanted" must be borrowed
                                          from the antecedent.

Detection rule (Universal Dependencies). The parser, lacking a head
for the gapped clause, typically attaches its arguments via ``conj`` to
some token in the antecedent clause — most often the deepest plausible
host (an ``obj``/``obl``/``xcomp``/``cop``-predicate of the antecedent
verb). Two complementary signals fire on those structural artefacts:

  - **Signal A** — a non-``VERB``/``AUX`` token has *two or more*
    ``conj`` children that are themselves non-``VERB``/``AUX``. The
    parser collapsed two gapped arguments (e.g. subject + object) onto
    a single host because the missing verb couldn't anchor them.

  - **Signal B** — a non-``VERB``/``AUX`` ``conj`` token under a
    non-``VERB``/``AUX`` parent has at least one child whose deprel
    is in ``{nsubj, csubj, nsubj:pass, appos, nmod, flat}`` — a
    second gapped argument hanging off the conj as if it were itself
    a clause head.

  - **Signal C** — same shape as Signal B but with a ``VERB``/``AUX``
    parent. German Stanza parses tend to attach the gap anchor
    directly as a ``conj`` of the matrix verb (rather than to one of
    the verb's arguments, as English Stanza usually does), so the
    Signal-B pattern needs the parent constraint relaxed for German.
    The conj token itself is still required to be non-verbal, which
    is what keeps well-formed verbal coordinations (*Jill bought a
    t-shirt and **she bought** Pat some shorts*) out of the findings
    list — both conjuncts are verbal there.

When the candidate ``conj`` token itself is a finite verb (or AUX),
we always decline to fire — that's a well-formed verbal coordination
regardless of parent.

Known limitations of v1 (not yet covered):

  - Parses that flatten the gapped material into the antecedent
    clause with no ``conj`` at all ("Always do it with your left
    hand, never with your right").
  - Parses where the gapped subject is attached as ``appos`` of an
    antecedent argument rather than ``conj`` ("…because of the pay,
    Bill because …").
  - Parses where the conj's parent is a participle/non-finite ``VERB``
    that itself heads the antecedent argument ("With Jill intent on
    resigning and Pat on following …").

Annotation:

  - ``GrammarAnomaly(description="Ellipsis", category="gapped_coordination")``
    on the gapped clause span. The finding additionally carries the
    antecedent verb's offsets and form, so the CAS writer (or a
    normalizer) can act on them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import tree_lang
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

VERBAL_UPOS = frozenset({"VERB", "AUX"})

# Relations under a ``conj`` head that signal a *second* gapped
# argument rather than an internal modifier of the conj head itself.
# ``flat`` is included for parser collapses like "Mr Leonard Nike"
# where flat chains the gapped arguments — within a coordinated
# subtree this is rarely a benign name continuation.
GAPPED_ARG_RELS = frozenset(
    {"nsubj", "csubj", "nsubj:pass", "appos", "nmod", "flat"}
)


@dataclass(frozen=True)
class GappedCoordinationFinding:
    begin: int
    end: int
    text: str
    antecedent_begin: int
    antecedent_end: int
    antecedent_text: str
    signal: str  # "A", "B", or "C"
    lang: str


def detect_gapped_coordination(
    doc, *, restrict_to_lang: str | None = None
) -> list[GappedCoordinationFinding]:
    """Find gapped-coordination cases in a udapi ``Document``."""
    findings: list[GappedCoordinationFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if lang is None:
            logger.debug(f"sentence {tree.sent_id}: no `# lang =` tag, skipping")
            continue
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue
        findings.extend(_classify_tree(tree, lang))
    return findings


def _classify_tree(tree, lang: str) -> list[GappedCoordinationFinding]:
    out: list[GappedCoordinationFinding] = []
    reported: set[int] = set()  # token ords already inside a reported cluster

    for node in tree.descendants:
        if node.deprel != "conj":
            continue
        if node.upos in VERBAL_UPOS:
            continue
        parent = node.parent
        if parent is None or parent.is_root():
            continue
        # Avoid emitting a duplicate when Signal A's sibling cluster is
        # entered through a non-first sibling.
        if node.ord in reported:
            continue

        signal = _which_signal(node, parent)
        if signal is None:
            continue

        if signal == "A":
            cluster = [
                c for c in parent.children
                if c.deprel == "conj" and c.upos not in VERBAL_UPOS
            ]
        else:  # "B" or "C"
            cluster = [node]

        # Mark every token under the cluster as reported so we don't
        # re-emit when iteration reaches other anchors in the same cluster.
        cluster_nodes: list = []
        for c in cluster:
            cluster_nodes.append(c)
            cluster_nodes.extend(c.descendants)
        for n in cluster_nodes:
            reported.add(n.ord)

        phrase_nodes = [n for n in cluster_nodes if n.upos != "PUNCT"]
        if not phrase_nodes:
            continue
        begin = min(token_offsets(n)[0] for n in phrase_nodes)
        end = max(token_offsets(n)[1] for n in phrase_nodes)
        text = " ".join(
            n.form for n in sorted(phrase_nodes, key=lambda n: n.ord)
        )

        antecedent = _antecedent_verb(parent)
        if antecedent is None:
            continue
        ant_begin, ant_end = token_offsets(antecedent)

        out.append(GappedCoordinationFinding(
            begin=begin, end=end, text=text,
            antecedent_begin=ant_begin, antecedent_end=ant_end,
            antecedent_text=antecedent.form,
            signal=signal, lang=lang,
        ))
        logger.debug(
            f"GappedCoord[{lang}, {signal}]: gap={text!r}, "
            f"antecedent={antecedent.form!r}"
        )
    return out


def _which_signal(node, parent) -> str | None:
    """Return ``'A'``, ``'B'``, or ``'C'`` if a gap should be reported
    anchored on ``node``, else ``None``.

    The caller has already filtered to non-verbal ``conj`` ``node``s.
    Signal A and B require a non-verbal parent; Signal C is the
    parent-is-verbal variant of B (German Stanza tends to attach the
    gap anchor directly to the matrix verb)."""
    parent_is_verbal = parent.upos in VERBAL_UPOS
    has_gapped_arg = any(c.deprel in GAPPED_ARG_RELS for c in node.children)

    if not parent_is_verbal:
        non_v_conj_siblings = [
            c for c in parent.children
            if c.deprel == "conj" and c.upos not in VERBAL_UPOS
        ]
        if len(non_v_conj_siblings) >= 2:
            return "A"
        if has_gapped_arg:
            return "B"
        return None

    # Verbal parent: only Signal C may fire.
    if has_gapped_arg:
        return "C"
    return None


def _antecedent_verb(host):
    """Walk from the conj host up to the tree root and return the
    *outermost* (matrix) verbal predicate found on the path.

    Intermediate ``xcomp``/``advcl`` verbs along the path are *not*
    the antecedent — they're embedded predicates of the matrix verb
    that gapped coordinations typically borrow from. For example, in
    "His father wanted him to marry Sue, but his mother Louise" the
    immediate verbal ancestor of "mother" is the xcomp "marry", but
    the missing predicate that needs to be copied is the matrix verb
    "wanted". We collect every verbal ancestor on the path and pick
    the one closest to the root.

    In copular antecedents ("Kim is an engineer …") the structural
    head is a NOUN/ADJ/PROPN; the verbal element sits on it as a
    ``cop`` child, so we also check each visited node for a ``cop``
    child."""
    verbs: list = []
    cur = host
    while cur is not None and not cur.is_root():
        if cur.upos in VERBAL_UPOS:
            verbs.append(cur)
        else:
            for ch in cur.children:
                if ch.deprel == "cop" and ch.upos in VERBAL_UPOS:
                    verbs.append(ch)
                    break
        cur = cur.parent
    if not verbs:
        return None
    return verbs[-1]
