"""Pure sluicing detector.

Operates on a udapi document and returns findings; no CAS dependency.

Detection rule (Universal Dependencies). X is the sluice remnant, G the
question-embedding governor:
  - X is a wh-word for the sentence's language — or a phrase headed by a
    non-wh word that has a wh-word child ("wie viele": head ``viele``,
    wh-child ``wie``).
  - X has no subject child (nsubj/csubj/nsubj:pass) and no verbal child —
    a sluice remnant is a bare wh-phrase, not a clause.
  - X attaches to G by one of:
      * STRICT, language-neutral: ``ccomp``, or ``advmod`` when X follows G;
      * BROADENED: ``obj``/``iobj``/``obl``/``conj``/``advmod``/``mark``/
        ``appos`` — accepted only when G is a known question-embedding
        predicate (see ``EMBEDDING_PREDICATES_BY_LANG``). Elliptical
        sluices routinely parse-degrade onto these relations; the
        embedding-predicate lexicon keeps precision.
    A governor that is itself an embedded clause head (deprel ``ccomp``/
    ``xcomp``/``acl``/``csubj``) is the verb of a *full* embedded question,
    not the embedding predicate — its wh dependent is not a sluice.

Each tree's language is read from its ``# lang =`` comment. Trees
without a language tag, with a tag whose lexicon is unsupported, or
that don't match ``restrict_to_lang`` (when given) are skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.sluicing_wh import (
    embedding_predicates,
    wh_words,
)
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

SUBJECT_RELS = {"nsubj", "csubj", "nsubj:pass"}

# Relations the remnant degrades onto in elliptical parses; accepted only
# when the governor is a known question-embedding predicate.
BROAD_RELS = {"obj", "iobj", "obl", "conj", "advmod", "mark", "appos"}

# A governor bearing one of these is itself an embedded clause head (the
# verb of a full embedded question), not the embedding predicate.
EMBEDDED_VERB_RELS = {"ccomp", "xcomp", "acl", "acl:relcl", "csubj"}


@dataclass(frozen=True)
class SluicingFinding:
    x_begin: int
    x_end: int
    g_begin: int
    g_end: int
    x_text: str
    g_text: str
    lang: str


def _wh_node(node, wh: frozenset[str]):
    """Return the wh-word in the remnant headed by ``node`` — ``node``
    itself, or a wh-word child (for "wie viele", head ``viele`` with the
    wh-child ``wie``). ``None`` if the remnant carries no wh-word."""
    if (node.form or "").lower() in wh:
        return node
    for child in node.children:
        if (child.form or "").lower() in wh:
            return child
    return None


def _embedding_governor(node):
    """The question-embedding predicate above the remnant. For a
    coordinated remnant ("... sucht und wozu") the remnant attaches via
    ``conj`` to a conjunct head, so the embedding predicate is one level
    further up."""
    g = node.parent
    if node.deprel == "conj" and g is not None and g.parent is not None:
        return g.parent
    return g


def _is_embedding(node, embed: frozenset[str]) -> bool:
    """True if ``node`` is a question-embedding predicate that is not
    itself an embedded clause head."""
    if node is None or node.is_root():
        return False
    if node.deprel in EMBEDDED_VERB_RELS:
        return False
    lemma = (node.lemma or "").lower()
    form = (node.form or "").lower()
    return lemma in embed or form in embed


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
        embed = embedding_predicates(lang)

        for node in tree.descendants:
            wh_n = _wh_node(node, wh)
            if wh_n is None:
                continue

            # A sluice remnant is a bare wh-phrase, not a clause.
            if any(child.deprel in SUBJECT_RELS for child in node.children):
                continue
            if any(child.upos == "VERB" for child in node.children):
                continue

            g = node.parent
            if g is None or g.is_root():
                continue

            # (a) strict, language-neutral path — the parse is "clean".
            if node.deprel == "ccomp" or (
                node.deprel == "advmod" and node.ord > g.ord
            ):
                governor = g
            # (b) broadened path — the parse degraded; license it with a
            # known question-embedding governor.
            elif node.deprel in BROAD_RELS:
                governor = _embedding_governor(node)
                if not _is_embedding(governor, embed):
                    continue
            else:
                continue

            # Remnant span: the wh-word plus any modifiers it heads
            # (so "wie viele" is reported whole, not just "viele");
            # punctuation children are left out.
            remnant = sorted(
                [node, *(c for c in node.children if c.upos != "PUNCT")],
                key=lambda n: n.ord,
            )
            x_begin = min(token_offsets(n)[0] for n in remnant)
            x_end = max(token_offsets(n)[1] for n in remnant)
            g_begin, g_end = token_offsets(governor)

            findings.append(
                SluicingFinding(
                    x_begin=x_begin, x_end=x_end,
                    g_begin=g_begin, g_end=g_end,
                    x_text=" ".join(n.form for n in remnant),
                    g_text=governor.form,
                    lang=lang,
                )
            )
            logger.debug(
                f"Sluicing[{lang}]: G={governor.form!r} --> "
                f"X={' '.join(n.form for n in remnant)!r}"
            )
    return findings
