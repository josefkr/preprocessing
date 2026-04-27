"""Pure passive-construction detector.

Operates on a udapi document and returns findings; no CAS dependency.

Two kinds of passive are detected:

**Canonical** (``kind="canonical"``):
  - Token X has ``deprel == aux:pass`` → the passive auxiliary.
  - V = X's parent is the lexical (head) verb.
  - Optionally, S = a child of V with ``deprel`` in
    {nsubj:pass, csubj:pass} is the passive subject.

**Short** (``kind="short"``):
  - V is a passive participle (per-language XPOS set, with a fallback
    to ``VerbForm=Part``).
  - V has no aux child of any kind (rules out canonical passives,
    active perfects, modals, etc.).
  - V has an ``obl`` / ``obl:agent`` child A whose ``case`` child P
    has a form in the language's agent-preposition set
    (e.g. "by" / "von" / "durch" / "par" / "por").

Each tree's ``# lang =`` comment is honored. Sentences whose language
has no passive lexicon are skipped for the short-passive path with a
debug log; canonical detection still runs (it is purely structural).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.passive import (
    is_passive_participle,
    passive_agent_preps,
)
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

PASSIVE_SUBJ_RELS = {"nsubj:pass", "csubj:pass"}
AGENT_OBL_RELS = {"obl", "obl:agent"}


@dataclass(frozen=True)
class PassiveRole:
    begin: int
    end: int
    text: str


@dataclass(frozen=True)
class PassiveFinding:
    kind: str  # "canonical" or "short"
    verb: PassiveRole
    aux: PassiveRole | None
    subject: PassiveRole | None
    agent: PassiveRole | None
    agent_marker: PassiveRole | None
    lang: str | None


def _role(node) -> PassiveRole:
    b, e = token_offsets(node)
    return PassiveRole(begin=b, end=e, text=node.form)


def _detect_canonical(tree, lang) -> list[PassiveFinding]:
    findings: list[PassiveFinding] = []
    for node in tree.descendants:
        if node.deprel != "aux:pass":
            continue
        v = node.parent
        if v is None or v.is_root():
            continue

        subj_node = next(
            (c for c in v.children if c.deprel in PASSIVE_SUBJ_RELS),
            None,
        )

        findings.append(
            PassiveFinding(
                kind="canonical",
                verb=_role(v),
                aux=_role(node),
                subject=_role(subj_node) if subj_node is not None else None,
                agent=None,
                agent_marker=None,
                lang=lang,
            )
        )
        logger.debug(
            f"Passive[canonical,{lang}]: aux={node.form!r} verb={v.form!r} "
            f"subject={subj_node.form if subj_node else None!r}"
        )
    return findings


def _detect_short(tree, lang) -> list[PassiveFinding]:
    if lang is None:
        return []
    try:
        agent_preps = passive_agent_preps(lang)
    except UnsupportedLanguage as e:
        logger.debug(str(e))
        return []

    findings: list[PassiveFinding] = []
    for v in tree.descendants:
        if not is_passive_participle(v, lang):
            continue
        # Skip if V has any aux child (canonical passive, perfect, modal, ...).
        if any(c.udeprel == "aux" for c in v.children):
            continue

        agent_node = None
        marker_node = None
        for c in v.children:
            if c.deprel not in AGENT_OBL_RELS:
                continue
            mk = next(
                (gc for gc in c.children
                 if gc.deprel == "case" and gc.form
                 and gc.form.lower() in agent_preps),
                None,
            )
            if mk is not None:
                agent_node = c
                marker_node = mk
                break

        if agent_node is None:
            continue

        findings.append(
            PassiveFinding(
                kind="short",
                verb=_role(v),
                aux=None,
                subject=None,
                agent=_role(agent_node),
                agent_marker=_role(marker_node),
                lang=lang,
            )
        )
        logger.debug(
            f"Passive[short,{lang}]: verb={v.form!r} "
            f"marker={marker_node.form!r} agent={agent_node.form!r}"
        )
    return findings


def detect_passive(
    doc, *, restrict_to_lang: str | None = None
) -> list[PassiveFinding]:
    """Find passive constructions (canonical + short) in a udapi ``Document``."""
    findings: list[PassiveFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue
        findings.extend(_detect_canonical(tree, lang))
        findings.extend(_detect_short(tree, lang))
    return findings
