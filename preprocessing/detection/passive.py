"""Pure passive-construction detector.

Operates on a udapi document and returns findings; no CAS dependency.

Two kinds of passive are detected:

**Canonical** (``kind="canonical"``):
  - Token X has ``deprel == aux:pass`` → the passive auxiliary.
  - V = X's parent is the lexical (head) verb.
  - Optionally, S = a child of V with ``deprel`` in
    {nsubj:pass, csubj:pass} is the passive subject.
  - Optionally, an agent: a child of V with ``deprel`` in
    {obl, obl:agent} whose ``case`` child is an agent preposition
    (by / von / durch / ...). This is the "agentful" passive that the
    normalizer rewrites to active; it is the common case in the target
    data ("Der Motor wird *vom Mechaniker* repariert."). 
    Agent extraction needs the language's lexicon, 
    so it is skipped (agent stays ``None``) for sentences 
    with no ``# lang =`` tag or an unsupported language.

**Short** (``kind="short"``):
  - V is specifically a passive participle (per-language XPOS set, 
    with a fallback to ``VerbForm=Part``).
  - V has no aux child of any kind (rules out canonical passives,
    active perfects, modals, etc.).
  - V has an ``obl`` / ``obl:agent`` child A whose ``case`` child P
    has a form in the relevant language's agent-preposition set
    (e.g. "by" / "von" / "durch" / "par" / "por").

Each tree's ``# lang =`` comment is honored. Sentences whose language
has no passive lexicon are skipped for the short-passive path with a
debug log message; canonical detection still runs (it is purely structural).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.passive import (
    is_passive_participle,
    passive_agent_preps,
)
from preprocessing.detection.offsets import span_offsets, token_offsets

logger = logging.getLogger(__name__)

PASSIVE_SUBJ_RELS = {"nsubj:pass", "csubj:pass"}
AGENT_OBL_RELS = {"obl", "obl:agent"}


def _agent_preps_or_empty(lang) -> frozenset[str]:
    """Agent-preposition lexicon for ``lang``, or empty if unavailable.

    Canonical-passive detection is purely structural and runs even for
    languages that lack a passive lexicon (or have no ``# lang =`` tag).
    However, in that case we simply can't recognise an agent, 
    and so we return an empty set rather than raising.
    """
    if lang is None:
        return frozenset()
    try:
        return passive_agent_preps(lang)
    except UnsupportedLanguage as e:
        logger.debug(str(e))
        return frozenset()


def _find_agent(verb, agent_preps: frozenset[str]):
    """Return ``(agent_node, marker_node)`` for V's by/von/durch agent PP.

    The agent is an ``obl`` / ``obl:agent`` child of the (participle) verb
    ``verb`` whose own ``case`` child is one of the language's agent
    prepositions. Returns ``(None, None)`` when there is no such child
    (e.g. an agentless passive, or no lexicon for the language).
    """
    if not agent_preps:
        return None, None
    for child in verb.children:
        if child.deprel not in AGENT_OBL_RELS:
            continue
        mk = next(
            (grandchild for grandchild in child.children
             if grandchild.deprel == "case" and _is_agent_prep(grandchild, agent_preps)),
            None,
        )
        if mk is not None:
            return child, mk
    return None, None


def _is_agent_prep(node, agent_preps: frozenset[str]) -> bool:
    """Whether a ``case`` node is an agent preposition.

    Matches on lemma OR surface form, lower-cased. 
    The lemma check is what catches German contractions: 
    Stanza expands ``vom`` into two tokens that both keep the surface form 
    ``vom`` but carry the lemma ``von`` (the expected ``von dem``), 
    so a form-only test would miss ``vom``/``zum``-style contracted agents."""
    for attr in (node.lemma, node.form):
        if attr and attr.lower() in agent_preps:
            return True
    return False


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
    # Full-phrase spans (head token + its whole subtree), as opposed to the
    # head-only spans in ``subject`` / ``agent``. The passive→active normalizer
    # quotes these so it gets "by Trading Officer Craig Methven" / "vom
    # Mechaniker" rather than just the head noun. ``None`` when the
    # corresponding head role is absent.
    subject_phrase: PassiveRole | None = None
    agent_phrase: PassiveRole | None = None


def _role(node) -> PassiveRole:
    b, e = token_offsets(node)
    return PassiveRole(begin=b, end=e, text=node.form)


def _phrase_role(node) -> PassiveRole:
    """A role spanning ``node``'s whole subtree (the head plus all descendants),
    e.g. the full agent or subject NP including its determiner/modifiers."""
    nodes = list(node.descendants(add_self=True))
    b, e = span_offsets(nodes)
    return PassiveRole(begin=b, end=e, text=" ".join(n.form for n in nodes))


def _detect_canonical(tree, lang) -> list[PassiveFinding]:
    agent_preps = _agent_preps_or_empty(lang)
    findings: list[PassiveFinding] = []
    for node in tree.descendants:
        if node.deprel != "aux:pass":
            continue
        verb = node.parent
        if verb is None or verb.is_root():
            continue

        subj_node = next(
            (child for child in verb.children if child.deprel in PASSIVE_SUBJ_RELS),
            None,
        )
        agent_node, marker_node = _find_agent(verb, agent_preps)

        findings.append(
            PassiveFinding(
                kind="canonical",
                verb=_role(verb),
                aux=_role(node),
                subject=_role(subj_node) if subj_node is not None else None,
                agent=_role(agent_node) if agent_node is not None else None,
                agent_marker=(
                    _role(marker_node) if marker_node is not None else None
                ),
                lang=lang,
                subject_phrase=(
                    _phrase_role(subj_node) if subj_node is not None else None
                ),
                agent_phrase=(
                    _phrase_role(agent_node) if agent_node is not None else None
                ),
            )
        )
        logger.debug(
            f"Passive[canonical,{lang}]: aux={node.form!r} verb={verb.form!r} "
            f"subject={subj_node.form if subj_node else None!r} "
            f"agent={agent_node.form if agent_node else None!r}"
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
    for passive_verb_cand in tree.descendants:
        if not is_passive_participle(passive_verb_cand, lang):
            continue
        # Skip if V has any aux child (canonical passive, perfect, modal, ...).
        if any(child.udeprel == "aux" for child in passive_verb_cand.children):
            continue

        agent_node, marker_node = _find_agent(passive_verb_cand, agent_preps)
        if agent_node is None:
            continue

        findings.append(
            PassiveFinding(
                kind="short",
                verb=_role(passive_verb_cand),
                aux=None,
                subject=None,
                agent=_role(agent_node),
                agent_marker=_role(marker_node),
                lang=lang,
                subject_phrase=None,
                agent_phrase=_phrase_role(agent_node),
            )
        )
        logger.debug(
            f"Passive[short,{lang}]: verb={passive_verb_cand.form!r} "
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
