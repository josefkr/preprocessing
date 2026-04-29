"""Pure cleft-sentence detector.

Operates on a udapi document and returns findings; no CAS dependency.

Two cleft kinds are detected today, both for English:

**it-clefts** (``kind="it_cleft"``):
  - F is a nominal (UPOS in {NOUN, PRON, PROPN}) with deprel ``root``
    or ``ccomp``.
  - F has a child with ``deprel == cop`` whose lemma is in the
    language's ``copula_lemmas`` (English: ``be``).
  - F has a child with ``deprel == acl:relcl``.
  - F has a child with ``deprel ∈ {nsubj, expl}`` whose lemma is in
    the language's ``it_lemmas`` (English: ``it``).

  Roles:
    - ``focus``: F together with the modifiers in its subtree that
      fall linearly between the cop/it pair and the start of the
      relative clause.
    - ``presupposition``: linear span of the ``acl:relcl`` subtree.
    - ``cleft_token``: the cleft pronoun ``it``.

**wh-clefts** (``kind="wh_cleft"``):
  - W has a lemma in the language's ``wh_lemmas`` (English: ``what``).
  - W has a child with ``deprel == acl:relcl``.
  - W's deprel is ``nsubj`` or ``nsubj:outer``.
  - W's head H is either a NOUN, or a VERB whose form equals its lemma
    (a heuristic for uninflected base/infinitive forms).

  Roles:
    - ``focus``: just the head H token.
    - ``presupposition``: linear span of the ``acl:relcl`` subtree of W.
    - ``cleft_token``: the wh-pronoun (``what``).

Trees are language-tagged via ``# lang =``; sentences whose language
has no cleft lexicon are skipped with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.clefts import CleftLexicon, clefts_lexicon
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

ADMISSIBLE_FOCUS_UPOS = frozenset({"NOUN", "PRON", "PROPN"})
ADMISSIBLE_FOCUS_DEPRELS = frozenset({"root", "ccomp"})
ADMISSIBLE_IT_DEPRELS = frozenset({"nsubj", "expl"})
ADMISSIBLE_WH_DEPRELS = frozenset({"nsubj", "nsubj:outer"})


@dataclass(frozen=True)
class CleftRole:
    begin: int
    end: int
    text: str


@dataclass(frozen=True)
class CleftFinding:
    kind: str  # "it_cleft" | "wh_cleft"
    focus: CleftRole
    presupposition: CleftRole
    cleft_token: CleftRole
    lang: str


def detect_clefts(
    doc, *, restrict_to_lang: str | None = None
) -> list[CleftFinding]:
    """Find cleft constructions in a udapi ``Document``."""
    findings: list[CleftFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if lang is None:
            logger.debug(
                f"sentence {tree.sent_id}: no `# lang =` tag, skipping"
            )
            continue
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue
        try:
            lex = clefts_lexicon(lang)
        except UnsupportedLanguage as e:
            logger.warning(str(e))
            continue

        for node in tree.descendants:
            f = _check_it_cleft(node, lex, lang)
            if f is not None:
                findings.append(f)
                logger.debug(
                    f"Cleft[it,{lang}]: focus={f.focus.text!r} "
                    f"presupposition={f.presupposition.text!r}"
                )
                continue
            f = _check_wh_cleft(node, lex, lang)
            if f is not None:
                findings.append(f)
                logger.debug(
                    f"Cleft[wh,{lang}]: focus={f.focus.text!r} "
                    f"presupposition={f.presupposition.text!r}"
                )
    return findings


def _lemma(node) -> str:
    return (node.lemma or "").lower()


def _check_it_cleft(f, lex: CleftLexicon, lang: str):
    if f.upos not in ADMISSIBLE_FOCUS_UPOS:
        return None
    if f.deprel not in ADMISSIBLE_FOCUS_DEPRELS:
        return None

    cop_node = next(
        (c for c in f.children
         if c.deprel == "cop" and _lemma(c) in lex.copula_lemmas),
        None,
    )
    if cop_node is None:
        return None

    relcl_node = next(
        (c for c in f.children if c.deprel == "acl:relcl"), None
    )
    if relcl_node is None:
        return None

    it_node = next(
        (c for c in f.children
         if c.deprel in ADMISSIBLE_IT_DEPRELS and _lemma(c) in lex.it_lemmas),
        None,
    )
    if it_node is None:
        return None

    relcl_subtree = [relcl_node] + list(relcl_node.descendants)
    relcl_min_ord = min(n.ord for n in relcl_subtree)
    relcl_begin = min(token_offsets(n)[0] for n in relcl_subtree)
    relcl_end = max(token_offsets(n)[1] for n in relcl_subtree)

    after_boundary_ord = max(cop_node.ord, it_node.ord)
    f_subtree = [f] + list(f.descendants)
    focus_nodes = [
        n for n in f_subtree
        if after_boundary_ord < n.ord < relcl_min_ord
    ]
    if not focus_nodes:
        return None
    focus_begin = min(token_offsets(n)[0] for n in focus_nodes)
    focus_end = max(token_offsets(n)[1] for n in focus_nodes)

    it_begin, it_end = token_offsets(it_node)

    return CleftFinding(
        kind="it_cleft",
        focus=CleftRole(begin=focus_begin, end=focus_end, text=f.form),
        presupposition=CleftRole(
            begin=relcl_begin, end=relcl_end, text=relcl_node.form
        ),
        cleft_token=CleftRole(begin=it_begin, end=it_end, text=it_node.form),
        lang=lang,
    )


def _check_wh_cleft(w, lex: CleftLexicon, lang: str):
    if _lemma(w) not in lex.wh_lemmas:
        return None
    if w.deprel not in ADMISSIBLE_WH_DEPRELS:
        return None

    relcl_node = next(
        (c for c in w.children if c.deprel == "acl:relcl"), None
    )
    if relcl_node is None:
        return None

    h = w.parent
    if h is None or h.is_root():
        return None
    if h.upos == "NOUN":
        pass
    elif h.upos == "VERB" and (h.form or "") == (h.lemma or ""):
        pass
    else:
        return None

    relcl_subtree = [relcl_node] + list(relcl_node.descendants)
    relcl_begin = min(token_offsets(n)[0] for n in relcl_subtree)
    relcl_end = max(token_offsets(n)[1] for n in relcl_subtree)

    h_begin, h_end = token_offsets(h)
    w_begin, w_end = token_offsets(w)

    return CleftFinding(
        kind="wh_cleft",
        focus=CleftRole(begin=h_begin, end=h_end, text=h.form),
        presupposition=CleftRole(
            begin=relcl_begin, end=relcl_end, text=relcl_node.form
        ),
        cleft_token=CleftRole(begin=w_begin, end=w_end, text=w.form),
        lang=lang,
    )
