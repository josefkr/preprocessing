"""Pure nominal-head ellipsis detector.

Operates on a udapi document and returns findings; no CAS dependency.

Detection rules (all subtypes require a core nominal relation —
``nsubj``, ``nsubj:pass``, ``obj``, ``iobj``, ``nmod``):

  - ``none``: form in lexicon's ``none_forms``, no dependents.
  - ``every_one`` / ``elder``: matches a :class:`FixedPattern` — the
    token's form equals ``head_form`` and it has a ``det`` child whose
    form equals ``det_form``. The token's own deprel must not be in
    the pattern's ``excluded_deprels``.
  - ``quantifier``: ``upos=ADJ`` with form in ``quantifier_forms``,
    only optional ``det`` child.
  - ``numeral``: ``upos=NUM`` with only optional ``det`` / ``amod``
    children.
  - ``comparative``: ``xpos`` in ``comparative_xpos`` (e.g. JJR/JJS),
    deprel != ``amod``, has a ``det`` child whose form is in
    ``definite_articles``.
  - ``adjective``: ``upos=ADJ`` and ``xpos`` in ``adjective_xpos``
    (base-form JJ), deprel != ``amod``, only optional ``det`` child,
    parent is a verb.

Subtypes are tried most-specific first; the first match wins so that
e.g. "the elder" is tagged ``elder`` rather than ``comparative`` or
``adjective``.

Each tree's ``# lang =`` comment is honored. Sentences whose language
has no lexicon entry are skipped with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.nominal_ellipsis import (
    FixedPattern,
    NominalEllipsisLexicon,
    nominal_ellipsis_lexicon,
)
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

CORE_RELS = frozenset({"nsubj", "nsubj:pass", "obj", "iobj", "nmod"})


@dataclass(frozen=True)
class NominalEllipsisFinding:
    begin: int
    end: int
    text: str
    subtype: str
    deprel: str
    lang: str


def detect_nominal_ellipsis(
    doc, *, restrict_to_lang: str | None = None
) -> list[NominalEllipsisFinding]:
    """Find nominal-head ellipsis cases in a udapi ``Document``."""
    findings: list[NominalEllipsisFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if lang is None:
            logger.debug(
                f"sentence {tree.sent_id}: no `# lang =` tag, skipping"
            )
            continue
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue
        classify = _classifier_for_lang(lang)
        if classify is None:
            continue

        for node in tree.descendants:
            f = classify(node)
            if f is not None:
                findings.append(f)
                logger.debug(
                    f"NominalEllipsis[{f.subtype},{lang}]: "
                    f"{node.form!r} deprel={node.deprel}"
                )
    return findings


def _classifier_for_lang(lang: str):
    """Return a ``node -> NominalEllipsisFinding | None`` classifier for
    ``lang``, or ``None`` if the language is unsupported.

    German has its own rule module (``nominal_ellipsis_de``) because German
    Stanza output — STTS XPOS, quantifiers tagged DET/PIS, degree only in
    ``feats`` — does not fit the lexicon-driven English-style checks. Other
    languages use the lexicon-driven :func:`_classify`."""
    if lang == "de":
        # Imported lazily to avoid an import cycle: nominal_ellipsis_de
        # imports NominalEllipsisFinding from this module.
        from preprocessing.detection.nominal_ellipsis_de import classify_de

        return lambda node: classify_de(node, lang)
    try:
        lex = nominal_ellipsis_lexicon(lang)
    except UnsupportedLanguage as e:
        logger.warning(str(e))
        return None
    return lambda node: _classify(node, lex, lang)


def _classify(node, lex: NominalEllipsisLexicon, lang: str):
    for check in (
        _check_none,
        _check_fixed_pattern,
        _check_quantifier,
        _check_numeral,
        _check_comparative,
        _check_adjective,
    ):
        finding = check(node, lex, lang)
        if finding is not None:
            return finding
    return None


def _form_lower(node) -> str:
    return (node.form or "").lower()


def _is_core_rel(node) -> bool:
    return node.deprel in CORE_RELS


def _children_only_in(node, allowed: set[str]) -> bool:
    return all(c.deprel in allowed for c in node.children)


def _make(node, subtype: str, lang: str) -> NominalEllipsisFinding:
    begin, end = token_offsets(node)
    return NominalEllipsisFinding(
        begin=begin,
        end=end,
        text=node.form,
        subtype=subtype,
        deprel=node.deprel,
        lang=lang,
    )


def _check_none(node, lex, lang):
    if _form_lower(node) not in lex.none_forms:
        return None
    if list(node.children):
        return None
    if not _is_core_rel(node):
        return None
    return _make(node, "none", lang)


def _check_fixed_pattern(node, lex, lang):
    form = _form_lower(node)
    for pat in lex.fixed_patterns:
        if form != pat.head_form:
            continue
        if node.deprel in pat.excluded_deprels:
            continue
        if not _is_core_rel(node):
            continue
        det_match = any(
            c.deprel == "det" and _form_lower(c) == pat.det_form
            for c in node.children
        )
        if not det_match:
            continue
        return _make(node, pat.subtype, lang)
    return None


def _check_quantifier(node, lex, lang):
    if node.upos != "ADJ":
        return None
    if _form_lower(node) not in lex.quantifier_forms:
        return None
    if not _children_only_in(node, {"det"}):
        return None
    if not _is_core_rel(node):
        return None
    return _make(node, "quantifier", lang)


def _check_numeral(node, lex, lang):
    if node.upos != "NUM":
        return None
    if not _children_only_in(node, {"det", "amod"}):
        return None
    if not _is_core_rel(node):
        return None
    return _make(node, "numeral", lang)


def _check_comparative(node, lex, lang):
    if not node.xpos or node.xpos not in lex.comparative_xpos:
        return None
    if node.deprel == "amod":
        return None
    if not _is_core_rel(node):
        return None
    has_def_art = any(
        c.deprel == "det" and _form_lower(c) in lex.definite_articles
        for c in node.children
    )
    if not has_def_art:
        return None
    return _make(node, "comparative", lang)


def _check_adjective(node, lex, lang):
    if node.upos != "ADJ":
        return None
    if not node.xpos or node.xpos not in lex.adjective_xpos:
        return None
    if node.deprel == "amod":
        return None
    if not _is_core_rel(node):
        return None
    if not _children_only_in(node, {"det"}):
        return None
    if node.parent is None or node.parent.upos != "VERB":
        return None
    return _make(node, "adjective", lang)
