"""German nominal-head ellipsis detector rules.

Companion to ``nominal_ellipsis.py``: that module owns the udapi traversal,
the :class:`NominalEllipsisFinding` dataclass and the entry point
``detect_nominal_ellipsis``, which dispatches per-sentence language and calls
:func:`classify_de` here for German (``# lang = de``) trees.

German needs its own rules: the English detector is built around
Penn-Treebank XPOS (``JJ/JJR/JJS``) and quantifiers being ``upos=ADJ`` —
neither holds for German Stanza output. The signals below were verified
against Stanza's German model on a sample of ASLAN data:

  - quantifier      : ``xpos=PIS``  (substituting indefinite pronoun; the
                      attributive twin is ``PIAT``) minus inherent pronouns
  - possessive pron.: ``xpos=PPOSS`` (substituting possessive; twin ``PPOSAT``)
  - cardinal        : ``upos=NUM`` acting as a head
  - ordinal         : ``xpos=ADJA`` with ``NumType=Ord``
  - comparative     : ``xpos=ADJA`` with ``Degree=Cmp``
  - superlative     : ``xpos=ADJA`` with ``Degree=Sup``, or — since Stanza
                      often drops ``Degree`` on a nominalised superlative
                      (``das älteste``) — a ``-st-`` superlative suffix
  - adjective       : ``xpos=ADJA`` acting as a nominal head (catch-all)
  - demonstrative   : a definite article (``xpos=ART``) immediately followed
                      by a preposition — the head noun is elided
                      (``der [Vater] von Kim``, ``das [Gasthaus] mit ...``)

An elliptical site *acts as a nominal head*. Because elliptical structures
get parser-degraded deprels (``appos``/``obl``/``conj``/``ccomp``), "head"
is a denylist of modifier relations, not the English ``CORE_RELS`` whitelist.

Subtypes are tried from most to least specific; the first match wins. Note that the
``adjective`` catch-all guarantees recall of any ``ADJA``-headed site even
when the finer ordinal/comparative/superlative cue is missing — the specific
checks only refine the subtype label.
"""

from __future__ import annotations

from preprocessing.detection.nominal_ellipsis import NominalEllipsisFinding
from preprocessing.detection.offsets import token_offsets

# Definite-article forms — for the demonstrative-pronoun heuristic.
DEFINITE_ARTICLES = frozenset({"der", "die", "das", "dem", "den", "des"})

# Tagged PIS but *inherently* pronominal — not nominal ellipsis. Everything
# else tagged PIS is treated as quantifier ellipsis (PIS is a closed class).
# These are forms not lemmas.
INHERENT_INDEF_PRONOUNS = frozenset(
    {
        "nichts",
        "etwas",
        "alles",
        "man",
        "jemand",
        "niemand",
        "wer",
        "was",
        "irgendwer",
        "irgendwas",
        "irgendetwas",
        "irgendjemand",
        "niemanden",
        "wen",
        "irgendjemanden",
        "irgendjemandem",
    }
)

# Relations a token bears as a *modifier* inside another phrase rather than
# as a (nominal) head. An ellipsis site acts as a head — but elliptical
# structures get parser-degraded deprels, so this is a denylist, not the
# English CORE_RELS whitelist.
_MODIFIER_RELS = frozenset(
    {
        "det",
        "amod",
        "nummod",
        "case",
        "advmod",
        "compound",
        "flat",
        "fixed",
        "cc",
        "mark",
        "punct",
        "aux",
        "cop",
        "expl",
    }
)

# Inflected German superlative endings (stem + ``-st-`` + agreement).
# This is not perfect probably but a good enough approximation for now.
_SUPERLATIVE_SUFFIXES = ("ste", "sten", "stes", "stem", "ster")


def _form_lower(node) -> str:
    return (node.form or "").lower()


def _feat(node, key: str) -> str:
    """Value of a UD morphological feature, or '' if absent."""
    return node.feats[key]


def _is_head_like(node) -> bool:
    """True if the token acts as a nominal head rather than a pre-nominal
    modifier — the structural hallmark of an elided-head site."""
    return node.deprel not in _MODIFIER_RELS


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


# --- per-subtype checks (most specific first) ------------------------------


def _check_demonstrative(node):
    """Definite article immediately followed by a preposition — the head
    noun is elided ("der [Vater] von Kim", "das [Gasthaus] mit ...")."""
    if node.xpos != "ART" or _form_lower(node) not in DEFINITE_ARTICLES:
        return None
    if node.deprel == "det":  # a genuine pre-nominal article
        return None
    nxt = node.next_node
    if nxt is None or nxt.upos != "ADP" or nxt.deprel != "case":
        return None
    return "demonstrative_pronoun"


def _check_possessive(node):
    """Substituting possessive pronoun (STTS PPOSS: 'seiner', 'meiner');
    the attributive twin is PPOSAT ('sein Blick')."""
    if node.xpos != "PPOSS" or not _is_head_like(node):
        return None
    return "possessive_pronoun"


def _check_quantifier(node):
    """Substituting indefinite pronoun (STTS PIS: 'viele', 'wenige',
    'einige', 'manche', 'jedes', 'keiner'); the attributive twin is PIAT.
    Inherent indefinite pronouns ('nichts', 'etwas', ...) are excluded."""
    if node.xpos != "PIS" or not _is_head_like(node):
        return None
    if _form_lower(node) in INHERENT_INDEF_PRONOUNS:
        return None
    return "quantifier"


def _check_cardinal(node):
    """Bare cardinal numeral acting as a head ("Paul vier [Meilen]"). The
    attributive twin is a 'nummod' modifier and is excluded by _is_head_like."""
    if node.upos != "NUM" or not _is_head_like(node):
        return None
    if _feat(node, "NumType") == "Ord":  # ordinals are handled via ADJA
        return None
    # A numeral heading a copula clause ("das Kind ist acht") is a predicate
    # (an age, a count), not an elided-head NP — exclude it.
    if any(c.deprel == "cop" for c in node.children):
        return None
    return "cardinal"


def _check_ordinal(node):
    """Ordinal: attributive-adjective tag (ADJA) with NumType=Ord acting as
    a head ("nach dem zweiten [Teil]")."""
    if node.xpos != "ADJA" or not _is_head_like(node):
        return None
    if _feat(node, "NumType") != "Ord":
        return None
    return "ordinal"


def _check_comparative(node):
    """Comparative: ADJA with Degree=Cmp acting as a head
    ("das neuere [Paar]")."""
    if node.xpos != "ADJA" or not _is_head_like(node):
        return None
    if _feat(node, "Degree") != "Cmp":
        return None
    return "comparative"


def _check_superlative(node):
    """Superlative: ADJA with Degree=Sup acting as a head. Stanza's German
    model frequently drops Degree on a nominalised superlative ('das
    älteste'), so fall back to the inflected '-st-' suffix when Degree is
    absent. Best-effort: a missed superlative is still caught as 'adjective'."""
    if node.xpos != "ADJA" or not _is_head_like(node):
        return None
    degree = _feat(node, "Degree")
    if degree == "Sup":
        return "superlative"
    if degree == "" and _form_lower(node).endswith(_SUPERLATIVE_SUFFIXES):
        return "superlative"
    return None


def _check_adjective(node):
    """Catch-all: an attributive-adjective tag (ADJA) acting as a nominal
    head, once ordinal/comparative/superlative are ruled out ("der große
    [Gorilla]", "die veganen [Produkte]"). A normal attributive adjective is
    an 'amod' modifier and is excluded by _is_head_like."""
    if node.xpos != "ADJA" or not _is_head_like(node):
        return None
    return "adjective"


_CHECKS = (
    _check_demonstrative,
    _check_possessive,
    _check_quantifier,
    _check_cardinal,
    _check_ordinal,
    _check_comparative,
    _check_superlative,
    _check_adjective,
)


def classify_de(node, lang: str = "de"):
    """Classify a single udapi node as a German nominal-ellipsis site, or
    return ``None``. Subtypes are tried most-specific first; first match wins."""
    for check in _CHECKS:
        subtype = check(node)
        if subtype is not None:
            return _make(node, subtype, lang)
    return None
