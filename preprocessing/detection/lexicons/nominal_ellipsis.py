"""Per-language lexicon for nominal-head ellipsis detection.

Holds the language-specific vocabulary and tag sets that drive the
detector's seven subtypes:

  - ``quantifier``: bare quantifier-like ADJ (en: many/several/few).
  - ``none``: bare ``none``.
  - ``numeral``: bare NUM (e.g. "He lost two").
  - ``every_one``: idiomatic ``every one`` pattern.
  - ``elder``: idiomatic ``the elder`` pattern.
  - ``comparative``: comparative/superlative ADJ with definite article.
  - ``adjective``: bare base-form ADJ used as a core argument.

Add a language by adding an entry to ``LEXICONS_BY_LANG``.
"""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage


@dataclass(frozen=True)
class FixedPattern:
    """An idiomatic ``(det, head)`` pair that triggers nominal ellipsis.

    Matches a token whose ``form`` (lowercased) equals ``head_form`` and
    that has a child with ``deprel == "det"`` whose form equals
    ``det_form``. The head's own ``deprel`` must not be in
    ``excluded_deprels`` (typically rules out attributive uses).
    """

    head_form: str
    det_form: str
    subtype: str
    excluded_deprels: frozenset[str]


@dataclass(frozen=True)
class NominalEllipsisLexicon:
    quantifier_forms: frozenset[str]
    none_forms: frozenset[str]
    fixed_patterns: tuple[FixedPattern, ...]
    definite_articles: frozenset[str]
    comparative_xpos: frozenset[str]
    adjective_xpos: frozenset[str]


LEXICONS_BY_LANG: dict[str, NominalEllipsisLexicon] = {
    "en": NominalEllipsisLexicon(
        quantifier_forms=frozenset({"many", "several", "few"}),
        none_forms=frozenset({"none"}),
        fixed_patterns=(
            FixedPattern(
                head_form="one",
                det_form="every",
                subtype="every_one",
                excluded_deprels=frozenset({"amod", "nummod"}),
            ),
            FixedPattern(
                head_form="elder",
                det_form="the",
                subtype="elder",
                excluded_deprels=frozenset({"amod"}),
            ),
        ),
        definite_articles=frozenset({"the"}),
        comparative_xpos=frozenset({"JJR", "JJS"}),
        adjective_xpos=frozenset({"JJ"}),
    ),
}


def nominal_ellipsis_lexicon(lang: str) -> NominalEllipsisLexicon:
    """Return the lexicon for ``lang`` or raise :class:`UnsupportedLanguage`."""
    try:
        return LEXICONS_BY_LANG[lang]
    except KeyError as e:
        raise UnsupportedLanguage(
            f"nominal_ellipsis: no lexicon for language {lang!r}"
        ) from e
