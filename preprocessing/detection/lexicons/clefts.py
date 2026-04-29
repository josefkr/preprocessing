"""Per-language lexicon for cleft detection.

Holds the language-specific lemmas that drive cleft detection. For
English these are the copula ``be`` and the cleft pronoun ``it``;
other languages will plug in their own (e.g. German ``sein`` / ``es``,
French ``être`` / ``ce``).
"""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage


@dataclass(frozen=True)
class CleftLexicon:
    copula_lemmas: frozenset[str]
    it_lemmas: frozenset[str]
    wh_lemmas: frozenset[str]


CLEFT_LEXICONS_BY_LANG: dict[str, CleftLexicon] = {
    "en": CleftLexicon(
        copula_lemmas=frozenset({"be"}),
        it_lemmas=frozenset({"it"}),
        wh_lemmas=frozenset({"what"}),
    ),
}


def clefts_lexicon(lang: str) -> CleftLexicon:
    """Return the cleft lexicon for ``lang`` or raise :class:`UnsupportedLanguage`."""
    try:
        return CLEFT_LEXICONS_BY_LANG[lang]
    except KeyError as e:
        raise UnsupportedLanguage(
            f"clefts: no lexicon for language {lang!r}"
        ) from e
