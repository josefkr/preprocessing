"""Wh-word lexicons for sluicing detection, keyed by ISO 639-1 code.

Add a language by adding an entry to ``WH_WORDS_BY_LANG``. Words are
matched case-insensitively against ``node.form``.
"""

from __future__ import annotations

from preprocessing.detection.language import UnsupportedLanguage

WH_WORDS_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset({"who", "what", "why", "when", "how", "whose", "where"}),
    "de": frozenset(
        {
            "wer",
            "wen",
            "wem",
            "wessen",
            "was",
            "wann",
            "warum",
            "wieso",
            "weshalb",
            "wozu",
            "wie",
            "wo",
            "woher",
            "wohin",
            "welcher",
            "welche",
            "welches",
            "welchem",
            "welchen",
        }
    ),
    "fr": frozenset(
        {
            "qui",
            "que",
            "quoi",
            "pourquoi",
            "quand",
            "comment",
            "où",
            "quel",
            "quelle",
            "quels",
            "quelles",
        }
    ),
    "es": frozenset(
        {
            "quién",
            "quiénes",
            "qué",
            "cuándo",
            "cómo",
            "dónde",
            "cuál",
            "cuáles",
            "cuyo",
            "cuya",
            "cuyos",
            "cuyas",
        }
    ),
}


def wh_words(lang: str) -> frozenset[str]:
    """Return the wh-word set for ``lang`` or raise :class:`UnsupportedLanguage`."""
    try:
        return WH_WORDS_BY_LANG[lang]
    except KeyError as e:
        raise UnsupportedLanguage(
            f"sluicing: no wh-word lexicon for language {lang!r}"
        ) from e
