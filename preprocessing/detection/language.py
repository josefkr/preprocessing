"""Language detection for detection pipelines.

Wraps ``lingua-language-detector``. Returns ISO 639-1 codes restricted
to the project's supported set ({en, de, fr, es}). Detection results
below the confidence threshold are reported as ``None`` so callers
can decide to skip rather than guess.
"""

from __future__ import annotations

import logging
import re

from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

SUPPORTED_LANGS: frozenset[str] = frozenset({"en", "de", "fr", "es"})
CONFIDENCE_THRESHOLD: float = 0.7

_LANG_TO_LINGUA: dict[str, Language] = {
    "en": Language.ENGLISH,
    "de": Language.GERMAN,
    "fr": Language.FRENCH,
    "es": Language.SPANISH,
}
_LINGUA_TO_LANG: dict[Language, str] = {v: k for k, v in _LANG_TO_LINGUA.items()}


class UnsupportedLanguage(Exception):
    """Raised when a detector has no resources for the requested language."""


_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(
            *_LANG_TO_LINGUA.values()
        ).build()
    return _detector


def detect_language(text: str) -> str | None:
    """Return ISO 639-1 code of the most likely language, or ``None``.

    ``None`` means the text was empty or detection confidence was below
    :data:`CONFIDENCE_THRESHOLD` — caller should skip rather than guess.
    """
    if not text or not text.strip():
        return None
    confs = _get_detector().compute_language_confidence_values(text)
    if not confs:
        return None
    top = confs[0]
    if top.value < CONFIDENCE_THRESHOLD:
        logger.debug(
            f"language detection low confidence ({top.value:.2f}); "
            f"sample={text[:60]!r}"
        )
        return None
    return _LINGUA_TO_LANG.get(top.language)


_LANG_RE = re.compile(r"^\s*lang\s*=\s*(\S+)\s*$", re.MULTILINE)


def tree_lang(tree) -> str | None:
    """Return the ISO code from a udapi tree's ``# lang =`` comment, or None."""
    if not tree.comment:
        return None
    m = _LANG_RE.search(tree.comment)
    return m.group(1) if m else None
