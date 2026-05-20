"""Lexicons for sluicing detection, keyed by ISO 639-1 code.

Two lexicons:
  * ``WH_WORDS_BY_LANG`` — the wh-words a sluice remnant can consist of.
  * ``EMBEDDING_PREDICATES_BY_LANG`` — predicate lemmas that embed an
    indirect question. The detector uses these to license a *broadened*
    relation gate: an elliptical sluice often parse-degrades so the
    remnant is no longer a clean ``ccomp`` of its governor (it lands on
    ``obj``/``conj``/``mark``/...); accepting those relations is only
    safe when the governor is a known question-embedding predicate.

Add a language by adding entries to the two maps. Words are matched
case-insensitively against ``node.form`` (wh-words) and against
``node.lemma`` / ``node.form`` (embedding predicates).
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
            "wie",
            "wo",
            "woher",
            "wohin",
            "welcher",
            "welche",
            "welches",
            "welchem",
            "welchen",
            # productive "wo(r)+P" interrogative R-pronouns
            "wozu",
            "wodurch",
            "womit",
            "wofür",
            "woran",
            "worauf",
            "worüber",
            "wovon",
            "wonach",
            "worin",
            "wobei",
            "worum",
            "wovor",
            "wogegen",
            "woraus",
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


# Question-embedding predicate lemmas (matched case-insensitively against
# both ``node.lemma`` and ``node.form``). Deliberately conservative: verbs
# that genuinely take an indirect question. Verbs like ``wollen`` / ``meinen``
# / ``sein`` are excluded so a full embedded clause whose wh-word happens to
# land on a broadened relation is not mistaken for a sluice.
EMBEDDING_PREDICATES_BY_LANG: dict[str, frozenset[str]] = {
    "de": frozenset(
        {
            "abfragen",
            "abklären",
            "ablesen",
            "abstimmen",
            "abwägen",
            "ahnen",
            "auslosen",
            "befragen",
            "bestimmen",
            "entscheiden",
            "erfahren",
            "erinnern",
            "erklären",
            "erörtern",
            "festlegen",
            "fragen",
            "herausfinden",
            "klären",
            "melden",
            "mitteilen",
            "nachvollziehen",
            "raten",
            "sagen",
            "vergessen",
            "verraten",
            "verstehen",
            "weiss",  # non-standard form for when lemmatization goes wrong 
            "wissen",
            "witttern",
            "zeigen",
            "überlegen",
        }
    ),
    "en": frozenset(
        {
            "decide",
            "tell",
            "weigh",
            "settle",
            "determine",
            "remember",
            "explain",
            "deliberate",
            "ask",
            "report",
            "understand",
            "disclose",
            "know",
            "show",
        }
    ),
    
}

# Nouns that can head an embedded question (e.g. "no idea why", "die Frage
# warum"). Used by the detector's nominal relation gate (acl/nmod) and also
# admitted into the broadened verbal gate so e.g. "what" attached as ``obj``
# of "idea" is recognised. Matched case-insensitively against ``node.lemma``
# and ``node.form``.
EMBEDDING_NOUNS_BY_LANG: dict[str, frozenset[str]] = {
    "de": frozenset(
        {
            "ahnung",
            "frage",
            "grund",
            "erklärung",
            "entscheidung",
            "idee",
        }
    ),
    "en": frozenset(
        {
            "clue",
            "guess",
            "idea",
            "question",
            "reason",
            "explanation",
        }
    ),
}

# TODO consult **List of clause embedding predicates from ZAS/IDS**
# Verbal predicates that can embed questions under negation: nicht glauben
#  Adjectival predicates not yet covered: überrascht.a ;
# Multiword expressions not yet covered:
# Gedanken machen.mwe ; begreiflich machen.mwe
#
# reflexives
# s. vorstellen


def embedding_predicates(lang: str) -> frozenset[str]:
    """Question-embedding predicate lemmas/forms for ``lang``.

    Returns an empty set for languages without an entry — the broadened
    relation gate then simply never fires for that language, leaving the
    strict ``ccomp``/``advmod`` detection unchanged."""
    return EMBEDDING_PREDICATES_BY_LANG.get(lang, frozenset())


def embedding_nouns(lang: str) -> frozenset[str]:
    """Question-embedding noun lemmas/forms for ``lang``.

    Returns an empty set for languages without an entry — the nominal
    relation gate then simply never fires for that language."""
    return EMBEDDING_NOUNS_BY_LANG.get(lang, frozenset())
