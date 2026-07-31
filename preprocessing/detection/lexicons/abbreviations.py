"""Per-language data for abbreviation detection (German, English).

Detection is deliberately lexicon-light: *which* long form an abbreviation stands
for is a corpus-level decision (see ``resolution/abbreviations/harvest.py``), so
this module only holds what is needed to decide whether an all-caps token is an
abbreviation **candidate** at all.

Three gates, each answering a distinct false-positive class observed in real exam
answers:

* ``FUNCTION_WORDS_BY_LANG`` — an all-caps function word is emphasis, not an
  abbreviation ("Test **VOR** dem Lernen", "**NACH** einem Test"). Note the
  general abbreviation lexicons do not even list *vor*/*nach*, so membership
  already screens most of these; the list matters for the ~139 German all-caps
  abbreviations that *are* listed and collide with a real word (``AN``, ``AM``,
  ``ALS``, ``AB``, ``ALL``).
* ``ENUM_RUN_MIN`` / ``ENUM_SEPARATORS`` — a run of adjacent all-caps tokens
  separated by nothing but punctuation is an enumeration, not a list of
  abbreviations ("in randomisierter Reihenfolge lernen bsp.: ADE - BEC - CBA").
  An LLM asked about these answers "yes, abbreviation" and even invents an
  expansion, so it has to be settled structurally.
* ``GLOSS_RE`` — long forms in the scraped Wiktionary lists that are *glosses*
  rather than substitutable expansions ("Abkürzung für das AFIS-ALKIS-Modell",
  "ohne Plural: Die offizielle Abkürzung von Aargau lautet AG."). Roughly a
  quarter of the German all-caps rows are of this kind.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: Candidate shape: an all-caps letter run. Two-letter forms are included on
#: purpose — a general lexicon is hopeless for them (636 German forms carry
#: ~17k senses) but corpus evidence resolves them well (KG, IT, ID, IR).
CANDIDATE_RE = re.compile(r"\b[A-ZÄÖÜ]{2,6}\b")

#: Closed-class words whose all-caps spelling is emphasis. Kept as an explicit
#: list so the detector needs no morphology; the normalization side additionally
#: consults SMOR, which generalises this to any closed-class reading.
FUNCTION_WORDS_BY_LANG: dict[str, frozenset[str]] = {
    "de": frozenset({
        "ab", "aber", "all", "alle", "als", "also", "am", "an", "auch", "auf",
        "aus", "bei", "beim", "bis", "da", "damit", "dann", "das", "dass",
        "dem", "den", "denn", "der", "des", "die", "dies", "diese", "doch",
        "dort", "du", "durch", "ein", "eine", "er", "es", "etwa", "für",
        "gegen", "hat", "hier", "ich", "ihr", "im", "in", "ist", "ja", "je",
        "kein", "man", "mehr", "mit", "nach", "nicht", "noch", "nun", "nur",
        "ob", "oder", "ohne", "schon", "sehr", "sein", "sie", "so", "sind",
        "über", "um", "und", "uns", "unter", "vom", "von", "vor", "war",
        "was", "weil", "wenn", "wer", "wie", "wir", "wo", "zu", "zum", "zur",
    }),
    "en": frozenset({
        "a", "all", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "he", "her", "his", "if", "in", "into", "is", "it", "its",
        "no", "not", "of", "on", "or", "our", "out", "she", "so", "than",
        "that", "the", "their", "then", "there", "they", "this", "to", "up",
        "us", "was", "we", "were", "what", "when", "which", "who", "with",
        "you", "your",
    }),
}

#: **Lexicalised abbreviations**, expanded from a closed list rather than from
#: corpus or lexicon evidence. This is a second candidate shape entirely: the
#: all-caps rule cannot see ``i.e.``, yet dotted forms are the *most frequent*
#: abbreviation class in English prose. In the MohlerMihalcea answers they outnumber
#: genuine all-caps abbreviations 15 to 3.
#:
#: Membership is deliberately narrow — only forms whose long form is uncontroversial
#: and whose expansion is a normalization rather than a change of register:
#:
#: * ``etc.`` is excluded although it is the single most frequent form here (10
#:   occurrences). "et cetera" is a transliteration, not an expansion, and "and so
#:   on" rewrites the author's phrasing.
#: * Titles (``Dr.``, ``Mr.``, ``Prof.``) are excluded for the same reason: nobody
#:   writes "Doctor Smith" as the normalized form of "Dr. Smith".
#: * ``w/`` and ``w/o`` are included because they appear in the data and, being
#:   solidus forms, they compose correctly inside a word: replacing ``w/`` in
#:   ``w/in`` yields ``within``.
LEXICALISED_BY_LANG: dict[str, dict[str, str]] = {
    "en": {
        "e.g.": "for example",
        "i.e.": "that is",
        "vs.": "versus",
        "vs": "versus",
        "cf.": "compare",
        "w/o": "without",
        "w/": "with",
    },
    "de": {},
}

#: Matches any key of :data:`LEXICALISED_BY_LANG`, longest first so that ``w/o``
#: wins over ``w/`` and ``vs.`` over ``vs``. Built per language on demand.
_LEXICALISED_RE_CACHE: dict[str, re.Pattern] = {}


def lexicalised_pattern(lang: str | None) -> re.Pattern | None:
    """Compiled alternation over the closed list for ``lang``, or ``None``."""
    key = lang or ""
    if key not in _LEXICALISED_RE_CACHE:
        table = LEXICALISED_BY_LANG.get(key) or {}
        if not table:
            _LEXICALISED_RE_CACHE[key] = None
        else:
            # Longest first; a trailing "." is part of the form, so it must not be
            # treated as a word boundary. A leading boundary is still required, to
            # keep "vs" out of "vs" inside "advs".
            alts = "|".join(re.escape(k) for k in
                            sorted(table, key=len, reverse=True))
            _LEXICALISED_RE_CACHE[key] = re.compile(rf"(?<![\w.]) ({alts})".replace(" ", ""),
                                                    re.IGNORECASE)
    return _LEXICALISED_RE_CACHE[key]


def lexicalised_expansion(lang: str | None, form: str) -> str | None:
    """Long form for a lexicalised abbreviation, case-matched to ``form``.

    Sentence-initial ``I.e.`` yields ``That is``; everything else is lower-cased as
    listed. Only the first character is transferred — an all-caps ``I.E.`` is far
    more likely emphasis or a typo than a request for ``THAT IS``.
    """
    table = LEXICALISED_BY_LANG.get(lang or "") or {}
    exp = table.get(form.lower())
    if exp is None:
        return None
    return exp[0].upper() + exp[1:] if form[:1].isupper() else exp


#: Fewest adjacent all-caps tokens that make a run an enumeration.
ENUM_RUN_MIN = 3

#: What may sit between two members of such a run — punctuation and space only.
#: This is what separates an enumeration ("ADE - BEC - CBA") from a genuine
#: series of abbreviations ("test (IT)/rehearsal (IR)/distraction (ID)"), whose
#: members have intervening words.
ENUM_SEPARATOR_RE = re.compile(r"^[\s\-–—,;:/·.()\[\]]*$")

#: A parenthesised all-caps token right after a noun phrase: the abbreviation is
#: glossed, so it must be left alone. Deliberately does **not** require the
#: initials to match — "Konditionierter Reiz (CS)" is *conditioned stimulus*, so
#: its initials against the German gloss are K+R. Missing such a gloss is worse
#: than missing an expansion: it lets a wrong expansion through instead.
GLOSS_CONTEXT_RE = re.compile(
    r"[^\W\d_]{3,}[^()]{0,40}?\s*\(\s*([A-ZÄÖÜ]{2,6})\s*\)"
)

#: Scraped long forms that are definitions rather than substitutable expansions.
GLOSS_RE = re.compile(
    r"Abkürzung für|steht für|Kurzwort für|kurz für|Kurzform|bezeichnet|"
    r"Netzjargon|Unterscheidungszeichen|Kfz-Kennzeichen|Kraftfahrzeugkennzeichen|"
    r"offizielle Abkürzung|ohne Plural|abbreviation of|initialism of|"
    r"short for|:", re.I
)

#: Longest a substitutable expansion may be, in words.
MAX_EXPANSION_WORDS = 6


#: Above this fraction of all-caps word tokens, capitalisation carries no signal.
MAX_CAPS_RATIO = 0.3

#: …but only once there are enough tokens for the ratio to mean anything. Student
#: answers are short (a median of ~60 characters in the Hagen data), so a single
#: abbreviation in a three-word sentence already exceeds 30 % — "Die VP kamen."
#: is 33 % all-caps and is obviously not shouted. Applying the ratio to such texts
#: suppressed legitimate detections.
MIN_TOKENS_FOR_CAPS_RATIO = 10


def caps_suppressed(words: list[str]) -> bool:
    """True if this text's capitalisation is uninformative (mostly-caps text).

    Requires both a minimum token count and a high ratio; see
    :data:`MIN_TOKENS_FOR_CAPS_RATIO`.
    """
    alpha = [w for w in words if w.isalpha()]
    if len(alpha) < MIN_TOKENS_FOR_CAPS_RATIO:
        return False
    caps = sum(1 for w in alpha if w.isupper() and len(w) > 1)
    return caps / len(alpha) > MAX_CAPS_RATIO


def is_function_word(lang: str | None, lower: str) -> bool:
    """True if the lower-cased form is a closed-class word in ``lang``."""
    return lower in FUNCTION_WORDS_BY_LANG.get(lang or "", frozenset())


#: Languages whose emphasis capitalisation is checked morphologically rather than
#: against a function-word list. German gets this from SMOR on the normalization side
#: (any all-closed-class reading is emphasis); English uses the affix-stripping
#: acceptor in ``english_morph``, because its emphasis targets *content* words.
_DICTIONARY_VETO_LANGS = frozenset({"en"})


def _english_words():
    """The English wellformedness oracle (see ``english_morph``). Lazy and optional."""
    from preprocessing.detection.lexicons.english_morph import get_english_words

    return get_english_words()


def is_emphasis(lang: str | None, form: str, *, has_evidence: bool = False,
                has_strong_evidence: bool = False) -> bool:
    """True if an all-caps ``form`` is emphasis rather than an abbreviation.

    Two mechanisms, because the languages differ in what gets shouted:

    * a **function word** in caps, for any language with a list ("Test *VOR* dem
      Lernen"). This is what German needs.
    * an ordinary **English word** in caps -- known or morphologically derivable. Measured on the
      MohlerMihalcea answers, the function-word list catches *none* of the real
      false positives — ``AFTER``, ``WHILE``, ``FIRST``, ``ALWAYS``, ``ANSWER``,
      ``BEFORE``, ``MAIN`` — because English emphasises *content* words. A word list
      catches all seven while leaving ``OOP``, ``OOA`` and ``RUP`` standing.

    Evidence stands the vetoes down, but the two vetoes require **different
    strengths**, and conflating them breaks one language or the other:

    * ``has_evidence`` — anything vouches for the form, *including* a general
      lexicon entry. Enough to override the dictionary check, since genuine acronyms
      that are also words (``CAT``, ``SAD``, ``AIDS``) are exactly what that check
      would otherwise discard.
    * ``has_strong_evidence`` — an in-context definition or a corpus acronym match;
      a lexicon entry does **not** count. Only this overrides the function-word
      veto. It has to be the stricter bar because ~139 German all-caps
      abbreviations are listed in the lexicon *and* collide with a real word
      (``AN``, ``AM``, ``ALS``, ``AB``, ``ALL``); letting lexicon presence through
      would un-veto all of them. Conversely English needs *some* route through,
      because ``IT`` and ``US`` are both function words and among the most common
      English abbreviations — so "the corpus spells it out somewhere" is what earns
      them their reading.

    With no corpus at all both flags are ``False`` and the vetoes are at their most
    conservative, which is the right default for a detector that only annotates.
    """
    lower = form.lower()
    if is_function_word(lang, lower):
        return not has_strong_evidence
    if has_evidence or has_strong_evidence:
        return False
    if (lang or "") not in _DICTIONARY_VETO_LANGS:
        return False
    # `is_wordlike`, not a plain list lookup: English derivations the list has never
    # seen are still ordinary words, and treating them as abbreviation candidates
    # would be wrong. `reusability` occurs in our own English data and is absent from
    # the wordlist; `objectoriented` and `oop` are correctly rejected.
    return _english_words().is_wordlike(lower)


#: The English scrape separates a long form from its explanatory gloss with a double
#: pipe: ``bareback||without a condom``, ``cam-to-cam||direct video communication by
#: webcam``. The German file has none, which is why this went unnoticed — 44 English
#: rows were passing the substitutability test with the gloss still attached, so an
#: expansion could carry "||" into the text.
GLOSS_SEPARATOR = "||"


def substitutable_part(long_form: str) -> str:
    """The part of a scraped long form that can stand in running text.

    Drops an explanatory tail after :data:`GLOSS_SEPARATOR`; returns the input
    otherwise. Separate from :func:`is_substitutable` so a loader can *recover* the
    usable half rather than discard the whole row — of the 44 affected English rows,
    most have a perfectly good long form before the separator.
    """
    lf = (long_form or "").strip()
    head = lf.split(GLOSS_SEPARATOR, 1)[0].strip()
    return head or lf


def is_substitutable(long_form: str) -> bool:
    """True if a scraped long form can be substituted into running text.

    Applied to :func:`substitutable_part`, so a row whose gloss tail makes it look
    too long is judged on its long form alone.
    """
    lf = substitutable_part(long_form)
    if not lf or lf.startswith("("):
        return False
    if GLOSS_RE.search(lf):
        return False
    return len(lf.split()) <= MAX_EXPANSION_WORDS


def enumeration_offsets(text: str) -> set[int]:
    """Start offsets of all-caps tokens belonging to an enumeration run."""
    matches = list(CANDIDATE_RE.finditer(text))
    out: set[int] = set()
    run: list[re.Match] = []

    def flush(group):
        if len(group) >= ENUM_RUN_MIN:
            out.update(m.start() for m in group)

    for m in matches:
        if run and ENUM_SEPARATOR_RE.match(text[run[-1].end():m.start()]):
            run.append(m)
        else:
            flush(run)
            run = [m]
    flush(run)
    return out


def glossed_offsets(text: str, restrict_to: set[str] | None = None) -> set[int]:
    """Start offsets of candidates that sit at their own gloss."""
    out: set[int] = set()
    for m in GLOSS_CONTEXT_RE.finditer(text):
        if restrict_to is None or m.group(1) in restrict_to:
            out.add(m.start(1))
    return out
