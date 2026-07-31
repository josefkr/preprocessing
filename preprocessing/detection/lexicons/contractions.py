"""Per-language lexicon for contraction / clitic detection (English, German).

Two mechanisms, because parsers represent contractions two ways:

**Clitics** (English ``do``+``n't``, German ``mir``+``'s``). Not multiword
tokens — the tokenizer emits ordinary adjacent tokens with their own offsets, so
expansion is a *lexical* substitution on the clitic, disambiguated by its lemma:

* EN ``'s`` is ``is`` (lemma *be*), ``has`` (lemma *have*) or ``us`` ("let's"),
  and as a **possessive** is not a contraction at all; EN ``'d`` is ``would`` or
  ``had``. Possessive ``'s`` is simply absent from the table, so it is never
  expanded.
* DE ``'s`` is the reduced ``es`` (lemma *es*): "mir's" -> "mir es", "geht's" ->
  "geht es". Meaning-preserving, like the English clitics.

**Prep+article** (German ``vom`` = ``von dem``, ``im`` = ``in dem``, …). These
*are* UD multiword tokens; the parser supplies the expansion, so no lexicon of
expansions is needed. This module only holds the *exception list* of lexicalised
cases where expanding is wrong or unidiomatic — consulted by the normalizer, and
only when prep+article expansion is opted into.
"""

from __future__ import annotations

from preprocessing.detection.language import UnsupportedLanguage

#: Clitic surface forms per language.
CLITICS_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset({"n't", "'m", "'re", "'ve", "'ll", "'s", "'d"}),
    "de": frozenset({"'s"}),
}

#: ``(clitic form, clitic lemma) -> expansion`` per language. The lemma
#: disambiguates; a pair that is absent (notably English possessive ``'s``) is
#: left untouched.
CLITIC_EXPANSIONS_BY_LANG: dict[str, dict[tuple[str, str], str]] = {
    "en": {
        ("n't", "not"): "not",
        ("'m", "be"): "am",
        ("'re", "be"): "are",
        ("'ve", "have"): "have",
        ("'ll", "will"): "will",
        ("'s", "be"): "is",
        ("'s", "have"): "has",
        ("'s", "we"): "us",      # "let's" -> "let us"
        ("'s", "us"): "us",
        ("'d", "would"): "would",
        ("'d", "have"): "had",
    },
    "de": {
        ("'s", "es"): "es",      # "mir's" -> "mir es", "geht's" -> "geht es"
    },
}

#: Hosts whose own form changes under expansion. UD splits English "can't" as
#: "ca" + "n't", so expanding the clitic alone would yield "*ca not*". German
#: clitic hosts do not change ("mir" stays "mir").
HOST_EXPANSIONS_BY_LANG: dict[str, dict[str, str]] = {
    "en": {"ca": "can", "wo": "will", "sha": "shall"},
    "de": {},
}

#: Lexicalised German prep+article contractions where expansion is wrong or
#: unidiomatic, keyed by ``(contraction surface, following word)`` (both
#: lower-cased). Consulted by the normalizer only when prep+article expansion is
#: enabled. **Deliberately conservative and incomplete**: many prep+article
#: cases are genuinely ambiguous between an idiomatic and a compositional
#: reading ("vom Kuchen" / "von dem Kuchen" are both fine), so only clearly
#: non-compositional cases belong here.
PREP_ARTICLE_EXCEPTIONS_BY_LANG: dict[str, frozenset[tuple[str, str]]] = {
    "de": frozenset({
        ("zum", "beispiel"),
        ("zum", "glück"),
        ("zum", "teil"),
        ("im", "allgemeinen"),
        ("im", "grunde"),
        ("im", "falle"),
        ("am", "besten"),
        ("am", "ehesten"),
        ("am", "meisten"),
    }),
}


#: Clipped indefinite-article forms per language → their full form. These are
#: *standalone* tokens (not clitics, not MWTs): the colloquial German article is
#: written with its leading "ei" dropped ("nen Krampf" = "einen Krampf"). Keyed
#: on the surface (a leading apostrophe is stripped before lookup), because
#: Stanza does **not** lemmatise these to "ein" — it guesses junk lemmas ("nie",
#: "nit", "'nerer") — so lemma is useless and only the surface is reliable.
#:
#: Scope note: ``nen``/``nem``/``ner`` are distinctive (not otherwise German
#: words) and their full form is fully determined by the surface — the detector
#: expands them unconditionally. ``ne`` (=eine) is also here but the detector
#: only treats it as an article when a noun phrase follows (otherwise it is the
#: tag question "ne?"). Bare ``n`` is deliberately absent — its expansion is
#: context-dependent ("n bisschen"→ein vs "n Kleinwagen"→einen) — and handled
#: separately.
CLIPPED_ARTICLES_BY_LANG: dict[str, dict[str, str]] = {
    "de": {"nen": "einen", "nem": "einem", "ner": "einer", "ne": "eine"},
}

#: Clipped forms that are ambiguous with a non-article word and so require the
#: detector's right-context noun-phrase check ("ne" = *eine* vs the tag "ne?").
CLIPPED_ARTICLES_NEED_NP_CONTEXT: dict[str, frozenset[str]] = {
    "de": frozenset({"ne"}),
}


def clipped_article_needs_np(lang: str, surface: str) -> bool:
    """True if this clipped form is only an article when followed by an NP."""
    return surface.lstrip("'").lower() in CLIPPED_ARTICLES_NEED_NP_CONTEXT.get(
        lang, frozenset()
    )


#: Maximally-reduced clipped articles whose full form is **not** determined by
#: the surface and must be inferred from the following noun's morphology
#: ("n bisschen" → ein, "n Kleinwagen" → einen). Per language.
CLIPPED_ARTICLES_INFERRED_BY_LANG: dict[str, frozenset[str]] = {
    "de": frozenset({"n"}),
}

#: German indefinite-article paradigm, keyed by (Gender, Case) for the singular.
#: There is no indefinite plural article, so Number=Plur has no form.
EIN_PARADIGM: dict[tuple[str, str], str] = {
    ("Masc", "Nom"): "ein", ("Masc", "Acc"): "einen",
    ("Masc", "Dat"): "einem", ("Masc", "Gen"): "eines",
    ("Neut", "Nom"): "ein", ("Neut", "Acc"): "ein",
    ("Neut", "Dat"): "einem", ("Neut", "Gen"): "eines",
    ("Fem", "Nom"): "eine", ("Fem", "Acc"): "eine",
    ("Fem", "Dat"): "einer", ("Fem", "Gen"): "einer",
}


def clipped_article_is_inferred(lang: str, surface: str) -> bool:
    """True if this clipped form's full form must be inferred from morphology."""
    return surface.lstrip("'").lower() in CLIPPED_ARTICLES_INFERRED_BY_LANG.get(
        lang, frozenset()
    )


def inflect_indefinite_article(
    lang: str, gender: str | None, case: str | None, number: str | None
) -> str | None:
    """Full indefinite article for the given morphology, or ``None`` when it
    can't be determined (missing gender/case, or plural — no indefinite plural
    article). Only the singular paradigm exists.

    ⚠ Accuracy is bounded by the parser's morphology: German **case** is often
    mis-assigned (accusative/dative syncretism), so this can return the wrong
    form. Callers should treat it as best-effort.
    """
    if lang != "de":
        return None
    if number == "Plur":
        return None
    if not gender or not case:
        return None
    return EIN_PARADIGM.get((gender, case))


def clipped_article_expansion(lang: str, surface: str) -> str | None:
    """Full indefinite article for a clipped surface form ("nen" -> "einen",
    "'nem" -> "einem"), or ``None`` if it isn't a known clipped article."""
    table = CLIPPED_ARTICLES_BY_LANG.get(lang)
    if not table:
        return None
    return table.get(surface.lstrip("'").lower())


def contraction_clitics(lang: str) -> frozenset[str]:
    """Clitic forms for ``lang``, or raise :class:`UnsupportedLanguage`."""
    try:
        return CLITICS_BY_LANG[lang]
    except KeyError as e:
        raise UnsupportedLanguage(
            f"contractions: no clitic lexicon for language {lang!r}"
        ) from e


def clitic_expansion(lang: str, form: str, lemma: str) -> str | None:
    """Expansion for a clitic, or ``None`` when this form/lemma pair is not an
    expandable contraction (e.g. English possessive ``'s``)."""
    table = CLITIC_EXPANSIONS_BY_LANG.get(lang)
    if table is None:
        raise UnsupportedLanguage(
            f"contractions: no clitic lexicon for language {lang!r}"
        )
    return table.get((form.lower(), (lemma or "").lower()))


#: Colloquial **clipped forms**: single written words that stand for a
#: multi-word sequence ("gonna" = going to, "lemme" = let me). A fourth
#: mechanism, distinct from clitics and from the German clipped articles:
#:
#: * they are not apostrophe clitics, so the (form, lemma) clitic table can't
#:   reach them, and Stanza's lemma is usually just the surface;
#: * the tokenizer is **inconsistent** about them — some stay one token
#:   ("kinda", "lemme", "'em"), others are split ("gonna" -> gon+na,
#:   "sorta" -> sort+a, "lotsa" -> lot+sa). The detector therefore matches the
#:   whole *written word*: one token, or an adjacent token sequence whose joined
#:   surface is a key here. Keying on the full word is what makes this safe —
#:   adding "a" or "na" as bare clitics would fire on ordinary words.
#:
#: Deliberately excluded because they are ambiguous with ordinary English or
#: need context we don't model:
#:   * "ain't" — needs subject agreement (I am not / that is not / you are not),
#:     so it is left to the clitic path rather than guessed;
#:   * "canny" — a real adjective ("a canny investor"), only *sometimes* Scots
#:     "cannot";
#:   * "betcha", and "wanna" before a noun ("I wanna cookie" = want *a*), where
#:     the intended expansion isn't recoverable from the surface.
CLIPPED_FORMS_BY_LANG: dict[str, dict[str, str]] = {
    "en": {
        # of-reductions
        "kinda": "kind of", "sorta": "sort of", "typa": "type of",
        "lotsa": "lots of", "outta": "out of", "hella": "hell of",
        # to-reductions
        "gonna": "going to", "wanna": "want to", "gotta": "have got to",
        "hafta": "have to", "tryna": "trying to",
        # have-reductions
        "shoulda": "should have", "coulda": "could have",
        "woulda": "would have", "musta": "must have",
        # verb + pronoun
        "lemme": "let me", "gimme": "give me",
        # whole-clause reductions
        "dunno": "do not know", "iunno": "I do not know",
        "wassup": "what is up", "innit": "is it not", "init": "is it not",
        "twas": "it was", "'twas": "it was", "ima": "I am",
        # lexical clippings
        "y'all": "you all", "ma'am": "madam", "'em": "them",
        "ol'": "old", "a'ight": "alright", "cuz": "because",
        "'cause": "because", "coz": "because",
        # Scots negation
        "shouldnay": "should not", "couldnay": "could not",
        "cannae": "cannot", "dinnae": "do not",
    },
}

#: Longest clipped form measured in tokens, so the detector knows how far to
#: look when joining adjacent tokens ("gon"+"na").
CLIPPED_FORM_MAX_TOKENS = 2


def clipped_form_expansion(lang: str, surface: str) -> str | None:
    """Expansion for a colloquial clipped form ("gonna" -> "going to"), or
    ``None``. Case-insensitive; the caller restores the original casing."""
    table = CLIPPED_FORMS_BY_LANG.get(lang)
    if not table:
        return None
    return table.get(surface.lower())


#: Whole-contraction expansions that are **not** simply
#: ``host + " " + clitic``, keyed by ``(host form, clitic form)`` (lower-cased).
#: English writes *can* + *not* solid, so "can't" expands to the single word
#: "cannot"; the other negated modals stay two words ("won't" → "will not",
#: "shan't" → "shall not", "couldn't" → "could not").
CONTRACTION_OVERRIDES_BY_LANG: dict[str, dict[tuple[str, str], str]] = {
    "en": {
        ("ca", "n't"): "cannot",
    },
}


def contraction_override(lang: str, host: str, clitic: str) -> str | None:
    """Expansion for the whole host+clitic word when it isn't the default
    ``host + " " + clitic`` composition ("can't" -> "cannot"), else ``None``."""
    table = CONTRACTION_OVERRIDES_BY_LANG.get(lang)
    if not table:
        return None
    return table.get(((host or "").lower(), (clitic or "").lower()))


def gdropped_expansion(lang: str, surface: str) -> str | None:
    """Restore a *g-dropped* participle written with an elision apostrophe
    ("talkin'" → "talking", "nothin'" → "nothing"), else ``None``.

    Only the apostrophe-marked spelling is handled here, and deliberately so:

    * the trailing ``'`` explicitly marks the dropped *g*, and no standard
      English word ends in ``in'``, so this is a safe productive rule — it does
      not need a dictionary, and the tokenizer keeps a closing quote as its own
      token ("the cabin'" never arises), so quotes can't trigger it;
    * it can't rely on the parse: Stanza tags these inconsistently
      ("stayin'" → VBG/*stay*, but "talkin'" → ADJ and "sayin" → SCONJ), and
      the lemma is often just the surface;
    * the **bare** spelling without an apostrophe ("stickin", "sayin") is left
      to :class:`PySpellCheckerNormalizer`, which already resolves it correctly
      via its dictionary while leaving real ``-in`` words (coin, cabin, within,
      begin, ruin, thin, protein, robin) untouched. A blind ``-in`` → ``-ing``
      rule here would corrupt those.
    """
    if lang != "en":
        return None
    s = surface or ""
    if len(s) <= 3 or not s.lower().endswith("in'"):
        return None
    stem = s[:-3]
    if not stem[-1:].isalpha():
        return None
    return stem + ("ING" if s[-3:-1].isupper() else "ing")


def clitic_expansion_in_context(
    lang: str,
    form: str,
    lemma: str,
    next_xpos: str | None = None,
    next_lemma: str | None = None,
) -> str | None:
    """Expansion for a clitic, using the following participle to settle cases
    the lemma cannot.

    Stanza lemmatises the perfect auxiliaries ``'s``/``'d`` as *be*/*would*, so
    the lemma table alone yields the ungrammatical "he is been" / "there would
    been". Only **provably safe** overrides are applied here — ones whose
    alternative is not grammatical English:

    * ``'d`` + past participle → *had*. ``would`` is always followed by a bare
      infinitive, never a participle ("there'd been" → "there had been").
    * ``'s`` + ``been`` → *has*. "is been" is never grammatical
      ("he's been lucky" → "he has been lucky").

    ``'s`` before any **other** participle is left to the lemma table, because
    it is genuinely ambiguous between the perfect ("he's eaten") and the passive
    ("the job's finished") and the parser's ``aux``/``aux:pass`` labels are not
    reliable enough to separate them.

    ``next_xpos``/``next_lemma`` describe the first following token that is not
    an adverb or particle, so intervening negation ("he's not been well") and
    adverbs ("he's already been") don't hide the participle.
    """
    f = (form or "").lower()
    if lang == "en" and next_xpos == "VBN":
        if f == "'d":
            return "had"
        if f == "'s" and (next_lemma or "").lower() == "be":
            return "has"
    return clitic_expansion(lang, form, lemma)


def host_expansion_in_context(
    lang: str,
    form: str,
    subject_lemma: str | None = None,
    subject_number: str | None = None,
) -> str | None:
    """Replacement for an irregular clitic host, using the subject where the
    host's own form doesn't determine it.

    ``ain't`` is tokenised ``ai`` + ``n't``, and ``ai`` stands for whichever
    form of *be* the subject requires — "I ain't" → *am*, "that ain't" → *is*,
    "you/we/they ain't" → *are*. Without this the rewrite would emit the
    ungrammatical "that ai not". Everything else falls back to the fixed table.
    """
    if lang == "en" and (form or "").lower() == "ai":
        subj = (subject_lemma or "").lower()
        if subj == "i":
            return "am"
        if subj in ("you", "we", "they") or subject_number == "Plur":
            return "are"
        return "is"
    return host_expansion(lang, form)


def host_expansion(lang: str, form: str) -> str | None:
    """Replacement for an irregular host form ("ca" -> "can"), else ``None``."""
    return HOST_EXPANSIONS_BY_LANG.get(lang, {}).get(form.lower())


def is_lexicalised_prep_article(lang: str, surface: str, following: str | None) -> bool:
    """True for a prep+article contraction that should not be expanded
    (idiomatic), given the immediately following word."""
    if following is None:
        return False
    exceptions = PREP_ARTICLE_EXCEPTIONS_BY_LANG.get(lang, frozenset())
    return (surface.lower(), following.lower()) in exceptions
