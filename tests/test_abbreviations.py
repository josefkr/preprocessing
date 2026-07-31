"""Tests for the pure abbreviation detector and its CAS adapter.

Detection is separate from expansion: which long form an abbreviation stands for
is decided corpus-wide (``resolution/abbreviations/harvest.py``), so the detector
reports candidates with or without suggestions. What is pinned here are the three
gates and the annotation shape.
"""

import pytest
from udapi.core.document import Document

from preprocessing.detection.abbreviations import (
    AbbreviationExpansion,
    detect_abbreviations,
)

T_GA = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly"


def _doc(text: str, rows: list[tuple]) -> Document:
    """rows: (form, upos, begin, end) — only forms/offsets matter here."""
    lines = ["# sent_id = 1", f"# text = {text}", "# lang = de"]
    for i, (form, upos, b, e) in enumerate(rows, 1):
        head = 0 if i == 1 else 1
        dep = "root" if i == 1 else "dep"
        lines.append("\t".join([
            str(i), form, form.lower(), upos, "_", "_", str(head), dep, "_",
            f"t_start={b}|t_end={e}",
        ]))
    doc = Document()
    doc.from_conllu_string("\n".join(lines) + "\n\n")
    return doc


def test_candidate_detected_with_expansion():
    doc = _doc("Die KG war klein.", [
        ("Die", "DET", 0, 3), ("KG", "NOUN", 4, 6),
        ("war", "AUX", 7, 10), ("klein", "ADJ", 11, 16), (".", "PUNCT", 16, 17),
    ])
    exp = {"KG": [AbbreviationExpansion("Kontrollgruppe", "corpus_acronym", 0.9)]}
    f = detect_abbreviations(doc, restrict_to_lang="de", expansions=exp)
    assert len(f) == 1
    assert (f[0].text, f[0].begin, f[0].end) == ("KG", 4, 6)
    assert [e.form for e in f[0].expansions] == ["Kontrollgruppe"]
    assert not f[0].defined_in_context


def test_candidate_reported_without_any_expansion():
    """Standalone annotation is useful before a corpus harvest exists."""
    doc = _doc("Die KG war klein.", [
        ("Die", "DET", 0, 3), ("KG", "NOUN", 4, 6),
        ("war", "AUX", 7, 10), ("klein", "ADJ", 11, 16), (".", "PUNCT", 16, 17),
    ])
    f = detect_abbreviations(doc, restrict_to_lang="de")
    assert len(f) == 1 and f[0].expansions == ()


def test_function_word_in_caps_is_vetoed():
    # "Test VOR dem Lernen" — emphasis capitalisation, not an abbreviation.
    doc = _doc("Test VOR dem Lernen", [
        ("Test", "NOUN", 0, 4), ("VOR", "ADP", 5, 8),
        ("dem", "DET", 9, 12), ("Lernen", "NOUN", 13, 19),
    ])
    assert detect_abbreviations(doc, restrict_to_lang="de") == []


def test_enumeration_run_is_vetoed():
    # "ADE - BEC - CBA" labels learning materials; a punctuation-separated run of
    # all-caps tokens is never a list of abbreviations.
    doc = _doc("Reihenfolge ADE - BEC - CBA", [
        ("Reihenfolge", "NOUN", 0, 11), ("ADE", "NOUN", 12, 15),
        ("-", "PUNCT", 16, 17), ("BEC", "NOUN", 18, 21),
        ("-", "PUNCT", 22, 23), ("CBA", "NOUN", 24, 27),
    ])
    assert detect_abbreviations(doc, restrict_to_lang="de") == []


def test_genuine_series_with_intervening_words_is_kept():
    # "test (IT)/rehearsal (IR)" — words between the members, so not an
    # enumeration. This is the case the enumeration gate must NOT eat.
    # Enough lower-case context that the all-caps ratio stays under the gate,
    # as in a real answer.
    text = "drei Gruppen lernten und zwar test (IT)/rehearsal (IR) danach"
    doc = _doc(text, [
        ("drei", "NUM", 0, 4), ("Gruppen", "NOUN", 5, 12),
        ("lernten", "VERB", 13, 20), ("und", "CCONJ", 21, 24),
        ("zwar", "ADV", 25, 29), ("test", "NOUN", 30, 34),
        ("(", "PUNCT", 35, 36), ("IT", "NOUN", 36, 38), (")", "PUNCT", 38, 39),
        ("/", "PUNCT", 39, 40), ("rehearsal", "NOUN", 40, 49),
        ("(", "PUNCT", 50, 51), ("IR", "NOUN", 51, 53), (")", "PUNCT", 53, 54),
        ("danach", "ADV", 55, 61),
    ])
    got = {f.text for f in detect_abbreviations(doc, restrict_to_lang="de")}
    assert got == {"IT", "IR"}


def test_glossed_occurrence_is_flagged_not_dropped():
    # "Konditionierte Reiz (CS)" — the phenomenon is present but must not be
    # normalized, so it is reported with defined_in_context set.
    doc = _doc("Der Konditionierte Reiz (CS) kam", [
        ("Der", "DET", 0, 3), ("Konditionierte", "ADJ", 4, 18),
        ("Reiz", "NOUN", 19, 23), ("(", "PUNCT", 24, 25),
        ("CS", "NOUN", 25, 27), (")", "PUNCT", 27, 28),
        ("kam", "VERB", 29, 32),
    ])
    f = detect_abbreviations(doc, restrict_to_lang="de")
    assert len(f) == 1 and f[0].text == "CS"
    assert f[0].defined_in_context


def test_restrict_to_lang_filters():
    doc = _doc("Die KG war klein.", [
        ("Die", "DET", 0, 3), ("KG", "NOUN", 4, 6),
        ("war", "AUX", 7, 10), ("klein", "ADJ", 11, 16), (".", "PUNCT", 16, 17),
    ])
    assert detect_abbreviations(doc, restrict_to_lang="en") == []


# --- CAS adapter ------------------------------------------------------------

def _cas(text, rows):
    """A CAS with the layers ``view_to_conllu`` needs: sentences, tokens, POS,
    lemmas and a dependency per token (the first token is the root)."""
    from cassis import Cas
    from py_lift.dkpro import T_DEP, T_LEMMA, T_POS, T_SENT, T_TOKEN
    from py_lift.util import get_lift_typesystem

    ts = get_lift_typesystem()
    cas = Cas(sofa_string=text, document_language="de", typesystem=ts)
    Sent, Tok, POS, Lem, Dep = (
        ts.get_type(t) for t in (T_SENT, T_TOKEN, T_POS, T_LEMMA, T_DEP)
    )
    cas.add(Sent(begin=0, end=len(text)))
    toks = []
    for form, upos, b, e in rows:
        tok = Tok(begin=b, end=e)
        cas.add(tok)
        toks.append(tok)
        cas.add(POS(begin=b, end=e, coarseValue=upos, PosValue=upos))
        cas.add(Lem(begin=b, end=e, value=form.lower()))
    for i, (_f, _u, b, e) in enumerate(rows):
        gov = toks[0]
        cas.add(Dep(begin=b, end=e, Governor=gov, Dependent=toks[i],
                    DependencyType="root" if i == 0 else "dep"))
    return cas


def test_cas_adapter_writes_ranked_suggestions():
    """Every candidate expansion becomes a SuggestedAction, so ambiguity survives
    into the annotation instead of being collapsed at write time."""
    from preprocessing.detection.cas_adapter import find_and_annotate

    text = "Die VP kamen."
    cas = _cas(text, [("Die", "DET", 0, 3), ("VP", "NOUN", 4, 6),
                      ("kamen", "VERB", 7, 12), (".", "PUNCT", 12, 13)])
    assert find_and_annotate(cas, cas.typesystem,
                             phenomenon="abbreviation", lang="de") == 1
    (ga,) = list(cas.select(T_GA))
    assert (ga.begin, ga.end, ga.category) == (4, 6, "abbreviation")
    # no corpus lexicon was supplied, so there are no suggestions yet
    assert list(ga.suggestions.elements) == []


def test_annotated_cas_serialises_to_xmi():
    """Regression: `suggestions` must be an FSArray — a bare value only explodes
    at serialisation time."""
    from cassis import load_cas_from_xmi
    from py_lift.util import get_lift_typesystem

    from preprocessing.detection.cas_adapter import find_and_annotate

    text = "Die KG kam."
    cas = _cas(text, [("Die", "DET", 0, 3), ("KG", "NOUN", 4, 6),
                      ("kam", "VERB", 7, 10), (".", "PUNCT", 10, 11)])
    find_and_annotate(cas, cas.typesystem, phenomenon="abbreviation", lang="de")
    xmi = cas.to_xmi()
    reloaded = load_cas_from_xmi(xmi, typesystem=get_lift_typesystem())
    assert len(list(reloaded.select(T_GA))) == 1


def test_registered_for_annotate_cli_and_pylift():
    from preprocessing.detection.cas_adapter import DETECTOR_REGISTRY
    from preprocessing.detection.lift_annotators import ANNOTATORS

    assert "abbreviation" in DETECTOR_REGISTRY
    assert "abbreviation" in ANNOTATORS


# --- English: emphasis veto, closed list, lexicon hygiene -----------------

class TestEnglishEmphasisVeto:
    """English emphasises *content* words, so the function-word list is not enough.

    Measured on the MohlerMihalcea answers, the candidates are dominated by emphasis
    capitalisation — AFTER, WHILE, FIRST, ALWAYS, ANSWER, BEFORE, MAIN — and the
    function-word list catches none of them. A dictionary check catches all seven
    while leaving OOP, OOA and RUP standing.
    """

    @pytest.mark.parametrize("form", ["MAIN", "AFTER", "WHILE", "FIRST",
                                      "ALWAYS", "ANSWER", "BEFORE"])
    def test_ordinary_words_are_emphasis(self, form):
        from preprocessing.detection.lexicons.abbreviations import is_emphasis
        assert is_emphasis("en", form) is True

    @pytest.mark.parametrize("form", ["OOP", "OOA", "RUP"])
    def test_genuine_abbreviations_survive(self, form):
        from preprocessing.detection.lexicons.abbreviations import is_emphasis
        assert is_emphasis("en", form) is False

    def test_lexicon_evidence_rescues_a_word_like_acronym(self):
        """CAT/SAD/AIDS are words *and* acronyms; a bare dictionary veto loses them."""
        from preprocessing.detection.lexicons.abbreviations import is_emphasis
        assert is_emphasis("en", "CAT") is True
        assert is_emphasis("en", "CAT", has_evidence=True) is False

    def test_function_words_need_strong_evidence(self):
        """IT and US are function words *and* common abbreviations.

        A lexicon entry is not enough — the corpus has to spell it out — because ~139
        German all-caps abbreviations are lexicon-listed *and* collide with a real
        word, and letting lexicon presence through would un-veto all of them.
        """
        from preprocessing.detection.lexicons.abbreviations import is_emphasis
        assert is_emphasis("en", "IT") is True
        assert is_emphasis("en", "IT", has_evidence=True) is True
        assert is_emphasis("en", "IT", has_strong_evidence=True) is False

    def test_german_is_unaffected(self):
        """The German inventory must not shift: AN/AM stay vetoed on lexicon alone."""
        from preprocessing.detection.lexicons.abbreviations import is_emphasis
        assert is_emphasis("de", "VOR") is True
        assert is_emphasis("de", "AN", has_evidence=True) is True
        assert is_emphasis("de", "KG", has_strong_evidence=True) is False
        # VP is not a German function word, so the dictionary path never applies.
        assert is_emphasis("de", "VP", has_evidence=True) is False


class TestLexicalisedClosedList:
    """Dotted/solidus forms: a second candidate shape, needing no corpus.

    They are the *most frequent* abbreviation class in English prose — 15 occurrences
    against 3 genuine all-caps abbreviations in this dataset.
    """

    def test_expands_listed_forms(self):
        from preprocessing.detection.lexicons.abbreviations import (
            lexicalised_expansion,
        )
        assert lexicalised_expansion("en", "i.e.") == "that is"
        assert lexicalised_expansion("en", "e.g.") == "for example"
        assert lexicalised_expansion("en", "vs.") == "versus"

    def test_sentence_initial_case_is_transferred(self):
        from preprocessing.detection.lexicons.abbreviations import (
            lexicalised_expansion,
        )
        assert lexicalised_expansion("en", "I.e.") == "That is"

    def test_etc_and_titles_are_excluded_on_purpose(self):
        """`etc.` is the most frequent form here and is still left alone.

        "et cetera" is a transliteration and "and so on" rewrites the author's
        phrasing — neither is a normalization.
        """
        from preprocessing.detection.lexicons.abbreviations import (
            lexicalised_expansion,
        )
        assert lexicalised_expansion("en", "etc.") is None
        assert lexicalised_expansion("en", "Dr.") is None

    def test_solidus_forms_compose_inside_a_word(self):
        """Replacing `w/` in `w/in` must yield `within`, and `w/o` must win over `w/`."""
        from preprocessing.detection.lexicons.abbreviations import (
            lexicalised_expansion,
            lexicalised_pattern,
        )
        pat = lexicalised_pattern("en")
        text = "a module w/in a program, w/o errors"
        got = [(m.group(0), lexicalised_expansion("en", m.group(0)))
               for m in pat.finditer(text)]
        assert got == [("w/", "with"), ("w/o", "without")]

    def test_no_false_match_inside_words(self):
        from preprocessing.detection.lexicons.abbreviations import lexicalised_pattern
        assert [m.group(0) for m in lexicalised_pattern("en").finditer("advs novs")] == []

    def test_german_has_no_list(self):
        from preprocessing.detection.lexicons.abbreviations import lexicalised_pattern
        assert lexicalised_pattern("de") is None


class TestGlossSeparator:
    """The English scrape uses `||` to separate a long form from its gloss.

    44 rows were passing substitutability with the gloss attached, so an expansion
    could carry "||" into the text. The German file contains none, which is why this
    went unnoticed.
    """

    def test_gloss_tail_is_dropped(self):
        from preprocessing.detection.lexicons.abbreviations import substitutable_part
        assert substitutable_part("bareback||without a condom") == "bareback"

    def test_form_without_a_gloss_is_untouched(self):
        from preprocessing.detection.lexicons.abbreviations import substitutable_part
        assert substitutable_part("object-oriented programming") == \
            "object-oriented programming"

    def test_long_form_recovered_from_an_over_long_row(self):
        """Judged on the long form alone, this row is usable; with the gloss it was
        10 words and discarded by the length cap."""
        from preprocessing.detection.lexicons.abbreviations import (
            is_substitutable,
            substitutable_part,
        )
        row = "order of play||the schedule of contests in a tennis event"
        assert is_substitutable(row) is True
        assert substitutable_part(row) == "order of play"


class TestEnglishWellformednessAcceptor:
    """Affix stripping over a wordlist — the English stand-in for SMOR's *rejection*.

    SMOR's value as a gate is that `analyse("Unkt")` is empty. A plain English wordlist
    cannot do that job because it misses ordinary derivations: `reusability`,
    `debuggability`, `subclassable` and `decompounding` are all absent, and
    `reusability` occurs in our own English answer data. Two neural tools were
    evaluated for this (CompoundPiece, PaReNT) and rejected because both are
    transducers that always emit an answer, so neither can reject.
    """

    @staticmethod
    def _w():
        from preprocessing.detection.lexicons.english_morph import get_english_words
        return get_english_words()

    @pytest.mark.parametrize("word", [
        "reusability",      # -ability -> reusable -> reuse   (two peels)
        "debuggability",    # needs consonant undoubling
        "subclassable",     # sub- prefix + -able
        "decompounding",    # de- prefix + -ing
        "unfriendable",     # un- prefix + -able
        "re-usability",     # hyphen is a word boundary
    ])
    def test_accepts_derivations_absent_from_the_wordlist(self, word):
        w = self._w()
        if not w.available:
            pytest.skip("English wordlist unavailable")
        assert w.knows(word) is False, "fixture assumes this is not listed verbatim"
        assert w.is_wordlike(word) is True

    @pytest.mark.parametrize("word", [
        "objectoriented",   # a run-together phrase, not a word
        "Unkt",             # what the German spell gate produces from "Unit"
        "oop", "ooa", "rup",  # the genuine abbreviations must NOT look wordlike
        "re-xyzzy",         # every hyphen component must be wordlike
        "abcable", "zzness",  # a real suffix on a non-stem
        "behaltensleistung",  # German, not English
    ])
    def test_rejects_non_words(self, word):
        w = self._w()
        if not w.available:
            pytest.skip("English wordlist unavailable")
        assert w.is_wordlike(word) is False

    def test_over_acceptance_is_deliberate(self):
        """`catness` is unattested but wellformed, and is accepted on purpose.

        SMOR accepts the equivalent German non-attested-but-wellformed compounds. The
        question is "could this be an ordinary English word?", not "is it listed".
        """
        w = self._w()
        if not w.available:
            pytest.skip("English wordlist unavailable")
        assert w.is_wordlike("catness") is True

    def test_absent_wordlist_is_conservative_not_wrong(self):
        """With no wordlist, everything is unknown — so a veto stands down."""
        from preprocessing.detection.lexicons.english_morph import EnglishWords
        w = EnglishWords()
        w._checked, w._impl = True, None       # simulate absence
        assert w.available is False
        assert w.is_wordlike("reusability") is False

    def test_veto_uses_the_acceptor(self):
        """Derivations must be vetoed as emphasis; abbreviations must not."""
        from preprocessing.detection.lexicons.abbreviations import is_emphasis
        if not self._w().available:
            pytest.skip("English wordlist unavailable")
        assert is_emphasis("en", "REUSABILITY") is True
        assert is_emphasis("en", "SUBCLASSABLE") is True
        assert is_emphasis("en", "OOP") is False
