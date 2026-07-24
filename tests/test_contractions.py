"""Tests for the pure contraction/clitic detector (English)."""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.contractions import detect_contractions

FIXTURES = Path(__file__).parent / "fixtures" / "contractions"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text())
    return doc


def test_regular_negation():
    # "He wouldn't say." -> would + n't
    findings = detect_contractions(_load("positive_negation_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "wouldn't"
    assert (f.begin, f.end) == (3, 11)
    assert f.host.text == "would" and f.clitic.text == "n't"
    assert f.expansion == "would not"
    assert f.lang == "en"


def test_irregular_host_is_expanded_too():
    # "She can't swim." — UD splits this as "ca" + "n't", so expanding only the
    # clitic would yield "*ca not*"; the host must become "can".
    findings = detect_contractions(_load("positive_irregular_host_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "can't"
    assert f.host.text == "ca"
    assert f.expansion == "can not"


def test_copula_s_is_is():
    # "He's happy." — clitic lemma 'be' disambiguates to "is". The host keeps
    # its original surface casing, so the rewrite stays sentence-initial.
    findings = detect_contractions(_load("positive_copula_s_en.conllu"))
    assert len(findings) == 1
    assert findings[0].expansion == "He is"


def test_perfect_s_is_has():
    # "She's gone." — same surface "'s", but lemma 'have' -> "has".
    findings = detect_contractions(_load("positive_perfect_s_en.conllu"))
    assert len(findings) == 1
    assert findings[0].expansion == "She has"


def test_possessive_s_is_not_a_contraction():
    # "John's car is new." — possessive 's is not a two-word contraction and
    # must never be expanded.
    assert detect_contractions(_load("negative_possessive_s_en.conllu")) == []


def test_uncontracted_form_is_not_flagged():
    # "He would not say." — already expanded; nothing adjacent to flag.
    assert detect_contractions(_load("negative_not_adjacent_en.conllu")) == []


# --- CAS adapter (what `annotate.py --phenomenon contraction` writes) --------

T_GA = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly"
T_LP = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.LexicalPhrase"

# NB: the CAS must carry the *real* text and offsets. A sofa rebuilt by
# space-joining token forms would separate "ca" from "n't" and the contraction
# would (correctly) no longer be one — adjacency is what defines it.
_TEXT = "She can't swim."
_ROWS = [
    ("She", "she", "PRON", "PRP", 4, "nsubj", 0, 3),
    ("ca", "can", "AUX", "MD", 4, "aux", 4, 6),
    ("n't", "not", "PART", "RB", 4, "advmod", 6, 9),
    ("swim", "swim", "VERB", "VB", 0, "root", 10, 14),
    (".", ".", "PUNCT", ".", 4, "punct", 14, 15),
]


def _cas():
    from cassis import Cas
    from py_lift.dkpro import T_DEP, T_LEMMA, T_POS, T_SENT, T_TOKEN
    from py_lift.util import get_lift_typesystem

    ts = get_lift_typesystem()
    cas = Cas(sofa_string=_TEXT, document_language="en", typesystem=ts)
    Sent, Tok, POS, Dep, Lem = (
        ts.get_type(t) for t in (T_SENT, T_TOKEN, T_POS, T_DEP, T_LEMMA)
    )
    cas.add(Sent(begin=0, end=len(_TEXT)))
    toks = []
    for form, lemma, upos, xpos, _h, _d, b, e in _ROWS:
        tok = Tok(begin=b, end=e)
        cas.add(tok)
        toks.append(tok)
        cas.add(POS(begin=b, end=e, coarseValue=upos, PosValue=xpos))
        cas.add(Lem(begin=b, end=e, value=lemma))
    for i, (_f, _l, _u, _x, h, d, b, e) in enumerate(_ROWS):
        gov = toks[i] if h == 0 else toks[h - 1]
        cas.add(Dep(begin=b, end=e, Governor=gov, Dependent=toks[i],
                    DependencyType="root" if h == 0 else d))
    return cas


def test_cas_adapter_writes_contraction_annotations():
    from preprocessing.detection.cas_adapter import (
        existing_annotations,
        find_and_annotate,
    )

    cas = _cas()
    assert find_and_annotate(
        cas, cas.typesystem, phenomenon="contraction", lang="en"
    ) == 1

    gas = [(a.begin, a.end, a.category) for a in cas.select(T_GA)]
    assert gas == [(4, 9, "contraction")]
    # `suggestions` is an FSArray of SuggestedAction, not a string.
    (anomaly,) = list(cas.select(T_GA))
    assert [s.replacement for s in anomaly.suggestions.elements] == ["can not"]
    lps = sorted((p.begin, p.end, p.text) for p in cas.select(T_LP))
    assert lps == [(4, 6, "Contraction_host"), (6, 9, "Contraction_clitic")]
    # the registry signature finds them again (skip/--replace on re-runs)
    assert len(existing_annotations(cas, "contraction")) == 3


def test_annotated_cas_serialises_to_xmi():
    """Regression: `suggestions` is an FSArray. Assigning a bare string works
    in memory and only explodes on serialisation ("'str' object has no
    attribute 'elements'"), so the round-trip must be exercised."""
    from preprocessing.detection.cas_adapter import find_and_annotate

    cas = _cas()
    find_and_annotate(cas, cas.typesystem, phenomenon="contraction", lang="en")
    xmi = cas.to_xmi()
    assert "can not" in xmi

    from cassis import load_cas_from_xmi
    from py_lift.util import get_lift_typesystem

    reloaded = load_cas_from_xmi(xmi, typesystem=get_lift_typesystem())
    (anomaly,) = list(reloaded.select(T_GA))
    assert [s.replacement for s in anomaly.suggestions.elements] == ["can not"]


# --- German clipped indefinite articles (nen / nem / ner) --------------------

def _de_doc(text: str, rows: list[tuple]):
    """Build a udapi doc from (id, form, lemma, upos, xpos, head, deprel,
    begin, end) rows tagged as German."""
    lines = ["# sent_id = 1", f"# text = {text}", "# lang = de"]
    for i, form, lemma, upos, xpos, head, dep, b, e in rows:
        lines.append("\t".join([str(i), form, lemma, upos, xpos, "_",
                                str(head), dep, "_", f"t_start={b}|t_end={e}"]))
    from udapi.core.document import Document
    doc = Document()
    doc.from_conllu_string("\n".join(lines) + "\n\n")
    return doc


import pytest as _pytest


@_pytest.mark.parametrize("surface,expansion,b,e", [
    ("nen", "einen", 7, 10),
    ("nem", "einem", 7, 10),
    ("ner", "einer", 7, 10),
    ("'nen", "einen", 7, 11),
])
def test_de_clipped_article_detected(surface, expansion, b, e):
    # "Er hat <surface> Hund."  (lemma from Stanza is junk — we use surface)
    text = f"Er hat {surface} Hund."
    nb = 7 + len(surface)              # noun begin after the clipped form + space
    rows = [
        (1, "Er", "er", "PRON", "PPER", 2, "nsubj", 0, 2),
        (2, "hat", "haben", "VERB", "VVFIN", 0, "root", 3, 6),
        (3, surface, "nie", "DET", "PIAT", 4, "det", b, e),
        (4, "Hund", "Hund", "NOUN", "NN", 2, "obj", nb + 1, nb + 5),
        (5, ".", ".", "PUNCT", "$.", 2, "punct", nb + 5, nb + 6),
    ]
    findings = detect_contractions(_de_doc(text, rows), restrict_to_lang="de")
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "clipped_article"
    assert f.text == surface
    assert (f.begin, f.end) == (b, e)
    assert f.expansion == expansion


def test_de_clipped_article_preserves_sentence_initial_case():
    # "Nen Kaffee bitte." -> "Einen ..."
    rows = [
        (1, "Nen", "nie", "DET", "PIAT", 2, "det", 0, 3),
        (2, "Kaffee", "Kaffee", "NOUN", "NN", 0, "root", 4, 10),
        (3, "bitte", "bitte", "PART", "PTKANT", 2, "discourse", 11, 16),
        (4, ".", ".", "PUNCT", "$.", 2, "punct", 16, 17),
    ]
    findings = detect_contractions(_de_doc("Nen Kaffee bitte.", rows),
                                   restrict_to_lang="de")
    assert [f.expansion for f in findings] == ["Einen"]


def test_english_has_no_clipped_articles():
    # "ner"/"nen" must not fire on English text.
    rows = [
        (1, "hat", "hat", "X", "X", 0, "root", 0, 3),
        (2, "nen", "nen", "X", "X", 1, "dep", 4, 7),
    ]
    doc_lines = ["# sent_id = 1", "# text = hat nen", "# lang = en",
                 "1\that\that\tX\tX\t_\t0\troot\t_\tt_start=0|t_end=3",
                 "2\tnen\tnen\tX\tX\t_\t1\tdep\t_\tt_start=4|t_end=7"]
    from udapi.core.document import Document
    doc = Document(); doc.from_conllu_string("\n".join(doc_lines) + "\n\n")
    assert [f for f in detect_contractions(doc, restrict_to_lang="en")
            if f.kind == "clipped_article"] == []


# --- "ne" (=eine) with the right-context NP gate -----------------------------

def test_de_ne_expands_before_noun():
    # "hab ne Karre ."  ne(4-6) Karre(7-12)
    rows = [
        (1, "hab", "haben", "VERB", "VVFIN", 0, "root", 0, 3),
        (2, "ne", "ein", "ADV", "ADV", 3, "det", 4, 6),
        (3, "Karre", "Karre", "NOUN", "NN", 1, "obj", 7, 12),
        (4, ".", ".", "PUNCT", "$.", 1, "punct", 12, 13),
    ]
    f = detect_contractions(_de_doc("hab ne Karre.", rows), restrict_to_lang="de")
    assert [(x.kind, x.text, x.expansion) for x in f] == [
        ("clipped_article", "ne", "eine")
    ]


def test_de_ne_expands_across_adjective():
    # "hab ne alte Karre ."  premodifier before the noun
    rows = [
        (1, "hab", "haben", "VERB", "VVFIN", 0, "root", 0, 3),
        (2, "ne", "ein", "ADV", "ADV", 4, "det", 4, 6),
        (3, "alte", "alt", "ADJ", "ADJA", 4, "amod", 7, 11),
        (4, "Karre", "Karre", "NOUN", "NN", 1, "obj", 12, 17),
        (5, ".", ".", "PUNCT", "$.", 1, "punct", 17, 18),
    ]
    f = detect_contractions(_de_doc("hab ne alte Karre.", rows), restrict_to_lang="de")
    assert [x.expansion for x in f] == ["eine"]


def test_de_ne_tag_not_expanded():
    # "Das ist gut , ne"  — tag question: punctuation/end follows, no NP.
    rows = [
        (1, "Das", "der", "PRON", "PDS", 3, "nsubj", 0, 3),
        (2, "ist", "sein", "AUX", "VAFIN", 3, "cop", 4, 7),
        (3, "gut", "gut", "ADJ", "ADJD", 0, "root", 8, 11),
        (4, ",", ",", "PUNCT", "$,", 5, "punct", 11, 12),
        (5, "ne", "ein", "ADV", "ADV", 3, "discourse", 13, 15),
    ]
    f = detect_contractions(_de_doc("Das ist gut, ne", rows), restrict_to_lang="de")
    assert [x for x in f if x.kind == "clipped_article"] == []


# --- bare "n": full form inferred from the following noun's morphology --------

def _doc_from_conllu(lines: list[str]):
    from udapi.core.document import Document
    doc = Document()
    doc.from_conllu_string("\n".join(lines) + "\n\n")
    return doc


def _n_doc(noun: str, feats: str, nb: int):
    # "hat n <noun>."  — n at 4-5, noun at nb
    return _doc_from_conllu([
        "# sent_id = 1", f"# text = hat n {noun}.", "# lang = de",
        "1\that\thaben\tVERB\tVVFIN\t_\t0\troot\t_\tt_start=0|t_end=3",
        "2\tn\tn\tADJ\tADJA\tDegree=Pos\t3\tamod\t_\tt_start=4|t_end=5",
        f"3\t{noun}\t{noun}\tNOUN\tNN\t{feats}\t1\tobj\t_\tt_start={nb}|t_end={nb+len(noun)}",
        f"4\t.\t.\tPUNCT\t$.\t_\t1\tpunct\t_\tt_start={nb+len(noun)}|t_end={nb+len(noun)+1}",
    ])


def test_de_n_masc_acc_is_einen():
    f = detect_contractions(_n_doc("Kleinwagen", "Case=Acc|Gender=Masc|Number=Sing", 6),
                            restrict_to_lang="de")
    assert [(x.text, x.expansion, x.kind, x.inferred) for x in f] == [
        ("n", "einen", "clipped_article", True)
    ]


def test_de_n_neut_is_ein():
    f = detect_contractions(_n_doc("Kind", "Case=Acc|Gender=Neut|Number=Sing", 6),
                            restrict_to_lang="de")
    assert [x.expansion for x in f] == ["ein"]


def test_de_n_masc_dat_is_einem():
    f = detect_contractions(_n_doc("Mann", "Case=Dat|Gender=Masc|Number=Sing", 6),
                            restrict_to_lang="de")
    assert [x.expansion for x in f] == ["einem"]


def test_de_n_skipped_when_case_missing():
    # No Case feature -> can't inflect -> no finding (leave "n" contracted).
    f = detect_contractions(_n_doc("Sache", "Gender=Fem|Number=Sing", 6),
                            restrict_to_lang="de")
    assert [x for x in f if x.kind == "clipped_article"] == []


def test_de_n_skipped_when_plural():
    # No indefinite plural article.
    f = detect_contractions(_n_doc("Autos", "Case=Acc|Gender=Neut|Number=Plur", 6),
                            restrict_to_lang="de")
    assert [x for x in f if x.kind == "clipped_article"] == []


def test_de_n_skipped_when_no_noun_follows():
    # "n" tagged ADV with no nominal head after it -> not an article.
    doc = _doc_from_conllu([
        "# sent_id = 1", "# text = passt schon n bisschen", "# lang = de",
        "1\tpasst\tpassen\tVERB\tVVFIN\t_\t0\troot\t_\tt_start=0|t_end=5",
        "2\tschon\tschon\tADV\tADV\t_\t1\tadvmod\t_\tt_start=6|t_end=11",
        "3\tn\tn\tNUM\tCARD\tNumType=Card\t4\tnummod\t_\tt_start=12|t_end=13",
        "4\tbisschen\tbisschen\tADV\tADV\t_\t1\tadvmod\t_\tt_start=14|t_end=22",
    ])
    assert [x for x in detect_contractions(doc, restrict_to_lang="de")
            if x.kind == "clipped_article"] == []
