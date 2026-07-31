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
    # clitic would yield "*ca not*"; the host must become "can" — and English writes it solid: "cannot".
    findings = detect_contractions(_load("positive_irregular_host_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "can't"
    assert f.host.text == "ca"
    assert f.expansion == "cannot"


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
    assert [s.replacement for s in anomaly.suggestions.elements] == ["cannot"]
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
    assert "cannot" in xmi

    from cassis import load_cas_from_xmi
    from py_lift.util import get_lift_typesystem

    reloaded = load_cas_from_xmi(xmi, typesystem=get_lift_typesystem())
    (anomaly,) = list(reloaded.select(T_GA))
    assert [s.replacement for s in anomaly.suggestions.elements] == ["cannot"]


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


# --- clitic disambiguation by the following participle ----------------------
# Stanza lemmatises the perfect auxiliaries "'s"/"'d" as *be*/*would*, which the
# lemma table alone turns into the ungrammatical "is been"/"would been". Only
# provably safe overrides are applied (the alternative isn't English).

def _en_doc(lines: list[str]):
    return _doc_from_conllu(["# sent_id = 1", "# lang = en"] + lines)


def test_s_before_been_is_has():
    # "He's been lucky." — Stanza gives lemma 'be' (-> "is"), but "is been" is
    # never grammatical, so the following participle forces "has".
    doc = _en_doc([
        "# text = He's been lucky.",
        "1\tHe\the\tPRON\tPRP\t_\t4\tnsubj\t_\tt_start=0|t_end=2",
        "2\t's\tbe\tAUX\tVBZ\t_\t4\taux\t_\tt_start=2|t_end=4",
        "3\tbeen\tbe\tAUX\tVBN\t_\t4\tcop\t_\tt_start=5|t_end=9",
        "4\tlucky\tlucky\tADJ\tJJ\t_\t0\troot\t_\tSpaceAfter=No|t_start=10|t_end=15",
        "5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tt_start=15|t_end=16",
    ])
    findings = detect_contractions(doc)
    assert len(findings) == 1
    assert findings[0].expansion == "He has"


def test_s_before_been_skips_intervening_negation():
    # "He's not been well." — the negation must not hide the participle.
    doc = _en_doc([
        "# text = He's not been well.",
        "1\tHe\the\tPRON\tPRP\t_\t5\tnsubj\t_\tt_start=0|t_end=2",
        "2\t's\tbe\tAUX\tVBZ\t_\t5\taux\t_\tt_start=2|t_end=4",
        "3\tnot\tnot\tPART\tRB\t_\t5\tadvmod\t_\tt_start=5|t_end=8",
        "4\tbeen\tbe\tAUX\tVBN\t_\t5\tcop\t_\tt_start=9|t_end=13",
        "5\twell\twell\tADJ\tJJ\t_\t0\troot\t_\tSpaceAfter=No|t_start=14|t_end=18",
        "6\t.\t.\tPUNCT\t.\t_\t5\tpunct\t_\tt_start=18|t_end=19",
    ])
    findings = detect_contractions(doc)
    assert len(findings) == 1
    assert findings[0].expansion == "He has"


def test_d_before_participle_is_had():
    # "there'd been a process." — "would" never precedes a participle, so
    # "'d" + VBN is the past perfect regardless of the 'would' lemma.
    doc = _en_doc([
        "# text = there'd been a process.",
        "1\tthere\tthere\tPRON\tEX\t_\t3\texpl\t_\tt_start=0|t_end=5",
        "2\t'd\twould\tAUX\tVBD\t_\t3\taux\t_\tt_start=5|t_end=7",
        "3\tbeen\tbe\tVERB\tVBN\t_\t0\troot\t_\tt_start=8|t_end=12",
        "4\ta\ta\tDET\tDT\t_\t5\tdet\t_\tt_start=13|t_end=14",
        "5\tprocess\tprocess\tNOUN\tNN\t_\t3\tnsubj\t_\tSpaceAfter=No|t_start=15|t_end=22",
        "6\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\tt_start=22|t_end=23",
    ])
    findings = detect_contractions(doc)
    assert len(findings) == 1
    assert findings[0].expansion == "there had"


def test_copula_s_before_adjective_stays_is():
    # "He's lucky." — no participle follows, so the lemma decision ("is") holds.
    doc = _en_doc([
        "# text = He's lucky.",
        "1\tHe\the\tPRON\tPRP\t_\t3\tnsubj\t_\tt_start=0|t_end=2",
        "2\t's\tbe\tAUX\tVBZ\t_\t3\tcop\t_\tt_start=2|t_end=4",
        "3\tlucky\tlucky\tADJ\tJJ\t_\t0\troot\t_\tSpaceAfter=No|t_start=5|t_end=10",
        "4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\tt_start=10|t_end=11",
    ])
    findings = detect_contractions(doc)
    assert len(findings) == 1
    assert findings[0].expansion == "He is"


def test_s_before_other_participle_left_to_lemma():
    # "The job's finished." — genuinely ambiguous (perfect "he's eaten" vs
    # passive "the job's finished"), so the lemma decision stands rather than
    # guessing "has".
    doc = _en_doc([
        "# text = The job's finished.",
        "1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\tt_start=0|t_end=3",
        "2\tjob\tjob\tNOUN\tNN\t_\t4\tnsubj:pass\t_\tt_start=4|t_end=7",
        "3\t's\tbe\tAUX\tVBZ\t_\t4\taux:pass\t_\tt_start=7|t_end=9",
        "4\tfinished\tfinish\tVERB\tVBN\t_\t0\troot\t_\tSpaceAfter=No|t_start=10|t_end=18",
        "5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tt_start=18|t_end=19",
    ])
    findings = detect_contractions(doc)
    assert len(findings) == 1
    assert findings[0].expansion == "job is"


# --- English clipped forms ("gonna", "lemme", "'em") ------------------------
# Matched as whole written words, which the tokenizer may keep as one token or
# split ("gonna" -> gon+na).

def test_clipped_form_single_token():
    # "He is kinda funny." — one token.
    doc = _en_doc([
        "# text = He is kinda funny.",
        "1\tHe\the\tPRON\tPRP\t_\t4\tnsubj\t_\tt_start=0|t_end=2",
        "2\tis\tbe\tAUX\tVBZ\t_\t4\tcop\t_\tt_start=3|t_end=5",
        "3\tkinda\tkinda\tADV\tRB\t_\t4\tadvmod\t_\tt_start=6|t_end=11",
        "4\tfunny\tfunny\tADJ\tJJ\t_\t0\troot\t_\tSpaceAfter=No|t_start=12|t_end=17",
        "5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tt_start=17|t_end=18",
    ])
    f = [x for x in detect_contractions(doc) if x.kind == "clipped_form"]
    assert len(f) == 1
    assert (f[0].text, f[0].expansion) == ("kinda", "kind of")


def test_clipped_form_split_across_two_tokens():
    # "I gonna go." — Stanza splits "gonna" into "gon" + "na"; the whole
    # written word must still be matched.
    doc = _en_doc([
        "# text = I gonna go.",
        "1\tI\tI\tPRON\tPRP\t_\t4\tnsubj\t_\tt_start=0|t_end=1",
        "2\tgon\tgo\tVERB\tVBG\t_\t4\taux\t_\tt_start=2|t_end=5",
        "3\tna\tto\tPART\tTO\t_\t4\tmark\t_\tt_start=5|t_end=7",
        "4\tgo\tgo\tVERB\tVB\t_\t0\troot\t_\tSpaceAfter=No|t_start=8|t_end=10",
        "5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tt_start=10|t_end=11",
    ])
    f = [x for x in detect_contractions(doc) if x.kind == "clipped_form"]
    assert len(f) == 1
    assert (f[0].text, f[0].expansion) == ("gonna", "going to")
    assert (f[0].begin, f[0].end) == (2, 7)


def test_clipped_form_glued_to_previous_word_gets_a_space():
    # "Take'em now." — written glued to the host, so the expansion must carry a
    # separating space or the rewrite yields "Takethem".
    doc = _en_doc([
        "# text = Take'em now.",
        "1\tTake\ttake\tVERB\tVB\t_\t0\troot\t_\tt_start=0|t_end=4",
        "2\t'em\tthey\tPRON\tPRP\t_\t1\tobj\t_\tt_start=4|t_end=7",
        "3\tnow\tnow\tADV\tRB\t_\t1\tadvmod\t_\tSpaceAfter=No|t_start=8|t_end=11",
        "4\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\tt_start=11|t_end=12",
    ])
    f = [x for x in detect_contractions(doc) if x.kind == "clipped_form"]
    assert len(f) == 1
    assert f[0].expansion == " them"


def test_clipped_form_sentence_initial_is_capitalised():
    # "twas someone." — lower-case in the source, but it opens the sentence.
    doc = _en_doc([
        "# text = twas someone.",
        "1\ttwas\ttwis\tAUX\tVBD\t_\t2\tcop\t_\tt_start=0|t_end=4",
        "2\tsomeone\tsomeone\tPRON\tNN\t_\t0\troot\t_\tSpaceAfter=No|t_start=5|t_end=12",
        "3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tt_start=12|t_end=13",
    ])
    f = [x for x in detect_contractions(doc) if x.kind == "clipped_form"]
    assert len(f) == 1
    assert f[0].expansion == "It was"


def test_aint_host_agrees_with_subject():
    # "ai" + "n't": the host is whichever form of *be* the subject needs, so the
    # rewrite is never the ungrammatical "that ai not".
    def _aint(subj, subj_lemma, subj_xpos):
        n = len(subj)
        return _en_doc([
            f"# text = {subj} ain't funny.",
            f"1\t{subj}\t{subj_lemma}\tPRON\t{subj_xpos}\t_\t4\tnsubj\t_\tt_start=0|t_end={n}",
            f"2\tai\tbe\tAUX\tVBP\t_\t4\tcop\t_\tt_start={n+1}|t_end={n+3}",
            f"3\tn't\tnot\tPART\tRB\t_\t4\tadvmod\t_\tt_start={n+3}|t_end={n+6}",
            f"4\tfunny\tfunny\tADJ\tJJ\t_\t0\troot\t_\tSpaceAfter=No|t_start={n+7}|t_end={n+12}",
            f"5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tt_start={n+12}|t_end={n+13}",
        ])
    got = {}
    for subj, lemma, xpos in [("That", "that", "DT"), ("I", "I", "PRP"),
                              ("You", "you", "PRP")]:
        f = [x for x in detect_contractions(_aint(subj, lemma, xpos))
             if x.kind == "clitic"]
        assert len(f) == 1, (subj, f)
        got[subj] = f[0].expansion
    # The finding spans "ain't" only (host "ai" + clitic "n't"), so the subject
    # is not part of the expansion.
    assert got == {"That": "is not", "I": "am not", "You": "are not"}


# --- g-dropping ("talkin'" -> "talking") ------------------------------------
# Only the apostrophe-marked spelling is handled here; the bare form
# ("stickin") is left to the dictionary-backed spelling normalizer.

def test_gdropping_single_token():
    doc = _en_doc([
        "# text = He was talkin' loudly.",
        "1\tHe\the\tPRON\tPRP\t_\t3\tnsubj\t_\tt_start=0|t_end=2",
        "2\twas\tbe\tAUX\tVBD\t_\t3\taux\t_\tt_start=3|t_end=6",
        "3\ttalkin'\ttalkin'\tADJ\tJJ\t_\t0\troot\t_\tt_start=7|t_end=14",
        "4\tloudly\tloudly\tADV\tRB\t_\t3\tadvmod\t_\tSpaceAfter=No|t_start=15|t_end=21",
        "5\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\tt_start=21|t_end=22",
    ])
    f = [x for x in detect_contractions(doc) if x.kind == "clipped_form"]
    assert len(f) == 1
    assert (f[0].text, f[0].expansion) == ("talkin'", "talking")


def test_gdropping_when_apostrophe_is_a_separate_token():
    # Stanza sometimes splits "stayin'" into "stayin" + "'", so the joined span
    # must be reachable too. The stem is a VERB, so the rule applies.
    doc = _en_doc([
        "# text = I am stayin' here.",
        "1\tI\tI\tPRON\tPRP\t_\t3\tnsubj\t_\tt_start=0|t_end=1",
        "2\tam\tbe\tAUX\tVBP\t_\t3\taux\t_\tt_start=2|t_end=4",
        "3\tstayin\tstayin\tVERB\tVBG\t_\t0\troot\t_\tSpaceAfter=No|t_start=5|t_end=11",
        "4\t'\t'\tPUNCT\t``\t_\t3\tpunct\t_\tt_start=11|t_end=12",
        "5\there\there\tADV\tRB\t_\t3\tadvmod\t_\tSpaceAfter=No|t_start=13|t_end=17",
        "6\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\tt_start=17|t_end=18",
    ])
    f = [x for x in detect_contractions(doc) if x.kind == "clipped_form"]
    assert len(f) == 1
    assert (f[0].text, f[0].expansion) == ("stayin'", "staying")
    assert (f[0].begin, f[0].end) == (5, 12)   # spans the apostrophe too


def test_gdropping_does_not_fire_on_a_closing_quote():
    # "the cabin'" — a NOUN followed by a closing quote must not become
    # "cabing". Only a verb stem is trusted for the joined form.
    doc = _en_doc([
        "# text = the cabin' here",
        "1\tthe\tthe\tDET\tDT\t_\t2\tdet\t_\tt_start=0|t_end=3",
        "2\tcabin\tcabin\tNOUN\tNN\t_\t0\troot\t_\tSpaceAfter=No|t_start=4|t_end=9",
        "3\t'\t'\tPUNCT\t''\t_\t2\tpunct\t_\tt_start=9|t_end=10",
        "4\there\there\tADV\tRB\t_\t2\tadvmod\t_\tt_start=11|t_end=15",
    ])
    assert [x for x in detect_contractions(doc) if x.kind == "clipped_form"] == []
