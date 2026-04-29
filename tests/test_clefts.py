"""Tests for the pure cleft detector."""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.clefts import detect_clefts

FIXTURES = Path(__file__).parent / "fixtures" / "clefts"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text())
    return doc


# ---- it-clefts ----------------------------------------------------------


def test_simple_it_cleft():
    findings = detect_clefts(_load("positive_simple_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "it_cleft"
    assert f.lang == "en"
    assert (f.focus.begin, f.focus.end) == (7, 11)
    assert (f.presupposition.begin, f.presupposition.end) == (12, 30)
    assert f.cleft_token.text == "It"
    assert (f.cleft_token.begin, f.cleft_token.end) == (0, 2)


def test_it_cleft_with_modifier():
    findings = detect_clefts(_load("positive_with_modifier_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "it_cleft"
    assert (f.focus.begin, f.focus.end) == (7, 19)
    assert (f.presupposition.begin, f.presupposition.end) == (20, 28)
    assert (f.cleft_token.begin, f.cleft_token.end) == (0, 2)


def test_no_relative_clause_does_not_match():
    findings = detect_clefts(_load("negative_no_relcl_en.conllu"))
    assert findings == []


def test_no_it_does_not_match():
    findings = detect_clefts(_load("negative_no_it_en.conllu"))
    assert findings == []


# ---- wh-clefts ----------------------------------------------------------


def test_wh_cleft_with_noun_focus():
    # "What surprised me was the answer." — focus is the NOUN "answer".
    findings = detect_clefts(_load("positive_wh_noun_focus_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "wh_cleft"
    assert f.lang == "en"
    assert f.focus.text.lower() == "answer"
    assert (f.focus.begin, f.focus.end) == (26, 32)
    # Presupposition spans the relative clause "surprised me".
    assert (f.presupposition.begin, f.presupposition.end) == (5, 17)
    # The wh-pronoun.
    assert f.cleft_token.text == "What"
    assert (f.cleft_token.begin, f.cleft_token.end) == (0, 4)


def test_wh_cleft_with_uninflected_verb_focus():
    # "What he wanted was to leave." — focus is "leave" (VERB, form==lemma).
    findings = detect_clefts(_load("positive_wh_verb_focus_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "wh_cleft"
    assert f.focus.text.lower() == "leave"
    assert (f.focus.begin, f.focus.end) == (22, 27)
    # Presupposition spans "he wanted".
    assert (f.presupposition.begin, f.presupposition.end) == (5, 14)
    assert f.cleft_token.text == "What"


def test_wh_no_relcl_does_not_match():
    # "What was the answer?" — no acl:relcl child of "What".
    findings = detect_clefts(_load("negative_wh_no_relcl_en.conllu"))
    assert findings == []


def test_wh_inflected_verb_focus_skipped():
    # "What he wanted was leaving." — "leaving" is VERB but form != lemma.
    findings = detect_clefts(
        _load("negative_wh_inflected_verb_focus_en.conllu")
    )
    assert findings == []


# ---- filtering ---------------------------------------------------------


def test_restrict_to_lang_filters():
    findings = detect_clefts(
        _load("positive_simple_en.conllu"), restrict_to_lang="de"
    )
    assert findings == []
