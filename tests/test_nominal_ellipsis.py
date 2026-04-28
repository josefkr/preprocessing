"""Tests for the pure nominal-head ellipsis detector."""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.nominal_ellipsis import detect_nominal_ellipsis

FIXTURES = Path(__file__).parent / "fixtures" / "nominal_ellipsis"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text())
    return doc


def _only(name: str):
    findings = detect_nominal_ellipsis(_load(name))
    assert len(findings) == 1, findings
    return findings[0]


def test_quantifier():
    f = _only("positive_quantifier.conllu")
    assert f.subtype == "quantifier"
    assert f.text.lower() == "several"
    assert f.deprel == "nsubj"


def test_none():
    f = _only("positive_none.conllu")
    assert f.subtype == "none"
    assert f.text.lower() == "none"


def test_numeral():
    f = _only("positive_numeral.conllu")
    assert f.subtype == "numeral"
    assert f.text.lower() == "two"
    assert f.deprel == "obj"


def test_every_one():
    f = _only("positive_every_one.conllu")
    assert f.subtype == "every_one"
    assert f.text.lower() == "one"
    assert f.deprel == "obj"


def test_comparative():
    f = _only("positive_comparative.conllu")
    assert f.subtype == "comparative"
    assert f.text.lower() == "larger"
    assert f.deprel == "obj"


def test_elder():
    # "The elder is married." — should be tagged `elder`, not
    # `comparative` or `adjective`, thanks to specificity ordering.
    f = _only("positive_elder.conllu")
    assert f.subtype == "elder"
    assert f.text.lower() == "elder"


def test_adjective():
    f = _only("positive_adjective.conllu")
    assert f.subtype == "adjective"
    assert f.text.lower() == "small"
    assert f.deprel == "obj"


def test_amod_modifier_is_not_ellipsis():
    findings = detect_nominal_ellipsis(_load("negative_amod.conllu"))
    assert findings == []


def test_restrict_to_lang_filters_other_languages():
    findings = detect_nominal_ellipsis(
        _load("positive_quantifier.conllu"), restrict_to_lang="de"
    )
    assert findings == []
