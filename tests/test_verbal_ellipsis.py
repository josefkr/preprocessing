"""Tests for the pure verbal-ellipsis detector."""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.verbal_ellipsis import detect_verbal_ellipsis

FIXTURES = Path(__file__).parent / "fixtures" / "verbal_ellipsis"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text())
    return doc


def test_positive_aux_as_root():
    # "He did." — 'did' is AUX but attached as root, not aux/aux:pass/cop.
    findings = detect_verbal_ellipsis(_load("positive_aux_as_root.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text.lower() == "did"
    assert (f.begin, f.end) == (3, 6)
    assert f.deprel == "root"


def test_negative_normal_aux():
    # "He is running." — 'is' is AUX and attached as 'aux'.
    findings = detect_verbal_ellipsis(_load("negative_normal_aux.conllu"))
    assert findings == []


def test_negative_copula():
    # "He is happy." — 'is' is AUX and attached as 'cop'.
    findings = detect_verbal_ellipsis(_load("negative_copula.conllu"))
    assert findings == []
