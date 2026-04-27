"""Tests for the pure subject-sharing detector."""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.subject_sharing import detect_subject_sharing

FIXTURES = Path(__file__).parent / "fixtures" / "subject_sharing"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text())
    return doc


def test_positive_basic_subject_sharing():
    findings = detect_subject_sharing(_load("positive_basic.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.x_text.lower() == "danced"
    assert f.y_text.lower() == "sang"
    assert (f.x_begin, f.x_end) == (13, 19)
    assert (f.y_begin, f.y_end) == (4, 8)
    assert len(f.shared_subjects) == 1
    s = f.shared_subjects[0]
    assert s.text.lower() == "she"
    assert (s.begin, s.end) == (0, 3)


def test_negative_right_conjunct_has_own_subject():
    # "She sang and he danced." — the right conjunct has its own subject.
    findings = detect_subject_sharing(_load("negative_own_subject.conllu"))
    assert findings == []
