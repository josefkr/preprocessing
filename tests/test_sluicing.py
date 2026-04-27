"""Tests for the pure sluicing detector.

Each fixture declares its language via ``# lang =`` so the detector
can pick the right wh-word lexicon. Add positive_*.conllu / negative_*
files for new cases.
"""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.sluicing import detect_sluicing

FIXTURES = Path(__file__).parent / "fixtures" / "sluicing"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text())
    return doc


def test_positive_basic_sluicing_en():
    findings = detect_sluicing(_load("positive_basic.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.x_text.lower() == "why"
    assert f.g_text.lower() == "say"
    assert (f.x_begin, f.x_end) == (15, 18)
    assert (f.g_begin, f.g_end) == (11, 14)
    assert f.lang == "en"


def test_negative_full_embedded_question():
    findings = detect_sluicing(_load("negative_full_question.conllu"))
    assert findings == []


def test_positive_de_sluicing():
    findings = detect_sluicing(_load("positive_de.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.x_text.lower() == "warum"
    assert f.g_text.lower() == "sagte"
    assert f.lang == "de"


def test_mixed_doc_runs_all_supported_sentences():
    findings = detect_sluicing(_load("mixed_en_de.conllu"))
    langs = sorted(f.lang for f in findings)
    assert langs == ["de", "en"]


def test_mixed_doc_filtered_by_restrict_to_lang():
    findings = detect_sluicing(_load("mixed_en_de.conllu"), restrict_to_lang="de")
    assert [f.lang for f in findings] == ["de"]
    assert findings[0].x_text.lower() == "warum"
