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


def test_positive_de_wodurch_lexicon():
    """`wodurch` — a productive German wo(r)+P interrogative — is in the
    German wh-lexicon and is detected as a sluice remnant."""
    findings = detect_sluicing(_load("positive_de_wodurch.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.x_text.lower() == "wodurch"
    assert f.g_text.lower() == "sagte"
    assert f.lang == "de"


def test_positive_de_obj_remnant():
    """A wh-remnant attached as `obj` (not `ccomp`) is recovered via the
    broadened relation gate, licensed by a question-embedding governor."""
    findings = detect_sluicing(_load("positive_de_obj_remnant.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.x_text.lower() == "was"
    assert f.g_text.lower() == "weiß"


def test_positive_de_multiword_remnant():
    """A multi-word remnant ("wie viele") is detected via its wh-word child,
    and the whole span is reported."""
    findings = detect_sluicing(_load("positive_de_multiword.conllu"))
    assert len(findings) == 1
    assert findings[0].x_text.lower() == "wie viele"


def test_negative_de_full_embedded_question():
    """A full embedded question — wh-word on `obj`, governor an embedded
    verb (deprel `ccomp`) with its own subject — is not a sluice. Guards
    the precision of the broadened relation gate."""
    findings = detect_sluicing(_load("negative_de_full_embedded.conllu"))
    assert findings == []
