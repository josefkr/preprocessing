"""Tests for the pure bare-wh-question detector.

Each fixture declares its language via ``# lang =`` so the detector
picks the right wh-word lexicon. Add positive_*.conllu / negative_*
files for new shapes.
"""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.bare_questions import detect_bare_questions

FIXTURES = Path(__file__).parent / "fixtures" / "bare_questions"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text())
    return doc


def test_positive_en_why():
    """Single wh-adverb root, no other content."""
    findings = detect_bare_questions(_load("positive_en_why.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "Why"
    assert f.wh_form == "Why"
    assert (f.begin, f.end) == (0, 3)
    assert f.lang == "en"


def test_positive_en_what_for():
    """Stranded preposition: 'What' is root, 'for' attaches as `case`.
    Span covers the whole wh-phrase including the postposed preposition."""
    findings = detect_bare_questions(_load("positive_en_what_for.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "What for"
    assert f.wh_form == "What"
    assert (f.begin, f.end) == (0, 8)


def test_positive_en_for_what():
    """Pied-piped preposition: wh-word is the root, the preposition is its
    `case` child."""
    findings = detect_bare_questions(_load("positive_en_for_what.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "For what"
    assert f.wh_form == "what"


def test_positive_en_what_man():
    """Wh-determiner + noun: root is the noun, wh-word attaches as `det`."""
    findings = detect_bare_questions(_load("positive_en_what_man.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "What man"
    assert f.wh_form == "What"


def test_positive_en_how_viable():
    """Wh-adverb + adjective: root is the adjective, 'How' attaches as
    `advmod`."""
    findings = detect_bare_questions(_load("positive_en_how_viable.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "How viable"
    assert f.wh_form == "How"


def test_negative_en_morris_who():
    """'Morris who?' is an echo question — 'who' attaches by `appos`,
    not a wh-phrase modifier slot, so it must NOT fire."""
    findings = detect_bare_questions(_load("negative_en_morris_who.conllu"))
    assert findings == []


def test_negative_en_full_question():
    """'Why did he do it?' is a complete question (finite verb + subject)
    and must NOT fire."""
    findings = detect_bare_questions(_load("negative_en_full_question.conllu"))
    assert findings == []


# --- German -----------------------------------------------------------------


def test_positive_de_warum():
    """German bare wh-adverb root — direct analogue of 'Why?'."""
    findings = detect_bare_questions(_load("positive_de_warum.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "Warum"
    assert f.wh_form == "Warum"
    assert (f.begin, f.end) == (0, 5)
    assert f.lang == "de"


def test_positive_de_an_wen():
    """German pied-piped preposition: wh-pronoun is the root, the
    preposition is its `case` child — analogue of 'For what?'."""
    findings = detect_bare_questions(_load("positive_de_an_wen.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "An wen"
    assert f.wh_form == "wen"
    assert f.lang == "de"


def test_positive_de_welcher_mann():
    """German wh-determiner + noun: root is the noun, wh-word attaches
    as `det` — analogue of 'What man?'."""
    findings = detect_bare_questions(_load("positive_de_welcher_mann.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "Welcher Mann"
    assert f.wh_form == "Welcher"
    assert f.lang == "de"


def test_positive_de_wie_gross():
    """German wh-adverb + predicative adjective: root is the adjective,
    'Wie' attaches as `advmod` — analogue of 'How viable?'."""
    findings = detect_bare_questions(_load("positive_de_wie_gross.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.text == "Wie groß"
    assert f.wh_form == "Wie"
    assert f.lang == "de"


def test_negative_de_schmidt_wer():
    """German echo question: 'wer' attaches by `appos` to a proper
    name, not a wh-phrase modifier slot — must NOT fire (analogue of
    'Morris who?')."""
    findings = detect_bare_questions(_load("negative_de_schmidt_wer.conllu"))
    assert findings == []


def test_negative_de_full_question():
    """German full finite question (finite verb + subject) — must NOT
    fire (analogue of 'Why did he do it?')."""
    findings = detect_bare_questions(_load("negative_de_full_question.conllu"))
    assert findings == []
