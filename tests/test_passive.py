"""Tests for the pure passive-construction detector.

Covers both canonical (aux:pass present) and short (participle + agent
PP) passives, with regression checks against false positives like
active perfect-tense clauses.
"""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.passive import detect_passive

FIXTURES = Path(__file__).parent / "fixtures" / "passive"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text())
    return doc


# ---- canonical passives -------------------------------------------------


def test_positive_with_subject_en():
    findings = detect_passive(_load("positive_with_subject_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "canonical"
    assert f.lang == "en"
    assert f.aux is not None and f.aux.text.lower() == "was"
    assert f.verb.text.lower() == "eaten"
    assert f.subject is not None and f.subject.text.lower() == "cake"
    assert f.agent is None
    assert f.agent_marker is None


def test_positive_no_subject_de():
    findings = detect_passive(_load("positive_no_subject_de.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "canonical"
    assert f.aux is not None and f.aux.text.lower() == "wird"
    assert f.verb.text.lower() == "getanzt"
    assert f.subject is None


def test_negative_active():
    findings = detect_passive(_load("negative_active.conllu"))
    assert findings == []


# ---- short passives -----------------------------------------------------


def test_short_passive_en_by():
    findings = detect_passive(_load("positive_short_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "short"
    assert f.lang == "en"
    assert f.verb.text.lower() == "read"
    assert f.aux is None
    assert f.subject is None
    assert f.agent is not None and f.agent.text.lower() == "john"
    assert f.agent_marker is not None and f.agent_marker.text.lower() == "by"


def test_short_passive_de_durch():
    findings = detect_passive(_load("positive_short_de_durch.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "short"
    assert f.verb.text.lower() == "bedrängt"
    assert f.agent is not None and f.agent.text.lower() == "lärm"
    assert f.agent_marker is not None and f.agent_marker.text.lower() == "durch"


def test_short_passive_de_attributive_von():
    # Participle used attributively (acl modifier of "Kräfte").
    findings = detect_passive(_load("positive_short_attributive_de.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "short"
    assert f.verb.text.lower() == "angeführten"
    assert f.agent is not None and f.agent.text.lower() == "kurden"
    assert f.agent_marker is not None and f.agent_marker.text.lower() == "von"


def test_active_perfect_does_not_trigger_short_passive():
    # "He has eaten by then." — VBN with `by` obl, but V has aux=has,
    # so the short-passive rule must reject it.
    findings = detect_passive(_load("negative_active_perfect_en.conllu"))
    assert findings == []


# ---- filtering ----------------------------------------------------------


def test_restrict_to_lang_filters():
    findings = detect_passive(
        _load("positive_no_subject_de.conllu"), restrict_to_lang="en"
    )
    assert findings == []
