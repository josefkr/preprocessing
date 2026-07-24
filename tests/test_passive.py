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
    findings = detect_passive(_load("negative_active_en.conllu"))
    assert findings == []


# ---- canonical passives WITH an agent (agentful) ------------------------


def test_canonical_passive_en_agent():
    # "The cake was eaten by John." — aux:pass present (canonical) AND a
    # by-agent. The agent must be extracted on the canonical path too.
    findings = detect_passive(_load("positive_canonical_agent_en.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "canonical"
    assert f.lang == "en"
    assert f.aux is not None and f.aux.text.lower() == "was"
    assert f.verb.text.lower() == "eaten"
    assert f.subject is not None and f.subject.text.lower() == "cake"
    assert f.agent is not None and f.agent.text.lower() == "john"
    assert f.agent_marker is not None and f.agent_marker.text.lower() == "by"
    # Full-phrase spans cover the whole NP (head + subtree), incl. the marker.
    assert f.subject_phrase is not None and f.subject_phrase.text == "The cake"
    assert f.agent_phrase is not None and f.agent_phrase.text == "by John"


def test_canonical_passive_de_agent_mwt():
    # Same sentence in the representation freshly ingested data now produces:
    # a real UD multiword token whose sub-words carry their true forms. The
    # agent marker surfaces as the proper preposition "von" (not the
    # contraction), i.e. without needing the "vom" lexicon workaround.
    findings = detect_passive(_load("positive_canonical_agent_de_mwt.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "canonical"
    assert f.lang == "de"
    assert f.aux is not None and f.aux.text.lower() == "wird"
    assert f.verb.text.lower() == "repariert"
    assert f.subject is not None and f.subject.text.lower() == "motor"
    assert f.agent is not None and f.agent.text.lower() == "mechaniker"
    assert f.agent_marker is not None and f.agent_marker.text.lower() == "von"
    # The span still covers the contraction in the sofa (15:29), since the
    # sub-words share the parent multiword token's offsets.
    assert f.agent_phrase is not None
    assert (f.agent_phrase.begin, f.agent_phrase.end) == (15, 29)


def test_canonical_passive_de_agent():
    # LEGACY shape: data ingested before MWTPart existed, where both expanded
    # sub-words surface as "vom" (see the fixture's note). Kept so the
    # already-processed corpus stays covered; the agent is recognised via the
    # contracted form in PASSIVE_AGENT_PREPS_BY_LANG.
    findings = detect_passive(_load("positive_canonical_agent_de.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "canonical"
    assert f.lang == "de"
    assert f.aux is not None and f.aux.text.lower() == "wird"
    assert f.verb.text.lower() == "repariert"
    assert f.subject is not None and f.subject.text.lower() == "motor"
    assert f.agent is not None and f.agent.text.lower() == "mechaniker"
    assert f.agent_marker is not None and f.agent_marker.text.lower() == "vom"
    # Agent phrase spans the contracted marker through the head (15:29).
    assert f.agent_phrase is not None
    assert (f.agent_phrase.begin, f.agent_phrase.end) == (15, 29)
    assert f.subject_phrase is not None and f.subject_phrase.text == "Der Motor"


def test_canonical_passive_no_agent_stays_none():
    # Regression: an agentless canonical passive still has agent == None.
    findings = detect_passive(_load("positive_with_subject_en.conllu"))
    assert len(findings) == 1
    assert findings[0].agent is None
    assert findings[0].agent_marker is None


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
