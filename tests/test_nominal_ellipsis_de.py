"""Tests for the German nominal-head ellipsis rules (``nominal_ellipsis_de``).

The English detector and the German one share an entry point
(``detect_nominal_ellipsis``) but not their rules: German has its own module
because Stanza's German output uses STTS XPOS (``PIS``/``ADJA``/``ART``…) and
carries degree/number only in ``feats`` — see ``nominal_ellipsis_de.py``.

The fixtures are authentic Stanza German parses (one sentence each, chosen so
exactly one elided-head site fires), except ``positive_possessive_de`` which is
hand-authored: Stanza's current German model tags substituting possessives
``PPER``/``PIS`` and never emits ``PPOSS``, so the ``PPOSS`` rule can only be
covered with a synthetic parse.
"""

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
    f = findings[0]
    assert f.lang == "de"
    return f


def test_quantifier():
    # "Viele kamen." — substituting indefinite pronoun (STTS PIS).
    f = _only("positive_quantifier_de.conllu")
    assert f.subtype == "quantifier"
    assert f.text == "Viele"
    assert f.deprel == "nsubj"


def test_cardinal():
    # "Paul kaufte vier." — bare cardinal acting as a head.
    f = _only("positive_cardinal_de.conllu")
    assert f.subtype == "cardinal"
    assert f.text == "vier"
    assert f.deprel == "obj"


def test_ordinal():
    # "Er nahm den zweiten." — ADJA with NumType=Ord as a head.
    f = _only("positive_ordinal_de.conllu")
    assert f.subtype == "ordinal"
    assert f.text == "zweiten"
    assert f.deprel == "obj"


def test_comparative():
    # "Sie wählte das neuere." — ADJA with Degree=Cmp as a head.
    f = _only("positive_comparative_de.conllu")
    assert f.subtype == "comparative"
    assert f.text == "neuere"
    assert f.deprel == "obj"


def test_superlative():
    # "Das älteste ist verkauft." — Stanza drops Degree on the nominalised
    # superlative, so the "-ste" suffix fallback is what classifies it.
    f = _only("positive_superlative_de.conllu")
    assert f.subtype == "superlative"
    assert f.text == "älteste"
    assert f.deprel == "nsubj"


def test_adjective():
    # "Der große gewann." — ADJA head with no ordinal/comparative/superlative
    # cue, caught by the catch-all.
    f = _only("positive_adjective_de.conllu")
    assert f.subtype == "adjective"
    assert f.text == "große"
    assert f.deprel == "nsubj"


def test_demonstrative_pronoun():
    # "Die aus Berlin kamen zuerst." — definite article acting as a head
    # (deprel != det) immediately followed by a preposition; the noun is elided.
    f = _only("positive_demonstrative_de.conllu")
    assert f.subtype == "demonstrative_pronoun"
    assert f.text == "Die"
    assert f.deprel == "nsubj"


def test_possessive_pronoun():
    # "Seiner ist kaputt." — substituting possessive (STTS PPOSS). Synthetic
    # fixture: see module docstring.
    f = _only("positive_possessive_de.conllu")
    assert f.subtype == "possessive_pronoun"
    assert f.text == "Seiner"
    assert f.deprel == "nsubj"


def test_attributive_adjective_is_not_ellipsis():
    # "Der große Mann gewann." — "große" is an amod modifier of "Mann", not a
    # head, so it is not an elided-head site.
    findings = detect_nominal_ellipsis(_load("negative_amod_de.conllu"))
    assert findings == []


def test_restrict_to_lang_filters_other_languages():
    findings = detect_nominal_ellipsis(
        _load("positive_quantifier_de.conllu"), restrict_to_lang="en"
    )
    assert findings == []
