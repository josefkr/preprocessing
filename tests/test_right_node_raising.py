"""Tests for the right-node-raising detector (coordination subset).

Fixtures are real Stanza parses (with t_start/t_end MISC), covering clausal RNR
(distinct subjects), stranded-preposition RNR (shared subject), and a non-RNR
VP-coordination look-alike that must NOT fire.
"""

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.right_node_raising import detect_right_node_raising

FIXTURES = Path(__file__).parent / "fixtures" / "right_node_raising"


def _load(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text(encoding="utf-8"))
    return doc


def test_clausal_rnr_distinct_subjects():
    # "Sam likes but Sue dislikes opera." — shared object "opera".
    findings = detect_right_node_raising(_load("positive_clausal_sam.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "coordination" and f.trigger == "distinct_subjects"
    assert f.left_predicate.text.lower() == "likes"
    assert f.right_predicate.text.lower() == "dislikes"
    assert f.shared_arg.text.lower() == "opera"


def test_clausal_rnr_multiword_shared_arg():
    # "Fred prepares and Susan eats the food." — shared object "the food".
    findings = detect_right_node_raising(_load("positive_clausal_fred.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.left_predicate.text.lower() == "prepares"
    assert f.right_predicate.text.lower() == "eats"
    assert f.shared_arg.text.lower() == "the food"


def test_stranded_preposition_rnr():
    # "She knew of but never mentioned my other work." — shared subject, but
    # "knew of __" strands a preposition -> RNR.
    findings = detect_right_node_raising(_load("positive_stranded_prep.conllu"))
    assert len(findings) == 1
    f = findings[0]
    assert f.trigger == "stranded_prep"
    assert f.left_predicate.text.lower() == "knew"
    assert f.right_predicate.text.lower() == "mentioned"
    assert f.shared_arg.text.lower() == "my other work"


def test_vp_coordination_not_rnr():
    # "John went and bought a fridge." — shared subject, "went" intransitive,
    # no stranded prep -> ordinary VP-coordination, NOT RNR.
    findings = detect_right_node_raising(_load("negative_vp_coord.conllu"))
    assert findings == []


def test_restrict_to_lang_filters():
    findings = detect_right_node_raising(
        _load("positive_clausal_sam.conllu"), restrict_to_lang="de"
    )
    assert findings == []
