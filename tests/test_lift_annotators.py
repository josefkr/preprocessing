"""Smoke tests for the py_lift-style detector-annotator wrappers.

For each wrapped structural detector, verify that
``SE_<Phenom>Annotator(language).process(cas)``:

1. passes py_lift's ``SEL_BaseAnnotator`` validation (typesystem / required
   parse types / language), and
2. writes exactly the same annotations as the current adapter path
   (``find_and_annotate_<phenomenon>``, which is what
   ``annotate.py --phenomenon <phenomenon>`` invokes).

Each CAS is built from an existing ``.conllu`` detector fixture using the
**LIFT typesystem** — the same one ``annotate.py`` loads XMIs with
(``get_lift_typesystem()``), which also carries ``GrammarAnomaly`` /
``LexicalPhrase``. Building with the LIFT singleton is what lets
``require_same_typesystem`` pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cassis import Cas
from py_lift.annotators.api import UnsupportedLanguageError
from py_lift.dkpro import T_DEP, T_LEMMA, T_POS, T_SENT, T_TOKEN
from py_lift.util import get_lift_typesystem

from preprocessing.detection import lift_annotators as la
from preprocessing.detection.cas_adapter import (
    find_and_annotate_abbreviations,
    find_and_annotate_bare_questions,
    find_and_annotate_clefts,
    find_and_annotate_gapped_coordination,
    find_and_annotate_nominal_ellipsis,
    find_and_annotate_passive,
    find_and_annotate_right_node_raising,
    find_and_annotate_sluicing,
    find_and_annotate_subject_sharing,
    find_and_annotate_suspended_composition,
    find_and_annotate_verbal_ellipsis,
)

FIXTURES = Path(__file__).parent / "fixtures"

T_GRAMMAR_ANOMALY = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly"
T_LEXICAL_PHRASE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.LexicalPhrase"

# (phenomenon key, annotator class, adapter fn, fixture (relative), language).
# One positive fixture per phenomenon; gapped-coordination is multi-sentence.
CASES = [
    ("abbreviation", la.SE_AbbreviationAnnotator, find_and_annotate_abbreviations,
     "abbreviations/positive_kg_de.conllu", "de"),
    ("suspended_composition", la.SE_SuspendedCompositionAnnotator,
     find_and_annotate_suspended_composition,
     "suspended_composition/positive_de.conllu", "de"),
    ("sluicing", la.SE_SluicingAnnotator, find_and_annotate_sluicing,
     "sluicing/positive_en_basic.conllu", "en"),
    ("bare_questions", la.SE_BareQuestionsAnnotator, find_and_annotate_bare_questions,
     "bare_questions/positive_de_an_wen.conllu", "de"),
    ("nominal_ellipsis", la.SE_NominalEllipsisAnnotator, find_and_annotate_nominal_ellipsis,
     "nominal_ellipsis/positive_adjective_en.conllu", "en"),
    ("verbal_ellipsis", la.SE_VerbalEllipsisAnnotator, find_and_annotate_verbal_ellipsis,
     "verbal_ellipsis/positive_aux_as_root_en.conllu", "en"),
    ("gapped_coordination", la.SE_GappedCoordinationAnnotator, find_and_annotate_gapped_coordination,
     "gapped_coordination/en_examples.conllu", "en"),
    ("passive", la.SE_PassiveAnnotator, find_and_annotate_passive,
     "passive/positive_canonical_agent_de.conllu", "de"),
    ("subject_sharing", la.SE_SubjectSharingAnnotator, find_and_annotate_subject_sharing,
     "subject_sharing/positive_basic_en.conllu", "en"),
    ("clefts", la.SE_CleftsAnnotator, find_and_annotate_clefts,
     "clefts/positive_simple_en.conllu", "en"),
    ("right_node_raising", la.SE_RightNodeRaisingAnnotator, find_and_annotate_right_node_raising,
     "right_node_raising/positive_clausal_fred_en.conllu", "en"),
]


def _parse_conllu(text: str):
    """Parse a (possibly multi-sentence) CoNLL-U fixture into per-sentence dicts.

    We ignore the fixtures' MISC offsets (some are hand-authored and don't
    align to the sofa) and rebuild the sofa from the token forms with
    contiguous offsets — detection uses forms/lemmas/tree structure, and both
    the reference and wrapper paths share the same CAS, so absolute offsets are
    irrelevant to the comparison.
    """
    sentences: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        if line.startswith("# text ="):
            cur = {"text": line.split("=", 1)[1].strip(), "lang": None, "rows": []}
            sentences.append(cur)
            continue
        if line.startswith("# lang =") and cur is not None:
            cur["lang"] = line.split("=", 1)[1].strip()
            continue
        if not line.strip() or line.startswith("#") or cur is None:
            continue
        cols = line.split("\t")
        # Skip CoNLL-U multiword-token ranges ("3-4") and empty nodes ("3.1").
        if "-" in cols[0] or "." in cols[0]:
            continue
        cur["rows"].append(
            {
                "id": int(cols[0]),
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                "xpos": cols[4],
                "head": int(cols[6]),
                "deprel": cols[7],
            }
        )
    return sentences


def _build_cas(fixture_rel: str, language: str) -> Cas:
    """Build a parsed CAS (Sentence/Token/POS/Lemma/Dependency) from a fixture,
    using the LIFT typesystem exactly as annotate.py does. The sofa is rebuilt
    from token forms (space-joined per sentence, newline between sentences), so
    forms/lemmas are exact regardless of the fixture's MISC offsets."""
    sentences = _parse_conllu((FIXTURES / fixture_rel).read_text())
    assert sentences, f"no sentences parsed from {fixture_rel}"

    ts = get_lift_typesystem()
    Sent = ts.get_type(T_SENT)
    Tok = ts.get_type(T_TOKEN)
    POS = ts.get_type(T_POS)
    Dep = ts.get_type(T_DEP)
    Lemma = ts.get_type(T_LEMMA)

    # Assign contiguous offsets from the forms; track each sentence's span.
    base = 0
    sofa_parts = []
    for s in sentences:
        pos = base
        spans = []
        for r in s["rows"]:
            spans.append((pos, pos + len(r["form"])))
            pos += len(r["form"]) + 1  # single space between tokens
        s["spans"] = spans
        text = " ".join(r["form"] for r in s["rows"])
        s["span"] = (base, base + len(text))
        sofa_parts.append(text)
        base += len(text) + 1  # newline between sentences

    cas = Cas(sofa_string="\n".join(sofa_parts), document_language=language, typesystem=ts)

    for s in sentences:
        cas.add(Sent(begin=s["span"][0], end=s["span"][1]))
        tokens_by_id = {}
        for r, (b, e) in zip(s["rows"], s["spans"]):
            tok = Tok(begin=b, end=e)
            cas.add(tok)
            tokens_by_id[r["id"]] = tok
            cas.add(POS(begin=b, end=e, coarseValue=r["upos"], PosValue=r["xpos"]))
            cas.add(Lemma(begin=b, end=e, value=r["lemma"]))
        for r, (b, e) in zip(s["rows"], s["spans"]):
            dependent = tokens_by_id[r["id"]]
            if r["head"] == 0:  # DKPro convention: root's Governor points to itself.
                cas.add(Dep(begin=b, end=e, Governor=dependent,
                            Dependent=dependent, DependencyType="root"))
            else:
                cas.add(Dep(begin=b, end=e, Governor=tokens_by_id[r["head"]],
                            Dependent=dependent, DependencyType=r["deprel"]))

    return cas


def _grammar_anomalies(cas: Cas):
    return sorted(
        (a.begin, a.end, getattr(a, "category", None), getattr(a, "description", None))
        for a in cas.select(T_GRAMMAR_ANOMALY)
    )


def _lexical_phrases(cas: Cas):
    return sorted(
        (p.begin, p.end, getattr(p, "text", None))
        for p in cas.select(T_LEXICAL_PHRASE)
    )


@pytest.mark.parametrize(
    "phenomenon,annotator_cls,adapter,fixture,language",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_wrapper_matches_adapter(phenomenon, annotator_cls, adapter, fixture, language):
    # Reference path: the adapter the CLI (`annotate.py -p <phenomenon>`) calls.
    cas_ref = _build_cas(fixture, language)
    n_ref = adapter(cas_ref, cas_ref.typesystem, lang=language)
    assert n_ref > 0, f"expected the reference adapter to detect {phenomenon}"

    # Wrapper path: goes through SEL_BaseAnnotator.process() validation.
    cas_wrap = _build_cas(fixture, language)
    added = annotator_cls(language).process(cas_wrap)

    assert added is True
    assert _grammar_anomalies(cas_wrap) == _grammar_anomalies(cas_ref)
    assert _lexical_phrases(cas_wrap) == _lexical_phrases(cas_ref)


def test_registry_covers_all_structural_detectors():
    # Every wrapper is reachable by its DETECTOR_REGISTRY phenomenon key.
    assert set(la.ANNOTATORS) == {c[0] for c in CASES}
    for key, cls in la.ANNOTATORS.items():
        assert set(cls.requires_types) == {T_TOKEN, T_POS, T_DEP, T_SENT}
        assert cls.supported_languages  # non-empty (restriction declared)


def test_wrapper_rejects_unsupported_language():
    # Clefts and RNR are EN-only; strict mode rejects an unsupported language.
    with pytest.raises(UnsupportedLanguageError):
        la.SE_CleftsAnnotator("de")
    with pytest.raises(UnsupportedLanguageError):
        la.SE_RightNodeRaisingAnnotator("de")
