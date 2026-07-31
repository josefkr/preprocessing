"""Tests for the pure suspended-composition detector (German Ergänzungsstrich).

Detection is deliberately resource-free — no morphology, no lexicon — so all of
these run anywhere. What needs checking is that the *surface* regex and the
surface->sofa offset mapping survive both of Stanza's tokenisations of a stub, since
that inconsistency is the whole reason the detector does not use the token stream.
"""

from __future__ import annotations

from pathlib import Path

from udapi.core.document import Document

from preprocessing.detection.suspended_composition import (
    detect_suspended_composition,
)

FIXTURES = Path(__file__).parent / "fixtures" / "suspended_composition"


def _doc(name: str) -> Document:
    doc = Document()
    doc.from_conllu_string((FIXTURES / name).read_text(encoding="utf-8"))
    return doc


class TestDetection:
    def test_finds_both_stubs(self):
        f = detect_suspended_composition(_doc("positive_de.conllu"),
                                         restrict_to_lang="de")
        assert [x.stub for x in f] == ["Vereins-", "be-"]

    def test_offsets_span_a_split_stub(self):
        """`Vereins-` is two tokens (`Vereins` 4-11 and `-` 11-12).

        The finding must cover both, so a consumer highlighting the anomaly gets the
        whole truncated conjunct rather than half of it.
        """
        f = detect_suspended_composition(_doc("positive_de.conllu"),
                                         restrict_to_lang="de")
        assert (f[0].begin, f[0].end) == (4, 12)

    def test_offsets_span_a_whole_token_stub(self):
        """`be-` arrives as a single token; same finding shape either way."""
        f = detect_suspended_composition(_doc("positive_de.conllu"),
                                         restrict_to_lang="de")
        assert (f[1].begin, f[1].end) == (54, 57)

    def test_internal_hyphens_are_not_suspensions(self):
        """`Horror-Videos` and `Sachsen-Anhalt` must not fire.

        The hyphen has to be word-final; this is the only thing separating an
        omission mark from an ordinary hyphenated compound or a place name.
        """
        f = detect_suspended_composition(_doc("negative_de.conllu"),
                                         restrict_to_lang="de")
        assert f == []

    def test_language_restriction(self):
        assert detect_suspended_composition(_doc("positive_de.conllu"),
                                            restrict_to_lang="en") == []

    def test_unresolved_without_a_resolver(self):
        """No resolver means sites but no completions — annotation still useful."""
        f = detect_suspended_composition(_doc("positive_de.conllu"),
                                         restrict_to_lang="de")
        assert all(x.completed is None and x.basis is None for x in f)
        assert all(x.donor is None for x in f)


class TestWithResolver:
    def test_resolver_attaches_completion_and_donor(self):
        """The optional resolver hook fills in the completion and donor offsets.

        Skipped unless the German resources are configured; the detector itself
        never needs them.
        """
        import pytest

        try:
            import sys

            sys.path.insert(
                0, str(Path(__file__).resolve().parents[2] / "normalization")
            )
            from aslan_normalization._dwds import get_headwords
            from aslan_normalization._smor import get_morphology
            from aslan_normalization._splitter import get_splitter
            from aslan_normalization._suspended import find_sites
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"normalization resources unavailable: {e}")

        morph, hw, sp = get_morphology(), get_headwords(), get_splitter()
        if not getattr(morph, "configured", False) or not hw.available:
            pytest.skip("needs ASLAN_SMOR_AUTOMATON and the DWDS headword list")

        def resolver(text, lang="de"):
            return find_sites(text, lang=lang, morph=morph, headwords=hw,
                              splitter=sp)

        f = detect_suspended_composition(_doc("positive_de.conllu"),
                                         restrict_to_lang="de",
                                         resolver=resolver)
        by_stub = {x.stub: x for x in f}
        assert by_stub["be-"].completed == "beladen"
        assert by_stub["be-"].donor == "entladen"
        # The donor is a real span in the sofa, not just a string.
        assert by_stub["be-"].donor_begin is not None


class TestMirrorDirection:
    """Leading-hyphen stubs: the shared constituent is the modifier.

    "Energieerzeugung und -verteilung" = Energieerzeugung und Energieverteilung.
    The donor *precedes* the stub here, the opposite of the trailing-hyphen case.
    """

    def test_finds_leading_hyphen_stub(self):
        f = detect_suspended_composition(_doc("positive_mirror_de.conllu"),
                                         restrict_to_lang="de")
        assert ("-verteilung", "shared_modifier") in [(x.stub, x.direction) for x in f]

    def test_both_directions_in_one_sentence(self):
        """"Frauen- und Kinderhandel , Drogenhandel und -konsum" has one of each.

        Neither pattern may assume it is the only one present, and the findings must
        come back in document order regardless of which pattern matched them.
        """
        f = detect_suspended_composition(_doc("positive_mirror_de.conllu"),
                                         restrict_to_lang="de")
        got = [(x.stub, x.direction) for x in f]
        assert got == [
            ("-verteilung", "shared_modifier"),
            ("Frauen-", "shared_head"),
            ("-konsum", "shared_modifier"),
        ]
        assert [x.begin for x in f] == sorted(x.begin for x in f)

    def test_leading_hyphen_offsets(self):
        f = detect_suspended_composition(_doc("positive_mirror_de.conllu"),
                                         restrict_to_lang="de")
        by = {x.stub: x for x in f}
        assert (by["-verteilung"].begin, by["-verteilung"].end) == (56, 67)
