"""Tests for the pure gapped-coordination detector.

The fixture `en_examples.conllu` is the Stanza parse of the English
TSV (`data/en_dummy/gapped_coordination.tsv`), with `# lang = en`
injected per sentence. Sentences 1–25 are the elliptical originals
(positives); sentences 26 and 27 are the parsed *resolutions* of the
last two TSV rows (negatives — well-formed verbal coordinations).

Per-sentence expectations are kept in a table at the top of the file
so that updates to the v1 rule (Signal A + Signal B) only need to
flip ✓/✗ in one place. Sentences marked ``False`` in EXPECTED_HITS
are misses that v1 explicitly does not catch (parser flattens the
gapped material into the antecedent clause, or attaches the gapped
subject as ``appos`` rather than ``conj``, etc.).
"""

from pathlib import Path

import pytest
from udapi.core.document import Document

from preprocessing.detection.gapped_coordination import detect_gapped_coordination

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "gapped_coordination"
    / "en_examples.conllu"
)

DE_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "gapped_coordination"
    / "de_examples.conllu"
)


# Per-sentence expected detector behaviour for v1 (Signal A + Signal B).
# True  = detector fires on this sentence (intended positive).
# False = detector does NOT fire on this sentence. For sent_ids 1..25
#         that's a known v1 miss (recorded so we can track recall over
#         time); for 26..27 it's the desired behaviour (these are the
#         well-formed coordinations parsed from the TSV's "correction"
#         column, so they MUST NOT fire).
EXPECTED_HITS: dict[int, bool] = {
    1: True,    # "Mr Leonard a coffee" — Signal A
    2: True,    # "Sue a mojito" — Signal A
    3: True,    # "Mr Leonard Nike" — Signal B (flat chain)
    4: True,    # "$5 to Pat" — Signal B (nmod under conj $)
    5: True,    # "Pat a barrister" — Signal A
    6: False,   # parser made "son" a compound of "medicine"; only one conj
    7: False,   # extra "an hour later" attaches via advmod to verb
    8: False,   # "the following year" attaches via obl:unmarked to verb
    9: True,    # "his mother Louise" — Signal B (flat Louise)
    10: False,  # "to join …" attaches via advcl to antecedent verb
    11: False,  # "Bill" attached as appos (not conj)
    12: True,   # "on Tuesday in Bonn" — Signal B (nmod Bonn under Tuesday)
    13: False,  # parser flattened "never with your right" into the verb
    14: False,  # conj parent is a participle VERB (resigning)
    15: True,   # "Bob a necklace …" — Signal B (appos necklace under Bob)
    16: False,  # parser made "my wife" an obj of "win", not a conj
    17: True,   # "of Pat irrelevant" — Signal B (nsubj Pat under irrelevant)
    18: True,   # "nor Jill hers" — Signal B (flat hers under Jill)
    19: True,   # "Pat just a t-shirt" — Signal A
    20: False,  # "on Tuesday" attaches via obl to verb (not under Pat)
    21: False,  # "to catch a bus" attaches via advcl to xcomp verb
    22: False,  # parser flattened into the antecedent advcl
    23: True,   # "Sue a postcard on Friday" — Signal B (nsubj Sue under postcard)
    24: False,  # "on Friday" attaches via obl to verb (not under Sue)
    25: True,   # "Pat some shorts" — Signal A
    26: False,  # well-formed coordination (parsed from TSV correction) — MUST not fire
    27: False,  # well-formed coordination (parsed from TSV correction) — MUST not fire
}


@pytest.fixture(scope="module")
def all_findings():
    """Run the detector once over the whole fixture document and
    index its findings by sentence id."""
    doc = Document()
    doc.from_conllu_string(FIXTURE.read_text())
    findings = detect_gapped_coordination(doc)
    # Bucket findings by sentence id: each tree's sentence-internal
    # ``t_start``/``t_end`` offsets restart at 0, so any finding whose
    # ``begin`` falls in [0, max_token_end] of a tree belongs to it.
    by_sid: dict[int, list] = {int(t.sent_id): [] for t in doc.trees}
    for tree in doc.trees:
        sid = int(tree.sent_id)
        ords = {n.ord for n in tree.descendants}
        # The detector emits each finding inside one tree; collect them
        # by checking which tree contains the offset of the leftmost token.
        max_end = max(int(n.misc["t_end"]) for n in tree.descendants)
        for f in findings:
            if 0 <= f.begin <= max_end and f not in sum(by_sid.values(), []):
                by_sid[sid].append(f)
                # remove from findings list so we don't double-count
        # We have to actually filter findings down once per tree;
        # simpler: re-detect per-tree below.
    # The "0 <= begin <= max_end" check is true for almost every finding
    # (every tree starts at 0), so re-detect per tree to be safe.
    by_sid = {}
    for tree in doc.trees:
        sub = Document()
        sub.from_conllu_string(_tree_to_conllu(tree))
        by_sid[int(tree.sent_id)] = detect_gapped_coordination(sub)
    return by_sid


def _tree_to_conllu(tree) -> str:
    """Re-serialise a single udapi tree as a self-contained CoNLL-U
    block (so the detector can be re-run per sentence). Preserves the
    ``t_start``/``t_end`` MISC fields the offset helper needs."""
    lines = []
    if tree.sent_id:
        lines.append(f"# sent_id = {tree.sent_id}")
    if tree.text:
        lines.append(f"# text = {tree.text}")
    lines.append("# lang = en")
    for n in tree.descendants:
        head = n.parent.ord if n.parent and not n.parent.is_root() else 0
        feats = str(n.feats) if n.feats else "_"
        deps = "_"
        if n.misc:
            misc = "|".join(f"{k}={v}" for k, v in n.misc.items())
        else:
            misc = "_"
        lines.append(
            f"{n.ord}\t{n.form}\t{n.lemma or '_'}\t{n.upos or '_'}\t"
            f"{n.xpos or '_'}\t{feats}\t{head}\t{n.deprel}\t{deps}\t{misc}"
        )
    return "\n".join(lines) + "\n\n"


@pytest.mark.parametrize("sid", sorted(EXPECTED_HITS))
def test_per_sentence(all_findings, sid):
    expected = EXPECTED_HITS[sid]
    actual_findings = all_findings.get(sid, [])
    actual = bool(actual_findings)
    assert actual == expected, (
        f"sentence {sid}: expected detector to "
        f"{'fire' if expected else 'not fire'}, got {len(actual_findings)} "
        f"finding(s)"
    )


def test_recall_summary(all_findings):
    """Track v1's recall on the elliptical originals (sentences 1..25).
    Adjust the threshold downward only after a deliberate detector
    change — this guard catches accidental precision/recall regressions
    on the v1 rule."""
    positives = [sid for sid in range(1, 26) if EXPECTED_HITS[sid]]
    misses = [sid for sid in range(1, 26) if not EXPECTED_HITS[sid]]
    caught = sum(1 for sid in positives if all_findings.get(sid))
    assert caught == len(positives), (
        f"v1 recall regressed: only {caught} of {len(positives)} "
        f"expected-positive sentences fired."
    )
    # Sanity: no parsed *correction* (26, 27) should fire — those are
    # the well-formed verbal coordinations.
    for neg in (26, 27):
        assert not all_findings.get(neg), (
            f"sentence {neg} fired but it is a well-formed coordination "
            "(parsed from a TSV correction)."
        )


def test_signal_a_emits_one_finding_per_cluster(all_findings):
    """Signal A collapses sibling conj children into a single finding;
    we should not emit duplicates anchored on each sibling."""
    for sid in (1, 2, 5, 19, 25):
        assert len(all_findings.get(sid, [])) == 1, (
            f"sent {sid}: Signal A should emit exactly one finding "
            f"covering the whole cluster, got {len(all_findings.get(sid, []))}"
        )


# ---------------------------------------------------------------------------
# German fixture (parses of `data/de_dummy/de_gapped_coordination.txt`,
# `# lang = de` and `t_start/t_end` injected). Sentences 1–24, 26, 28 are
# the elliptical originals; sentences 2, 25, 27 are full questions
# (negative controls — must NOT fire).
# ---------------------------------------------------------------------------

DE_EXPECTED_HITS: dict[int, bool] = {
    1: False,   # G2 shape — Kaffee attaches as 2nd `obj` of wollte, not under conj
    2: False,   # full question — negative control
    3: False,   # G2 shape — Mojito as 2nd `obj` of kaufte
    4: True,    # "Herr Leonard Nike" — Signal B (flat)
    5: True,    # "Pat 50 Dollar" — Signal C (conj of verb, nmod child)
    6: True,    # "Pat Rechtsanwältin" — Signal B (nsubj)
    7: True,    # "ihr Sohn Medizin" — Signal B (appos)
    8: False,   # G2 shape — "eine Stunde später" attaches to traf via amod/obl
    9: True,    # "ihre Eltern im Jahr darauf" — Signal C (nmod under conj)
    10: True,   # "seine Mutter Louise jedoch Mary" — Signal C (appos under conj)
    11: False,  # parser sees two full inf-clauses, no gap in tree
    12: False,  # parser attaches Willy as appos of Firma (not conj)
    13: True,   # "am Dienstag in Bonn" — Signal C (nmod under Dienstag)
    14: False,  # Hand conj of Hand, no GAPPED_ARG_RELS children
    15: True,   # "Bob … eine Halskette" — Signal C (nsubj+nmod under Halskette)
    16: False,  # parser made Frau an appos of gewinnt (not conj)
    17: True,   # "die Kritik an Pat irrelevant" — Signal B (nsubj)
    18: False,  # Jill conj of beendet but has only advmod child
    19: True,   # "Pat nur ein T-Shirt" — Signal C (nmod under Pat)
    20: False,  # G2 shape — second `am Dienstag` attaches to war
    21: False,  # parser sees two full inf-clauses, no gap in tree
    22: False,  # parser flattened the gap into an advcl chain (no conj)
    23: True,   # "Sue … eine Postkarte" — Signal C (nsubj+nmod under Postkarte)
    24: True,   # "Sue am Freitag einen Brief" — Signal C (nmod under Sue)
    25: False,  # full question — negative control
    26: True,   # "Pat Shorts" — Signal B (flat)
    27: False,  # full question — negative control
    28: True,   # "Pat Shorts" — Signal B (flat)
}


@pytest.fixture(scope="module")
def de_all_findings():
    """Run the detector over the German fixture, bucketed by sentence id."""
    doc = Document()
    doc.from_conllu_string(DE_FIXTURE.read_text())
    by_sid: dict[int, list] = {}
    for tree in doc.trees:
        sub = Document()
        sub.from_conllu_string(_tree_to_conllu(tree))
        by_sid[int(tree.sent_id)] = detect_gapped_coordination(sub)
    return by_sid


@pytest.mark.parametrize("sid", sorted(DE_EXPECTED_HITS))
def test_de_per_sentence(de_all_findings, sid):
    expected = DE_EXPECTED_HITS[sid]
    actual_findings = de_all_findings.get(sid, [])
    actual = bool(actual_findings)
    assert actual == expected, (
        f"German sentence {sid}: expected detector to "
        f"{'fire' if expected else 'not fire'}, got {len(actual_findings)} "
        f"finding(s)"
    )


def test_de_recall_summary(de_all_findings):
    """Track v1+SignalC recall on the German elliptical originals."""
    positives = [
        sid for sid in DE_EXPECTED_HITS
        if DE_EXPECTED_HITS[sid] and sid not in {2, 25, 27}
    ]
    caught = sum(1 for sid in positives if de_all_findings.get(sid))
    assert caught == len(positives), (
        f"German recall regressed: only {caught} of {len(positives)} "
        f"expected-positive sentences fired."
    )
    # Negative controls (the three full questions in the TXT) MUST not fire.
    for neg in (2, 25, 27):
        assert not de_all_findings.get(neg), (
            f"German sentence {neg} is a full question but the detector fired."
        )


def test_de_signal_c_used(de_all_findings):
    """Signal C is the German-specific reason for adding the
    parent-is-VERB branch; at least a few German positives should be
    flagged with signal=='C' (regression guard against Signal C being
    silently disabled)."""
    c_count = sum(
        1
        for sid, findings in de_all_findings.items()
        if findings and findings[0].signal == "C"
    )
    assert c_count >= 5, (
        f"Expected Signal C to fire on at least 5 German sentences, "
        f"got {c_count}"
    )


def test_antecedent_resolution(all_findings):
    """The antecedent field on each finding should pick the right verb."""
    expected_antecedents = {
        1: "wanted",
        2: "bought",
        3: "wore",
        4: "gave",
        5: "is",        # cop-predicate; ancestor walk lands on the AUX
        9: "wanted",
        12: "been",     # cop AUX above the predicative location
        15: "given",
        17: "were",     # cop AUX above the ADJ predicate
        18: "finished",
        19: "have",
        23: "sent",
        25: "bought",
    }
    for sid, want in expected_antecedents.items():
        findings = all_findings.get(sid, [])
        assert findings, f"sent {sid}: expected one finding"
        assert findings[0].antecedent_text == want, (
            f"sent {sid}: expected antecedent {want!r}, "
            f"got {findings[0].antecedent_text!r}"
        )
