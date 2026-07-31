"""Apply detectors to a CAS view and write annotations.

Each ``find_and_annotate_*`` entry point handles language plumbing
identically:

- ``lang=None, mixed=False`` → detect once on the whole sofa text;
  every sentence is tagged with that language. Skip detection if
  confidence is below threshold or the language is unsupported.
- ``lang=<X>, mixed=False`` → trust the user; every sentence is tagged
  ``<X>``. Optionally verify by running doc-level detection and
  warning on mismatch (the user remains authoritative).
- ``lang=None, mixed=True`` → detect each sentence independently; only
  sentences whose language is supported get a tag (others are skipped).
- ``lang=<X>, mixed=True`` → detect each sentence; the detector then
  filters to sentences whose detected language equals ``<X>``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cassis
from py_lift.dkpro import T_SENT
from udapi.core.document import Document

from preprocessing.detection.cas_conllu import view_to_conllu
from preprocessing.detection.language import (
    SUPPORTED_LANGS,
    detect_language,
)
from preprocessing.detection.bare_questions import (
    BareQuestionFinding,
    detect_bare_questions,
)
from preprocessing.detection.clefts import CleftFinding, detect_clefts
from preprocessing.detection.abbreviations import (
    AbbreviationFinding,
    detect_abbreviations,
)
from preprocessing.detection.contractions import (
    ContractionFinding,
    detect_contractions,
)
from preprocessing.detection.gapped_coordination import (
    GappedCoordinationFinding,
    detect_gapped_coordination,
)
from preprocessing.detection.nominal_ellipsis import (
    NominalEllipsisFinding,
    detect_nominal_ellipsis,
)
from preprocessing.detection.passive import PassiveFinding, detect_passive
from preprocessing.detection.suspended_composition import (
    SuspensionFinding,
    detect_suspended_composition,
)
from preprocessing.detection.right_node_raising import (
    RNRFinding,
    detect_right_node_raising,
)
from preprocessing.detection.sluicing import SluicingFinding, detect_sluicing
from preprocessing.detection.subject_sharing import (
    SubjectSharingFinding,
    detect_subject_sharing,
)
from preprocessing.detection.verbal_ellipsis import (
    VerbalEllipsisFinding,
    detect_verbal_ellipsis,
)

logger = logging.getLogger(__name__)

T_GRAMMAR_ANOMALY = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly"
T_LEXICAL_PHRASE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.LexicalPhrase"
T_SUGGESTED_ACTION = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.SuggestedAction"


def _resolve_sentence_langs(view, lang: str | None, mixed: bool) -> list[str | None] | None:
    """Compute the per-sentence ``# lang =`` tags to embed.

    Returns ``None`` if the view has no sentences. Otherwise returns a
    list aligned with the view's sentences. Each entry is either an
    ISO code or ``None`` (no tag).
    """
    sentences = list(view.select(T_SENT))
    if not sentences:
        return None

    sofa = view.sofa_string or ""
    n = len(sentences)

    if mixed:
        per: list[str | None] = []
        for sent in sentences:
            text = sofa[sent.begin:sent.end]
            detected = detect_language(text)
            if detected is None:
                per.append(None)
            elif detected not in SUPPORTED_LANGS:
                per.append(None)
            else:
                per.append(detected)
        return per

    # Monolingual mode: one language for the whole document.
    if lang is not None:
        # Trust the user; verify and warn on mismatch.
        detected = detect_language(sofa)
        if detected is not None and detected != lang:
            logger.warning(
                f"--lang={lang!r} but document language detected as {detected!r}; "
                "proceeding with user-supplied language."
            )
        return [lang] * n

    detected = detect_language(sofa)
    if detected is None or detected not in SUPPORTED_LANGS:
        logger.warning(
            "Could not confidently detect document language; "
            "skipping detection. Pass --lang explicitly to override."
        )
        return [None] * n
    return [detected] * n


def _build_doc(
    view, *, phenomenon: str, lang: str | None, mixed: bool
) -> tuple[Document | None, str | None]:
    """Convert ``view`` to a udapi Document, returning ``(doc, restrict_to_lang)``.

    ``restrict_to_lang`` is the value the detector should filter on;
    it is non-None only in mixed-mode with an explicit ``--lang``.
    Returns ``(None, None)`` when the view has no sentences.
    """
    sentence_langs = _resolve_sentence_langs(view, lang, mixed)
    if sentence_langs is None:
        logger.warning(
            f"View has no Sentence annotations; skipping {phenomenon} detection. "
            "Run sentence segmentation on this view first."
        )
        return None, None

    conllu_str = view_to_conllu(view, sentence_langs=sentence_langs)
    doc = Document()
    doc.from_conllu_string(conllu_str)
    restrict = lang if (mixed and lang is not None) else None
    return doc, restrict


def _write_abbreviations(view, ts, findings: list[AbbreviationFinding]) -> None:
    """A ``GrammarAnomaly`` per abbreviation candidate, carrying **every** ranked
    expansion as a ``SuggestedAction``.

    This is the first writer to use the ``suggestions`` FSArray for genuine
    ambiguity rather than a single rewrite: German short forms routinely have
    several readings (``VP`` has 22 in Wiktionary alone), and collapsing them at
    annotation time would throw away exactly what a later disambiguation step or
    a human reader needs. ``certainty`` carries the harvest's confidence, so the
    ranking survives too.

    An occurrence that sits at its own gloss ("Konditionierter Reiz (CS)") is
    still annotated — the phenomenon *is* there — but under the separate category
    ``abbreviation_defined``, so a normalizer can see at a glance that it must be
    left alone.
    """
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    SA = ts.get_type(T_SUGGESTED_ACTION)
    FSArray = ts.get_type("uima.cas.FSArray")
    for f in findings:
        actions = []
        for exp in f.expansions:
            action = SA(begin=f.begin, end=f.end, replacement=exp.form,
                        certainty=float(exp.certainty))
            view.add(action)
            actions.append(action)
        view.add(GA(
            begin=f.begin, end=f.end,
            description="Abbreviation",
            category="abbreviation_defined" if f.defined_in_context
                     else "abbreviation",
            suggestions=FSArray(elements=actions),
        ))



# Named per-phenomenon entry points. These are thin back-compat delegations to
# the generic :func:`find_and_annotate` (driven by :data:`DETECTOR_REGISTRY`,
# defined at the bottom of this module). Several normalizers import them by name.
def find_and_annotate_abbreviations(view, ts, *, lang=None, mixed=False) -> int:
    """Named entry point for the abbreviation detector (py_lift wrapper, CLI)."""
    return find_and_annotate(
        view, ts, phenomenon="abbreviation", lang=lang, mixed=mixed
    )


def find_and_annotate_sluicing(view, ts, *, lang=None, mixed=False) -> int:
    """Detect sluicing on ``view`` and add CAS annotations for each finding."""
    return find_and_annotate(view, ts, phenomenon="sluicing", lang=lang, mixed=mixed)


def find_and_annotate_subject_sharing(view, ts, *, lang=None, mixed=False) -> int:
    """Detect subject-sharing conjuncts on ``view`` and add annotations."""
    return find_and_annotate(view, ts, phenomenon="subject_sharing", lang=lang, mixed=mixed)


def find_and_annotate_verbal_ellipsis(view, ts, *, lang=None, mixed=False) -> int:
    """Detect verbal ellipsis on ``view`` and add CAS annotations."""
    return find_and_annotate(view, ts, phenomenon="verbal_ellipsis", lang=lang, mixed=mixed)


def find_and_annotate_passive(view, ts, *, lang=None, mixed=False) -> int:
    """Detect passive constructions on ``view`` and add CAS annotations."""
    return find_and_annotate(view, ts, phenomenon="passive", lang=lang, mixed=mixed)


def find_and_annotate_nominal_ellipsis(view, ts, *, lang=None, mixed=False) -> int:
    """Detect nominal-head ellipsis on ``view`` and add CAS annotations."""
    return find_and_annotate(view, ts, phenomenon="nominal_ellipsis", lang=lang, mixed=mixed)


def find_and_annotate_clefts(view, ts, *, lang=None, mixed=False) -> int:
    """Detect cleft constructions on ``view`` and add CAS annotations."""
    return find_and_annotate(view, ts, phenomenon="clefts", lang=lang, mixed=mixed)


def find_and_annotate_bare_questions(view, ts, *, lang=None, mixed=False) -> int:
    """Detect bare wh-question sentences on ``view`` and add CAS annotations."""
    return find_and_annotate(view, ts, phenomenon="bare_questions", lang=lang, mixed=mixed)


def find_and_annotate_gapped_coordination(view, ts, *, lang=None, mixed=False) -> int:
    """Detect gapped-coordination clauses on ``view`` and add CAS annotations."""
    return find_and_annotate(view, ts, phenomenon="gapped_coordination", lang=lang, mixed=mixed)


def find_and_annotate_right_node_raising(view, ts, *, lang=None, mixed=False) -> int:
    """Detect (coordination-subset) right node raising and add CAS annotations."""
    return find_and_annotate(view, ts, phenomenon="right_node_raising", lang=lang, mixed=mixed)


def find_and_annotate_suspended_composition(
    view, ts, *, lang=None, mixed=False
) -> int:
    """Detect suspended composition (Ergänzungsstrich) and add CAS annotations.

    Annotation-only: it records the sites and, where the resources are available,
    the completion as a ``SuggestedAction``. Applying the rewrite is the
    normalizer's job (``aslan_normalization.suspended_composition``).
    """
    return find_and_annotate(view, ts, phenomenon="suspended_composition",
                             lang=lang, mixed=mixed)


def _write_sluicing(view, ts, findings: list[SluicingFinding]) -> None:
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    for f in findings:
        view.add(GA(
            begin=f.x_begin, end=f.x_end,
            description="Ellipsis", category="sluicing",
        ))
        view.add(LP(begin=f.g_begin, end=f.g_end, text="QEmbedder"))


def _write_bare_questions(
    view, ts, findings: list[BareQuestionFinding]
) -> None:
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    for f in findings:
        view.add(GA(
            begin=f.begin, end=f.end,
            description="Ellipsis", category="bare_wh",
        ))


def _write_gapped_coordination(
    view, ts, findings: list[GappedCoordinationFinding]
) -> None:
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    for f in findings:
        view.add(GA(
            begin=f.begin, end=f.end,
            description="Ellipsis", category="gapped_coordination",
        ))
        view.add(LP(
            begin=f.antecedent_begin, end=f.antecedent_end,
            text="GappedAntecedent",
        ))


def _write_subject_sharing(
    view, ts, findings: list[SubjectSharingFinding]
) -> None:
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    for f in findings:
        view.add(GA(
            begin=f.x_begin, end=f.x_end,
            description="Ellipsis", category="right_conj_subject",
        ))
        for s in f.shared_subjects:
            view.add(LP(begin=s.begin, end=s.end, text="Shared_subject"))


def _write_verbal_ellipsis(
    view, ts, findings: list[VerbalEllipsisFinding]
) -> None:
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    for f in findings:
        view.add(GA(
            begin=f.begin, end=f.end,
            description="Ellipsis", category="auxiliary",
        ))


_CLEFT_TOKEN_TEXT = {
    "it_cleft": "Cleft_it",
    "wh_cleft": "Cleft_wh",
}


def _write_clefts(view, ts, findings: list[CleftFinding]) -> None:
    LP = ts.get_type(T_LEXICAL_PHRASE)
    for f in findings:
        view.add(LP(
            begin=f.focus.begin, end=f.focus.end, text="Cleft_focus",
        ))
        view.add(LP(
            begin=f.presupposition.begin, end=f.presupposition.end,
            text="Cleft_presupposition",
        ))
        view.add(LP(
            begin=f.cleft_token.begin, end=f.cleft_token.end,
            text=_CLEFT_TOKEN_TEXT.get(f.kind, "Cleft_it"),
        ))


def _write_nominal_ellipsis(
    view, ts, findings: list[NominalEllipsisFinding]
) -> None:
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    for f in findings:
        view.add(GA(
            begin=f.begin, end=f.end,
            description="Ellipsis",
            category=f"nominal_head_{f.subtype}",
        ))


def _write_right_node_raising(
    view, ts, findings: list[RNRFinding]
) -> None:
    """A ``GrammarAnomaly`` spanning the RNR construction (non-final predicate
    through the shared right-edge constituent) plus a ``LexicalPhrase`` on each
    role (the two predicates and the shared constituent)."""
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    for f in findings:
        view.add(GA(
            begin=f.left_predicate.begin, end=f.shared_arg.end,
            description="Ellipsis", category="right_node_raising",
        ))
        view.add(LP(
            begin=f.left_predicate.begin, end=f.left_predicate.end,
            text="RNR_left_predicate",
        ))
        view.add(LP(
            begin=f.right_predicate.begin, end=f.right_predicate.end,
            text="RNR_right_predicate",
        ))
        view.add(LP(
            begin=f.shared_arg.begin, end=f.shared_arg.end,
            text="RNR_shared_arg",
        ))


def _write_contractions(view, ts, findings: list[ContractionFinding]) -> None:
    """A ``GrammarAnomaly`` over the whole contraction ("wouldn't") plus a
    ``LexicalPhrase`` on each part. The expansion ("would not") is carried as a
    DKPro ``SuggestedAction`` in the anomaly's ``suggestions``, so consumers can
    read it without re-running the lexicon.

    ``suggestions`` is an ``FSArray`` (of ``SuggestedAction``), not a string —
    assigning a bare string only fails later, when the CAS is serialised to XMI.
    """
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    SA = ts.get_type(T_SUGGESTED_ACTION)
    FSArray = ts.get_type("uima.cas.FSArray")
    for f in findings:
        action = SA(begin=f.begin, end=f.end, replacement=f.expansion, certainty=1.0)
        view.add(action)
        view.add(GA(
            begin=f.begin, end=f.end,
            description="Contraction", category="contraction",
            suggestions=FSArray(elements=[action]),
        ))
        # Clitic findings mark the two surface parts; prep+article findings are
        # a single multiword surface token, so they carry no host/clitic parts.
        if f.host is not None:
            view.add(LP(begin=f.host.begin, end=f.host.end, text="Contraction_host"))
        if f.clitic is not None:
            view.add(LP(begin=f.clitic.begin, end=f.clitic.end, text="Contraction_clitic"))


def _write_suspended_composition(
    view, ts, findings: list[SuspensionFinding]
) -> None:
    """A ``GrammarAnomaly`` over the truncated conjunct, plus the donor it borrows
    from as a ``LexicalPhrase``.

    The category distinguishes what a consumer may do with the site:
    ``suspended_composition`` carries a completion in ``suggestions`` and can be
    normalized; ``suspended_composition_unresolved`` marks a real site whose split
    could not be settled (ambiguous, or the morphology/attestation resources were
    absent). Recording the second kind rather than dropping it is the point — the
    phenomenon is present either way, and a silently missing annotation is
    indistinguishable from a clean sentence.

    ``Suspension_donor`` is emitted only when a donor was identified. It is often
    several tokens away — real examples put it six away — so it is genuinely
    informative rather than derivable from adjacency. Which *side* it sits on
    encodes the direction: before the stub for a shared modifier
    ("Energieerzeugung und -verteilung"), after it for a shared head ("Sonn- und
    Feiertagen").
    """
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    SA = ts.get_type(T_SUGGESTED_ACTION)
    FSArray = ts.get_type("uima.cas.FSArray")
    for f in findings:
        suggestions = []
        if f.completed is not None:
            action = SA(begin=f.begin, end=f.end, replacement=f.completed,
                        certainty=1.0 if f.basis == "unique" else 0.7)
            view.add(action)
            suggestions.append(action)
        view.add(GA(
            begin=f.begin, end=f.end,
            # Direction rides in the description rather than the category, so the
            # registry's `ga_categories` (which drive re-run detection and
            # --replace) stay stable while consumers can still tell the two apart.
            # It is also derivable from geometry: the donor precedes the stub for a
            # shared modifier and follows it for a shared head.
            description=f"Suspended composition ({f.direction.replace('_', ' ')})",
            category=("suspended_composition" if f.completed is not None
                      else "suspended_composition_unresolved"),
            suggestions=FSArray(elements=suggestions),
        ))
        if f.donor_begin is not None and f.donor_end is not None:
            view.add(LP(begin=f.donor_begin, end=f.donor_end,
                        text="Suspension_donor"))


def _write_passive(view, ts, findings: list[PassiveFinding]) -> None:
    LP = ts.get_type(T_LEXICAL_PHRASE)
    for f in findings:
        view.add(LP(begin=f.verb.begin, end=f.verb.end, text="Passive_verb"))
        if f.aux is not None:
            view.add(LP(begin=f.aux.begin, end=f.aux.end, text="Passive_aux"))
        if f.subject is not None:
            view.add(LP(
                begin=f.subject.begin, end=f.subject.end,
                text="Passive_subject",
            ))
        if f.agent is not None:
            view.add(LP(
                begin=f.agent.begin, end=f.agent.end,
                text="Passive_agent",
            ))
        if f.agent_marker is not None:
            view.add(LP(
                begin=f.agent_marker.begin, end=f.agent_marker.end,
                text="Passive_agent_marker",
            ))


# --- detector registry -----------------------------------------------------
#
# Single source of truth mapping a phenomenon name to its pure detector, its CAS
# writer, and the annotation signature it produces (used to skip/replace on
# re-runs). The generic `annotate.py` CLI and :func:`find_and_annotate` /
# :func:`existing_annotations` are all driven by this table, so adding a new
# structural detector is one entry here plus a `detect_X` + `_write_X` — no new
# `add_X.py` script.


@dataclass(frozen=True)
class DetectorSpec:
    """How to run and identify one structural detector.

    ``detect`` is a pure ``detect_X(doc, *, restrict_to_lang=…)``; ``write`` is
    ``_write_X(view, ts, findings)``. The remaining fields describe the
    annotations the writer emits, so a re-run can find (and, with ``--replace``,
    remove) what a previous run created:
    ``ga_categories`` — exact ``GrammarAnomaly.category`` values;
    ``ga_category_prefixes`` — category prefixes (e.g. ``nominal_head_``);
    ``lp_texts`` — ``LexicalPhrase.text`` role labels.
    """

    name: str
    detect: Callable
    write: Callable
    label: str  # human-readable, for logging / language warnings
    ga_categories: frozenset[str] = frozenset()
    ga_category_prefixes: tuple[str, ...] = ()
    lp_texts: frozenset[str] = frozenset()


DETECTOR_REGISTRY: dict[str, DetectorSpec] = {
    "sluicing": DetectorSpec(
        "sluicing", detect_sluicing, _write_sluicing, "sluicing",
        ga_categories=frozenset({"sluicing"}), lp_texts=frozenset({"QEmbedder"}),
    ),
    "subject_sharing": DetectorSpec(
        "subject_sharing", detect_subject_sharing, _write_subject_sharing,
        "subject-sharing",
        ga_categories=frozenset({"right_conj_subject"}),
        lp_texts=frozenset({"Shared_subject"}),
    ),
    "verbal_ellipsis": DetectorSpec(
        "verbal_ellipsis", detect_verbal_ellipsis, _write_verbal_ellipsis,
        "verbal-ellipsis", ga_categories=frozenset({"auxiliary"}),
    ),
    "passive": DetectorSpec(
        "passive", detect_passive, _write_passive, "passive",
        lp_texts=frozenset({
            "Passive_verb", "Passive_aux", "Passive_subject",
            "Passive_agent", "Passive_agent_marker",
        }),
    ),
    "nominal_ellipsis": DetectorSpec(
        "nominal_ellipsis", detect_nominal_ellipsis, _write_nominal_ellipsis,
        "nominal-ellipsis", ga_category_prefixes=("nominal_head_",),
    ),
    "clefts": DetectorSpec(
        "clefts", detect_clefts, _write_clefts, "clefts",
        lp_texts=frozenset({
            "Cleft_focus", "Cleft_presupposition", "Cleft_it", "Cleft_wh",
        }),
    ),
    "bare_questions": DetectorSpec(
        "bare_questions", detect_bare_questions, _write_bare_questions,
        "bare-questions", ga_categories=frozenset({"bare_wh"}),
    ),
    "gapped_coordination": DetectorSpec(
        "gapped_coordination", detect_gapped_coordination,
        _write_gapped_coordination, "gapped-coordination",
        ga_categories=frozenset({"gapped_coordination"}),
        lp_texts=frozenset({"GappedAntecedent"}),
    ),
    "abbreviation": DetectorSpec(
        "abbreviation", detect_abbreviations, _write_abbreviations, "abbreviation",
        ga_categories=frozenset({"abbreviation", "abbreviation_defined"}),
    ),
    "contraction": DetectorSpec(
        "contraction", detect_contractions, _write_contractions, "contraction",
        ga_categories=frozenset({"contraction"}),
        lp_texts=frozenset({"Contraction_host", "Contraction_clitic"}),
    ),
    "suspended_composition": DetectorSpec(
        "suspended_composition", detect_suspended_composition,
        _write_suspended_composition, "suspended composition",
        ga_categories=frozenset({
            "suspended_composition", "suspended_composition_unresolved",
        }),
        lp_texts=frozenset({"Suspension_donor"}),
    ),
    "right_node_raising": DetectorSpec(
        "right_node_raising", detect_right_node_raising,
        _write_right_node_raising, "right-node-raising",
        ga_categories=frozenset({"right_node_raising"}),
        lp_texts=frozenset({
            "RNR_left_predicate", "RNR_right_predicate", "RNR_shared_arg",
        }),
    ),
}


def find_and_annotate(
    view, ts: cassis.TypeSystem, *, phenomenon: str,
    lang: str | None = None, mixed: bool = False,
) -> int:
    """Detect ``phenomenon`` on ``view`` and write its CAS annotations.

    Generic replacement for the per-phenomenon ``find_and_annotate_X``
    functions (which now delegate here). Returns the number of findings.
    """
    try:
        spec = DETECTOR_REGISTRY[phenomenon]
    except KeyError:
        raise ValueError(
            f"unknown phenomenon {phenomenon!r}; known: "
            f"{', '.join(sorted(DETECTOR_REGISTRY))}"
        ) from None
    doc, restrict = _build_doc(
        view, phenomenon=spec.label, lang=lang, mixed=mixed
    )
    if doc is None:
        return 0
    findings = spec.detect(doc, restrict_to_lang=restrict)
    spec.write(view, ts, findings)
    return len(findings)


def existing_annotations(view, phenomenon: str) -> list:
    """Annotations a previous run of ``phenomenon``'s detector would have
    created on ``view`` — used to skip already-annotated views and, with
    ``--replace``, to remove them before re-running."""
    spec = DETECTOR_REGISTRY[phenomenon]
    out: list = []
    if spec.ga_categories or spec.ga_category_prefixes:
        for a in view.select(T_GRAMMAR_ANOMALY):
            cat = getattr(a, "category", None) or ""
            if cat in spec.ga_categories or any(
                cat.startswith(p) for p in spec.ga_category_prefixes
            ):
                out.append(a)
    if spec.lp_texts:
        for lp in view.select(T_LEXICAL_PHRASE):
            if getattr(lp, "text", None) in spec.lp_texts:
                out.append(lp)
    return out
