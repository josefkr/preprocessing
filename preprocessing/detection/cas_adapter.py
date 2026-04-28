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

import cassis
from py_lift.dkpro import T_SENT
from udapi.core.document import Document

from preprocessing.detection.cas_conllu import view_to_conllu
from preprocessing.detection.language import (
    SUPPORTED_LANGS,
    detect_language,
)
from preprocessing.detection.nominal_ellipsis import (
    NominalEllipsisFinding,
    detect_nominal_ellipsis,
)
from preprocessing.detection.passive import PassiveFinding, detect_passive
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


def find_and_annotate_sluicing(
    view, ts: cassis.TypeSystem, *, lang: str | None = None, mixed: bool = False
) -> int:
    """Detect sluicing on ``view`` and add CAS annotations for each finding."""
    doc, restrict = _build_doc(
        view, phenomenon="sluicing", lang=lang, mixed=mixed
    )
    if doc is None:
        return 0
    findings = detect_sluicing(doc, restrict_to_lang=restrict)
    _write_sluicing(view, ts, findings)
    return len(findings)


def find_and_annotate_subject_sharing(
    view, ts: cassis.TypeSystem, *, lang: str | None = None, mixed: bool = False
) -> int:
    """Detect subject-sharing conjuncts on ``view`` and add annotations."""
    doc, restrict = _build_doc(
        view, phenomenon="subject-sharing", lang=lang, mixed=mixed
    )
    if doc is None:
        return 0
    findings = detect_subject_sharing(doc, restrict_to_lang=restrict)
    _write_subject_sharing(view, ts, findings)
    return len(findings)


def find_and_annotate_verbal_ellipsis(
    view, ts: cassis.TypeSystem, *, lang: str | None = None, mixed: bool = False
) -> int:
    """Detect verbal ellipsis on ``view`` and add CAS annotations."""
    doc, restrict = _build_doc(
        view, phenomenon="verbal-ellipsis", lang=lang, mixed=mixed
    )
    if doc is None:
        return 0
    findings = detect_verbal_ellipsis(doc, restrict_to_lang=restrict)
    _write_verbal_ellipsis(view, ts, findings)
    return len(findings)


def find_and_annotate_passive(
    view, ts: cassis.TypeSystem, *, lang: str | None = None, mixed: bool = False
) -> int:
    """Detect passive constructions on ``view`` and add CAS annotations."""
    doc, restrict = _build_doc(
        view, phenomenon="passive", lang=lang, mixed=mixed
    )
    if doc is None:
        return 0
    findings = detect_passive(doc, restrict_to_lang=restrict)
    _write_passive(view, ts, findings)
    return len(findings)


def find_and_annotate_nominal_ellipsis(
    view, ts: cassis.TypeSystem, *, lang: str | None = None, mixed: bool = False
) -> int:
    """Detect nominal-head ellipsis on ``view`` and add CAS annotations."""
    doc, restrict = _build_doc(
        view, phenomenon="nominal-ellipsis", lang=lang, mixed=mixed
    )
    if doc is None:
        return 0
    findings = detect_nominal_ellipsis(doc, restrict_to_lang=restrict)
    _write_nominal_ellipsis(view, ts, findings)
    return len(findings)


def _write_sluicing(view, ts, findings: list[SluicingFinding]) -> None:
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    for f in findings:
        view.add(GA(
            begin=f.x_begin, end=f.x_end,
            description="Ellipsis", category="sluicing",
        ))
        view.add(LP(begin=f.g_begin, end=f.g_end, text="QEmbedder"))


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
