"""py_lift-style annotator wrappers around our structural detectors.

This is the Option-1 rollout from ``PYLIFT_INTEGRATION.md``: expose our
existing ``find_and_annotate_*`` adapters through py_lift's uniform annotator
contract (:class:`py_lift.annotators.api.SEL_BaseAnnotator`) so each detector
becomes drop-in usable wherever a py_lift annotator is expected, and gains
py_lift's TypeSystem / required-type / language validation for free.

The pure detector core (``detect_<phenomenon>``) and the CAS adapter
(``find_and_annotate_<phenomenon>``) are **untouched** — a wrapper's
``_process`` merely delegates to the adapter. Both entry points stay available:

* ``SE_<Phenom>Annotator(language).process(cas)`` — the py_lift path
  (single ``language`` in ``__init__``, with validation);
* ``find_and_annotate_<phenomenon>(view, ts, lang=…, mixed=…)`` — the existing
  path, which is the only one that keeps our per-sentence ``--mixed`` mode.

All structural detectors run through the same CAS→CoNLL-U→udapi bridge, which
needs Token / POS / Dependency / Sentence annotations, so ``requires_types`` is
declared once on the base. Only ``supported_languages`` (and the adapter) vary
per phenomenon; the values mirror the language wiring in ``PHENOMENA_COVERAGE.md``.

The external-tool annotators (coreference, RWSE, spelling, EDUs) are **not**
wrapped here — they keep their own ``add_*.py`` scripts because they wrap a
service/model rather than a structural detector.
"""

from __future__ import annotations

from typing import Callable

from cassis import Cas
from py_lift.annotators.api import SEL_BaseAnnotator
from py_lift.decorators import requires_types, supported_languages
from py_lift.dkpro import T_DEP, T_POS, T_SENT, T_TOKEN

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


@requires_types(T_TOKEN, T_POS, T_DEP, T_SENT)
class _DetectorAnnotator(SEL_BaseAnnotator):
    """Base wrapper: run a ``find_and_annotate_*`` adapter as a py_lift annotator.

    Subclasses set :attr:`_adapter` (a ``find_and_annotate_<phenomenon>``) and
    declare ``@supported_languages``. ``process(cas)`` (inherited) validates the
    CAS typesystem, the required parse types, and the language before
    ``_process`` runs the detector on the CAS's default view.
    """

    #: The ``find_and_annotate_<phenomenon>(view, ts, *, lang=…)`` adapter.
    _adapter: Callable[..., int]

    def _process(self, cas: Cas) -> bool:
        # The CAS is its own default (_Initial) view; the adapter writes back
        # into it. self.ts is the LIFT singleton, which require_same_typesystem
        # has already confirmed is the CAS's typesystem.
        n = type(self)._adapter(cas, self.ts, lang=self.language)
        return n > 0


@supported_languages("en", "de")
class SE_SluicingAnnotator(_DetectorAnnotator):
    """Sluicing detector as a py_lift annotator (EN/DE)."""

    _adapter = staticmethod(find_and_annotate_sluicing)


@supported_languages("en", "de")
class SE_BareQuestionsAnnotator(_DetectorAnnotator):
    """Bare wh-question detector as a py_lift annotator (EN/DE)."""

    _adapter = staticmethod(find_and_annotate_bare_questions)


@supported_languages("en", "de")
class SE_NominalEllipsisAnnotator(_DetectorAnnotator):
    """Nominal-head ellipsis detector as a py_lift annotator (EN/DE)."""

    _adapter = staticmethod(find_and_annotate_nominal_ellipsis)


@supported_languages("en", "de")
class SE_VerbalEllipsisAnnotator(_DetectorAnnotator):
    """Verbal-ellipsis (VPE) detector as a py_lift annotator (EN/DE)."""

    _adapter = staticmethod(find_and_annotate_verbal_ellipsis)


@supported_languages("en", "de")
class SE_GappedCoordinationAnnotator(_DetectorAnnotator):
    """Gapped-coordination detector as a py_lift annotator (EN/DE)."""

    _adapter = staticmethod(find_and_annotate_gapped_coordination)


@supported_languages("en", "de")
class SE_PassiveAnnotator(_DetectorAnnotator):
    """Passive detector (canonical + short) as a py_lift annotator (EN/DE)."""

    _adapter = staticmethod(find_and_annotate_passive)


@supported_languages("en", "de")
class SE_SubjectSharingAnnotator(_DetectorAnnotator):
    """Subject-sharing detector as a py_lift annotator (structural; EN/DE)."""

    _adapter = staticmethod(find_and_annotate_subject_sharing)


@supported_languages("en")
class SE_CleftsAnnotator(_DetectorAnnotator):
    """Cleft detector (it-clefts + wh-clefts) as a py_lift annotator (EN)."""

    _adapter = staticmethod(find_and_annotate_clefts)


@supported_languages("en")
class SE_RightNodeRaisingAnnotator(_DetectorAnnotator):
    """Right-node-raising detector as a py_lift annotator (EN, coordination subset).

    Writes the same ``GrammarAnomaly(category="right_node_raising")`` + ``RNR_*``
    ``LexicalPhrase`` annotations as ``annotate.py --phenomenon right_node_raising``.
    """

    _adapter = staticmethod(find_and_annotate_right_node_raising)


@supported_languages("de", "en")
class SE_AbbreviationAnnotator(_DetectorAnnotator):
    """Abbreviation detector as a py_lift annotator (DE, EN).

    Writes ``GrammarAnomaly(category="abbreviation")`` — or
    ``"abbreviation_defined"`` where the long form accompanies the occurrence —
    with every ranked expansion carried as a ``SuggestedAction``. Standalone
    detection is useful even without a corpus: the candidates and their gate
    decisions are recorded, and expansions are attached later by the
    corpus-level harvest.
    """

    _adapter = staticmethod(find_and_annotate_abbreviations)


@supported_languages("de")
class SE_SuspendedCompositionAnnotator(_DetectorAnnotator):
    """Suspended-composition detector as a py_lift annotator (DE).

    Writes ``GrammarAnomaly(category="suspended_composition")`` — or
    ``"…_unresolved"`` where the split could not be settled — with the completion
    carried as a ``SuggestedAction``, plus ``LexicalPhrase(text="Suspension_donor")``
    over the conjunct the material was borrowed from.

    German-only for now: resolution leans on German morphology and a German
    attestation lexicon. English has the same construction ("pre- and post-war"),
    but nothing equivalent to SMOR to resolve it with.
    """

    _adapter = staticmethod(find_and_annotate_suspended_composition)


#: Phenomenon key (as in ``DETECTOR_REGISTRY`` / ``annotate.py --phenomenon``)
#: → its py_lift annotator wrapper. Keep in sync when a detector is added.
ANNOTATORS: dict[str, type[_DetectorAnnotator]] = {
    "abbreviation": SE_AbbreviationAnnotator,
    "suspended_composition": SE_SuspendedCompositionAnnotator,
    "sluicing": SE_SluicingAnnotator,
    "bare_questions": SE_BareQuestionsAnnotator,
    "nominal_ellipsis": SE_NominalEllipsisAnnotator,
    "verbal_ellipsis": SE_VerbalEllipsisAnnotator,
    "gapped_coordination": SE_GappedCoordinationAnnotator,
    "passive": SE_PassiveAnnotator,
    "subject_sharing": SE_SubjectSharingAnnotator,
    "clefts": SE_CleftsAnnotator,
    "right_node_raising": SE_RightNodeRaisingAnnotator,
}
