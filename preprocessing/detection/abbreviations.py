"""Pure abbreviation detector (German, English).

Operates on a udapi document and returns findings; no CAS dependency.

An **abbreviation candidate** is an all-caps letter run of 2–6 characters that
survives three gates, each answering a false-positive class seen in real exam
answers (see ``lexicons/abbreviations.py`` for the data behind them):

1. it is not a **function word** in all caps — "Test *VOR* dem Lernen" is
   emphasis, not an abbreviation;
2. it is not part of an **enumeration** — "bsp.: ADE - BEC - CBA - D" lists
   learning materials, and a run of adjacent all-caps tokens separated only by
   punctuation is never a list of abbreviations;
3. the document is not **mostly capitalised**, in which case capitalisation
   carries no signal at all.

Detection is separate from *expansion*: which long form an abbreviation stands
for is a corpus-level decision (a definition in one answer resolves a bare use in
another), made by ``resolution/abbreviations/harvest.py``. This detector
therefore annotates candidates whether or not an expansion is known, and accepts
an optional ``expansions`` map to attach ranked suggestions when one is.

Occurrences that sit at **their own gloss** ("Konditionierter Reiz (CS)") are
flagged rather than dropped: the phenomenon is present, but normalizing it would
be wrong, so the annotation records both facts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from preprocessing.detection.language import tree_lang
from preprocessing.detection.lexicons.abbreviations import (
    CANDIDATE_RE,
    caps_suppressed,
    enumeration_offsets,
    glossed_offsets,
    is_function_word,
)
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AbbreviationExpansion:
    """One candidate long form for an abbreviation."""

    form: str
    source: str = "lexicon"     # definition | corpus_acronym | lexicon | llm
    certainty: float = 0.0


@dataclass(frozen=True)
class AbbreviationFinding:
    begin: int
    end: int
    text: str                   # the abbreviation as written
    lang: str | None
    #: Ranked candidate expansions; empty when nothing could be proposed. The CAS
    #: writer turns these into ``SuggestedAction``s, so ambiguity survives into
    #: the annotation instead of being silently collapsed.
    expansions: tuple[AbbreviationExpansion, ...] = ()
    #: True when the long form appears next to this occurrence, which means the
    #: abbreviation must be left as it is.
    defined_in_context: bool = False


def detect_abbreviations(
    doc,
    *,
    restrict_to_lang: str | None = None,
    expansions: dict[str, list[AbbreviationExpansion]] | None = None,
) -> list[AbbreviationFinding]:
    """Find abbreviation candidates in a udapi ``Document``.

    Args:
        doc: parsed document; only token forms and offsets are used.
        restrict_to_lang: skip sentences in other languages.
        expansions: optional ``abbreviation -> ranked expansions`` map from the
            corpus-level harvest. Without it, candidates are still reported (with
            no suggestions), which is what makes standalone annotation useful.
    """
    findings: list[AbbreviationFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue
        nodes = list(tree.descendants)
        if not nodes or caps_suppressed([n.form or "" for n in nodes]):
            continue

        # Gates 2 and 3 are defined over the sentence's surface, so reconstruct
        # it once and index the candidates by offset.
        sentence = tree.text or tree.get_sentence()
        enum = enumeration_offsets(sentence) if sentence else set()
        glossed = glossed_offsets(sentence) if sentence else set()
        # map sentence-relative offset -> was it inside an enumeration / a gloss
        surface_positions: dict[str, list[int]] = {}
        for m in CANDIDATE_RE.finditer(sentence or ""):
            surface_positions.setdefault(m.group(0), []).append(m.start())

        seen: dict[str, int] = {}
        for node in nodes:
            form = node.form or ""
            if not CANDIDATE_RE.fullmatch(form):
                continue
            if is_function_word(lang, form.lower()):
                logger.debug(f"Abbreviation[{lang}]: {form!r} vetoed (function word)")
                continue
            # Line the token up with its n-th surface occurrence, so the
            # sentence-level gates can be consulted per occurrence.
            idx = seen.get(form, 0)
            seen[form] = idx + 1
            positions = surface_positions.get(form, [])
            pos = positions[idx] if idx < len(positions) else None
            if pos is not None and pos in enum:
                logger.debug(
                    f"Abbreviation[{lang}]: {form!r} vetoed (enumeration label)"
                )
                continue
            try:
                begin, end = token_offsets(node)
            except ValueError:
                continue
            cands = tuple((expansions or {}).get(form, ()))
            findings.append(
                AbbreviationFinding(
                    begin=begin,
                    end=end,
                    text=form,
                    lang=lang,
                    expansions=cands,
                    defined_in_context=(pos is not None and pos in glossed),
                )
            )
            logger.debug(
                f"Abbreviation[{lang}]: {form!r} [{begin}:{end}] "
                f"{len(cands)} expansion(s)"
                f"{' (glossed)' if findings[-1].defined_in_context else ''}"
            )
    return findings
