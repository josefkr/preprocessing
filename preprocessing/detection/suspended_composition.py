"""Pure suspended-composition detector (German Ergänzungsstrich).

Operates on a udapi document and returns findings; no CAS dependency.

A *suspended compound* writes a shared right-hand constituent once, on the last
conjunct, and marks its omission on the earlier ones with a hyphen: "Sonn- und
Feiertagen" = Sonntagen und Feiertagen; "be- und entladen" = beladen und entladen.
It spans four host categories — noun compounds, separable and inseparable verbal
prefixes, adjectives, and numerals — so a noun-only rule misses roughly half.

German suspends in **both** directions, and the hyphen's position says which:
a *trailing* hyphen shares the head and the donor follows ("Sonn- und Feiertagen"),
a *leading* hyphen shares the modifier and the donor precedes ("Energieerzeugung und
-verteilung"). Both occur in one sentence often enough to matter — "Frauen- und
Kinderhandel , Drogenhandel und -konsum" has one of each — so neither pattern may
assume it is the only one present.

**Why this detector reads the surface rather than the token stream.** Stanza
handles these tokens badly *and* inconsistently, tagging the stub ``PUNCT`` and
splitting it or not depending on the word — in one sentence it kept ``An-`` whole
while splitting ``Um-`` into ``Um`` + ``-``::

    Dabei be- und entladen   ->  ('be-', PUNCT) ('und', CCONJ) ('entladen', VERB)
    Die Vereins- und ...     ->  ('Vereins', NOUN) ('-', PUNCT) ('und', CCONJ)

So the stub cannot be located by form, POS, or dependency. It is found by regex on
the sentence surface and then mapped back to sofa offsets through whichever tokens
it overlaps, which is robust to either tokenisation. This is also why suspended
composition must be resolved **before** any parse-dependent step: every downstream
structural detector inherits the damage.

Detection is deliberately separate from *resolution*. Deciding which part of the
donor completes the stub needs morphology and an attestation lexicon (see
``aslan_normalization._suspended``), and is genuinely ambiguous for a minority of
sites. This detector therefore reports the site — and the donor it believes the
material comes from — whether or not the completion can be settled, and takes an
optional ``resolver`` to attach one when it can.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import tree_lang
from preprocessing.detection.lexicons.suspended_composition import (
    LEAD_STUB_RE,
    STUB_RE,
    coordinators,
)
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuspensionFinding:
    """One suspended conjunct."""

    begin: int
    end: int
    stub: str                  # as written, hyphen included ("Sonn-", "-verteilung")
    lang: str | None
    #: ``shared_head`` — trailing hyphen, donor follows ("Sonn- und Feiertagen");
    #: ``shared_modifier`` — leading hyphen, donor precedes ("Energieerzeugung und
    #: -verteilung"). Both can occur in one sentence.
    direction: str = "shared_head"
    #: Offsets of the conjunct the shared material comes from, when one was found.
    donor_begin: int | None = None
    donor_end: int | None = None
    donor: str | None = None
    #: The completed form ("Sonntagen"), when a resolver supplied one.
    completed: str | None = None
    #: How the resolver chose the split, for triage: ``unique`` | ``parallel`` |
    #: ``score`` | ``attested``. ``None`` when unresolved.
    basis: str | None = None


def _text_to_sofa(tree) -> tuple[str, list[tuple[int, int, int, int]]]:
    """``(sentence_text, spans)`` where each span is ``(t_begin, t_end, s_begin, s_end)``.

    Maps positions in the sentence surface onto sofa offsets. Built by walking the
    tokens in order and locating each in the surface, so it does not assume the
    surface is a space-join of the tokens (it is not, around punctuation) nor that
    the tokenizer split the stub in any particular way.
    """
    text = tree.text or tree.get_sentence() or ""
    spans: list[tuple[int, int, int, int]] = []
    cursor = 0
    for node in tree.descendants:
        form = node.form or ""
        if not form:
            continue
        at = text.find(form, cursor)
        if at < 0:                      # surface and tokens disagree; skip the token
            continue
        cursor = at + len(form)
        try:
            s_begin, s_end = token_offsets(node)
        except ValueError:
            continue
        spans.append((at, at + len(form), s_begin, s_end))
    return text, spans


def _sofa_span(spans, t_begin: int, t_end: int) -> tuple[int, int] | None:
    """Sofa offsets of the surface range ``[t_begin, t_end)``, via overlapping tokens.

    Takes the first and last overlapping token, which is what makes the two
    tokenisations of a stub (``be-`` versus ``Vereins`` + ``-``) come out the same.
    """
    hit = [s for s in spans if s[0] < t_end and s[1] > t_begin]
    if not hit:
        return None
    return hit[0][2], hit[-1][3]


def detect_suspended_composition(
    doc,
    *,
    restrict_to_lang: str | None = None,
    resolver=None,
) -> list[SuspensionFinding]:
    """Find suspended conjuncts in a udapi ``Document``.

    Args:
        doc: parsed document. Only the sentence surface and token offsets are
            used — deliberately, see the module docstring.
        restrict_to_lang: skip sentences in other languages.
        resolver: optional ``find_sites(text, lang=…) -> [Site]`` callable
            (``aslan_normalization._suspended.find_sites``). When given, the
            donor and the completed form are attached; without it the sites are
            still reported, which is what makes standalone annotation useful.
    """
    findings: list[SuspensionFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue
        if not coordinators(lang):
            continue        # no coordinator inventory for this language
        text, spans = _text_to_sofa(tree)
        if not text or not spans:
            continue

        resolved = {}
        if resolver is not None:
            try:
                for site in resolver(text, lang=lang or "de"):
                    resolved[site.begin] = site
            except Exception as e:  # noqa: BLE001 — resolution is optional
                logger.debug(f"suspension resolver failed: {e}")

        for direction, pattern in (("shared_head", STUB_RE),
                                   ("shared_modifier", LEAD_STUB_RE)):
            for m in pattern.finditer(text):
                span = _sofa_span(spans, m.start(), m.end())
                if span is None:
                    continue
                site = resolved.get(m.start())
                r = getattr(site, "resolution", None) if site else None
                donor_span = None
                if r is not None:
                    # The donor is *after* the stub with a shared head and *before*
                    # it with a shared modifier, so search the matching side.
                    at = (text.rfind(r.donor, 0, m.start())
                          if direction == "shared_modifier"
                          else text.find(r.donor, m.end()))
                    if at >= 0:
                        donor_span = _sofa_span(spans, at, at + len(r.donor))
                findings.append(SuspensionFinding(
                    begin=span[0], end=span[1], stub=m.group(0), lang=lang,
                    direction=direction,
                    donor_begin=donor_span[0] if donor_span else None,
                    donor_end=donor_span[1] if donor_span else None,
                    donor=r.donor if r else None,
                    completed=r.completed if r else None,
                    basis=r.basis if r else None,
                ))
                logger.debug(
                    f"Suspension[{lang}/{direction}]: {m.group(0)!r} "
                    f"[{span[0]}:{span[1]}]"
                    + (f" -> {r.completed!r} ({r.basis})" if r else " (unresolved)")
                )
    findings.sort(key=lambda f: f.begin)
    return findings
