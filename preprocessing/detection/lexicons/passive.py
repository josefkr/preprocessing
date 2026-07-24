"""Per-language lexicons used by the passive detector.

- ``PASSIVE_AGENT_PREPS_BY_LANG``: prepositions that can mark the agent
  in a short passive (e.g. English "by", German "von"/"durch").
- ``PARTICIPLE_XPOS_BY_LANG``: XPOS tags that identify a passive
  participle. If the tagger doesn't emit XPOS or uses a different
  scheme, the detector falls back to the universal ``VerbForm=Part``
  feat.
"""

from __future__ import annotations

from preprocessing.detection.language import UnsupportedLanguage

PASSIVE_AGENT_PREPS_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset({"by"}),
    # "vom" is the contraction "von dem". Stanza expands it correctly to the
    # words "von" + "dem"; the duplicated surface form was OUR round-trip
    # (sub-words have no character span, so they inherited the parent's, and
    # FORM was re-derived from the covered text). Ingestion now records the
    # sub-word forms (preprocessing.mwt) and cas_conllu emits a real multiword
    # token, so freshly ingested data yields "von" and this entry is redundant
    # for it. KEEP "vom" anyway: data ingested before that change has no
    # MWTPart annotations and still surfaces the contracted form, and dropping
    # it would silently break German agent detection on that data. "vom" can
    # only mean "von + dem", so it never causes a false agent.
    "de": frozenset({"von", "durch", "vom"}),
    "fr": frozenset({"par"}),
    "es": frozenset({"por"}),
}

PARTICIPLE_XPOS_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset({"VBN"}),
    "de": frozenset({"VVPP"}),
    "fr": frozenset(),
    "es": frozenset(),
}


def passive_agent_preps(lang: str) -> frozenset[str]:
    """Return the agent-preposition set for ``lang`` or raise."""
    try:
        return PASSIVE_AGENT_PREPS_BY_LANG[lang]
    except KeyError as e:
        raise UnsupportedLanguage(
            f"passive (short): no agent-preposition lexicon for {lang!r}"
        ) from e


def is_passive_participle(node, lang: str) -> bool:
    """Return True if ``node`` looks like a (passive) participle.

    Uses the language's XPOS set when available, otherwise falls back
    to ``VerbForm=Part`` from the FEATS column.
    """
    xpos_set = PARTICIPLE_XPOS_BY_LANG.get(lang, frozenset())
    if node.xpos and node.xpos in xpos_set:
        return True
    try:
        return node.feats["VerbForm"] == "Part"
    except KeyError:
        return False
