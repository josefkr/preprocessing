"""Multiword-token (MWT) sub-word forms in the CAS.

Stanza expands a contraction into words that have **no character span of their
own** (``start_char/end_char == None``): German "vom" -> "von" + "dem", "im" ->
"in" + "dem"; likewise French "du", Italian "del". Ingestion therefore gives
every sub-word the *parent* token's span, and since a CAS ``Token``'s surface
form **is** its covered text, the sub-word forms are otherwise lost — both
sub-words of "vom" read "vom".

That has two visible consequences:

* form-based queries over German are silently incomplete (searching for "dem"
  never matches the article inside "im"/"vom"/"zum"), and
* CoNLL-U reconstructed from the CAS cannot be UD-valid, because the two rows
  repeat the surface form instead of being a multiword token.

:data:`T_MWT_PART` records the missing piece: one annotation per sub-word,
carrying its true ``form`` and its ``index`` within the parent token. The type
is injected into the CAS typesystem on demand (same approach as
``RWSENormalizer._ensure_rwse_type``), so it works whatever typesystem the CAS
was built with and is serialized with the XMI.

Data ingested before this existed simply has no ``MWTPart`` annotations;
readers must fall back to the covered text, which reproduces the old behaviour.
"""

from __future__ import annotations

T_MWT_PART = "org.aslan.type.MWTPart"

_ANNOTATION = "uima.tcas.Annotation"
_STRING = "uima.cas.String"
_INT = "uima.cas.Integer"


def ensure_mwt_part_type(ts):
    """Define :data:`T_MWT_PART` on ``ts`` if absent; return the type. Idempotent."""
    if ts.contains_type(T_MWT_PART):
        return ts.get_type(T_MWT_PART)
    t = ts.create_type(name=T_MWT_PART, supertypeName=_ANNOTATION)
    ts.create_feature(domainType=t, name="form", rangeType=_STRING)
    ts.create_feature(domainType=t, name="index", rangeType=_INT)
    return t


def add_mwt_part(view, ts, *, begin: int, end: int, form: str, index: int) -> None:
    """Record one MWT sub-word's true surface form at the parent token's span."""
    MWT = ensure_mwt_part_type(ts)
    view.add(MWT(begin=begin, end=end, form=form, index=index))


def mwt_forms_by_span(view) -> dict[tuple[int, int], list[str]]:
    """``(begin, end) -> [sub-word forms in order]`` for this view.

    Empty for data ingested before MWTPart existed, in which case callers keep
    using the covered text.
    """
    out: dict[tuple[int, int], list[tuple[int, str]]] = {}
    try:
        parts = list(view.select(T_MWT_PART))
    except Exception:  # noqa: BLE001 — type not in this CAS's typesystem
        return {}
    for p in parts:
        out.setdefault((p.begin, p.end), []).append(
            (getattr(p, "index", 0) or 0, getattr(p, "form", "") or "")
        )
    return {k: [f for _i, f in sorted(v)] for k, v in out.items()}
