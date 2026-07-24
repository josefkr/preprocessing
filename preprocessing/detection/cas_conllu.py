"""CAS view -> CoNLL-U conversion.

Builds a multi-sentence CoNLL-U string from a CAS view, preserving the
original character offsets of each token in the MISC column as
``t_start=<int>|t_end=<int>``. Detection modules can then operate on
udapi trees and still emit CAS annotations with correct char offsets.

Optionally embeds a ``# lang = <iso>`` comment per sentence so
detectors can read the language directly off the tree.

Why we have a local ``cas_to_str`` instead of using py_lift's: py_lift's
hardcodes the UPOS column to ``"FM"`` and writes the CAS ``PosValue``
into XPOS, which breaks udapi-based detectors that check ``node.upos``.
The local version reads the DKPro POS type's two fields:
``PosValue`` (fine, XPOS) and ``coarseValue`` (UD, UPOS), and falls
back to ``"FM"`` only when ``coarseValue`` is missing.
"""

from __future__ import annotations

import re

from py_lift.dkpro import T_DEP, T_LEMMA, T_MORPH, T_POS, T_SENT, T_TOKEN

from preprocessing.mwt import mwt_forms_by_span


def view_to_conllu(view, *, sentence_langs: list[str | None] | None = None) -> str:
    """Convert a CAS view to a CoNLL-U document string.

    Sentence segmentation is taken from the view's Sentence annotations.
    Returns an empty string if the view has no Sentence annotations
    (callers should warn the user and skip detection).

    If ``sentence_langs`` is given, it must match the number of
    sentences in the view; each non-None entry becomes a
    ``# lang = <iso>`` comment for that sentence. Sentences whose
    entry is ``None`` get no language comment.
    """
    sentences = list(view.select(T_SENT))
    if not sentences:
        return ""

    if sentence_langs is not None and len(sentence_langs) != len(sentences):
        raise ValueError(
            f"sentence_langs has {len(sentence_langs)} entries but view "
            f"has {len(sentences)} sentences"
        )

    blocks: list[str] = []
    for i, sent in enumerate(sentences, start=1):
        block = _cas_to_conllu_block(view, sent, sent_id=i)
        if sentence_langs is not None and sentence_langs[i - 1] is not None:
            block = _insert_lang_comment(block, sentence_langs[i - 1])
        blocks.append(block)

    return "\n\n".join(blocks) + "\n\n"


def _sort_feats(feats: str) -> str:
    """UD requires FEATS attributes sorted case-insensitively."""
    if feats == "_" or "=" not in feats:
        return feats
    return "|".join(sorted(feats.split("|"), key=lambda p: p.split("=", 1)[0].lower()))


def _no_space_after(token, sofa: str) -> bool:
    """True when the next character after this token is not whitespace, i.e.
    UD wants ``SpaceAfter=No`` (typically before punctuation)."""
    end = token.end
    if end is None or end >= len(sofa):
        return False
    return not sofa[end].isspace()


def _misc(token, sofa: str, *, no_space: bool) -> str:
    """MISC column: the offsets detectors rely on, plus SpaceAfter=No so the
    emitted CoNLL-U is UD-valid."""
    items = []
    if no_space:
        items.append("SpaceAfter=No")
    items.append(f"t_start={token.begin}|t_end={token.end}")
    return "|".join(items)


def _group_by_span(annotations) -> dict[tuple[int, int], list]:
    """Group annotations by ``(begin, end)``, each group ordered by ``xmiID``.

    Multiword-token sub-words (German "vom" -> von+dem, "im" -> in+dem) have no
    character span of their own, so ingestion gives them the *parent* token's
    span. Keying by offset alone therefore collapses them and silently drops one
    annotation — which loses the ADP tag of the case-marker in ~1 of 20 German
    tokens. ``xmiID`` is assigned in creation order, which is word order, so
    sorting by it restores the sub-word sequence (von before dem) and lets the
    i-th token pair with the i-th POS / Lemma / Morph of the same span.
    (``select()`` order is *not* stable for equal spans and must not be used.)
    """
    out: dict[tuple[int, int], list] = {}
    for a in annotations:
        out.setdefault((a.begin, a.end), []).append(a)
    for group in out.values():
        group.sort(key=lambda a: a.xmiID)
    return out


def _nth(group: list | None, index: int):
    """The ``index``-th annotation of a same-span group, or ``None``."""
    if not group:
        return None
    return group[index] if index < len(group) else group[0]


def _cas_to_conllu_block(view, sentence, *, sent_id: int) -> str:
    """Build a single CoNLL-U sentence block for ``sentence`` in ``view``.

    Replacement for py_lift's ``cas_to_str`` that respects DKPro's two
    POS fields: ``PosValue`` (XPOS) and ``coarseValue`` (UPOS).
    """
    tokens = [
        t for t in view.select_covered(T_TOKEN, sentence)
        if not re.match(r"^\s*$", t.get_covered_text())
    ]
    if not tokens:
        sent_text = re.sub(r"\n", " ", sentence.get_covered_text())
        return f"# sent_id = {sent_id}\n# text = {sent_text}\n"

    # Order same-span tokens (MWT sub-words) by creation order too, so the
    # CoNLL-U row order is deterministic and matches the word order.
    tokens.sort(key=lambda t: (t.begin, t.end, t.xmiID))
    id_map = {t.xmiID: i + 1 for i, t in enumerate(tokens)}

    pos_by_span = _group_by_span(view.select_covered(T_POS, sentence))
    lemma_by_span = _group_by_span(view.select_covered(T_LEMMA, sentence))
    morph_by_span = _group_by_span(view.select_covered(T_MORPH, sentence))
    # Each token's position within its own same-span group (0 for ordinary
    # tokens), used to pick the matching POS / Lemma / Morph.
    span_index = {}
    for group in _group_by_span(tokens).values():
        for i, t in enumerate(group):
            span_index[t.xmiID] = i
    dep_by_dep_id: dict[int, object] = {}
    for d in view.select(T_DEP):
        if d.Dependent is not None and d.Dependent.xmiID in id_map:
            dep_by_dep_id[d.Dependent.xmiID] = d

    # Sub-word surface forms for multiword tokens, when ingestion recorded them
    # (data ingested before MWTPart existed yields {} -> covered-text fallback).
    mwt_forms = mwt_forms_by_span(view)
    # A token is followed directly by the next one (no whitespace between) ->
    # UD requires SpaceAfter=No. Computed per surface position, so MWT
    # sub-words (which share a span) are handled by the range line instead.
    sofa = view.sofa_string or ""

    rows: list[str] = []
    emitted_ranges: set[tuple[int, int]] = set()
    for token in tokens:
        key = (token.begin, token.end)
        idx = span_index.get(token.xmiID, 0)
        group = _group_by_span(tokens).get(key, [token])

        # Multiword token: emit the UD range line once, before its sub-words.
        if len(group) > 1 and key not in emitted_ranges:
            emitted_ranges.add(key)
            first_id = id_map[group[0].xmiID]
            last_id = id_map[group[-1].xmiID]
            rows.append(
                "\t".join([
                    f"{first_id}-{last_id}",
                    token.get_covered_text(),   # the surface, e.g. "vom"
                    "_", "_", "_", "_", "_", "_", "_",
                    _misc(token, sofa, no_space=_no_space_after(token, sofa)),
                ])
            )
        pos = _nth(pos_by_span.get(key), idx)
        upos = (getattr(pos, "coarseValue", "") or "") if pos else ""
        xpos = (getattr(pos, "PosValue", "") or "") if pos else ""
        if not upos:
            upos = "FM"
        if not xpos:
            xpos = "_"

        lemma = _nth(lemma_by_span.get(key), idx)
        lemma_value = (getattr(lemma, "value", "") or "") if lemma else "_"

        morph = _nth(morph_by_span.get(key), idx)
        morph_value = (getattr(morph, "morphTag", "") or "") if morph else "_"
        if not morph_value:
            morph_value = "_"
        morph_value = _sort_feats(morph_value)

        dep = dep_by_dep_id.get(token.xmiID)
        if dep is None:
            raise RuntimeError(
                f"No dependency for token {token.get_covered_text()!r}"
            )
        # In DKPro CAS, the root's Governor points to itself.
        if dep.Governor.xmiID == token.xmiID:
            head = 0
            deprel = "root"
        else:
            head = id_map.get(dep.Governor.xmiID, 0)
            deprel = dep.DependencyType or "dep"

        # FORM: the sub-word's true surface when this is an MWT part and
        # ingestion recorded it; otherwise the covered text (old data, and all
        # ordinary tokens).
        form = token.get_covered_text()
        if len(group) > 1:
            forms = mwt_forms.get(key)
            if forms and idx < len(forms):
                form = forms[idx]
        # SpaceAfter belongs on the surface token: the range line for an MWT,
        # the token itself otherwise.
        misc = (
            f"t_start={token.begin}|t_end={token.end}"
            if len(group) > 1
            else _misc(token, sofa, no_space=_no_space_after(token, sofa))
        )
        rows.append(
            "\t".join(
                [
                    str(id_map[token.xmiID]),
                    form,
                    lemma_value,
                    upos,
                    xpos,
                    morph_value,
                    str(head),
                    deprel,
                    "_",
                    misc,
                ]
            )
        )

    sent_text = re.sub(r"\n", " ", sentence.get_covered_text())
    return (
        f"# sent_id = {sent_id}\n"
        f"# text = {sent_text}\n"
        + "\n".join(rows)
    )


def _insert_lang_comment(block: str, lang: str) -> str:
    """Insert ``# lang = <lang>`` after the existing ``# text =`` line."""
    return re.sub(
        r"^(# text = .*)$",
        rf"\1\n# lang = {lang}",
        block,
        count=1,
        flags=re.MULTILINE,
    )
