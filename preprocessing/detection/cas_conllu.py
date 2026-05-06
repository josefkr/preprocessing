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

    id_map = {t.xmiID: i + 1 for i, t in enumerate(tokens)}

    pos_by_offset = {
        (p.begin, p.end): p for p in view.select_covered(T_POS, sentence)
    }
    lemma_by_offset = {
        (l.begin, l.end): l for l in view.select_covered(T_LEMMA, sentence)
    }
    morph_by_offset = {
        (m.begin, m.end): m for m in view.select_covered(T_MORPH, sentence)
    }
    dep_by_dep_id: dict[int, object] = {}
    for d in view.select(T_DEP):
        if d.Dependent is not None and d.Dependent.xmiID in id_map:
            dep_by_dep_id[d.Dependent.xmiID] = d

    rows: list[str] = []
    for token in tokens:
        key = (token.begin, token.end)
        pos = pos_by_offset.get(key)
        upos = (getattr(pos, "coarseValue", "") or "") if pos else ""
        xpos = (getattr(pos, "PosValue", "") or "") if pos else ""
        if not upos:
            upos = "FM"
        if not xpos:
            xpos = "_"

        lemma = lemma_by_offset.get(key)
        lemma_value = (getattr(lemma, "value", "") or "") if lemma else "_"

        morph = morph_by_offset.get(key)
        morph_value = (getattr(morph, "morphTag", "") or "") if morph else "_"
        if not morph_value:
            morph_value = "_"

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

        misc = f"t_start={token.begin}|t_end={token.end}"
        rows.append(
            "\t".join(
                [
                    str(id_map[token.xmiID]),
                    token.get_covered_text(),
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
