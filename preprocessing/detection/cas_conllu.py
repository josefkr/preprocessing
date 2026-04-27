"""CAS view -> CoNLL-U conversion.

Builds a multi-sentence CoNLL-U string from a CAS view, preserving the
original character offsets of each token in the MISC column as
``t_start=<int>|t_end=<int>``. Detection modules can then operate on
udapi trees and still emit CAS annotations with correct char offsets.

Optionally embeds a ``# lang = <iso>`` comment per sentence so
detectors can read the language directly off the tree.
"""

from __future__ import annotations

import contextlib
import io
import re

from py_lift.dkpro import T_SENT
from py_lift.utils.conllu import cas_to_str


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
        with contextlib.redirect_stdout(io.StringIO()):
            block = cas_to_str(view, sent)
        block = re.sub(
            r"^# sent_id = \d+",
            f"# sent_id = {i}",
            block,
            count=1,
            flags=re.MULTILINE,
        )
        if sentence_langs is not None and sentence_langs[i - 1] is not None:
            block = _insert_lang_comment(block, sentence_langs[i - 1])
        blocks.append(block)

    return "\n\n".join(blocks) + "\n\n"


def _insert_lang_comment(block: str, lang: str) -> str:
    """Insert ``# lang = <lang>`` after the existing ``# text =`` line."""
    return re.sub(
        r"^(# text = .*)$",
        rf"\1\n# lang = {lang}",
        block,
        count=1,
        flags=re.MULTILINE,
    )
