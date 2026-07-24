"""Pure contraction / clitic detector (English + German).

Operates on a udapi document and returns findings; no CAS dependency. Three
mechanisms, because the families surface differently in the parse:

1. **Clitics** (English ``n't``/``'s``/…; German ``'s``). A clitic token
   written **adjacent** to the preceding token (host end == clitic begin, i.e.
   one written word "wouldn't"/"mir's"), whose ``(form, lemma)`` pair has an
   expansion in the lexicon. The lemma disambiguates (EN ``'s`` → is/has/us;
   DE ``'s`` → es); possessive ``'s`` is absent and never fires. The host may
   change too ("can't" = "ca"+"n't" → "can not"). Not UD multiword tokens.

2. **Prep+article** (German ``vom`` = ``von dem``). Genuine UD multiword
   tokens — read from the tree's MWTs (ADP+DET shape), expansion parser-supplied.

3. **Clipped indefinite articles** (German ``nen``/``nem``/``ner``/``ne``).
   Standalone tokens recognised by **surface** form (Stanza mislemmatises them),
   with a fully determined full form ("nen" → "einen"). ``ne`` (=eine) is
   ambiguous with the tag question "ne?", so it counts as an article only when a
   noun phrase follows ("ne Karre" → "eine Karre"; "…, ne?" is left alone).

Every finding carries a ``kind`` so the normalizer can apply its per-kind
policy (clitics + clipped articles always expand; prep+article is opt-in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.contractions import (
    clipped_article_expansion,
    clipped_article_is_inferred,
    clipped_article_needs_np,
    clitic_expansion,
    contraction_clitics,
    host_expansion,
    inflect_indefinite_article,
)
from preprocessing.detection.offsets import token_offsets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContractionPart:
    """One half of a contraction: the host ("would") or the clitic ("n't")."""

    begin: int
    end: int
    text: str


@dataclass(frozen=True)
class ContractionFinding:
    begin: int          # whole contraction span ("wouldn't", "vom")
    end: int
    text: str           # surface form
    expansion: str      # e.g. "would not", "mir es", "von dem"
    lang: str | None
    kind: str = "clitic"     # "clitic" | "prep_article" | "clipped_article"
    # Clitic findings carry the two surface parts (host + clitic); prep+article
    # and clipped-article findings are a single token, so both are None.
    host: ContractionPart | None = None
    clitic: ContractionPart | None = None
    # For prep+article only: the immediately following word, so the normalizer
    # can consult the lexicalised-exception list without re-parsing.
    following: str | None = None
    # True when the full form was inferred from the following noun's morphology
    # (bare "n" → ein/einen/…) rather than fixed by the surface. Parser case is
    # unreliable, so the normalizer gates these behind an opt-in flag.
    inferred: bool = False


def detect_contractions(
    doc, *, restrict_to_lang: str | None = None
) -> list[ContractionFinding]:
    """Find contractions/clitics in a udapi ``Document``."""
    findings: list[ContractionFinding] = []
    for tree in doc.trees:
        lang = tree_lang(tree)
        if restrict_to_lang is not None and lang != restrict_to_lang:
            continue

        # Prep+article contractions are multiword tokens (German "vom" =
        # "von dem"); the parser supplies the expansion. Data-driven and
        # language-agnostic — English simply has no such MWTs.
        findings.extend(_prep_article_findings(tree, lang))

        # Clipped indefinite articles (German "nen"/"nem"/"ner") are standalone
        # tokens; recognised by surface form (Stanza's lemma is unreliable).
        findings.extend(_clipped_article_findings(tree, lang))

        try:
            clitics = contraction_clitics(lang)
        except UnsupportedLanguage as e:
            logger.warning(str(e))
            continue

        nodes = list(tree.descendants)
        for i, node in enumerate(nodes):
            if i == 0 or (node.form or "").lower() not in clitics:
                continue
            host = nodes[i - 1]
            try:
                h_begin, h_end = token_offsets(host)
                c_begin, c_end = token_offsets(node)
            except ValueError:
                continue
            if h_end != c_begin:
                continue  # not written as one word — not a contraction

            expansion = clitic_expansion(lang, node.form, node.lemma or "")
            if expansion is None:
                # e.g. possessive "'s" — not a two-word contraction.
                continue
            host_form = host_expansion(lang, host.form) or host.form

            findings.append(
                ContractionFinding(
                    begin=h_begin,
                    end=c_end,
                    text=f"{host.form}{node.form}",
                    expansion=f"{host_form} {expansion}",
                    lang=lang,
                    kind="clitic",
                    host=ContractionPart(h_begin, h_end, host.form),
                    clitic=ContractionPart(c_begin, c_end, node.form),
                )
            )
            logger.debug(
                f"Contraction[{lang}]: {host.form!r}+{node.form!r} "
                f"[{h_begin}:{c_end}] -> {findings[-1].expansion!r}"
            )
    return findings


# Preposition + definite-article contraction shape (German "vom" = von[ADP] +
# dem[DET]): a two-word multiword token whose parts are a case-marking adposition
# and its determiner.
_PREP_ARTICLE_UPOS = ("ADP", "DET")


def _prep_article_findings(tree, lang: str | None) -> list[ContractionFinding]:
    """Prep+article contractions in ``tree``, read from its multiword tokens.

    The expansion is whatever the parser split the MWT into ("vom" -> "von dem"),
    so no per-language expansion lexicon is needed — only the ADP+DET shape gate,
    which keeps other multiword tokens out.
    """
    out: list[ContractionFinding] = []
    for mwt in tree.multiword_tokens:
        words = list(mwt.words)
        if len(words) != 2:
            continue
        if tuple(w.upos for w in words) != _PREP_ARTICLE_UPOS:
            continue
        try:
            begin, end = token_offsets(words[0])  # sub-words share the MWT span
        except ValueError:
            continue
        # word following the contraction in the tree (for the exception check)
        last_ord = words[-1].ord
        following = next(
            (n.form for n in tree.descendants if n.ord > last_ord), None
        )
        out.append(
            ContractionFinding(
                begin=begin,
                end=end,
                text=mwt.form,
                expansion=" ".join(w.form for w in words),
                lang=lang,
                kind="prep_article",
                following=following,
            )
        )
        logger.debug(
            f"Contraction[{lang}]: prep+article {mwt.form!r} "
            f"[{begin}:{end}] -> {out[-1].expansion!r}"
        )
    return out


def _clipped_article_findings(tree, lang: str | None) -> list[ContractionFinding]:
    """Clipped indefinite articles in ``tree`` (German "nen"/"nem"/"ner").

    Standalone tokens recognised purely by surface form — Stanza does not
    lemmatise them to "ein" — with a fully determined full form ("nen" ->
    "einen"). The token's own casing is transferred to the expansion so a
    sentence-initial "Nen" becomes "Einen".
    """
    nodes = list(tree.descendants)
    out: list[ContractionFinding] = []
    for i, node in enumerate(nodes):
        surface = node.form or ""
        inferred = False

        expansion = clipped_article_expansion(lang, surface)
        if expansion is not None:
            # Fixed-form clipped article (nen/nem/ner/ne). Ambiguous forms
            # ("ne" = eine vs the tag "ne?") need a following noun phrase.
            if clipped_article_needs_np(lang, surface) and not _np_follows(nodes, i):
                continue
        elif clipped_article_is_inferred(lang, surface):
            # Bare "n": full form inferred from the head noun's morphology.
            head = _np_head(nodes, i)
            if head is None:
                continue
            expansion = inflect_indefinite_article(
                lang,
                head.feats.get("Gender"),
                head.feats.get("Case"),
                head.feats.get("Number"),
            )
            if expansion is None:
                continue  # couldn't determine the form — leave it contracted
            inferred = True
        else:
            continue

        try:
            begin, end = token_offsets(node)
        except ValueError:
            continue
        first_alpha = next((c for c in surface if c.isalpha()), "")
        if first_alpha.isupper():
            expansion = expansion[:1].upper() + expansion[1:]
        out.append(
            ContractionFinding(
                begin=begin,
                end=end,
                text=surface,
                expansion=expansion,
                lang=lang,
                kind="clipped_article",
                inferred=inferred,
            )
        )
        logger.debug(
            f"Contraction[{lang}]: clipped article {surface!r} "
            f"[{begin}:{end}] -> {expansion!r}{' (inferred)' if inferred else ''}"
        )
    return out


# Right-context noun-phrase gate for ambiguous clipped forms ("ne").
_NP_PREMODIFIER_UPOS = frozenset({"ADJ", "ADV", "DET", "NUM", "PRON"})
_NP_HEAD_UPOS = frozenset({"NOUN", "PROPN"})


def _np_follows(nodes: list, idx: int) -> bool:
    """True if the tokens after ``nodes[idx]`` open a noun phrase — a nominal
    head reached across only premodifiers (adjective/adverb/determiner/…).

    Conservative: anything else first (punctuation, a verb, a conjunction, the
    sentence end) means no NP, so a tag-question "ne" is left unchanged. This
    trades the odd comma-containing NP for never turning a tag into "eine".
    """
    return _np_head(nodes, idx) is not None


def _np_head(nodes: list, idx: int):
    """The nominal head opened by the tokens after ``nodes[idx]`` (a NOUN/PROPN
    reached across only premodifiers), or ``None`` if no NP follows."""
    for w in nodes[idx + 1:]:
        if w.upos in _NP_HEAD_UPOS:
            return w
        if w.upos in _NP_PREMODIFIER_UPOS:
            continue
        return None
    return None
