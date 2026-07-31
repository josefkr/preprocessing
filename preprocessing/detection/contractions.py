"""Pure contraction / clitic detector (English + German).

Operates on a udapi document and returns findings; no CAS dependency. Four
mechanisms, because the families surface differently in the parse:

1. **Clitics** (English ``n't``/``'s``/…; German ``'s``). A clitic token
   written **adjacent** to the preceding token (host end == clitic begin, i.e.
   one written word "wouldn't"/"mir's"), whose ``(form, lemma)`` pair has an
   expansion in the lexicon. The lemma disambiguates (EN ``'s`` → is/has/us;
   DE ``'s`` → es); possessive ``'s`` is absent and never fires. The host may
   change too ("can't" = "ca"+"n't" → "cannot", written solid via the
   lexicon's whole-contraction overrides). Not UD multiword tokens.

2. **Prep+article** (German ``vom`` = ``von dem``). Genuine UD multiword
   tokens — read from the tree's MWTs (ADP+DET shape), expansion parser-supplied.

3. **Clipped indefinite articles** (German ``nen``/``nem``/``ner``/``ne``).
   Standalone tokens recognised by **surface** form (Stanza mislemmatises them),
   with a fully determined full form ("nen" → "einen"). ``ne`` (=eine) is
   ambiguous with the tag question "ne?", so it counts as an article only when a
   noun phrase follows ("ne Karre" → "eine Karre"; "…, ne?" is left alone).

4. **Clipped forms** (English ``gonna``/``lemme``/``'em``). Colloquial single
   written words standing for a multi-word sequence. Matched by **surface**, but
   the tokenizer is inconsistent about them — some stay one token ("kinda"),
   others are split ("gonna" → gon+na) — so the match is on the whole *written
   word*: one token or an adjacent run. Keying on the full word is what keeps
   this safe; registering "na"/"a" as bare clitics would fire on ordinary words.

Clitic expansion is also **context-sensitive** where the lemma is ambiguous:
``'s``/``'d`` before a participle are the perfect auxiliaries (*has*/*had*, not
*is*/*would*), and the host of ``ain't`` agrees with the subject
(*am*/*is*/*are*) — see the lexicon's ``*_in_context`` helpers.

Every finding carries a ``kind`` so the normalizer can apply its per-kind
policy (clitics, clipped articles and clipped forms always expand; prep+article
is opt-in).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing.detection.language import UnsupportedLanguage, tree_lang
from preprocessing.detection.lexicons.contractions import (
    CLIPPED_FORM_MAX_TOKENS,
    clipped_article_expansion,
    clipped_article_is_inferred,
    clipped_article_needs_np,
    clipped_form_expansion,
    clitic_expansion_in_context,
    contraction_clitics,
    contraction_override,
    gdropped_expansion,
    host_expansion_in_context,
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

        # Colloquial clipped forms (English "gonna", "lemme", "'em"): matched as
        # whole written words, which may be one token or an adjacent pair.
        findings.extend(_clipped_form_findings(tree, lang))

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

            nxt = _next_content(nodes, i)
            expansion = clitic_expansion_in_context(
                lang,
                node.form,
                node.lemma or "",
                next_xpos=nxt.xpos if nxt is not None else None,
                next_lemma=nxt.lemma if nxt is not None else None,
            )
            if expansion is None:
                # e.g. possessive "'s" — not a two-word contraction.
                continue
            subj = _clause_subject(host)
            host_form = (
                host_expansion_in_context(
                    lang,
                    host.form,
                    subject_lemma=subj.lemma if subj is not None else None,
                    subject_number=(
                        subj.feats.get("Number") if subj is not None else None
                    ),
                )
                or host.form
            )

            # Most contractions expand to "host clitic"; a few are written solid
            # ("can't" -> "cannot").
            whole = contraction_override(lang, host.form, node.form)
            if whole is not None and host.form[:1].isupper():
                whole = whole[:1].upper() + whole[1:]

            findings.append(
                ContractionFinding(
                    begin=h_begin,
                    end=c_end,
                    text=f"{host.form}{node.form}",
                    expansion=whole if whole is not None else f"{host_form} {expansion}",
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


# Tokens that may sit between a clitic and the participle that disambiguates it
# ("he's *not* been well", "he's *already* been here"), so they are skipped when
# looking for the following content word.
_CLITIC_CONTEXT_SKIP_UPOS = frozenset({"ADV", "PART"})


def _clause_subject(host):
    """The subject of the clause ``host`` is an auxiliary of, or ``None``.

    Used to inflect "ain't" ("ai" + "n't") for agreement. Looks for an ``nsubj``
    sibling under the host's head, which is where the subject sits for an
    auxiliary ("that" in "that ain't funny").
    """
    parent = getattr(host, "parent", None)
    if parent is None:
        return None
    for child in parent.children:
        if child.udeprel in ("nsubj", "expl"):
            return child
    return None


def _next_content(nodes: list, idx: int):
    """First token after ``nodes[idx]`` that is not an adverb/particle, else None."""
    for node in nodes[idx + 1:]:
        if node.upos in _CLITIC_CONTEXT_SKIP_UPOS:
            continue
        return node
    return None


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


def _gdropping_candidate(span: list, width: int) -> bool:
    """Whether the g-dropping rule may be applied to this token span.

    The tokenizer is inconsistent about the elision apostrophe: "talkin'" can
    come through as one token, but "stayin'" is sometimes split into
    ``stayin`` + ``'``. Both must be reachable, so joined spans are allowed —
    but a joined span could equally be an ordinary word followed by a **closing
    quote** ("the cabin'" → tokens ``cabin`` + ``'``), and expanding that would
    produce "cabing". A single token is unambiguous (quotes are always split
    off); a joined span is only trusted when the parser calls the stem a verb,
    which excludes cabin/thin/sin and friends.
    """
    if width == 1:
        return True
    return (
        (span[-1].form or "") == "'"
        and span[0].upos in ("VERB", "AUX")
    )


def _clipped_form_findings(tree, lang: str | None) -> list[ContractionFinding]:
    """Colloquial clipped forms in ``tree`` (English "gonna", "lemme", "'em").

    Matches the whole **written word**: either a single token, or a run of
    adjacent tokens (the tokenizer splits some of these — "gonna" into
    "gon"+"na"). Longest match wins, and matched tokens are not reused, so
    "gonna" is never also read as a bare "na".
    """
    nodes = list(tree.descendants)
    out: list[ContractionFinding] = []
    i = 0
    while i < len(nodes):
        matched = None
        # Longest match first, so a 2-token "gon"+"na" beats any 1-token read.
        for width in range(min(CLIPPED_FORM_MAX_TOKENS, len(nodes) - i), 0, -1):
            span = nodes[i:i + width]
            try:
                offsets = [token_offsets(n) for n in span]
            except ValueError:
                continue
            # The tokens must be written as one word (no whitespace between).
            if any(offsets[k][1] != offsets[k + 1][0] for k in range(len(span) - 1)):
                continue
            surface = "".join(n.form or "" for n in span)
            expansion = clipped_form_expansion(lang, surface)
            if expansion is None and _gdropping_candidate(span, width):
                # Productive rule rather than a table entry: g-dropping
                # ("talkin'" -> "talking").
                expansion = gdropped_expansion(lang, surface)
            if expansion is not None:
                matched = (span, offsets, surface, expansion)
                break
        if matched is None:
            i += 1
            continue

        span, offsets, surface, expansion = matched
        begin, end = offsets[0][0], offsets[-1][1]
        first_alpha = next((c for c in surface if c.isalpha()), "")
        # Capitalise when the source word is capitalised, or when the form opens
        # the sentence ("twas someone ..." -> "It was someone ...").
        if first_alpha.isupper() or i == 0:
            expansion = expansion[:1].upper() + expansion[1:]
        # A clipped form can be written glued to the previous word
        # ("Take'em"); the expansion then needs a separating space so the
        # rewrite doesn't run the words together ("Takethem").
        if i > 0:
            try:
                if token_offsets(nodes[i - 1])[1] == begin:
                    expansion = " " + expansion
            except ValueError:
                pass
        out.append(
            ContractionFinding(
                begin=begin,
                end=end,
                text=surface,
                expansion=expansion,
                lang=lang,
                kind="clipped_form",
            )
        )
        logger.debug(
            f"Contraction[{lang}]: clipped form {surface!r} "
            f"[{begin}:{end}] -> {expansion!r}"
        )
        i += len(span)
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
