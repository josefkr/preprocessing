"""Parser-behaviour contract: the Stanza facts our compensations rely on.

Several detectors and normalizers deliberately work *around* Stanza — surface
lookups instead of lemmas, features treated as hints, deprels distrusted,
whole-written-word matching instead of token matching. Each of those
compensations is only justified while the underlying Stanza behaviour holds.

Our other tests use hand-written CoNLL-U fixtures, so they would keep passing
while a Stanza upgrade silently changed the live pipeline. This module closes
that gap: it asserts the observed parser behaviour directly, so an upgrade fails
**loudly**, pointing at the compensation to re-examine — a failure here is not
necessarily a bug, it may mean "Stanza improved, this workaround can retire".

Each test names the compensation it underwrites. Skipped when Stanza or the
language models aren't available.
"""

from __future__ import annotations

import pytest

pytest.importorskip("stanza", reason="stanza not installed")

_PROCESSORS = "tokenize,pos,lemma,depparse"


def _pipeline(lang: str):
    import stanza

    try:
        return stanza.Pipeline(
            lang, processors=_PROCESSORS, download_method=None,
            verbose=False, tokenize_no_ssplit=True,
        )
    except Exception as e:  # noqa: BLE001 — model not downloaded, etc.
        pytest.skip(f"stanza {lang!r} model unavailable: {e}")


@pytest.fixture(scope="module")
def en():
    return _pipeline("en")


@pytest.fixture(scope="module")
def de():
    return _pipeline("de")


def _words(nlp, text):
    return list(nlp(text).sentences[0].words)


def _tokens(nlp, text):
    return list(nlp(text).sentences[0].tokens)


def _find(words, surface):
    return next(w for w in words if w.text == surface)


# --- tokenization ------------------------------------------------------------

def test_german_mwt_subwords_have_no_offsets(de):
    """Justifies: ``preprocessing/mwt.py`` (MWTPart) and the parent-span
    inheritance in ``preprocessing/stanza.py`` +
    ``aslan_normalization/_stanza_annotate.py``.

    German "vom" is a multiword token whose sub-words carry no character span of
    their own, so a CAS token's covered text would read "vom" for both halves
    and the true forms (von/dem) must be recorded separately.
    """
    tok = next(t for t in _tokens(de, "Der Motor wird vom Mechaniker repariert.")
               if t.text == "vom")
    assert len(tok.words) == 2, "‘vom’ is no longer expanded into two words"
    assert [w.text for w in tok.words] == ["von", "dem"]
    assert all(w.start_char is None and w.end_char is None for w in tok.words), (
        "MWT sub-words now carry their own offsets — MWTPart and the "
        "parent-span inheritance may be retirable"
    )


def test_english_clipped_forms_tokenise_inconsistently(en):
    """Justifies: whole-written-word matching in ``_clipped_form_findings``
    (``CLIPPED_FORMS_BY_LANG``).

    Some clipped forms stay one token, others are split, so matching on tokens
    alone cannot reach them uniformly.
    """
    assert [w.text for w in _words(en, "I gonna go.")][1:3] == ["gon", "na"], (
        "‘gonna’ no longer splits — the joined-span match may be simplifiable"
    )
    assert any(w.text == "kinda" for w in _words(en, "He is kinda funny.")), (
        "‘kinda’ no longer arrives as a single token"
    )


def test_aint_and_cant_split_into_non_word_hosts(en):
    """Justifies: ``host_expansion_in_context`` ("ai" → am/is/are) and
    ``HOST_EXPANSIONS_BY_LANG`` ("ca" → can)."""
    assert [w.text for w in _words(en, "That ain't funny.")][1:3] == ["ai", "n't"]
    assert [w.text for w in _words(en, "She can't swim.")][1:3] == ["ca", "n't"]


def test_closing_quote_is_split_from_its_word(en):
    """Justifies: treating a single token ending in ``in'`` as safe g-dropping.

    If a closing quote stayed attached ("cabin'"), the productive ``-in'`` rule
    would corrupt it into "cabing".
    """
    forms = [w.text for w in _words(en, "He entered 'the cabin' quietly.")]
    assert "cabin" in forms and "cabin'" not in forms


# --- lemmas ------------------------------------------------------------------

def test_german_clipped_articles_do_not_lemmatise_to_ein(de):
    """Justifies: surface-keyed ``CLIPPED_ARTICLES_BY_LANG`` instead of a
    lemma lookup."""
    for surface in ("nen", "nem", "ner"):
        w = _find(_words(de, f"Er hat {surface} Hund."), surface)
        assert w.lemma != "ein", (
            f"‘{surface}’ now lemmatises to ‘ein’ — the surface-keyed clipped "
            "article table could become lemma-driven"
        )


def test_english_perfect_auxiliary_clitics_lemmatise_as_be_and_would(en):
    """Justifies: ``clitic_expansion_in_context`` (participle-based has/had).

    The lemma does not distinguish the perfect auxiliary from the copula/modal,
    which is why "he's been" would otherwise expand to the ungrammatical
    "he is been".
    """
    assert _find(_words(en, "He's been lucky."), "'s").lemma == "be"
    assert _find(_words(en, "there'd been a process."), "'d").lemma == "would"


def test_g_dropped_forms_are_not_reliably_lemmatised(en):
    """Justifies: the productive ``-in'`` rule (not lemma reconstruction), and
    leaving bare ``-in`` to the dictionary-backed spelling normalizer."""
    assert _find(_words(en, "someone stickin their nose in."), "stickin").lemma \
        == "stickin", "‘stickin’ now lemmatises — g-dropping could use the lemma"


# --- morphological features --------------------------------------------------

def test_german_plural_noun_gender_is_unreliable(de):
    """Justifies: ``_smor.noun()`` treating gender as a *hint* and trying the
    other genders.

    "Motoren" is the plural of masculine *Motor*, but Stanza labels it Fem.
    """
    w = _find(_words(de, "Die Motoren wurden repariert."), "Motoren")
    assert w.feats and "Gender=Masc" not in w.feats, (
        f"plural gender now looks correct ({w.feats}) — the gender fallback in "
        "_smor.noun() may be unnecessary"
    )


def test_german_nominalised_superlative_drops_degree(de):
    """Justifies: the ``-st-`` suffix fallback in
    ``nominal_ellipsis_de._check_superlative``."""
    w = _find(_words(de, "Das älteste ist verkauft."), "älteste")
    assert "Degree=Sup" not in (w.feats or ""), (
        "Degree=Sup is now present — the -st- suffix fallback may be retirable"
    )


def test_german_substituting_possessive_is_never_tagged_pposs(de):
    """Documents that ``nominal_ellipsis_de._check_possessive`` (xpos == PPOSS)
    is unreachable with the current model, which is why its fixture is
    hand-authored rather than a real parse."""
    seen = set()
    for text in ("Meiner ist kaputt.", "Das ist meins.", "Er nahm seinen.",
                 "Ihrer war schneller."):
        seen.update(w.xpos for w in _words(de, text))
    assert "PPOSS" not in seen, (
        "PPOSS is now emitted — the possessive nominal-ellipsis rule is live "
        "again and could be tested against a real parse"
    )


# --- dependency labels -------------------------------------------------------

def test_perfect_vs_passive_auxiliary_labels_are_unreliable(en):
    """Justifies: rejecting ``aux`` vs ``aux:pass``/``cop`` as the has/had
    signal, in favour of rules whose alternative is ungrammatical.

    "He's eaten already" is a perfect (the clitic is *has*), yet the clitic is
    not labelled a plain ``aux``.
    """
    dep = _find(_words(en, "He's eaten already."), "'s").deprel
    assert dep != "aux", (
        f"the perfect auxiliary is now labelled {dep!r} — deprel may have become "
        "a usable signal for disambiguating 's/'d"
    )
