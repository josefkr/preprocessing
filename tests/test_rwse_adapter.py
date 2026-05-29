"""Tests for the RWSE CAS adapter (``add_rwse.annotate_rwse``).

The ``RWSE_Checker`` (which loads a transformer model) is replaced by a
fake whose ``in_confusion_sets`` / ``check_multi`` behaviour is
scripted, so these tests run without any model download or inference.
They verify:

  * an RWSE annotation is written (with suggestion + certainty) when the
    fake checker prefers a different confusion-set member;
  * tokens not in a confusion set, and tokens the checker leaves
    unchanged, produce no annotation;
  * only the targeted occurrence is masked when a confusion word repeats
    in the sentence;
  * leading-letter case is transferred from the original token onto the
    (lowercased) model suggestion;
  * the adapter errors when the view lacks Sentence / Token annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from add_rwse import annotate_rwse, MASK, T_SENT, T_TOKEN, T_RWSE
from preprocessing.util import get_aslan_typesystem


@dataclass
class _Tok:
    begin: int
    end: int


class _FakeChecker:
    """Scripted stand-in for ``RWSE_Checker``.

    ``confusion`` maps a lowercased surface to its confusion set.
    ``check_results`` maps a lowercased surface to a list of
    ``(token_str, score)`` pairs — exactly the per-candidate scores the
    real ``check_multi`` returns. A surface absent from
    ``check_results`` returns an empty list."""

    def __init__(self, confusion, check_results, case_sensitive=False):
        self.confusion = confusion
        self.check_results = check_results
        self.case_sensitive = case_sensitive
        self.seen_masked: list[str] = []

    def in_confusion_sets(self, token) -> bool:
        key = token if self.case_sensitive else token.lower()
        return key in self.confusion

    def check_multi(self, token, masked_sentence):
        # Record the masked sentence so a test can assert which
        # occurrence was masked.
        self.seen_masked.append(masked_sentence)
        key = token if self.case_sensitive else token.lower()
        return [
            {"token_str": ts_, "score": sc, "sequence": ""}
            for ts_, sc in self.check_results.get(key, [])
        ]


def _make_cas(text: str):
    from cassis import Cas

    ts = get_aslan_typesystem()
    cas = Cas(sofa_string=text, document_language="de", typesystem=ts)
    return cas, ts


def _add_sentence_and_tokens(cas, ts, spans: list[tuple[int, int]], sent=None):
    """Add a Sentence covering ``sent`` (default: whole sofa) plus Token
    annotations at the given char spans."""
    Sent = ts.get_type(T_SENT)
    Tok = ts.get_type(T_TOKEN)
    text = cas.sofa_string
    if sent is None:
        sent = (0, len(text))
    cas.add(Sent(begin=sent[0], end=sent[1]))
    for b, e in spans:
        cas.add(Tok(begin=b, end=e))


def _token_spans(text: str) -> list[tuple[int, int]]:
    """Whitespace token spans (good enough for these fixtures)."""
    spans = []
    i = 0
    for word in text.split(" "):
        # find the word from i (handles single spaces)
        b = text.index(word, i)
        spans.append((b, b + len(word)))
        i = b + len(word)
    return spans


# ----------------------------------------------------------------------
# Core behaviour
# ----------------------------------------------------------------------


def test_writes_rwse_when_checker_suggests_change():
    text = "ich gebe das Buch ab"
    #       0   4    9   13   18
    cas, ts = _make_cas(text)
    _add_sentence_and_tokens(cas, ts, _token_spans(text))

    checker = _FakeChecker(
        confusion={"das": ["das", "dass"]},
        check_results={"das": [("dass", 0.985), ("das", 0.004)]},
    )
    n = annotate_rwse(cas, ts, checker)
    assert n == 1

    rwse = list(cas.select(T_RWSE))
    assert len(rwse) == 1
    a = rwse[0]
    assert text[a.begin:a.end] == "das"
    assert a.suggestion == "dass"
    # certainty = log10(0.985) - log10(0.004) ≈ 2.39
    assert a.certainty == pytest.approx(2.39, abs=0.01)
    assert a.category == "rwse"


def test_no_annotation_when_checker_keeps_token():
    text = "ich gebe das Buch ab"
    cas, ts = _make_cas(text)
    _add_sentence_and_tokens(cas, ts, _token_spans(text))

    # "das" is a confusion word but scores far higher than "dass" here,
    # so no swap should be flagged.
    checker = _FakeChecker(
        confusion={"das": ["das", "dass"]},
        check_results={"das": [("das", 0.97), ("dass", 0.01)]},
    )
    n = annotate_rwse(cas, ts, checker)
    assert n == 0
    assert list(cas.select(T_RWSE)) == []


def test_tokens_outside_confusion_sets_ignored():
    text = "voll der schöne Tag"
    cas, ts = _make_cas(text)
    _add_sentence_and_tokens(cas, ts, _token_spans(text))

    # No token is in a confusion set → checker.check never matters.
    checker = _FakeChecker(confusion={"das": ["das", "dass"]}, check_results={})
    n = annotate_rwse(cas, ts, checker)
    assert n == 0


def test_only_targeted_occurrence_is_masked():
    text = "das ist das Auto"
    #       0   4   8   12
    cas, ts = _make_cas(text)
    _add_sentence_and_tokens(cas, ts, _token_spans(text))

    checker = _FakeChecker(
        confusion={"das": ["das", "dass"]},
        check_results={"das": [("dass", 0.9), ("das", 0.05)]},
    )
    annotate_rwse(cas, ts, checker)

    # "das" appears twice → two check() calls, each masking a
    # different occurrence (exactly one MASK token per call, at a
    # different position).
    assert len(checker.seen_masked) == 2
    masked_first, masked_second = checker.seen_masked
    assert masked_first == f"{MASK} ist das Auto"
    assert masked_second == f"das ist {MASK} Auto"


def test_case_transfer_capitalizes_suggestion():
    text = "die Rede war lang"
    #       0   4    9   13
    cas, ts = _make_cas(text)
    _add_sentence_and_tokens(cas, ts, _token_spans(text))

    # Confusion set is lowercased (ci file); model emits lowercase
    # "reede"; adapter should restore the capital R from "Rede".
    checker = _FakeChecker(
        confusion={"rede": ["rede", "reede"]},
        check_results={"rede": [("reede", 0.9), ("rede", 0.05)]},
    )
    n = annotate_rwse(cas, ts, checker)
    assert n == 1
    a = list(cas.select(T_RWSE))[0]
    assert text[a.begin:a.end] == "Rede"
    assert a.suggestion == "Reede"


def test_min_certainty_filters_weak_suggestions():
    text = "ich gebe das Buch ab"
    cas, ts = _make_cas(text)
    _add_sentence_and_tokens(cas, ts, _token_spans(text))

    # best=0.5, orig=0.4 → passes magnitude=1 (0.5 ≥ 0.4), but
    # certainty = log10(0.5/0.4) ≈ 0.097 < min_certainty 1.0 → filtered.
    checker = _FakeChecker(
        confusion={"das": ["das", "dass"]},
        check_results={"das": [("dass", 0.5), ("das", 0.4)]},
    )
    n = annotate_rwse(cas, ts, checker, magnitude=1, min_certainty=1.0)
    assert n == 0


# ----------------------------------------------------------------------
# Preconditions
# ----------------------------------------------------------------------


def test_raises_without_sentences():
    text = "das ist das Auto"
    cas, ts = _make_cas(text)
    # Tokens but no Sentence.
    Tok = ts.get_type(T_TOKEN)
    for b, e in _token_spans(text):
        cas.add(Tok(begin=b, end=e))
    checker = _FakeChecker(confusion={"das": ["das", "dass"]}, check_results={})
    with pytest.raises(ValueError, match="Sentence"):
        annotate_rwse(cas, ts, checker)


def test_raises_without_tokens():
    text = "das ist das Auto"
    cas, ts = _make_cas(text)
    Sent = ts.get_type(T_SENT)
    cas.add(Sent(begin=0, end=len(text)))
    checker = _FakeChecker(confusion={"das": ["das", "dass"]}, check_results={})
    with pytest.raises(ValueError, match="Token"):
        annotate_rwse(cas, ts, checker)
