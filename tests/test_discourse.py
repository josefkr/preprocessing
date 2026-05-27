"""Tests for the EDU segmenter and the ``add_edus.py`` annotator.

The HF model is mocked here — we don't want test runs to depend on
``~/.cache/huggingface`` being populated. The tests verify:

  * Word-level boundary predictions are correctly converted into
    character offsets relative to the source sentence (including
    sentences with multiple consecutive spaces, leading/trailing
    whitespace, and single-word "sentences").
  * The CLI helper ``annotate_edus`` adds ``ElementaryDiscourseUnit``
    annotations to the right sofa offsets given pre-existing
    ``Sentence`` annotations.
  * ``annotate_edus`` errors when the view has no ``Sentence``
    annotations (per spec — sentence segmentation is required upstream).
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from preprocessing.discourse import EduSegmenter, LABEL_B, LABEL_O
from preprocessing.util import get_aslan_typesystem
from add_edus import annotate_edus, T_EDU, T_SENT


# ----------------------------------------------------------------------
# Fake model harness — drives the segmenter without loading XLM-R.
# ----------------------------------------------------------------------


def _install_fake_model(segmenter: EduSegmenter, word_labels: list[int]) -> None:
    """Replace the segmenter's lazy-load step with a fake tokenizer +
    fake model that emit the given per-word labels.

    ``word_labels`` is a list with one label per *content* word
    (LABEL_O / LABEL_B). The fake also emits 0s for the two
    ``[LANG=]``/``[FRAME=]`` slots and a special-token at each end,
    matching the shape the real tokenizer would produce."""

    # ``word_ids`` returned by the real fast tokenizer: [None, 0, 1, ...,
    # n+1, None] where 0, 1 are LANG/FRAME and 2..n+1 are the content
    # words. We synthesize a one-subword-per-word version.
    def _fake_tokenize(inputs, **kwargs):
        # inputs = [LANG_TOKEN, FRAME_TOKEN, w1, w2, ...]
        n_inputs = len(inputs)
        # Build the word_ids: [None, 0, 1, ..., n_inputs-1, None]
        word_ids = [None] + list(range(n_inputs)) + [None]
        # Labels: one per subword. Special tokens get LABEL_O.
        labels = (
            [LABEL_O]      # <s>
            + [LABEL_O]    # [LANG=]
            + [LABEL_O]    # [FRAME=]
            + word_labels  # content words
            + [LABEL_O]    # </s>
        )
        # Pad/truncate to match the real-tokenizer behavior at 512.
        # For our tests we never hit that limit.
        enc = MagicMock()
        enc.word_ids = lambda: word_ids
        # Make the encoding "look like" a dict for ``items()`` iteration.
        enc.items = lambda: iter([("input_ids", MagicMock(to=lambda d: object()))])
        enc.__getitem__ = lambda self, k: MagicMock()
        # Stash labels on the encoding so the fake model can read them.
        enc._fake_labels = labels
        return enc

    fake_tokenizer = MagicMock(side_effect=_fake_tokenize)

    def _fake_forward(**model_inputs):
        # The encoding object isn't passed through; for the test we
        # just use the last constructed encoding.
        labels = _fake_tokenize.last_labels
        # Construct logits where the argmax matches ``labels``.
        # logits shape: (batch=1, seq_len, num_labels=2)
        import torch
        seq_len = len(labels)
        logits = torch.zeros(1, seq_len, 2)
        for i, lab in enumerate(labels):
            logits[0, i, lab] = 1.0
        out = MagicMock()
        out.logits = logits
        return out

    # Capture labels for the forward pass.
    _orig = _fake_tokenize

    def _capture(inputs, **kwargs):
        enc = _orig(inputs, **kwargs)
        _fake_tokenize.last_labels = enc._fake_labels
        return enc

    fake_tokenizer.side_effect = _capture
    _fake_tokenize.last_labels = None

    fake_model = MagicMock(side_effect=_fake_forward)
    fake_model.device = "cpu"
    fake_model.eval = lambda: fake_model
    fake_model.to = lambda d: fake_model

    segmenter._tokenizer = fake_tokenizer
    segmenter._model = fake_model
    segmenter._device = "cpu"


# ----------------------------------------------------------------------
# Offset bookkeeping
# ----------------------------------------------------------------------


def test_segment_single_word_returns_one_edu():
    seg = EduSegmenter(device="cpu")
    _install_fake_model(seg, word_labels=[LABEL_O])
    spans = seg.segment_sentence("Hello.")
    assert spans == [(0, 6)]


def test_segment_no_boundaries_returns_one_edu():
    seg = EduSegmenter(device="cpu")
    # Three words, no boundaries → one EDU spanning all.
    _install_fake_model(seg, word_labels=[LABEL_O, LABEL_O, LABEL_O])
    sent = "The dog ran."
    spans = seg.segment_sentence(sent)
    # First word starts at 0, last word "ran." ends at len(sent).
    assert spans == [(0, len(sent))]


def test_segment_internal_boundary_splits_sentence():
    seg = EduSegmenter(device="cpu")
    # Sentence: "While he ran, the dog barked."
    # Words:       0=While 1=he 2=ran, 3=the 4=dog 5=barked.
    # Boundary on word 3 → EDUs are [0..2] and [3..5].
    _install_fake_model(
        seg, word_labels=[LABEL_O, LABEL_O, LABEL_O, LABEL_B, LABEL_O, LABEL_O],
    )
    sent = "While he ran, the dog barked."
    spans = seg.segment_sentence(sent)
    # Find the actual character positions via the same regex the
    # segmenter uses, to keep the expectation parser-independent.
    word_matches = list(re.finditer(r"\S+", sent))
    expected = [
        (word_matches[0].start(), word_matches[2].end()),  # "While he ran,"
        (word_matches[3].start(), word_matches[5].end()),  # "the dog barked."
    ]
    assert spans == expected
    # Sanity: the actual surface strings.
    assert sent[spans[0][0]:spans[0][1]] == "While he ran,"
    assert sent[spans[1][0]:spans[1][1]] == "the dog barked."


def test_segment_leading_and_trailing_whitespace_ignored():
    seg = EduSegmenter(device="cpu")
    _install_fake_model(seg, word_labels=[LABEL_O, LABEL_O])
    sent = "   Hi there.   "
    spans = seg.segment_sentence(sent)
    # Begin starts at the first non-whitespace ("H" at index 3); end at
    # the period (index 12).
    assert spans == [(3, 12)]
    assert sent[spans[0][0]:spans[0][1]] == "Hi there."


def test_segment_multiple_spaces_between_words():
    """Multiple spaces between words must not shift the EDU offsets —
    the segmenter uses ``\\S+`` which skips runs of whitespace."""
    seg = EduSegmenter(device="cpu")
    _install_fake_model(seg, word_labels=[LABEL_O, LABEL_B, LABEL_O])
    sent = "a   b    c"
    spans = seg.segment_sentence(sent)
    assert spans == [(0, 1), (4, 10)]
    assert sent[spans[0][0]:spans[0][1]] == "a"
    assert sent[spans[1][0]:spans[1][1]] == "b    c"


def test_segment_blank_sentence_returns_empty():
    seg = EduSegmenter(device="cpu")
    assert seg.segment_sentence("") == []
    assert seg.segment_sentence("    ") == []


def test_segment_boundary_on_first_word_is_ignored():
    """Per the upstream logic, the first content word always starts the
    first EDU — a LABEL_B on the very first word doesn't create a
    zero-width EDU before it."""
    seg = EduSegmenter(device="cpu")
    _install_fake_model(seg, word_labels=[LABEL_B, LABEL_O])
    sent = "a b"
    spans = seg.segment_sentence(sent)
    assert spans == [(0, 3)]


# ----------------------------------------------------------------------
# add_edus.py integration — CLI helper, real CAS, fake model
# ----------------------------------------------------------------------


def _build_cas_with_sentences(sentences: list[tuple[int, int]]):
    """Build a CAS view with the given Sentence annotations.

    ``sentences`` is a list of (begin, end) sofa offsets. The sofa
    string is taken from the largest end."""
    from cassis import Cas

    if not sentences:
        return None
    end = max(e for _, e in sentences)
    text = "x" * end  # any character — sufficient since we only need offsets
    ts = get_aslan_typesystem()
    cas = Cas(sofa_string=text, document_language="en", typesystem=ts)
    Sent = ts.get_type(T_SENT)
    for b, e in sentences:
        cas.add(Sent(begin=b, end=e))
    return cas, ts


def test_annotate_edus_writes_at_correct_sofa_offsets():
    # Sentence at sofa offsets 10..23 (length 13: "While he ran,").
    # Then mock segmenter splits inside that sentence at word 3 in a
    # 6-word sentence.
    real_text = "PAD       While he ran, the dog barked.PAD"
    #            0123456789012345678901234567890123456789012
    sent_begin = 10
    sent_end = 39

    from cassis import Cas

    ts = get_aslan_typesystem()
    cas = Cas(sofa_string=real_text, document_language="en", typesystem=ts)
    Sent = ts.get_type(T_SENT)
    cas.add(Sent(begin=sent_begin, end=sent_end))

    seg = EduSegmenter(device="cpu")
    _install_fake_model(
        seg, word_labels=[LABEL_O, LABEL_O, LABEL_O, LABEL_B, LABEL_O, LABEL_O],
    )

    n = annotate_edus(cas, ts, seg)
    assert n == 2

    edus = sorted(cas.select(T_EDU), key=lambda x: x.begin)
    assert len(edus) == 2
    assert real_text[edus[0].begin:edus[0].end] == "While he ran,"
    assert real_text[edus[1].begin:edus[1].end] == "the dog barked."


def test_annotate_edus_raises_without_sentences():
    from cassis import Cas

    ts = get_aslan_typesystem()
    cas = Cas(sofa_string="Some text here.", document_language="en", typesystem=ts)
    seg = EduSegmenter(device="cpu")
    # Model is never called because we error before segmenting.
    with pytest.raises(ValueError, match="Sentence"):
        annotate_edus(cas, ts, seg)


def test_annotate_edus_skips_blank_sentences():
    """A Sentence annotation whose sofa range is whitespace-only must
    not produce a (begin, begin) EDU."""
    real_text = "Hello world."
    from cassis import Cas

    ts = get_aslan_typesystem()
    cas = Cas(sofa_string=real_text, document_language="en", typesystem=ts)
    Sent = ts.get_type(T_SENT)
    # One real sentence, one whitespace-only sentence
    cas.add(Sent(begin=0, end=12))
    cas.add(Sent(begin=12, end=12))  # zero-width, definitely blank

    seg = EduSegmenter(device="cpu")
    _install_fake_model(seg, word_labels=[LABEL_O, LABEL_O])

    n = annotate_edus(cas, ts, seg)
    assert n == 1
    edus = list(cas.select(T_EDU))
    assert len(edus) == 1
    assert real_text[edus[0].begin:edus[0].end] == "Hello world."
