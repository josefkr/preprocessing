"""Elementary Discourse Unit (EDU) segmenter wrapping the HF model
``poyum/test_discut``.

The model is an XLM-RoBERTa token classifier emitting per-word
labels: 0 = continuation, 1 = start of new EDU. This module wraps
the raw forward pass with subword-to-word alignment and returns
EDU boundaries as character offsets relative to the input sentence
string — so callers (``preprocessing/add_edus.py``) can simply add
each sentence's ``begin`` to map offsets into CAS sofa offsets.

The model accepts space-tokenized input. Languages without
inter-word whitespace (Chinese, Thai, Japanese) need to be
pre-tokenized before calling ``segment_sentence``.

Loading is offline by default (see ``_hf_offline``): the model must
have been downloaded once into the HF cache; subsequent runs do not
hit the network. The HF identifier of the segmenter is
:data:`MODEL_ID`.

Empirical findings (from the upstream ``segment_lite.py``):

  * The ``[LANG=]``/``[FRAME=]`` dummy tokens that the upstream
    custom pipeline prepended per sentence are *inert* on this
    checkpoint — predictions are bit-identical across language
    codes — so we keep them only for input-shape fidelity.
  * Label semantics differ by training corpus: 1 means EDU-start
    for RST/SDRT/dep/iso corpora (the relevant case for our
    multilingual setting), but means *connective-start* for the
    PDTB-only languages in the training mix (Italian luna, Thai
    tdtb, Nigerian Pidgin disconaija, Turkish tdb/tedm). We treat
    the labels as EDU boundaries everywhere; consumers should be
    aware that segmentation quality on the PDTB-only languages may
    be off-target.
"""

from __future__ import annotations

import logging
import re

# Force offline mode for HF Hub BEFORE transformers is imported. The
# import side-effect of this module is what sets the env vars.
from preprocessing import _hf_offline  # noqa: F401

logger = logging.getLogger(__name__)


MODEL_ID = "poyum/test_discut"
LABEL_O, LABEL_B = 0, 1
LANG_TOKEN, FRAME_TOKEN = "[LANG=]", "[FRAME=]"

# Max subword tokens the model accepts (including the [LANG=] and
# [FRAME=] dummy tokens). Longer sentences are truncated; trailing
# words then silently roll into the last predicted EDU, and we log a
# warning when this happens.
_MAX_TOKENS = 512


class EduSegmenter:
    """Wraps the HF EDU model for use in the preprocessing pipeline.

    Lazy-loads the model on first ``segment_sentence`` call and caches
    it for the instance's lifetime. Construct one instance per worker
    process.
    """

    def __init__(self, device: str | None = None):
        """Initialize.

        Args:
            device: Torch device string (``"cuda"`` / ``"cpu"``). When
                ``None`` (default), picks CUDA when available, else CPU.
        """
        self._device = device
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: E402
        from transformers import (  # noqa: E402
            AutoTokenizer,
            XLMRobertaForTokenClassification,
        )

        torch.set_grad_enabled(False)
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading EDU segmenter (%s) on %s...", MODEL_ID, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = (
            XLMRobertaForTokenClassification.from_pretrained(MODEL_ID)
            .eval()
            .to(self._device)
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_sentence(self, sentence: str) -> list[tuple[int, int]]:
        """Segment ``sentence`` into EDUs.

        Returns a list of ``(begin, end)`` character offsets relative
        to ``sentence``. A blank sentence returns ``[]``. A
        single-EDU sentence returns ``[(begin_of_first_word,
        end_of_last_word)]``.

        Truncation: sentences longer than ``_MAX_TOKENS`` subword
        tokens have their tail folded into the last predicted EDU
        (the model never sees the truncated words). A warning is
        logged in that case.
        """
        # Token positions: capture each whitespace-delimited word's
        # character span so we can map predictions back to offsets.
        word_matches = list(re.finditer(r"\S+", sentence))
        if not word_matches:
            return []
        words = [m.group() for m in word_matches]
        n_words = len(words)

        self._load()

        inputs = [LANG_TOKEN, FRAME_TOKEN] + words
        enc = self._tokenizer(
            inputs,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS,
        )
        word_ids = enc.word_ids()

        # Detect tail truncation: any word index past the highest one
        # the tokenizer mapped into a subword slot was dropped.
        last_seen_input_idx = max(
            (wid for wid in word_ids if wid is not None), default=-1,
        )
        # The first two positions in ``inputs`` are LANG/FRAME tokens,
        # so word index N in our caller-facing ``words`` list maps to
        # input index N + 2.
        last_seen_word_idx = last_seen_input_idx - 2
        if last_seen_word_idx < n_words - 1:
            logger.warning(
                "EDU segmenter input truncated at %d subword tokens: "
                "sentence has %d words but only the first %d received "
                "boundary predictions (tail folded into the last EDU).",
                _MAX_TOKENS, n_words, max(last_seen_word_idx + 1, 0),
            )

        model_inputs = {k: v.to(self._device) for k, v in enc.items()}
        preds = self._model(**model_inputs).logits[0].argmax(-1).tolist()

        # First-subword-per-word label, skipping specials.
        word_pred: dict[int, int] = {}
        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid in word_pred:
                continue
            word_pred[wid] = preds[tok_idx]

        # Group consecutive words into EDUs. The first word always
        # starts the first EDU; thereafter, a LABEL_B closes the
        # previous EDU at the prior word's end and starts a new one.
        edus: list[tuple[int, int]] = []
        edu_start_word = 0
        for word_idx in range(1, n_words):
            is_start = word_pred.get(word_idx + 2, LABEL_O) == LABEL_B
            if is_start:
                edus.append(
                    (
                        word_matches[edu_start_word].start(),
                        word_matches[word_idx - 1].end(),
                    )
                )
                edu_start_word = word_idx
        # Close the final EDU.
        edus.append(
            (
                word_matches[edu_start_word].start(),
                word_matches[-1].end(),
            )
        )
        return edus
