"""Force Hugging Face Hub into offline mode before ``transformers`` is imported.

Sets ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` (both with
``setdefault`` semantics so a caller's explicit override wins). Once
set, ``AutoTokenizer.from_pretrained(...)`` and
``AutoModel.from_pretrained(...)`` only use the local HF cache and do
not hit the network — which would otherwise hang on VPN-restricted
networks.

Model files are assumed to have been downloaded at least once **outside**
offline mode, into the default HF cache
(``~/.cache/huggingface/hub/``). To download the EDU segmenter
(``poyum/test_discut``) once, run **without** offline mode set::

    python -c "from transformers import AutoTokenizer, XLMRobertaForTokenClassification; \
        AutoTokenizer.from_pretrained('poyum/test_discut'); \
        XLMRobertaForTokenClassification.from_pretrained('poyum/test_discut')"

Import this module BEFORE any ``from transformers import ...`` line in
code paths that should be offline-only — e.g. it sits at the top of
``preprocessing.discourse``, which constructs the EDU segmenter.

Mirrors the pattern of ``aslan_normalization._stanza_offline`` for the
Stanza model loader.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
