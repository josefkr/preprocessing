"""Patch ``stanza.Pipeline`` to default ``download_method`` to ``NONE``
and point Stanza at the legacy ``~/stanza_resources/`` model cache.

Importing this module:

  1. Sets ``STANZA_RESOURCES_DIR`` (if not already set) to
     ``~/stanza_resources``. Stanza 1.11+ changed its default cache
     location to a versioned ``~/.cache/stanza/<version>/resources/``
     directory; this project keeps models under the older
     ``~/stanza_resources/`` path, so without this override Stanza
     reports "Cannot load model" even though the files are present.
  2. Patches ``stanza.Pipeline.__init__`` to default
     ``download_method=NONE`` (idempotent). Any subsequent
     ``stanza.Pipeline(...)`` call in the process will skip Stanza's
     online resources check, which otherwise hangs when the machine is
     on a VPN that doesn't route to the Stanza model server.

The model files are assumed to be already installed locally under
``~/stanza_resources/<lang>/``. If they aren't, run download once
outside the VPN:

    python -c "import stanza; stanza.download('en')"
    python -c "import stanza; stanza.download('de')"

(The downloads will respect ``STANZA_RESOURCES_DIR`` if it's set, so
they end up in the legacy directory.)

Any module that constructs a ``stanza.Pipeline`` (directly or
transitively via ``preprocessing.stanza.Stanza_Preprocessor``) should
``import preprocessing._stanza_offline`` at the top, so the patch is in
place before the *first* pipeline construction in the process.
``preprocessing.stanza`` caches the pipeline module-globally, so a late
import wouldn't take effect.

Both side effects use ``setdefault`` semantics: a caller can override
either via environment or by passing explicit kwargs to
``stanza.Pipeline``.

A sibling copy lives at ``aslan_normalization._stanza_offline`` in the
normalization fork; the two patches are idempotent, so importing both
in one process is safe.
"""

import os

os.environ.setdefault(
    "STANZA_RESOURCES_DIR",
    os.path.expanduser("~/stanza_resources"),
)


def _disable_stanza_downloads() -> None:
    import stanza
    from stanza.pipeline.core import DownloadMethod

    if getattr(stanza.Pipeline.__init__, "_no_download_patched", False):
        return

    _original_init = stanza.Pipeline.__init__

    def _patched_init(pipeline_self, *args, **kwargs):
        kwargs.setdefault("download_method", DownloadMethod.NONE)
        return _original_init(pipeline_self, *args, **kwargs)

    _patched_init._no_download_patched = True
    stanza.Pipeline.__init__ = _patched_init


_disable_stanza_downloads()
