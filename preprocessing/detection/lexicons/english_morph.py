"""English wellformedness oracle: affix stripping over a wordlist.

The English counterpart of the one SMOR property this codebase actually depends on —
the ability to **reject**. ``smor.analyse("Unkt")`` returns an empty tuple, and that
emptiness is what the German spelling gate and the suspended-composition
analysability gate are built on. SMOR gets it by *composing*: it recognises
``Zwischentestgruppe`` and ``Behaltensleistung`` without ever having seen them.

A plain English wordlist cannot do that. Measured against pyspellchecker's list:

===================  =========
form                 known?
===================  =========
``maintainability``  yes
``encapsulation``    yes
``reusability``      **no**
``debuggability``    **no**
``subclassable``     **no**
``decompounding``    **no**
===================  =========

All four misses are ordinary English derivations, and ``reusability`` occurs in our
own English answer data. So this module composes the way SMOR does, over the one
productive process English actually has in quantity: **derivation**. Peel a known
affix, ask the wordlist, recurse.

Two neural tools were evaluated for this job and rejected — CompoundPiece and PaReNT
— for the same structural reason: both are *transducers*, always emitting an answer,
so neither can reject. See the "English morphology" section of
``normalization/README.md``. A rule that can say no is worth more here than a
0.88-accuracy classifier that cannot.

**Deliberate over-acceptance.** ``catness`` is accepted: nobody writes it, but it is
morphologically wellformed English, and SMOR accepts the equivalent German
non-attested-but-wellformed compounds too. The question this module answers is "could
this be an ordinary English word?", not "is it in a dictionary" — that is exactly the
question an emphasis veto or an OOV gate needs.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

#: Derivational suffixes, each with the stem shapes peeling it may leave. Ordered
#: longest-first so ``ability`` is tried before ``ity`` and ``y``.
#:
#: The replacement lists encode English's orthographic joins: ``activity`` needs
#: ``ity`` -> ``e`` to reach *active*, ``creation`` needs ``ion`` -> ``e`` to reach
#: *create*. An empty string means "peel and stop".
SUFFIX_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ability", ("able", "")),          # reusability -> reusable
    ("ibility", ("ible", "")),
    ("ization", ("ize", "")),
    ("isation", ("ise", "")),
    ("ational", ("ate", "")),
    ("fulness", ("ful", "")),
    ("lessness", ("less", "")),
    ("ousness", ("ous", "")),
    ("iveness", ("ive", "")),
    ("ation", ("ate", "e", "")),        # documentation -> document
    ("ition", ("ite", "e", "")),
    ("ement", ("e", "")),
    ("ment", ("e", "")),
    ("ness", ("", "y")),                # happiness -> happy (via i->y undo)
    ("less", ("",)),
    ("able", ("e", "")),                # debuggable -> debug (via undoubling)
    ("ible", ("e", "")),
    ("tion", ("te", "t", "")),
    ("sion", ("de", "d", "")),
    ("ance", ("e", "")),
    ("ence", ("e", "")),
    ("ancy", ("ant", "")),
    ("ency", ("ent", "")),
    ("ship", ("",)),
    ("hood", ("",)),
    ("wise", ("",)),
    ("ward", ("",)),
    ("ised", ("ise", "")),
    ("ized", ("ize", "")),
    ("ings", ("e", "")),
    ("ings", ("",)),
    ("ful", ("",)),
    ("ist", ("e", "")),
    ("ism", ("e", "")),
    ("ion", ("e", "")),
    ("ity", ("e", "")),                 # activity -> active
    ("ive", ("e", "")),
    ("ous", ("e", "")),
    ("ise", ("", "y")),
    ("ize", ("", "y")),
    ("ify", ("", "y")),
    ("ing", ("e", "")),                 # programming -> program (undoubling)
    ("est", ("e", "")),
    ("ers", ("e", "")),
    ("ory", ("e", "")),
    ("ary", ("e", "")),
    ("ing", ("",)),
    ("ial", ("", "y")),
    ("al", ("e", "")),
    ("ly", ("", "le")),
    ("er", ("e", "")),                  # debugger -> debug, writer -> write
    ("or", ("e", "")),
    ("ed", ("e", "")),
    ("es", ("e", "")),
    ("s", ("",)),
    ("y", ("e", "")),
)

#: Productive English prefixes. Peeling these needs no orthographic repair, which is
#: why they are a flat list rather than rules.
PREFIXES: tuple[str, ...] = (
    "pseudo", "counter", "inter", "intra", "super", "trans", "under", "multi",
    "over", "semi", "anti", "auto", "micro", "macro", "post", "pre", "sub",
    "non", "mis", "dis", "out", "up", "un", "re", "de", "in", "im", "co",
)

#: How many affixes may be peeled in sequence. ``reusability`` needs two
#: (``ability`` -> *reusable*, then ``able`` -> *reuse*); three covers
#: ``un`` + ``reusable`` + ``ity``. Beyond that the search starts accepting noise.
MAX_DEPTH = 3

#: A stem shorter than this is not credible, and short strings are exactly where
#: spurious peels land (``ooa`` -> ``oo``).
MIN_STEM = 3


class EnglishWords:
    """Lazily-loaded wordlist plus the affix-stripping acceptor.

    Both halves live together because the acceptor is only meaningful relative to a
    wordlist: peeling is the *composition* rule, the list is the lexicon it composes
    over. Optional — with no wordlist available every method returns ``False``, so a
    caller relying on this to *veto* becomes conservative rather than wrong.
    """

    def __init__(self) -> None:
        self._impl = None
        self._checked = False

    @property
    def impl(self):
        if not self._checked:
            self._checked = True
            try:
                from spellchecker import SpellChecker

                self._impl = SpellChecker(language="en")
            except Exception as e:  # noqa: BLE001 — optional resource
                logger.debug(f"English wordlist unavailable ({e})")
                self._impl = None
        return self._impl

    @property
    def available(self) -> bool:
        return self.impl is not None

    @lru_cache(maxsize=8192)
    def knows(self, word: str) -> bool:
        """True if ``word`` is in the wordlist verbatim."""
        impl = self.impl
        if impl is None or not word:
            return False
        try:
            return word.lower() in impl
        except Exception:  # noqa: BLE001
            return False

    # --- the composing part -------------------------------------------------

    @staticmethod
    def _repairs(stem: str) -> list[str]:
        """Orthographic undo operations after a peel, most likely first.

        English spelling mutates at the join, so the peeled string is often not the
        stem: *debuggable* leaves ``debugg`` (doubled consonant), *happiness* leaves
        ``happi`` (y→i). Without these the acceptor rejects most real derivations.
        """
        out = [stem]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1].isalpha():
            out.append(stem[:-1])                       # debugg -> debug
        if stem.endswith("i"):
            out.append(stem[:-1] + "y")                 # happi -> happy
        if stem.endswith("ck"):
            out.append(stem[:-1])                       # panick -> panic
        return out

    def _candidate_stems(self, word: str):
        """Every stem ``word`` could reduce to by peeling one affix."""
        for suffix, replacements in SUFFIX_RULES:
            if not word.endswith(suffix) or len(word) - len(suffix) < MIN_STEM:
                continue
            base = word[: -len(suffix)]
            for repl in replacements:
                for cand in self._repairs(base + repl) if not repl else [base + repl]:
                    if len(cand) >= MIN_STEM:
                        yield cand
        for prefix in PREFIXES:
            if word.startswith(prefix) and len(word) - len(prefix) >= MIN_STEM:
                yield word[len(prefix):]

    @lru_cache(maxsize=8192)
    def derivable(self, word: str, _depth: int = 0) -> bool:
        """True if ``word`` reduces to a known word by peeling known affixes.

        This is the SMOR-analogue: it accepts wellformed forms the list has never
        seen (``reusability``, ``debuggability``, ``subclassable``) and **rejects**
        strings that are not English word shapes (``objectoriented``, ``Unkt``).
        Rejection is the whole point — see the module docstring.
        """
        w = (word or "").lower()
        if not w or _depth > MAX_DEPTH:
            return False
        if self.knows(w):
            return True
        if _depth == MAX_DEPTH:
            return False
        return any(self.derivable(stem, _depth + 1)
                   for stem in self._candidate_stems(w))

    def is_wordlike(self, word: str) -> bool:
        """True if ``word`` is a known or morphologically derivable English word.

        The method callers should use. Hyphenated forms are checked per component,
        since a hyphen is a word boundary in English (``re-usability``); every
        component must be wordlike, so ``re-xyzzy`` is rejected.
        """
        w = (word or "").strip().lower()
        if not w:
            return False
        if "-" in w:
            parts = [p for p in w.split("-") if p]
            if not parts:
                return False
            # A bound prefix component ("re-") is legitimate on its own.
            return all(self.derivable(p) or p in PREFIXES for p in parts)
        return self.derivable(w)


@lru_cache(maxsize=1)
def get_english_words() -> EnglishWords:
    """Process-wide oracle (the wordlist is loaded once)."""
    return EnglishWords()
