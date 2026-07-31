"""Per-language data for suspended composition (German).

A *suspended compound* writes a shared right-hand constituent only once, on the
last conjunct, and marks its omission elsewhere with a hyphen: "Sonn- und
Feiertagen" = Sonntagen und Feiertagen, "be- und entladen" = beladen und entladen.
The orthographic device is the so-called  *Ergänzungsstrich*.

Detection needs almost no data — a word-final hyphen plus a coordinator is the
whole signature — so this module holds only the three closed classes that the
resolution step cannot derive:

* ``COORDINATORS_BY_LANG`` — what may join the conjuncts. ``oder`` and bare commas
  matter as much as ``und``: real examples include "Geistes- oder
  Sozialwissenschaftler" and the three-way "An- , Um- oder Abmeldeprozedur".
* ``VERB_PARTICLES_BY_LANG`` — German separable and inseparable verbal prefixes.
  These are needed because the *surface splitter* is noun-trained and never
  proposes a body shorter than three characters, so it cannot see the boundary in
  ``Ab|meldeprozedur`` or ``ab|geschaltet``. As a closed class, they can simply
  be enumerated.
* ``STUB_RE`` / ``LEAD_STUB_RE`` / ``NON_STUB_RE`` — the surface shape of a
  truncated conjunct in either direction, and what disqualifies one.

German suspends in **both** directions, and the hyphen's position says which:

* trailing hyphen — the shared constituent is the **head**, written on the last
  conjunct: "Sonn- und Feiertagen". The donor *follows* the stub.
* leading hyphen — the shared constituent is the **modifier**, written on the
  first conjunct: "Kindergarten und -krippe", "Energieerzeugung und -verteilung".
  The donor *precedes* the stub.

Both can occur in one sentence ("Frauen- und Kinderhandel, Drogenhandel und
-konsum"), so neither direction can assume it is the only one present.

Why surface patterns rather than the parse: Stanza handles these tokens
inconsistently and badly. In one sentence it kept ``An-`` whole but split ``Um-``
into ``Um`` + ``-``, and it tags the stub ``PUNCT``:

    Dabei be- und entladen  ->  ('be-', PUNCT) ('und', CCONJ) ('entladen', VERB)
    Die Vereins- und ...    ->  ('Vereins', NOUN) ('-', PUNCT) ('und', CCONJ)

So the stub cannot be located through tokens or dependencies. This is also why
suspended composition has to be resolved *before* any parse-dependent step.
(TODO: We may look at what other parsers do with these casese.)
"""

from __future__ import annotations

import re

#: A truncated conjunct: letters (or digits, for "3- und 4-Zimmer-Wohnung") ending
#: in a hyphen at a word boundary. Internal hyphens do not qualify a token — the
#: hyphen must be **final**, which is what separates the stub ``Horror-`` from the
#: ordinary hyphenated compound ``Horror-Videos`` and from ``Sachsen-Anhalt``.
STUB_RE = re.compile(r"(?<![\w-])([\w][\w.]*?)-(?=\s|$|[,;])")

#: The **mirror** stub: a hyphen at the *start* of a word. Here the shared
#: constituent is the left one and the gap is in the later conjunct —
#: "Kindergarten und -krippe" = Kindergarten und Kinderkrippe. The donor therefore
#: *precedes* the stub, the opposite of :data:`STUB_RE`.
LEAD_STUB_RE = re.compile(r"(?<![\w-])-([\w][\w.]*)")

#: Discards a hyphen that is not a suspension: an em/en dash used as punctuation,
#: and a lone hyphen with no material before it.
NON_STUB_RE = re.compile(r"^[\W\d_]*$")

#: What may join suspended conjuncts. A comma alone is enough ("An- , Um- oder
#: Abmeldeprozedur"), so the resolver treats punctuation as a coordinator too.
COORDINATORS_BY_LANG: dict[str, frozenset[str]] = {
    "de": frozenset({"und", "oder", "bzw.", "beziehungsweise", "sowie", "wie",
                     "als", "aber", "&", "/"}),
    "en": frozenset({"and", "or", "&", "/"}),
}

#: German verbal prefixes: separable particles and inseparable prefixes together.
#: Longest first so that ``hinunter`` wins over ``hin``. Enumerated because the
#: surface splitter cannot propose two-character bodies, which is precisely where
#: it failed on real data (``Ab|meldeprozedur``, ``ab|geschaltet``).
VERB_PARTICLES_BY_LANG: dict[str, tuple[str, ...]] = {
    "de": (
        # separable, multi-syllable directionals first
        "gegenüber", "zusammen", "zurecht", "zurück", "hinunter", "herunter",
        "hinüber", "herüber", "hervor", "herein", "heraus", "herbei", "herab",
        "heran", "herauf", "hinein", "hinaus", "hinauf", "hinab", "entgegen",
        "voraus", "vorbei", "vorüber", "voran", "weiter", "wieder", "nieder",
        "empor", "durch", "hinter", "unter", "über", "gegen",
        # separable, short
        "auf", "aus", "ein", "fort", "mit", "nach", "vor", "weg", "zu", "los",
        "her", "hin", "ab", "an", "bei", "da", "dar", "um",
        # inseparable
        "miss", "wider", "zer", "ent", "ver", "be", "er", "ge", "emp", # empfangen!
    ),
    "en": (),
}

#: A body this short is only credible as a verbal particle, never as a compound
#: member; used to stop one- and two-letter fragments from winning.
MIN_BODY_LEN = 2


def coordinators(lang: str | None) -> frozenset[str]:
    return COORDINATORS_BY_LANG.get(lang or "", frozenset())


def verb_particles(lang: str | None) -> tuple[str, ...]:
    return VERB_PARTICLES_BY_LANG.get(lang or "", ())


def particle_prefix_lengths(word: str, lang: str | None = "de") -> tuple[int, ...]:
    """Lengths of the verbal prefixes ``word`` starts with, longest first.

    Supplies the split points the surface splitter structurally cannot see. Note
    this deliberately over-generates — ``Geschichtsverein`` starts with ``ge`` —
    so every candidate must still pass the caller's morphological and attestation
    gates. Over-generating is safe; missing the only correct boundary is not.
    """
    lower = word.lower()
    return tuple(
        len(p) for p in verb_particles(lang)
        if lower.startswith(p) and len(p) < len(word)
    )
