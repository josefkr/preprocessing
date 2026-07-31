# Detection Modules

This document describes the suite of annotators that detect linguistic
phenomena (sluicing, subject sharing, verbal ellipsis, passive,
nominal-head ellipsis, clefts, bare wh-questions, gapped coordination,
right node raising, contractions, abbreviations, suspended composition) in CAS
XMI files, how they are
organized, and how to extend them with new phenomena or new languages.

## Overview

Each phenomenon is implemented as three layers:

1. A **pure detector** that operates on a [udapi][udapi] document and
   returns plain-Python *findings*. It has no CAS dependency and is
   directly testable with `.conllu` fixtures.
2. A **CAS adapter** entry point that converts a CAS view into a
   CoNLL-U document, runs the detector, and writes the resulting
   findings back into the view as UIMA annotations. Each phenomenon is
   registered in `cas_adapter.DETECTOR_REGISTRY` (its detector, writer,
   and annotation signature).
3. A **single generic CLI** (`annotate.py --phenomenon <name>`) that loads
   XMI files, calls the adapter on each requested view, and saves the result.
   (There is no longer a per-phenomenon `add_<phenomenon>.py`; the external-tool
   annotators — coreference, RWSE, Stanza parses, EDUs, spelling — keep their
   own `add_*.py` scripts because they are not structural detectors.)

This split means detection logic can be developed and verified
without touching XMI/CAS at all, while CAS I/O is centralized in one
place.

[udapi]: https://github.com/udapi/udapi-python

## Package layout

```
preprocessing/detection/
  cas_conllu.py        CAS view → CoNLL-U string (one block per sentence,
                       offsets preserved in MISC, optional `# lang =`).
  offsets.py           Recovers (begin, end) offsets from a udapi node's
                       MISC. Handles single tokens and multi-token spans.
  language.py          Lingua wrapper, `tree_lang()` parser for
                       `# lang =`, `UnsupportedLanguage` exception.
  cli.py               Shared argparse plumbing (`--lang`, `--mixed`).
  cas_adapter.py       `find_and_annotate_*` entry points + writers.
                       Plumbs `lang`/`mixed` through to detectors.
  lift_annotators.py   py_lift-style `SE_<Phenom>Annotator` wrappers around
                       the adapters above (Option 1 of
                       ../PYLIFT_INTEGRATION.md). One shared base declares
                       requires_types once — every structural detector needs
                       the same Token/POS/Dependency/Sentence set — so a
                       subclass only names its adapter and its
                       supported_languages. `ANNOTATORS` maps the same
                       phenomenon keys `annotate.py --phenomenon` uses.
  sluicing.py          Pure sluicing detector.
  subject_sharing.py   Pure subject-sharing detector.
  verbal_ellipsis.py   Pure verbal-ellipsis detector.
  passive.py           Pure passive detector (canonical + short).
  nominal_ellipsis.py  Pure nominal-head ellipsis detector. Dispatches on
                       each sentence's language: German trees use the
                       rules in nominal_ellipsis_de.py, other languages
                       the lexicon-driven English-style checks.
  nominal_ellipsis_de.py  German nominal-head ellipsis rules — German
                       Stanza output (STTS XPOS, DET/PIS quantifiers)
                       does not fit the English Penn-Treebank rules.
  clefts.py            Pure cleft detector (English it-clefts + wh-clefts).
  bare_questions.py    Pure bare-wh-question detector. Whole sentence is
                       a wh-phrase ("Why?", "What for?", "What man?") —
                       no verb, no embedding governor.
  gapped_coordination.py
                       Pure gapped-coordination detector. A coordinated
                       clause whose main predicate is missing must
                       borrow it from the antecedent ("Paul wanted a
                       milk shake and Mr Leonard a coffee").
  right_node_raising.py
                       Pure right-node-raising detector (coordination
                       subset only). Two coordinated predicates share a
                       right-edge constituent elided from the non-final
                       conjunct ("Sam likes but Sue dislikes opera").
                       Lexicon-free / structural. Comparative &
                       subordinate RNR are left to the LLM normalizer.
  suspended_composition.py
                       Pure suspended-composition detector (German
                       Ergänzungsstrich). Two directions, told apart by
                       where the hyphen sits: trailing means the *head*
                       is shared and the donor follows ("Sonn- und
                       Feiertagen"); leading means the *modifier* is
                       shared and the donor precedes ("Energieerzeugung
                       und -verteilung"). Both can occur in one sentence,
                       so neither pattern assumes it is alone. Works on the sentence
                       *surface* and maps back to sofa offsets through the
                       overlapping tokens, because Stanza tags the stub
                       PUNCT and splits it or not depending on the word.
                       Takes an optional `resolver` hook to attach the
                       completion; reports the site either way.
  abbreviations.py     Pure abbreviation detector (German, English).
                       All-caps runs of 2-6 letters, gated four ways: not a
                       function word in caps ("Test VOR dem Lernen" is
                       emphasis), not part of a punctuation-separated
                       enumeration ("ADE - BEC - CBA"), not in
                       mostly-capitalised text, and occurrences beside
                       their own long form ("Konditionierter Reiz (CS)")
                       flagged rather than dropped. Which long form an
                       abbreviation stands for is decided corpus-wide, so
                       the detector takes an optional `expansions` map and
                       reports candidates with or without it.
  contractions.py      Pure contraction / clitic detector (English +
                       German). Four mechanisms: (1) a clitic token
                       ("n't", "'s", …) written *adjacent* to its host
                       ("wouldn't", "mir's"), whose (form, lemma) pair
                       has an expansion — English possessive "'s" is
                       absent and never fires; (2) German prep+article
                       multiword tokens ("vom" = von+dem), read from the
                       tree's MWTs (ADP+DET shape), with the parser's own
                       expansion; (3) German clipped indefinite articles
                       ("nen"/"nem"/"ner"/"ne"/"n"), standalone tokens
                       recognised by surface form — "ne" needs a following
                       NP, bare "n" is inferred from the noun's morphology;
                       (4) English colloquial clipped forms ("gonna",
                       "lemme", "'em"), matched as whole written words
                       (one token or an adjacent run, since the tokenizer
                       splits some of them), plus a productive g-dropping
                       rule for "talkin'" -> "talking".
  lexicons/
    sluicing_wh.py        Wh-words, question-embedding predicates
                          (verbal), and question-embedding nouns per
                          language.
    passive.py            Agent prepositions + participle XPOS sets.
    nominal_ellipsis.py   English quantifier forms, idiomatic patterns,
                          definite articles, comparative/JJ XPOS.
    clefts.py             Cleft-pronoun forms etc. per language.
    abbreviations.py      Candidate shape, the per-language function-word
                          veto list, the English dictionary-based emphasis
                          veto with its two evidence strengths, the closed
                          list of lexicalised forms (i.e./e.g./w/ -- a
                          second candidate shape the all-caps rule cannot
                          see, and the most frequent abbreviation class in
                          English prose), enumeration-run parameters, the gloss
                          context pattern, the substitutability filter for
                          scraped long forms (~a quarter of Wiktionary's
                          German all-caps rows are glosses, not
                          expansions), and the mostly-caps rule with its
                          minimum-token guard.
    contractions.py       Per-language clitic expansions keyed by
                          (form, lemma) (EN + DE), the irregular English
                          hosts ("ca" -> "can", "wo" -> "will"), the
                          German prep+article lexicalised-exception list,
                          the German clipped-article table + indefinite
                          paradigm (for inferring bare "n"), the English
                          clipped-forms table, the whole-contraction
                          overrides ("can't" -> "cannot"), and the
                          context-aware helpers that disambiguate
                          "'s"/"'d" before a participle and inflect
                          "ain't" for subject agreement.

annotate.py                One generic CLI for all structural detectors:
                           `annotate.py --phenomenon <name>` (choices come from
                           cas_adapter.DETECTOR_REGISTRY). Replaces the former
                           per-phenomenon add_<phenomenon>.py scripts.
```

`add_spelling_errors.py`, `add_coreference.py`, and `add_edus.py`
follow the same CLI template but wrap external annotators
(py_lift's spell checker, the Maverick coreference service, and the
HF model `poyum/test_discut` for Elementary Discourse Unit
segmentation respectively) rather than detectors in
`preprocessing/detection/`. They share the `--replace`
skip/overwrite semantics described in [CLI conventions](#cli-conventions).
`add_edus.py` additionally requires upstream `Sentence` annotations
on the chosen view — sentence segmentation is not auto-run.

## Pipeline per detector

```
CAS view                 view_to_conllu(view, sentence_langs=…)
  │                          │
  │ adapter computes lang per │
  │ sentence (auto / explicit │
  │ / per-sentence)           │
  ▼                          ▼
┌──────────────┐   ┌───────────────────────────┐   ┌───────────────────┐
│ Sentence     │ → │ CoNLL-U string with       │ → │ udapi.Document    │
│ annotations  │   │ `# lang =` per block,     │   │ (one tree/sent)   │
│              │   │ token offsets in MISC.    │   │                   │
└──────────────┘   └───────────────────────────┘   └─────────┬─────────┘
                                                              │
                                                    detect_<phenomenon>
                                                    (restrict_to_lang=…)
                                                              │
                                                              ▼
                                                  list[<…>Finding]
                                                              │
                                                              ▼
                                                    _write_<phenomenon>
                                                  → CAS annotations on view
```

The conversion step uses `py_lift.utils.conllu.cas_to_str` per
sentence; offsets are stored in the MISC column as `t_start=<int>|t_end=<int>`
so detectors can recover sofa positions for any token they encounter.

## Phenomena

Each row below summarizes the detection rule and the CAS annotations
the writer creates.

| Phenomenon | Module | CAS annotations (UIMA types) |
|---|---|---|
| Sluicing | `detection/sluicing.py` | `GrammarAnomaly(description="Ellipsis", category="sluicing")` on the wh-word, plus `LexicalPhrase(text="QEmbedder")` on the embedding predicate (verb or noun). |
| Subject sharing | `detection/subject_sharing.py` | `GrammarAnomaly(description="Ellipsis", category="right_conj_subject")` on the subjectless right conjunct, plus one `LexicalPhrase(text="Shared_subject")` per shared subject of the left conjunct (spanning the whole subject phrase — the head token and its subtree). |
| Verbal ellipsis | `detection/verbal_ellipsis.py` | `GrammarAnomaly(description="Ellipsis", category="auxiliary")` on the AUX token. |
| Passive | `detection/passive.py` | Up to five `LexicalPhrase`s per finding: `Passive_verb`, `Passive_aux`, `Passive_subject`, `Passive_agent`, `Passive_agent_marker`. Aux+subject for canonical passives; agent+marker for short passives; verb is always emitted. |
| Nominal-head ellipsis | `detection/nominal_ellipsis.py` (+ `nominal_ellipsis_de.py` for German) | `GrammarAnomaly(description="Ellipsis", category="nominal_head_<subtype>")`. English subtypes: `quantifier`, `none`, `numeral`, `every_one`, `comparative`, `elder`, `adjective`. German subtypes: `quantifier`, `cardinal`, `ordinal`, `comparative`, `superlative`, `adjective`, `possessive_pronoun`, `demonstrative_pronoun`. |
| Clefts | `detection/clefts.py` | Three `LexicalPhrase`s per finding — `Cleft_focus` over the focused phrase, `Cleft_presupposition` over the relative clause, plus the cleft pronoun (`Cleft_it` for it-clefts, `Cleft_wh` for wh-clefts). |
| Bare wh-questions | `detection/bare_questions.py` | `GrammarAnomaly(description="Ellipsis", category="bare_wh")` on the wh-phrase span. No second annotation — bare wh-questions have no governor. |
| Gapped coordination | `detection/gapped_coordination.py` | `GrammarAnomaly(description="Ellipsis", category="gapped_coordination")` on the gapped clause span, plus `LexicalPhrase(text="GappedAntecedent")` on the antecedent verb whose predicate the gap borrows. |
| Right node raising | `detection/right_node_raising.py` | `GrammarAnomaly(description="Ellipsis", category="right_node_raising")` spanning the construction (non-final predicate through the shared constituent), plus three `LexicalPhrase`s: `RNR_left_predicate` (non-final predicate), `RNR_right_predicate` (final predicate), `RNR_shared_arg` (the shared right-edge constituent). |
| Suspended composition (DE) | `detection/suspended_composition.py` | `GrammarAnomaly(description="Suspended composition (shared head|shared modifier)", category="suspended_composition")` over the truncated conjunct, with the completion as a `SuggestedAction` (`certainty` 1.0 for a deduced split, 0.7 for a preferred one); `category="suspended_composition_unresolved"` when the split could not be settled or the resources were absent — recorded rather than dropped, since a missing annotation is indistinguishable from a clean sentence. Plus `LexicalPhrase(text="Suspension_donor")` over the conjunct the material comes from, which is often several tokens away (six, in real data). |
| Abbreviations (DE/EN) | `detection/abbreviations.py` | `GrammarAnomaly(description="Abbreviation", category="abbreviation")` over the short form, or `category="abbreviation_defined"` when the long form accompanies it (present, but must not be normalized). **Every** candidate expansion is written as a `SuggestedAction` in `suggestions`, with `certainty` carrying the corpus harvest's confidence — the first writer to use that FSArray for genuine ambiguity rather than a single rewrite, since a German short form routinely has many readings. |
| Contractions (EN/DE) | `detection/contractions.py` | `GrammarAnomaly(description="Contraction", category="contraction")` over the whole contraction ("wouldn't", "mir's", "vom", "nen"), carrying the expansion as a `SuggestedAction` in `suggestions` ("would not", "mir es", "von dem", "einen"). **Clitic** findings also add `LexicalPhrase(text="Contraction_host")` + `LexicalPhrase(text="Contraction_clitic")` on the two parts; **prep+article** and **clipped-article** findings are a single surface token, so they carry no host/clitic phrases. |

### Detection rules (concise)

- **Sluicing.** The remnant X is a wh-word — or a phrase headed by a
  non-wh word that has a wh-word child (`wie viele`: head `viele`,
  wh-child `wie`). X has no subject child and no verbal child (a sluice
  remnant is a bare wh-phrase, not a clause). X attaches to its
  governor G by one of three paths:
  - **strict, language-neutral** — `ccomp`, or `advmod` when X follows G
    linearly (so fronted "Why did you ask?" is not flagged);
  - **broadened verbal/nominal** — `obj`/`iobj`/`obl`/`conj`/`advmod`/
    `mark`/`appos`, accepted *only* when G is a known
    question-embedding predicate *or* a known embedding noun ("no idea
    why", "die Frage warum"). Elliptical sluices routinely parse-degrade
    off `ccomp`; the embedding-predicate and embedding-noun lexicons
    keep the broadened path precise.
  - **nominal** — `acl`/`nmod`, relations that only make sense for noun
    governors. Accepted *only* when the parent is in the embedding-noun
    lexicon. Kept disjoint from the broadened verbal set so adding a
    noun lexicon entry doesn't license verbal governors on these
    relations.

  The wh-word list, the question-embedding-predicate (verbal) list, and
  the embedding-noun list are all per-language lexicons
  (`lexicons/sluicing_wh.py`).
- **Subject sharing.** X is `conj` of Y; Y has at least one subject
  child (`nsubj`/`csubj`/`nsubj:pass`); X has none.
- **Verbal ellipsis.** Token has POS `AUX` (matched in either UPOS or
  XPOS — see note below) and is attached by a deprel that is not in
  `{aux, aux:pass, cop}`.
- **Passive — canonical.** Token X has `deprel == aux:pass`; X's parent
  V is the lexical verb; an optional `nsubj:pass`/`csubj:pass` child of
  V is the passive subject.
- **Passive — short.** V is a passive participle (per-language XPOS
  set, with a `VerbForm=Part` fallback), V has *no* aux child of any
  deprel (rules out canonical passives and active perfects), and V has
  an `obl`/`obl:agent` child whose `case` form is in the language's
  agent-preposition set.
- **Nominal-head ellipsis (English).** Subtypes are tried in this order,
  first match wins (so e.g. "the elder" is `elder`, not `comparative` or
  `adjective`):
  1. `none` — form in `none_forms`, no dependents.
  2. `every_one` / `elder` — fixed `(det_form, head_form)` patterns
     from the lexicon, deprel not in `excluded_deprels`.
  3. `quantifier` — UPOS=ADJ, form in `quantifier_forms`, only optional
     `det` child.
  4. `numeral` — UPOS=NUM, only optional `det`/`amod` children.
  5. `comparative` — XPOS in `comparative_xpos` (e.g. `JJR`/`JJS`),
     deprel ≠ `amod`, has a `det` child whose form is a definite
     article.
  6. `adjective` — UPOS=ADJ, XPOS in `adjective_xpos` (e.g. `JJ`),
     deprel ≠ `amod`, only optional `det` child, parent is a verb.
  All subtypes additionally require a core nominal relation
  (`nsubj`/`nsubj:pass`/`obj`/`iobj`/`nmod`).
- **Clefts (English).** Currently it-clefts and wh-clefts:
  - **it-cleft** — a nominal focus F (UPOS in `{NOUN, PRON, PROPN}`,
    deprel `root` or `ccomp`) with a copula `be` child (`deprel=cop`),
    an `acl:relcl` child (the presupposition clause), and a child whose
    deprel is `nsubj`/`expl` with lemma `it`.
  - **wh-cleft** — analogous with a free-relative subject headed by a
    wh-word ("What he wanted was rest.").
- **Bare wh-questions.** A sentence-level rule: the whole sentence is a
  wh-phrase with no main predicate. Requirements:
  - At least one descendant has form `?`.
  - No descendant is `VERB`/`AUX`; no descendant has an
    `nsubj`/`csubj`/`nsubj:pass` child — i.e. nothing is a clause.
  - Either the sentence root *is* a wh-word, **or** the root has a
    wh-word child with deprel in `{det, advmod, amod, case}` — the
    modifier slots from which a wh-word can head a wh-phrase ("What
    man?" — `det`; "How viable?" — `advmod`; "For what?" — `case` on
    the wh-root).
  - Echo questions like "Morris who?" are excluded: there `who`
    attaches by `appos`/`parataxis`, *not* a wh-phrase modifier slot.
  Reuses the wh-word lexicon from sluicing — there's no separate
  bare-wh lexicon.
- **Gapped coordination.** A coordinated clause whose main predicate
  (verb, copula, or adjective) is missing has to borrow it from the
  antecedent — *Paul wanted a milk shake and Mr Leonard a coffee*. The
  parser, with no head for the gapped clause, typically attaches the
  gap's arguments via `conj` to some token in the antecedent clause
  (most often the deepest plausible host — `obj`/`obl`/`xcomp`/cop
  predicate). Two signals fire on those structural artefacts (v1):
  - **Signal A** — a non-`VERB`/`AUX` token has *two or more* `conj`
    children that are themselves non-`VERB`/`AUX`. The parser
    collapsed two gapped arguments (subject + object, etc.) onto a
    single host because no verb could anchor them.
  - **Signal B** — a non-verbal `conj` token under a non-verbal parent
    has at least one child with deprel in `{nsubj, csubj, nsubj:pass,
    appos, nmod, flat}` — a second gapped argument hung off the conj.
  - **Signal C** — same shape as B but with a *verbal* parent. German
    Stanza tends to attach the gap anchor directly as a `conj` of the
    matrix verb rather than to one of its arguments, so Signal B's
    parent constraint is relaxed for this case. The conj token itself
    is still required to be non-verbal, which keeps well-formed
    verbal coordinations (where *both* conjuncts are verbs) out of
    the findings.

  The antecedent verb is the *outermost* (matrix) verbal ancestor on
  the path to the root; for copular antecedents it's the `cop` child
  of the non-verbal head. Lexicon-free, language-agnostic. Coverage
  (Signal A + B + C): ~half of the parser-recoverable English cases
  and roughly the same on German. See `tests/test_gapped_coordination.py`
  for the per-sentence EN/DE EXPECTED_HITS tables.
- **Right node raising (coordination subset).** A `conj` links two
  predicates (UPOS in `{VERB, AUX, ADJ}`): V1 (non-final, earlier) and V2
  (final). V2 has a core-argument child (`obj`/`iobj`/`obl`/`xcomp`/
  `ccomp`) whose subtree reaches the sentence's right edge — the candidate
  shared constituent — and V1 lacks a *filled* core argument of that class
  (the gap). To separate genuine RNR from ordinary VP-coordination that
  merely looks like a gap ("John went and bought a fridge"), one of two
  further conditions must hold:
  - **clausal** — both conjuncts carry their own overt subject (distinct
    subjects ⇒ each conjunct is a clause), reported as trigger
    `distinct_subjects`; or
  - **stranded preposition** — V1 carries a stranded preposition (an
    `obl`/`obj` child that is a bare `ADP` with no nominal of its own,
    "knew of __"), reported as trigger `stranded_prep`.

  Lexicon-free / language-agnostic. **Only the coordination subset** is
  detected structurally; comparative RNR ("more X than Y") and subordinate
  RNR ("those who voted against … outnumbered those who voted for …") have
  no `conj` and are handled by the LLM normalizer instead
  (`aslan_normalization/right_node_raising.py`).
- **Contractions / clitics (English + German).** Three mechanisms, because
  parsers represent the families differently:

  1. **Clitics** (English `n't`/`'s`/`'re`/`'m`/`'ve`/`'ll`/`'d`; German `'s`).
     A clitic token **adjacent** to the preceding token — the host's end offset
     equals the clitic's begin offset, i.e. the two were written as one word.
     Adjacency is the defining signal: it separates a real contraction from an
     already-expanded "would not" or a stray apostrophe token. The
     `(form, lemma)` pair must have an expansion in `lexicons/contractions.py`;
     the lemma disambiguates English `'s` (*be* → "is", *have* → "has", *we/us*
     → "us" in "let's") and `'d` (*would* → "would", *have* → "had"), and German
     `'s` (*es* → "es", as in "mir's" → "mir es", "geht's" → "geht es").
     **Possessive `'s` is absent from that table, so it never fires** — it is not
     a contraction of two words. The host may change: UD splits "can't" into
     "ca" + "n't", so the finding reports "cannot" rather than "*ca not*" —
     English writes *can*+*not* solid, via the lexicon's whole-contraction
     overrides; the other negated modals stay two words ("will not").
     Clitics are **not** UD multiword tokens — each part has real char offsets.

  2. **Preposition+article** (German `vom` = von+dem, `im` = in+dem, `zum`,
     `zur`, `beim`, `ins`, …). These *are* UD multiword tokens, so the detector
     reads them from the tree's multiword tokens, keeps the two-word `ADP+DET`
     shape, and takes the expansion the parser already produced (persisted via
     `preprocessing/mwt.py`) — no expansion lexicon. The sub-words share the
     multiword token's character span.

  3. **Clipped indefinite articles** (German `nen`/`nem`/`ner`/`ne`/`n`). The
     colloquial indefinite article written with the leading "ei" dropped
     ("nen Krampf" = "einen Krampf") — a **standalone token**, neither clitic
     nor MWT. Stanza does not lemmatise these to *ein* (it guesses junk lemmas),
     so the detector keys on the **surface form** and transfers the token's own
     casing to the expansion ("Nen" → "Einen"). Three sub-cases by how
     determined the full form is: `nen`/`nem`/`ner` are distinctive and fully
     determined, so they fire unconditionally; `ne` is ambiguous with the tag
     question "ne?" and fires only when a **noun phrase follows** (a NOUN/PROPN
     head reached across premodifiers — "ne alte Karre" → "eine alte Karre";
     "…, ne?" is left alone); bare `n` has no surface-determined form
     ("n bisschen" → *ein* vs "n Kleinwagen" → *einen*), so its form is
     **inferred from the head noun's Gender+Case** via the indefinite paradigm
     (`inflect_indefinite_article`), and the finding is marked `inferred=True`.

  All three are always detected/annotated. Whether prep+article contractions
  are *expanded* is a normalizer-side policy (opt-in, with a lexicalised
  exception list), because expanding them is not reliably meaning-preserving
  (weak article vs demonstrative: "im Haus" vs "in dem Haus"). Clipped-article
  expansion is likewise gated: the surface-determined forms
  (`nen`/`nem`/`ner`, and `ne` with a following NP) expand by default, but the
  morphology-inferred bare `n` is **opt-in** (`expand_clipped_n`), because
  German case is often mis-parsed (accusative/dative syncretism) and the
  inferred article can be wrong.
- **Abbreviations.** An all-caps letter run of 2–6 characters, then three
  vetoes. Two-letter forms are included on purpose: a general lexicon is hopeless
  for them (636 German all-caps forms carry ~17k senses) but corpus evidence
  resolves them well (`KG`, `IT`, `ID`, `IR` in the Hagen exam data).

  Each veto answers a false-positive class actually observed, not a hypothetical
  one:

  1. **Function word in caps** — "Test *VOR* dem Lernen", "*NACH* einem Test" are
     emphasis. The general lexicons do not even list *vor*/*nach*, so membership
     screens most of these already; the explicit list earns its keep on the ~139
     German all-caps abbreviations that *do* collide with a real word (`AN`, `AM`,
     `ALS`, `AB`, `ALL`).
  2. **Enumeration run** — three or more adjacent all-caps tokens separated by
     nothing but punctuation ("bsp.: ADE - BEC - CBA - D") is a list of learning
     materials, not of abbreviations. This one *must* be structural: asked whether
     `ADE` is an abbreviation, an LLM answered yes and invented
     *Aufmerksamkeitsdefizit-Hyperaktivitätsstörung* while its own stated reason
     described the surrounding context as a sequence of learning materials. What
     distinguishes an enumeration from a genuine series ("test (IT)/rehearsal
     (IR)/distraction (ID)") is that the latter's members have intervening words.
  3. **Mostly-capitalised text** — above 30 % all-caps word tokens,
     capitalisation carries no signal. This needs a **minimum-token guard**
     (10 tokens): student answers are short, so "Die VP kamen." is already 33 %
     all-caps and is obviously not shouted. Without the guard the gate suppressed
     legitimate detections.

  A fourth condition *flags rather than rejects*: an occurrence sitting beside its
  own long form ("Konditionierter Reiz (CS)", Schwartz & Hearst-style) is real —
  the phenomenon is present — but expanding it would be wrong, so it is written
  with `category="abbreviation_defined"`. The gloss pattern deliberately does not
  require the initials to match, since "Konditionierter Reiz" is *conditioned
  stimulus* and its German initials are K+R; missing a gloss is worse than missing
  an expansion, because it lets a wrong expansion through.

  Detection is separate from *expansion*. Which long form an abbreviation stands
  for is a corpus-level decision — a definition in one answer resolves a bare use
  in another — made by `resolution/abbreviations/harvest.py`. The detector
  therefore reports candidates whether or not an expansion is known, and takes an
  optional `expansions` map to attach ranked `SuggestedAction`s when one is. This
  is why standalone annotation is still useful: it records the candidates and the
  gate decisions with no corpus at all.
- **Nominal-head ellipsis (German).** German Stanza emits STTS XPOS
  (which carries no degree) and tags quantifiers `DET`/`PIS`, so the
  English XPOS-based rules do not transfer; German has its own rule
  module, `nominal_ellipsis_de.py`. It keys on the STTS *substituting*
  tags — `PIS` for `quantifier` ellipsis, `PPOSS` for
  `possessive_pronoun` ellipsis; on `ADJA` acting as a nominal head for
  `adjective` / `comparative` / `superlative` / `ordinal` ellipsis
  (subtype refined via the `feats` `Degree`/`NumType`); on `NUM`
  cardinals; and on a definite article (`ART`) immediately followed by a
  preposition for `demonstrative_pronoun` ellipsis. "Acts as a head" is
  a denylist of modifier relations, not the English core-relation
  whitelist — elliptical structures routinely get parser-degraded
  deprels (`appos`/`obl`/`conj`/`ccomp`).

> **POS-column note.** `view_to_conllu` builds its CoNLL-U via a local
> serializer (`_cas_to_conllu_block` in `detection/cas_conllu.py`) that
> reads the DKPro POS type's two fields: `coarseValue` populates the
> UPOS column and `PosValue` populates the XPOS column. `add_stanza_parses.py`
> writes both fields (UD `word.upos` → `coarseValue`, fine-grained
> `word.xpos` → `PosValue`), so converted CAS data matches the
> UD-convention `.conllu` fixtures the detectors are tested against.
> Where `coarseValue` is missing, the UPOS column falls back to `"FM"`.
> Detectors that need to recognize POS tags accept matches in either
> column, so fixtures using only UPOS still work.

## Language handling

All detectors share the same multilingual interface:

- A `# lang = <iso>` comment is embedded in each CoNLL-U sentence;
  detectors read it via `language.tree_lang(tree)`.
- Each `find_and_annotate_*` adapter and CLI accepts `--lang` and
  `--mixed`. The behavior matrix:

| `--lang` | `--mixed` | Behavior |
|---------|---------|---------|
| set     | off    | Trust the user; tag every sentence with the given language. Run a single doc-level detection for verification and warn on mismatch. |
| set     | on     | Detect language per sentence; the adapter then filters detection to sentences whose detected language equals `--lang`. |
| unset   | off    | Detect language once on the document. If confidence < 0.7 or the language is unsupported, skip detection with a warning. |
| unset   | on     | Detect each sentence; sentences with confident, supported languages run, others are skipped silently. |

Language detection uses [`lingua-language-detector`][lingua] with a
fixed confidence threshold of 0.7. Supported languages are
{`en`, `de`, `fr`, `es`}.

[lingua]: https://github.com/pemistahl/lingua-py

When a detector reaches a sentence whose language has no lexicon
entry, it raises `UnsupportedLanguage`, which the per-phenomenon
detector catches and converts into a per-sentence skip with a
warning. Pure-structural detectors (subject sharing, verbal ellipsis,
canonical passive) do not need a lexicon and run unchanged on any
language; the `lang` value is still threaded through so findings can
be filtered consistently.

## CLI conventions

All structural detectors run through the one generic CLI:

```bash
python annotate.py --phenomenon <name> INPUT \
    [--view _InitialView spelling_normalized ...] \
    [--output OUTPUT_DIR] \
    [--lang {en,de,fr,es}] \
    [--mixed] \
    [--replace]
```

- `--phenomenon` is required; choices are the keys of
  `cas_adapter.DETECTOR_REGISTRY`.
- `INPUT` is a single XMI file or a directory of XMI files.
- `--output` writes annotated copies; omitted, the input files are
  overwritten in place.
- It is **idempotent by default**: views that already contain the chosen
  phenomenon's annotations are skipped. The skip signature comes from the
  registry entry (its `ga_categories` / `ga_category_prefixes` / `lp_texts`) —
  e.g. `category == "sluicing"` for sluicing,
  `text in {Passive_verb, Passive_aux, …}` for passive,
  `category == "bare_wh"` for bare wh-questions — via
  `cas_adapter.existing_annotations(view, phenomenon)`.
- `--replace` overrides the skip: it removes all annotations this detector
  would have created (signature-matched) and re-runs. Useful when iterating on a
  detector or lexicon and re-annotating an existing corpus in place. The
  external-tool annotators (`add_spelling_errors.py`, `add_coreference.py`,
  `add_edus.py`, `add_rwse.py`) keep their own scripts and share the same
  `--replace` flag.

A typical run order on a corpus is documented in
[`CALL_SEQUENCING.md`](./CALL_SEQUENCING.md).

## Testing

The repository ships hand-written `.conllu` fixtures under
`tests/fixtures/<phenomenon>/`. Tests load them via
`udapi.core.document.Document.from_conllu_string` and call the pure
detector directly:

```python
from udapi.core.document import Document
from preprocessing.detection.sluicing import detect_sluicing

doc = Document()
doc.from_conllu_string(Path("tests/fixtures/sluicing/positive_basic.conllu").read_text())
findings = detect_sluicing(doc)
```

Fixtures must declare `# lang = <iso>` per sentence; otherwise the
detector treats the sentence as having no language and skips it.

Run the detector test suite with:

```bash
poetry run pytest tests/test_sluicing.py tests/test_subject_sharing.py \
                  tests/test_verbal_ellipsis.py tests/test_passive.py \
                  tests/test_nominal_ellipsis.py tests/test_clefts.py \
                  tests/test_bare_questions.py tests/test_gapped_coordination.py \
                  tests/test_right_node_raising.py tests/test_abbreviations.py \
                  tests/test_suspended_composition.py
```

The py_lift annotator wrappers (`detection/lift_annotators.py`, see
[`PYLIFT_INTEGRATION.md`](../PYLIFT_INTEGRATION.md) §4) have their own suite,
`tests/test_lift_annotators.py`, which also asserts that the `ANNOTATORS` and
`DETECTOR_REGISTRY` tables do not drift apart.

## Adding a new phenomenon

1. Create `preprocessing/detection/<phenomenon>.py` with:
   - one or more `<…>Finding` dataclasses,
   - a pure `detect_<phenomenon>(doc, *, restrict_to_lang=None)`
     function that reads each tree's language via `tree_lang` and
     consults a lexicon if needed.
2. If the rule is lexically sensitive, add
   `preprocessing/detection/lexicons/<phenomenon>.py` with a
   `<…>_BY_LANG` table and a lookup function that raises
   `UnsupportedLanguage` on misses.
3. Add a `_write_<phenomenon>(view, ts, findings)` writer to
   `preprocessing/detection/cas_adapter.py` (it emits the appropriate UIMA
   annotations), then register the phenomenon in `DETECTOR_REGISTRY` with a
   `DetectorSpec` (its `detect_<phenomenon>`, the writer, and the annotation
   signature it produces: `ga_categories` / `ga_category_prefixes` / `lp_texts`).
   That's it — the generic `find_and_annotate`, `existing_annotations`, and the
   `annotate.py` CLI all pick it up automatically. (A thin
   `find_and_annotate_<phenomenon>` delegation is optional, only if some caller
   wants the named function.)
4. **No new CLI script needed** — `annotate.py --phenomenon <name>` works as soon
   as the registry entry exists. (Optionally add a row to
   `gen_annotation_script.py`'s `ANNOTATORS` table so it's included in generated
   annotation scripts.)
5. Add a wrapper to `preprocessing/detection/lift_annotators.py`: a
   `SE_<Phenom>Annotator(_DetectorAnnotator)` with `_adapter` set to the
   adapter and `@supported_languages(...)` declared, plus its entry in
   `ANNOTATORS`. `requires_types` is inherited, so this really is two lines —
   and `tests/test_lift_annotators.py` fails if the entry is missing, so the
   two registries cannot drift.
6. Add `tests/test_<phenomenon>.py` and `tests/fixtures/<phenomenon>/`
   with at least one positive and one negative `.conllu` fixture.

## Adding a new language

Most detectors are lexically driven; adding a new language usually
means adding a row to a per-phenomenon lexicon module:

| Lexicon | Add an entry to |
|---|---|
| `lexicons/sluicing_wh.py` | `WH_WORDS_BY_LANG` (closed wh-word list — shared by sluicing **and** bare-wh-question detection), `EMBEDDING_PREDICATES_BY_LANG` (question-embedding *predicate* lemmas — license the verbal broadened relation gate), and `EMBEDDING_NOUNS_BY_LANG` (question-embedding *noun* lemmas — license the nominal relation gate and noun governors on the broadened path). Absent ⇒ that path simply never fires for the language. |
| `lexicons/passive.py` | `PASSIVE_AGENT_PREPS_BY_LANG` and `PARTICIPLE_XPOS_BY_LANG` (or rely on the `VerbForm=Part` fallback). |
| `lexicons/nominal_ellipsis.py` | `LEXICONS_BY_LANG` with a complete `NominalEllipsisLexicon` — for languages whose Stanza output uses Penn-Treebank-like XPOS. German does **not** use this lexicon: its rules live in `detection/nominal_ellipsis_de.py` (see the German rule above). A language whose tagging resembles German's (STTS-style XPOS, no degree in the tag) likely needs a similar dedicated rule module rather than a lexicon row. |
| `lexicons/clefts.py` | Cleft-pronoun forms and copula lemmas per language. |
| `lexicons/suspended_composition.py` | `COORDINATORS_BY_LANG` (what may join the conjuncts — `oder` and bare commas matter as much as `und`) and `VERB_PARTICLES_BY_LANG`. The stub pattern itself is orthographic, not language-specific. English needs only the coordinator row to be *detected*; resolving it would additionally need a morphology and an attestation lexicon for that language. |
| `lexicons/abbreviations.py` | `FUNCTION_WORDS_BY_LANG` (the closed-class words whose all-caps spelling is emphasis rather than an abbreviation) and, if the language needs different ones, the gloss/enumeration patterns. The candidate shape itself (`CANDIDATE_RE`) is script-based, not language-based, so a language using the Latin alphabet needs only the function-word row. `GLOSS_RE` and `MAX_EXPANSION_WORDS` filter *scraped lexicon* long forms rather than text, so they belong to whichever lexicon is being harvested. |
| `lexicons/contractions.py` | Per-language clitic forms and their `(form, lemma) -> expansion` tables (`CLITICS_BY_LANG`, `CLITIC_EXPANSIONS_BY_LANG`), irregular host forms, the German prep+article `PREP_ARTICLE_EXCEPTIONS_BY_LANG` list, and the German clipped-article table (`CLIPPED_ARTICLES_BY_LANG` = nen/nem/ner/ne) plus the indefinite paradigm (`EIN_PARADIGM` + `inflect_indefinite_article`) used to infer bare `n`. Prep+article contractions themselves are multiword tokens (parser-supplied expansion), so they need no expansion entries — only the exception list. |

Also add the new ISO code to `SUPPORTED_LANGS` in
`preprocessing/detection/language.py` so the CLI's `--lang` choices
include it.

For detectors that aren't lexically driven (subject sharing, verbal
ellipsis, canonical passive), no language work is needed — they are
already structural and language-agnostic.
