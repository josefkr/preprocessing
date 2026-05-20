# Detection Modules

This document describes the suite of annotators that detect linguistic
phenomena (sluicing, subject sharing, verbal ellipsis, passive,
nominal-head ellipsis, clefts, bare wh-questions) in CAS XMI files, how
they are organized, and how to extend them with new phenomena or new
languages.

## Overview

Each phenomenon is implemented as three layers:

1. A **pure detector** that operates on a [udapi][udapi] document and
   returns plain-Python *findings*. It has no CAS dependency and is
   directly testable with `.conllu` fixtures.
2. A **CAS adapter** entry point that converts a CAS view into a
   CoNLL-U document, runs the detector, and writes the resulting
   findings back into the view as UIMA annotations.
3. A **CLI script** (`add_<phenomenon>.py`) that loads XMI files,
   calls the adapter on each requested view, and saves the result.

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
  lexicons/
    sluicing_wh.py        Wh-words, question-embedding predicates
                          (verbal), and question-embedding nouns per
                          language.
    passive.py            Agent prepositions + participle XPOS sets.
    nominal_ellipsis.py   English quantifier forms, idiomatic patterns,
                          definite articles, comparative/JJ XPOS.
    clefts.py             Cleft-pronoun forms etc. per language.

add_sluicing.py            Thin CLIs (one per phenomenon).
add_subject_sharing.py
add_verbal_ellipsis.py
add_passive.py
add_nominal_ellipsis.py
add_clefts.py
add_bare_questions.py
```

`add_spelling_errors.py` and `add_coreference.py` follow the same CLI
template but wrap external annotators (py_lift's spell checker and the
Maverick coreference service respectively) rather than detectors in
`preprocessing/detection/`. They share the `--replace` skip/overwrite
semantics described in [CLI conventions](#cli-conventions).

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
| Subject sharing | `detection/subject_sharing.py` | `GrammarAnomaly(description="Ellipsis", category="right_conj_subject")` on the subjectless right conjunct, plus one `LexicalPhrase(text="Shared_subject")` per shared subject of the left conjunct. |
| Verbal ellipsis | `detection/verbal_ellipsis.py` | `GrammarAnomaly(description="Ellipsis", category="auxiliary")` on the AUX token. |
| Passive | `detection/passive.py` | Up to five `LexicalPhrase`s per finding: `Passive_verb`, `Passive_aux`, `Passive_subject`, `Passive_agent`, `Passive_agent_marker`. Aux+subject for canonical passives; agent+marker for short passives; verb is always emitted. |
| Nominal-head ellipsis | `detection/nominal_ellipsis.py` (+ `nominal_ellipsis_de.py` for German) | `GrammarAnomaly(description="Ellipsis", category="nominal_head_<subtype>")`. English subtypes: `quantifier`, `none`, `numeral`, `every_one`, `comparative`, `elder`, `adjective`. German subtypes: `quantifier`, `cardinal`, `ordinal`, `comparative`, `superlative`, `adjective`, `possessive_pronoun`, `demonstrative_pronoun`. |
| Clefts | `detection/clefts.py` | Three `LexicalPhrase`s per finding — `Cleft_focus` over the focused phrase, `Cleft_presupposition` over the relative clause, plus the cleft pronoun (`Cleft_it` for it-clefts, `Cleft_wh` for wh-clefts). |
| Bare wh-questions | `detection/bare_questions.py` | `GrammarAnomaly(description="Ellipsis", category="bare_wh")` on the wh-phrase span. No second annotation — bare wh-questions have no governor. |

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

All `add_<phenomenon>.py` scripts share the same surface:

```bash
python add_<phenomenon>.py INPUT \
    [--view _InitialView spelling_normalized ...] \
    [--output OUTPUT_DIR] \
    [--lang {en,de,fr,es}] \
    [--mixed] \
    [--replace]
```

- `INPUT` is a single XMI file or a directory of XMI files.
- `--output` writes annotated copies; omitted, the input files are
  overwritten in place.
- Each script is **idempotent by default**: views that already contain
  the target annotation type are skipped (the skip-key is per script —
  e.g. `category == "sluicing"` for sluicing,
  `text in {Passive_verb, Passive_aux, …}` for passive,
  `category == "bare_wh"` for bare wh-questions).
- `--replace` overrides the skip: the script removes all annotations
  this detector would have created (signature-matched) and re-runs.
  Useful when iterating on a detector or lexicon and re-annotating an
  existing corpus in place. The same flag exists on
  `add_spelling_errors.py` and `add_coreference.py`.

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
                  tests/test_bare_questions.py
```

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
3. Add `find_and_annotate_<phenomenon>(view, ts, *, lang, mixed)` and
   `_write_<phenomenon>(view, ts, findings)` to
   `preprocessing/detection/cas_adapter.py`. The writer emits the
   appropriate UIMA annotations. Reuse `_build_doc` for the
   converter+udapi step.
4. Add `add_<phenomenon>.py` at the project root following the
   existing template (it should be a thin wrapper around the adapter
   plus `add_language_args`).
5. Add `tests/test_<phenomenon>.py` and `tests/fixtures/<phenomenon>/`
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

Also add the new ISO code to `SUPPORTED_LANGS` in
`preprocessing/detection/language.py` so the CLI's `--lang` choices
include it.

For detectors that aren't lexically driven (subject sharing, verbal
ellipsis, canonical passive), no language work is needed — they are
already structural and language-agnostic.
