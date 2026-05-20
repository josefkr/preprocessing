# Named-entity annotations from Stanza

The Stanza preprocessor (`preprocessing/stanza.py`) and the
`add_stanza_parses.py` script run Stanza's `ner` processor alongside
tokenization, POS, lemma, morphology and dependency parsing, and record
the named entities as CAS annotations.

## Annotation type

Each entity is written as one annotation of the generic DKPro type

```
de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity
```

(constant `T_NER` in `preprocessing/api.py`). The annotation spans the
entity's character offsets; the **raw Stanza label string** is stored
verbatim in the `value` feature — e.g. `value="PERSON"`, `value="GPE"`.

The DKPro typesystem also defines ~30 `NamedEntity` *subtypes*
(`Person`, `Organization`, `Gpe`, …), but the annotators deliberately
use only the base `NamedEntity` type with `value` set. This keeps the
code language-agnostic. Downstream code should treat `value` as a free
string, not assume a fixed inventory (see below).

## Labelsets are language-specific

Stanza's NER models use different label inventories per language, so the
set of `value` strings depends on the document language:

| Language | Stanza model trained on | `value` inventory |
|---|---|---|
| English (`en`) | OntoNotes | 18 classes: `PERSON`, `NORP`, `FAC`, `ORG`, `GPE`, `LOC`, `PRODUCT`, `EVENT`, `WORK_OF_ART`, `LAW`, `LANGUAGE`, `DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`, `ORDINAL`, `CARDINAL` |
| German (`de`) | CoNLL-2003 (German) | 4 classes: `PER`, `LOC`, `ORG`, `MISC` |

The two sets are **not** the same and not in a subset relation — German is
coarser and has a catch-all `MISC` that has no English equivalent.

## Back-filling onto already-parsed views

`add_stanza_parses.py` skips the parse layers (sentences / tokens / POS /
dependencies) for a view that already has token annotations, to avoid
duplicates. Named entities are checked **independently**: if a view has
tokens but no `NamedEntity` annotations, NER is still added. So running
`add_stanza_parses.py` over an already-parsed corpus adds in NER annos
without re-doing (or duplicating) the parse.

## Where it lives in code

- `preprocessing/stanza.py` — the `ner` processor in the pipeline and the
  pass that turns `doc.ents` into `NamedEntity` annotations.
- `add_stanza_parses.py` — the CLI that adds the annotations to XMI files
  (with the independent NER back-fill check).
- `preprocessing/api.py` — the `T_NER` type-name constant.
