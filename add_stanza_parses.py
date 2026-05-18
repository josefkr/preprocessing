#!/usr/bin/env python3
"""Add Stanza parse annotations to existing CAS XMI files.

This code loads XMI files, extracts the sofaString from a specified view,
runs the Stanza preprocessor on it, and writes the resulting annotations
(sentences, tokens, POS, lemma, morphology, dependencies) back to that view.

Usage:
    # Single file, default view (_InitialView), English:
    python add_stanza_parses.py input.xmi

    # Directory of XMI files, German, specific view:
    python add_stanza_parses.py ./xmi_dir/ --language de --view spelling_corrected

    # Multiple views processed in sequence:
    python add_stanza_parses.py ./xmi_dir/ --view view1 view2

    # Custom output directory (default: overwrite in place):
    python add_stanza_parses.py ./xmi_dir/ --output ./parsed/
"""

import argparse
import logging
import sys
from pathlib import Path

import cassis

from preprocessing.api import T_DEP, T_LEMMA, T_MORPH, T_NER, T_POS, T_SENT, T_TOKEN
from preprocessing.stanza import Stanza_Preprocessor
from preprocessing.util import get_aslan_typesystem

logger = logging.getLogger(__name__)

DEFAULT_VIEW = "_InitialView"


def add_stanza_annotations(preprocessor: Stanza_Preprocessor, view) -> None:
    """Run Stanza on a CAS view's text and add annotations to that view.

    This reuses the Stanza pipeline from the preprocessor but adds
    annotations directly to the given view instead of creating a new CAS.

    The parse part (sentences, tokens, POS, lemma, morphology,
    dependencies) is skipped if the view already contains token
    annotations, to avoid duplicates (e.g. if a normalizer already
    parsed the view). Named entities are checked independently, so NER
    is back-filled even into views that were already parsed before NER
    support existed.

    Args:
        preprocessor: An initialized Stanza_Preprocessor instance.
        view: A cassis CAS view whose sofaString will be parsed.
    """
    text = view.sofa_string
    if not text or not text.strip():
        logger.warning("View has empty sofaString, skipping.")
        return

    has_tokens = bool(list(view.select(T_TOKEN)))
    has_ner = bool(list(view.select(T_NER)))
    if has_tokens and has_ner:
        logger.info("View already has token and NER annotations, skipping.")
        return

    ts = preprocessor.ts
    pipeline = preprocessor._load_pipeline()
    doc = pipeline(text)

    # Type handles
    T = ts.get_type(T_TOKEN)
    S = ts.get_type(T_SENT)
    P = ts.get_type(T_POS)
    D = ts.get_type(T_DEP)
    L = ts.get_type(T_LEMMA)
    M = ts.get_type(T_MORPH)
    N = ts.get_type(T_NER)

    # Named entities are independent of the parse layers, so they can be
    # back-filled even when tokens already exist. The raw Stanza label is
    # stored verbatim in the 'value' feature.
    if not has_ner:
        for ent in doc.ents:
            view.add(N(begin=ent.start_char, end=ent.end_char, value=ent.type))

    if has_tokens:
        logger.info("View already has token annotations, skipping parse.")
        return

    # First pass: sentences
    for sentence in doc.sentences:
        first_word = sentence.tokens[0].words[0]
        last_word = sentence.tokens[-1].words[-1]
        view.add(S(begin=first_word.start_char, end=last_word.end_char))

    # Second pass: tokens and their annotations
    token_map = {}
    global_token_id = 0

    for sent_idx, sentence in enumerate(doc.sentences):
        for token_idx, token in enumerate(sentence.tokens):
            for word_idx, word in enumerate(token.words):
                begin = word.start_char
                end = word.end_char

                # DKPro convention: PosValue=xpos (fine), coarseValue=upos (UD).
                cas_pos = P(
                    begin=begin,
                    end=end,
                    PosValue=word.xpos or "",
                    coarseValue=word.upos or "",
                )
                view.add(cas_pos)

                cas_lemma = L(begin=begin, end=end, value=word.lemma or "")
                view.add(cas_lemma)

                if word.feats:
                    view.add(M(begin=begin, end=end, morphTag=word.feats))

                cas_token = T(
                    begin=begin,
                    end=end,
                    id=global_token_id,
                    pos=cas_pos,
                    lemma=cas_lemma,
                )
                view.add(cas_token)

                token_map[(sent_idx, token_idx, word_idx)] = (cas_token, word)
                global_token_id += 1

    # Third pass: dependency relations
    for sent_idx, sentence in enumerate(doc.sentences):
        for token_idx, token in enumerate(sentence.tokens):
            for word_idx, word in enumerate(token.words):
                dependent_anno, _ = token_map[(sent_idx, token_idx, word_idx)]
                head_idx = word.head

                if head_idx == 0:
                    governor_anno = dependent_anno
                else:
                    governor_anno = None
                    word_count = 0
                    for t_idx, t in enumerate(sentence.tokens):
                        for w_idx, w in enumerate(t.words):
                            word_count += 1
                            if word_count == head_idx:
                                if (sent_idx, t_idx, w_idx) in token_map:
                                    governor_anno, _ = token_map[
                                        (sent_idx, t_idx, w_idx)
                                    ]
                                break
                        if governor_anno is not None:
                            break

                    if governor_anno is None:
                        logger.warning(
                            f"Could not find head {head_idx} for '{word.text}' "
                            f"in sentence {sent_idx}"
                        )
                        continue

                view.add(
                    D(
                        begin=dependent_anno.begin,
                        end=dependent_anno.end,
                        Governor=governor_anno,
                        Dependent=dependent_anno,
                        DependencyType=word.deprel or "dep",
                        flavor="basic",
                    )
                )


def process_file(
    xmi_path: Path,
    preprocessor: Stanza_Preprocessor,
    ts: cassis.TypeSystem,
    views: list[str],
    output_path: Path,
) -> None:
    """Load an XMI file, add Stanza annotations to the specified views, and save."""
    with open(xmi_path, "rb") as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=ts)

    for view_name in views:
        try:
            view = cas.get_view(view_name)
        except Exception:
            logger.warning(f"{xmi_path.name}: view '{view_name}' not found, skipping.")
            continue

        logger.info(f"{xmi_path.name}: processing view '{view_name}'")
        add_stanza_annotations(preprocessor, view)

    cas.to_xmi(str(output_path), pretty_print=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add Stanza parse annotations to existing CAS XMI files."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a single XMI file or a directory of XMI files.",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="en",
        help="Language code for Stanza (default: en).",
    )
    parser.add_argument(
        "--view",
        "-v",
        nargs="+",
        default=[DEFAULT_VIEW],
        help=f"View name(s) to process (default: {DEFAULT_VIEW}). "
        "Multiple views are processed in sequence.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for parsed XMI files. "
        "If omitted, files are overwritten in place.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Collect input files
    input_path: Path = args.input
    if input_path.is_file():
        xmi_files = [input_path]
    elif input_path.is_dir():
        xmi_files = sorted(input_path.glob("*.xmi"))
        if not xmi_files:
            print(f"No .xmi files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Set up output directory
    output_dir: Path | None = args.output
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    ts = get_aslan_typesystem()
    preprocessor = Stanza_Preprocessor(language=args.language)

    success = 0
    errors = 0
    for i, xmi_file in enumerate(xmi_files, 1):
        out_path = (output_dir / xmi_file.name) if output_dir else xmi_file
        if i % 40 == 0:
            print(f"on {i}-th file ")
        try:
            process_file(xmi_file, preprocessor, ts, args.view, out_path)
            success += 1
        except Exception as e:
            errors += 1
            logger.error(f"{xmi_file.name}: {e}")

        if i % 100 == 0 or i == len(xmi_files):
            print(f"  [{i}/{len(xmi_files)}] processed")

    print(
        f"\nDone. {success} succeeded, {errors} failed out of {len(xmi_files)} files."
    )


if __name__ == "__main__":
    main()
