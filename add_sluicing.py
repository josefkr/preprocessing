#!/usr/bin/env python3
"""Add sluicing annotations to existing CAS XMI files.

Detects sluicing: cases where an embedded question consists of nothing
but the question word (wh-word).

Detection rule (in Universal Dependencies terms):
  - X is a wh-word (who, what, why, when, how, whose)
  - X is the Dependent of G via the 'ccomp' relation
  - X has NO child with a subject relation (nsubj, csubj, nsubj:pass)

Annotations added:
  - GrammarAnomaly on X: description="Ellipsis", category="sluicing"
  - LexicalPhrase on G: text="QEmbedder"

Usage:
    # Single file, default view (_InitialView):
    python add_sluicing.py input.xmi

    # Directory of XMI files, specific views:
    python add_sluicing.py ./xmi_dir/ --view _InitialView spelling_corrected

    # Custom output directory (default: overwrite in place):
    python add_sluicing.py ./xmi_dir/ --output ./annotated/
"""

import argparse
import logging
import sys
from pathlib import Path

import cassis

from py_lift.util import get_lift_typesystem

logger = logging.getLogger(__name__)

T_DEP = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
T_GRAMMAR_ANOMALY = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly"
T_LEXICAL_PHRASE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.LexicalPhrase"

DEFAULT_VIEW = "_InitialView"
WH_WORDS = {"who", "what", "why", "when", "how", "whose"}
SUBJECT_RELS = {"nsubj", "csubj", "nsubj:pass"}


def find_and_annotate_sluicing(view, ts) -> int:
    """Find sluicing cases and add annotations.

    Returns the number of sluicing annotations added.
    """
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    sofa = view.sofa_string or ""

    # Build a set of governor spans that have subject children
    # (begin, end) of nodes that govern a subject relation
    gov_has_subject: set[tuple[int, int]] = set()
    for dep in view.select(T_DEP):
        if dep.DependencyType in SUBJECT_RELS:
            gov = dep.Governor
            gov_has_subject.add((gov.begin, gov.end))

    count = 0
    for dep in view.select(T_DEP):
        # Accept ccomp, or advmod only when the wh-word follows its governor
        if dep.DependencyType == "ccomp":
            pass
        elif dep.DependencyType == "advmod":
            if dep.Dependent.begin <= dep.Governor.begin:
                # wh-word precedes governor ("Why did you ask?") — not sluicing
                continue
        else:
            continue

        X = dep.Dependent  # the wh-word (sluiced question)
        G = dep.Governor   # the embedding predicate

        # Check if X is a wh-word
        x_text = sofa[X.begin:X.end].lower()
        if x_text not in WH_WORDS:
            continue

        # Check that X has no subject child
        x_key = (X.begin, X.end)
        if x_key in gov_has_subject:
            continue

        # This is a sluicing case
        # Annotate X with GrammarAnomaly
        view.add(GA(
            begin=X.begin,
            end=X.end,
            description="Ellipsis",
            category="sluicing",
        ))

        # Annotate G with LexicalPhrase
        view.add(LP(
            begin=G.begin,
            end=G.end,
            text="QEmbedder",
        ))

        logger.debug(
            f"  Sluicing: G='{sofa[G.begin:G.end]}' -ccomp-> X='{x_text}'"
        )
        count += 1

    return count


def process_file(
    xmi_path: Path,
    ts: cassis.TypeSystem,
    views: list[str],
    output_path: Path,
) -> None:
    """Load an XMI file, detect sluicing on specified views, and save."""
    with open(xmi_path, "rb") as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=ts)

    for view_name in views:
        try:
            view = cas.get_view(view_name)
        except Exception:
            logger.warning(f"{xmi_path.name}: view '{view_name}' not found, skipping.")
            continue

        # Check for existing annotations to avoid duplicates
        existing = [
            a for a in view.select(T_GRAMMAR_ANOMALY)
            if getattr(a, "category", None) == "sluicing"
        ]
        if existing:
            logger.info(
                f"{xmi_path.name}: view '{view_name}' already has "
                f"{len(existing)} sluicing annotations, skipping."
            )
            continue

        count = find_and_annotate_sluicing(view, ts)
        logger.info(
            f"{xmi_path.name}: view '{view_name}' — {count} sluicing cases found"
        )

    cas.to_xmi(str(output_path), pretty_print=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add sluicing annotations to existing CAS XMI files."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a single XMI file or a directory of XMI files.",
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
        help="Output directory for annotated XMI files. "
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

    ts = get_lift_typesystem()

    success = 0
    errors = 0
    for i, xmi_file in enumerate(xmi_files, 1):
        out_path = (output_dir / xmi_file.name) if output_dir else xmi_file
        try:
            process_file(xmi_file, ts, args.view, out_path)
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
