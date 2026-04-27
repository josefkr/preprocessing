#!/usr/bin/env python3
"""Add subject-sharing annotations to coordinated clauses in CAS XMI files.

Detects coordinated clauses where the right conjunct (X) lacks its own
subject, sharing the subject (S) of the left conjunct (Y) instead.

Detection logic lives in :mod:`preprocessing.detection.subject_sharing`
and operates on udapi trees, so it can be tested independently with
``.conllu`` fixtures. This script is a thin CAS/CLI wrapper.

Annotations added:
  - GrammarAnomaly on X: description="Ellipsis", category="right_conj_subject"
  - LexicalPhrase on S: text="Shared_subject"

Usage:
    # Single file, default view (_InitialView):
    python add_subject_sharing.py input.xmi

    # Directory of XMI files, specific views:
    python add_subject_sharing.py ./xmi_dir/ --view _InitialView spelling_corrected

    # Custom output directory (default: overwrite in place):
    python add_subject_sharing.py ./xmi_dir/ --output ./annotated/
"""

import argparse
import logging
import sys
from pathlib import Path

import cassis

from py_lift.util import get_lift_typesystem

from preprocessing.detection.cas_adapter import (
    T_GRAMMAR_ANOMALY,
    find_and_annotate_subject_sharing,
)
from preprocessing.detection.cli import add_language_args

logger = logging.getLogger(__name__)

DEFAULT_VIEW = "_InitialView"


def process_file(
    xmi_path: Path,
    ts: cassis.TypeSystem,
    views: list[str],
    output_path: Path,
    *,
    lang: str | None,
    mixed: bool,
) -> None:
    """Load an XMI file, detect subject sharing on specified views, and save."""
    with open(xmi_path, "rb") as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=ts)

    for view_name in views:
        try:
            view = cas.get_view(view_name)
        except Exception:
            logger.warning(f"{xmi_path.name}: view '{view_name}' not found, skipping.")
            continue

        existing = [
            a for a in view.select(T_GRAMMAR_ANOMALY)
            if getattr(a, "category", None) == "right_conj_subject"
        ]
        if existing:
            logger.info(
                f"{xmi_path.name}: view '{view_name}' already has "
                f"{len(existing)} subject-sharing annotations, skipping."
            )
            continue

        count = find_and_annotate_subject_sharing(
            view, ts, lang=lang, mixed=mixed
        )
        logger.info(
            f"{xmi_path.name}: view '{view_name}' — {count} subject-sharing cases found"
        )

    cas.to_xmi(str(output_path), pretty_print=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add subject-sharing annotations to coordinated clauses in CAS XMI files."
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
    add_language_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

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

    output_dir: Path | None = args.output
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    ts = get_lift_typesystem()

    success = 0
    errors = 0
    for i, xmi_file in enumerate(xmi_files, 1):
        out_path = (output_dir / xmi_file.name) if output_dir else xmi_file
        try:
            process_file(
                xmi_file, ts, args.view, out_path,
                lang=args.lang, mixed=args.mixed,
            )
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
