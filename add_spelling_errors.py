#!/usr/bin/env python3
"""Add spelling error annotations to existing CAS XMI files.

Uses the SE_SpellErrorAnnotator from py_lift to detect spelling errors
and add SpellingAnomaly annotations to specified views.

Requires that the XMI files already contain token annotations.

Usage:
    # Single file, default view (_InitialView), German:
    python add_spelling_errors.py input.xmi --language de

    # Directory of XMI files, English, specific view:
    python add_spelling_errors.py ./xmi_dir/ --language en --view spelling_corrected

    # Multiple views processed in sequence:
    python add_spelling_errors.py ./xmi_dir/ --language de --view view1 view2

    # Custom output directory (default: overwrite in place):
    python add_spelling_errors.py ./xmi_dir/ --language de --output ./annotated/
"""

import argparse
import logging
import sys
from pathlib import Path

import cassis

from py_lift.annotators.misc import SE_SpellErrorAnnotator
from py_lift.dkpro import T_ANOMALY
from py_lift.util import get_lift_typesystem

logger = logging.getLogger(__name__)

DEFAULT_VIEW = "_InitialView"


def process_file(
    xmi_path: Path,
    annotator: SE_SpellErrorAnnotator,
    ts: cassis.TypeSystem,
    views: list[str],
    output_path: Path,
    *,
    replace: bool = False,
) -> None:
    """Load an XMI file, run spell checking on specified views, and save."""
    with open(xmi_path, "rb") as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=ts, lenient=True)

    for view_name in views:
        print(f"processing {view_name} in {xmi_path}")
        try:
            view = cas.get_view(view_name)
        except Exception:
            logger.warning(f"{xmi_path.name}: view '{view_name}' not found, skipping.")
            continue

        existing = list(view.select(T_ANOMALY))
        if existing:
            if not replace:
                logger.info(
                    f"{xmi_path.name}: view '{view_name}' already has "
                    f"{len(existing)} spelling anomalies, skipping "
                    "(use --replace to overwrite)."
                )
                continue
            for a in existing:
                view.remove(a)
            logger.info(
                f"{xmi_path.name}: view '{view_name}' — removed "
                f"{len(existing)} existing spelling anomalies."
            )

        logger.info(f"{xmi_path.name}: spell-checking view '{view_name}'")
        annotator.process(view)

    cas.to_xmi(str(output_path), pretty_print=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add spelling error annotations to existing CAS XMI files."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a single XMI file or a directory of XMI files.",
    )
    parser.add_argument(
        "--language",
        "-l",
        required=True,
        help="Language code for spell checking (e.g. en, de).",
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
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Remove existing spelling anomalies on each processed "
        "view before re-running the spell checker. Default: skip "
        "views that already have spelling anomalies.",
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
    annotator = SE_SpellErrorAnnotator(language=args.language)

    success = 0
    errors = 0
    for i, xmi_file in enumerate(xmi_files, 1):
        out_path = (output_dir / xmi_file.name) if output_dir else xmi_file
        try:
            process_file(
                xmi_file, annotator, ts, args.view, out_path,
                replace=args.replace,
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
