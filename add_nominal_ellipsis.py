#!/usr/bin/env python3
"""Add nominal-head ellipsis annotations to existing CAS XMI files.

Detects cases where the nominal head of an NP is missing and the
remaining surface element (a quantifier, numeral, comparative,
adjective, or specific idiom like "every one" / "the elder") stands
in for the elided noun.

Detection logic lives in :mod:`preprocessing.detection.nominal_ellipsis`
and operates on udapi trees, so it can be tested independently with
``.conllu`` fixtures. This script is a thin CAS/CLI wrapper.

Annotation added per finding:
  - GrammarAnomaly: description="Ellipsis",
    category=f"nominal_head_{subtype}" where subtype is one of
    quantifier, none, numeral, every_one, comparative, elder, adjective.

Usage:
    # Single file, default view (_InitialView):
    python add_nominal_ellipsis.py input.xmi

    # Directory of XMI files, specific views:
    python add_nominal_ellipsis.py ./xmi_dir/ --view _InitialView spelling_corrected

    # Custom output directory (default: overwrite in place):
    python add_nominal_ellipsis.py ./xmi_dir/ --output ./annotated/
"""

import argparse
import logging
import sys
from pathlib import Path

import cassis
from py_lift.util import get_lift_typesystem

from preprocessing.detection.cas_adapter import (
    T_GRAMMAR_ANOMALY,
    find_and_annotate_nominal_ellipsis,
)
from preprocessing.detection.cli import add_language_args

logger = logging.getLogger(__name__)

DEFAULT_VIEW = "_InitialView"


def _existing_nominal_ellipsis_annotations(view) -> list:
    """All annotations this detector would have created on ``view``."""
    return [
        a for a in view.select(T_GRAMMAR_ANOMALY)
        if (getattr(a, "category", "") or "").startswith("nominal_head_")
    ]


def process_file(
    xmi_path: Path,
    ts: cassis.TypeSystem,
    views: list[str],
    output_path: Path,
    *,
    lang: str | None,
    mixed: bool,
    replace: bool = False,
) -> None:
    """Load an XMI file, detect nominal-head ellipsis on the views, and save."""
    with open(xmi_path, "rb") as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=ts, lenient=True)

    for view_name in views:
        try:
            view = cas.get_view(view_name)
        except Exception:
            logger.warning(f"{xmi_path.name}: view '{view_name}' not found, skipping.")
            continue

        existing = _existing_nominal_ellipsis_annotations(view)
        if existing:
            if not replace:
                logger.info(
                    f"{xmi_path.name}: view '{view_name}' already has "
                    f"{len(existing)} nominal-ellipsis annotations, skipping "
                    "(use --replace to overwrite)."
                )
                continue
            for a in existing:
                view.remove(a)
            logger.info(
                f"{xmi_path.name}: view '{view_name}' — removed "
                f"{len(existing)} existing nominal-ellipsis annotations."
            )

        count = find_and_annotate_nominal_ellipsis(
            view, ts, lang=lang, mixed=mixed
        )
        logger.info(
            f"{xmi_path.name}: view '{view_name}' — {count} nominal-ellipsis cases found"
        )

    cas.to_xmi(str(output_path), pretty_print=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add nominal-head ellipsis annotations to existing CAS XMI files."
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
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Remove existing nominal-ellipsis annotations on each "
        "processed view before re-running the detector. Default: skip "
        "views that already have nominal-ellipsis annotations.",
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
