#!/usr/bin/env python3
"""Generic structural-phenomenon annotator for CAS XMI files.

One CLI for every structural detector, replacing the former per-phenomenon
``add_<phenomenon>.py`` scripts. The set of phenomena, their detectors, writers,
and annotation signatures live in
:data:`preprocessing.detection.cas_adapter.DETECTOR_REGISTRY`; this script is a
thin CAS/CLI wrapper around it (load XMI → per view: skip/replace → detect+write
→ save).

Detection logic stays in ``preprocessing.detection.<phenomenon>`` (pure udapi,
testable with ``.conllu`` fixtures); this script never touches it directly.

Usage:
    # Single file, default view (_InitialView):
    python annotate.py --phenomenon clefts input.xmi

    # Directory of XMI files, specific views:
    python annotate.py --phenomenon passive ./xmi_dir/ \
        --view _InitialView spelling_normalized

    # Custom output dir; re-annotate in place with --replace:
    python annotate.py --phenomenon sluicing ./xmi_dir/ --output ./out/ --replace

The external-tool annotators (coreference, RWSE, Stanza parses, EDUs, spelling)
are NOT structural detectors and keep their own dedicated scripts.
"""

import argparse
import logging
import sys
from pathlib import Path

import cassis
from py_lift.util import get_lift_typesystem

from preprocessing.detection.cas_adapter import (
    DETECTOR_REGISTRY,
    existing_annotations,
    find_and_annotate,
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
    phenomenon: str,
    lang: str | None,
    mixed: bool,
    replace: bool = False,
) -> None:
    """Load an XMI file, detect ``phenomenon`` on the given views, and save."""
    with open(xmi_path, "rb") as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=ts, lenient=True)

    for view_name in views:
        try:
            view = cas.get_view(view_name)
        except Exception:
            logger.warning(f"{xmi_path.name}: view '{view_name}' not found, skipping.")
            continue

        existing = existing_annotations(view, phenomenon)
        if existing:
            if not replace:
                logger.info(
                    f"{xmi_path.name}: view '{view_name}' already has "
                    f"{len(existing)} {phenomenon} annotations, skipping "
                    "(use --replace to overwrite)."
                )
                continue
            for a in existing:
                view.remove(a)
            logger.info(
                f"{xmi_path.name}: view '{view_name}' — removed "
                f"{len(existing)} existing {phenomenon} annotations."
            )

        count = find_and_annotate(
            view, ts, phenomenon=phenomenon, lang=lang, mixed=mixed
        )
        logger.info(
            f"{xmi_path.name}: view '{view_name}' — {count} {phenomenon} cases found"
        )

    cas.to_xmi(str(output_path), pretty_print=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add structural-phenomenon annotations to existing CAS XMI files."
    )
    parser.add_argument(
        "--phenomenon",
        "-p",
        required=True,
        choices=sorted(DETECTOR_REGISTRY),
        help="Which structural phenomenon to detect and annotate.",
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
        help="Remove existing annotations of the chosen phenomenon on each "
        "processed view before re-running the detector. Default: skip views "
        "that already have them.",
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
                phenomenon=args.phenomenon, lang=args.lang, mixed=args.mixed,
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
