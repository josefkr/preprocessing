#!/usr/bin/env python3
"""Add Elementary Discourse Unit (EDU) annotations to existing CAS XMI files.

For each ``Sentence`` annotation in the chosen view(s), runs the HF
``poyum/test_discut`` segmenter (see ``preprocessing.discourse``) and
writes one ``ElementaryDiscourseUnit`` annotation per emitted EDU
span. Sentences are required upstream — a view without ``Sentence``
annotations is skipped with an error.

The EDU type lives at::

    de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.ElementaryDiscourseUnit

(declared in ``preprocessing/data/TypeSystem.xml``; loaded here via
``preprocessing.util.get_aslan_typesystem``).

Usage:
    # Single file, default view (_InitialView):
    python add_edus.py input.xmi

    # Directory of XMI files, specific views:
    python add_edus.py ./xmi_dir/ --view _InitialView spelling_corrected

    # Custom output directory (default: overwrite in place):
    python add_edus.py ./xmi_dir/ --output ./annotated/

    # Force CPU even when CUDA is available:
    python add_edus.py input.xmi --cpu

    # Re-run on files that already have EDU annotations:
    python add_edus.py input.xmi --replace
"""

import argparse
import logging
import sys
from pathlib import Path

import cassis

from preprocessing.util import get_aslan_typesystem
from preprocessing.discourse import EduSegmenter

logger = logging.getLogger(__name__)

T_SENT = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
T_EDU = (
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.ElementaryDiscourseUnit"
)

DEFAULT_VIEW = "_InitialView"


def annotate_edus(view, ts, segmenter: EduSegmenter) -> int:
    """For each ``Sentence`` in ``view``, segment its text into EDUs
    and add one ``ElementaryDiscourseUnit`` annotation per span.

    Returns the number of EDU annotations added. Raises ``ValueError``
    if the view has no Sentence annotations.
    """
    sentences = sorted(view.select(T_SENT), key=lambda s: s.begin)
    if not sentences:
        raise ValueError(
            "view has no Sentence annotations; run sentence segmentation first"
        )

    EDU = ts.get_type(T_EDU)
    sofa = view.sofa_string or ""
    count = 0
    for sent in sentences:
        sent_text = sofa[sent.begin:sent.end]
        if not sent_text.strip():
            continue
        try:
            spans = segmenter.segment_sentence(sent_text)
        except Exception as e:
            logger.warning(
                f"EDU segmentation failed for sentence "
                f"[{sent.begin}:{sent.end}]: {e}"
            )
            continue
        for (b, e) in spans:
            view.add(EDU(begin=sent.begin + b, end=sent.begin + e))
            count += 1
    return count


def process_file(
    xmi_path: Path,
    ts: cassis.TypeSystem,
    views: list[str],
    output_path: Path,
    segmenter: EduSegmenter,
    *,
    replace: bool = False,
) -> None:
    """Load an XMI file, add EDU annotations to specified views, save."""
    with open(xmi_path, "rb") as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=ts)

    any_change = False
    for view_name in views:
        try:
            view = cas.get_view(view_name)
        except Exception:
            logger.warning(
                f"{xmi_path.name}: view '{view_name}' not found, skipping."
            )
            continue

        existing = list(view.select(T_EDU))
        if existing:
            if not replace:
                logger.info(
                    f"{xmi_path.name}: view '{view_name}' already has "
                    f"{len(existing)} EDU annotations, skipping "
                    "(use --replace to overwrite)."
                )
                continue
            for a in existing:
                view.remove(a)
            logger.info(
                f"{xmi_path.name}: view '{view_name}' — removed "
                f"{len(existing)} existing EDU annotations."
            )

        try:
            count = annotate_edus(view, ts, segmenter)
        except ValueError as e:
            logger.error(f"{xmi_path.name}: view '{view_name}': {e}")
            continue

        logger.info(
            f"{xmi_path.name}: view '{view_name}' — {count} EDUs annotated"
        )
        any_change = True

    if any_change:
        cas.to_xmi(str(output_path), pretty_print=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add EDU annotations to existing CAS XMI files."
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
        help="Remove existing EDU annotations on each processed view "
        "before re-running the segmenter. Default: skip views that "
        "already have EDU annotations.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Collect input files.
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

    ts = get_aslan_typesystem()
    segmenter = EduSegmenter(device="cpu" if args.cpu else None)

    success = 0
    errors = 0
    for i, xmi_file in enumerate(xmi_files, 1):
        out_path = (output_dir / xmi_file.name) if output_dir else xmi_file
        try:
            process_file(
                xmi_file, ts, args.view, out_path, segmenter,
                replace=args.replace,
            )
            success += 1
        except Exception as e:
            errors += 1
            logger.error(f"{xmi_file.name}: {e}")

        if i % 100 == 0 or i == len(xmi_files):
            print(f"  [{i}/{len(xmi_files)}] processed")

    print(
        f"\nDone. {success} succeeded, {errors} failed "
        f"out of {len(xmi_files)} files."
    )


if __name__ == "__main__":
    main()
