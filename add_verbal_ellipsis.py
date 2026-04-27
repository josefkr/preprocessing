#!/usr/bin/env python3
"""Add verbal ellipsis annotations to existing CAS XMI files.

Detects verbal ellipsis by finding tokens with POS=AUX that serve as
a non-auxiliary dependent (i.e. the Dependency linking them as Dependent
has a DependencyType other than "aux" or "aux:pass"). These are cases
where an auxiliary stands in for a missing main verb.

Each detected ellipsis is annotated with a GrammarAnomaly annotation
(description="Ellipsis", category="auxiliary").

Usage:
    # Single file, default view (_InitialView):
    python add_verbal_ellipsis.py input.xmi

    # Directory of XMI files, specific view:
    python add_verbal_ellipsis.py ./xmi_dir/ --view spelling_corrected

    # Multiple views:
    python add_verbal_ellipsis.py ./xmi_dir/ --view view1 view2

    # Custom output directory (default: overwrite in place):
    python add_verbal_ellipsis.py ./xmi_dir/ --output ./annotated/
"""

import argparse
import logging
import sys
from pathlib import Path

import cassis

from py_lift.util import get_lift_typesystem

logger = logging.getLogger(__name__)

T_POS = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
T_DEP = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
T_GRAMMAR_ANOMALY = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly"

DEFAULT_VIEW = "_InitialView"
AUX_DEP_TYPES = {"aux", "aux:pass", "cop"}


def find_and_annotate_ellipsis(view, ts) -> int:
    """Find verbal ellipsis cases in a view and add GrammarAnomaly annotations.

    Returns the number of ellipsis annotations added.
    """
    GA = ts.get_type(T_GRAMMAR_ANOMALY)

    # Collect spans of all AUX tokens
    aux_spans = set()
    for pos in view.select(T_POS):
        if getattr(pos, "PosValue", None) == "AUX":
            aux_spans.add((pos.begin, pos.end))

    if not aux_spans:
        return 0

    # Build a map: (begin, end) of Dependent -> DependencyType
    dep_types_by_dependent = {}
    for dep in view.select(T_DEP):
        dependent = getattr(dep, "Dependent", None)
        if dependent is not None:
            key = (dependent.begin, dependent.end)
            dep_types_by_dependent.setdefault(key, []).append(
                getattr(dep, "DependencyType", "")
            )

    count = 0
    for begin, end in sorted(aux_spans):
        dep_type_list = dep_types_by_dependent.get((begin, end), [])
        # If any dependency relation for this token is NOT aux/aux:pass,
        # it's a verbal ellipsis
        for dep_type in dep_type_list:
            if dep_type not in AUX_DEP_TYPES:
                view.add(GA(
                    begin=begin,
                    end=end,
                    description="Ellipsis",
                    category="auxiliary",
                ))
                sofa = view.sofa_string or ""
                logger.debug(
                    f"  Ellipsis: '{sofa[begin:end]}' [{begin}:{end}] "
                    f"dep={dep_type}"
                )
                count += 1
                break  # one annotation per token span

    return count


def process_file(
    xmi_path: Path,
    ts: cassis.TypeSystem,
    views: list[str],
    output_path: Path,
) -> None:
    """Load an XMI file, detect verbal ellipsis on specified views, and save."""
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
            if getattr(a, "description", None) == "Ellipsis"
        ]
        if existing:
            logger.info(
                f"{xmi_path.name}: view '{view_name}' already has "
                f"{len(existing)} ellipsis annotations, skipping."
            )
            continue

        count = find_and_annotate_ellipsis(view, ts)
        logger.info(f"{xmi_path.name}: view '{view_name}' — {count} ellipsis found")

    cas.to_xmi(str(output_path), pretty_print=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add verbal ellipsis annotations to existing CAS XMI files."
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
