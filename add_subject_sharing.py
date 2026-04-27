#!/usr/bin/env python3
"""Add subject-sharing annotations to coordinated clauses in CAS XMI files.

Detects coordinated clauses where the right conjunct (X) lacks its own
subject, sharing the subject (S) of the left conjunct (Y) instead.

Detection rule (in Universal Dependencies terms):
  - X is a dependent of Y via the 'conj' relation
  - Y has a child S with a subject relation (nsubj, csubj, nsubj:pass)
  - X has NO child with a subject relation

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

logger = logging.getLogger(__name__)

T_DEP = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
T_GRAMMAR_ANOMALY = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.GrammarAnomaly"
T_LEXICAL_PHRASE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.LexicalPhrase"

DEFAULT_VIEW = "_InitialView"
SUBJECT_RELS = {"nsubj", "csubj", "nsubj:pass"}


def find_and_annotate_subject_sharing(view, ts) -> int:
    """Find subject-sharing conjuncts and add annotations.

    Returns the number of conjunct annotations added (each match produces
    one GrammarAnomaly on X and one LexicalPhrase on S).
    """
    GA = ts.get_type(T_GRAMMAR_ANOMALY)
    LP = ts.get_type(T_LEXICAL_PHRASE)
    sofa = view.sofa_string or ""

    # Build a map: governor (begin, end) -> list of subject dependency annotations
    gov_subjects: dict[tuple[int, int], list] = {}
    for dep in view.select(T_DEP):
        if dep.DependencyType in SUBJECT_RELS:
            gov = dep.Governor
            key = (gov.begin, gov.end)
            gov_subjects.setdefault(key, []).append(dep)

    count = 0
    for dep in view.select(T_DEP):
        if dep.DependencyType != "conj":
            continue

        Y = dep.Governor   # head of conjunction (left conjunct)
        X = dep.Dependent  # right conjunct (lacks subject)

        y_key = (Y.begin, Y.end)
        x_key = (X.begin, X.end)

        y_subjs = gov_subjects.get(y_key, [])
        x_subjs = gov_subjects.get(x_key, [])

        if not y_subjs or x_subjs:
            # Y has no subject, or X has its own subject — skip
            continue

        # X is a conjunct that shares Y's subject
        # Annotate X with GrammarAnomaly
        view.add(GA(
            begin=X.begin,
            end=X.end,
            description="Ellipsis",
            category="right_conj_subject",
        ))

        # Annotate each subject S of Y with LexicalPhrase
        for subj_dep in y_subjs:
            S = subj_dep.Dependent
            # Avoid duplicate LexicalPhrase annotations on the same span
            view.add(LP(
                begin=S.begin,
                end=S.end,
                text="Shared_subject",
            ))
            logger.debug(
                f"  Subject sharing: Y='{sofa[Y.begin:Y.end]}' conj-> "
                f"X='{sofa[X.begin:X.end]}', shared S='{sofa[S.begin:S.end]}'"
            )

        count += 1

    return count


def process_file(
    xmi_path: Path,
    ts: cassis.TypeSystem,
    views: list[str],
    output_path: Path,
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

        # Check for existing annotations to avoid duplicates
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

        count = find_and_annotate_subject_sharing(view, ts)
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
