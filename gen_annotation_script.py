#!/usr/bin/env python3
"""Generate a per-dataset annotation bash script from a views.yaml.

A finished XMI directory carries a cas_explorer ``views.yaml`` describing,
in order, the views the pipeline produced (``_InitialView`` plus one per
normalization step). The annotators in this fork (``add_*.py``) should run
on a *prefix* of those views, by this rule:

    Run an annotator on the views from the start up to AND INCLUDING the
    view where its phenomenon is normalized; stop after that. If the
    phenomenon is not normalized in this pipeline (its normalized view is
    absent — or no normalizer exists for it yet), run on ALL views, so the
    phenomenon is still annotated everywhere it might occur.

This script reads a ``views.yaml``, applies that rule per annotator using
the spec table below, and emits a runnable bash script with the right
``--view`` list for each ``add_*.py`` call — so you no longer hand-edit a
per-dataset annotation script.

The spec table (:data:`ANNOTATORS`) is the single place to maintain:
each entry pairs an annotator with the view its phenomenon normalizes to
(``normalized_view``), or ``None`` when no normalizer exists yet. When a
normalizer is later added for a now-"foundational" annotator (e.g. a
passive normalizer writing ``passive_normalized``), just set that
annotator's ``normalized_view`` and it becomes gated automatically.

Usage:
    python gen_annotation_script.py --views output/my_run/views.yaml \\
        --language de --output output/my_run/perform_annotations.sh
    # or print to stdout:
    python gen_annotation_script.py --views output/my_run/views.yaml --language en
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# This fork's directory — the generated script cd's here before calling the
# annotators (so relative ``add_*.py`` and ``poetry run`` resolve correctly).
PREPROCESSING_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Annotator:
    """One annotator step and how to invoke it.

    Structural detectors run through the generic ``annotate.py`` (``script`` =
    ``"annotate.py"``, ``phenomenon`` = a ``DETECTOR_REGISTRY`` key); the
    external-tool annotators keep their own ``add_*.py`` script (``phenomenon``
    empty).

    ``normalized_view`` is the CAS view at which this annotator's phenomenon
    gets normalized (a normalizer's OUTPUT_VIEW_NAME); ``None`` means no
    normalizer exists for it — it then runs on all views. ``lang_flag`` is
    the script's language option (``--lang``, ``--language``, or ``None``
    for language-agnostic annotators).
    """

    name: str
    script: str
    normalized_view: str | None
    lang_flag: str | None = "--lang"
    note: str = ""
    phenomenon: str = ""


# Single source of truth. Order = emission order; stanza parses must come
# first (everything else needs the parse). coreference last (heaviest).
ANNOTATORS: list[Annotator] = [
    Annotator(
        "stanza parses", "add_stanza_parses.py", None, "--language",
        note="foundational: parse every view (prerequisite for the rest)",
    ),
    Annotator("RWSE", "add_rwse.py", "rwse_normalized", "--language"),
    Annotator(
        "OOV spelling errors", "add_spelling_errors.py", "spelling_normalized",
        "--language", note="needs tokenization, so after parsing",
    ),
    # Structural detectors — all run through the generic annotate.py CLI.
    Annotator("nominal ellipsis", "annotate.py", "nominal_ellipsis_resolved",
              "--lang", phenomenon="nominal_ellipsis"),
    Annotator("verbal ellipsis (VPE)", "annotate.py", "vpe_resolved",
              "--lang", phenomenon="verbal_ellipsis"),
    Annotator("shared subject / coordination ellipsis", "annotate.py",
              "coord_subjects_explicated", "--lang", phenomenon="subject_sharing"),
    Annotator("gapped coordination", "annotate.py",
              "gapped_coordination_resolved", "--lang",
              phenomenon="gapped_coordination"),
    Annotator("bare wh-questions", "annotate.py", "bare_questions_resolved",
              "--lang", phenomenon="bare_questions"),
    Annotator("sluicing", "annotate.py", "sluicing_resolved", "--lang",
              phenomenon="sluicing"),
    Annotator("passive", "annotate.py", None, "--lang", phenomenon="passive",
              note="no normalizer yet — give it a normalized_view to gate it"),
    Annotator("clefts", "annotate.py", None, "--lang", phenomenon="clefts"),
    Annotator("right node raising", "annotate.py", "right_node_raising_normalized",
              "--lang", phenomenon="right_node_raising"),
    Annotator("EDUs", "add_edus.py", None, None),
    Annotator("coreference", "add_coreference.py", "coref_normalized",
              "--language"),
]


def load_view_ids(views_yaml: Path) -> list[str]:
    """Read the ordered list of view ids from a cas_explorer views.yaml."""
    with open(views_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("view_order") or []
    ids = [e["id"] for e in entries if isinstance(e, dict) and e.get("id")]
    if not ids:
        raise ValueError(
            f"{views_yaml}: no 'view_order' entries with an 'id' found."
        )
    return ids


def views_for(annotator: Annotator, view_ids: list[str]) -> list[str]:
    """The view prefix this annotator runs on, per the gating rule."""
    nv = annotator.normalized_view
    if nv is not None and nv in view_ids:
        return view_ids[: view_ids.index(nv) + 1]
    return list(view_ids)


def _command(annotator: Annotator, views: list[str], runner: str) -> str:
    parts = [runner, annotator.script]
    if annotator.phenomenon:
        parts += ["--phenomenon", annotator.phenomenon]
    if annotator.lang_flag is not None:
        parts += [annotator.lang_flag, '"$LANG_CODE"']
    parts += ["--view", *views, "--", '"$XMIDIR"']
    return " ".join(parts)


def generate(view_ids: list[str], xmi_dir: Path, language: str,
             runner: str, views_yaml: Path) -> str:
    """Build the full bash script text."""
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by gen_annotation_script.py — do not hand-edit;",
        f"# regenerate from {views_yaml}.",
        "#",
        "# Each annotator runs on views up to & including the view where its",
        "# phenomenon is normalized; if that view is absent it runs on all",
        "# views. Views (in pipeline order):",
        f"#   {' '.join(view_ids)}",
        "",
        f"cd {PREPROCESSING_DIR}",
        "",
        f'XMIDIR="{xmi_dir}"',
        f'LANG_CODE="{language}"',
        "",
    ]
    for ann in ANNOTATORS:
        views = views_for(ann, view_ids)
        header = f"echo; echo '== {ann.name} =='"
        if ann.note:
            lines.append(f"# {ann.note}")
        lines.append(header)
        lines.append(_command(ann, views, runner))
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dataset's annotation bash script from its views.yaml."
    )
    parser.add_argument(
        "--views", type=Path, required=True,
        help="Path to the dataset's views.yaml.",
    )
    parser.add_argument(
        "--language", required=True,
        help="Language passed to the annotators (e.g. de, en).",
    )
    parser.add_argument(
        "--xmi-dir", type=Path, default=None,
        help="XMI directory the annotators process (default: the directory "
        "containing views.yaml).",
    )
    parser.add_argument(
        "--runner", default="poetry run -- python3",
        help="Command used to run each annotator (default: 'poetry run "
        "-- python3'). The ``--`` after ``poetry run`` is required: it "
        "stops Poetry's CLI from parsing flags like ``--language`` / "
        "``--view`` as its own options. Pass ``--runner python3`` if "
        "you're already in the activated preprocessing venv.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write the script here (default: print to stdout).",
    )
    args = parser.parse_args()

    if not args.views.is_file():
        print(f"views.yaml not found: {args.views}")
        sys.exit(1)

    try:
        view_ids = load_view_ids(args.views)
    except (ValueError, yaml.YAMLError) as e:
        print(f"Error reading views.yaml: {e}")
        sys.exit(1)

    xmi_dir = (args.xmi_dir or args.views.parent).resolve()
    script = generate(
        view_ids, xmi_dir, args.language, args.runner, args.views.resolve()
    )

    if args.output:
        args.output.write_text(script, encoding="utf-8")
        print(f"Wrote {args.output}")
        print(f"  views: {' '.join(view_ids)}")
    else:
        sys.stdout.write(script)


if __name__ == "__main__":
    main()
