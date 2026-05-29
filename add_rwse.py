#!/usr/bin/env python3
"""Add real-word spelling-error (RWSE) annotations to existing CAS XMI files.

A real-word spelling error is a real orthographic word confused with
another real word ("Rede" → "Reede", "das" → "dass") — the error is
in-vocabulary, so an ordinary OOV spellchecker can't catch it. This
script wraps the masked-language-model RWSE checker
(``rwse_checker.rwse.RWSE_Checker``) from the ``rwse-checker`` fork: for
every token that is a member of a configured *confusion set*, it masks
that token and asks the model whether a different member of the set
fits the context better. When it does, an ``RWSE`` anomaly annotation
is written over the token span, carrying the model's ``suggestion`` and
``certainty``.

Requires upstream ``Sentence`` and ``Token`` annotations on the view
(run sentence segmentation + tokenisation first) — a view without them
is skipped with an error.

The ``RWSE`` type
(``de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.RWSE``) ships in
``preprocessing/data/TypeSystem.xml`` (extends ``Anomaly`` with
``suggestion: String`` + ``certainty: Float``).

Usage:
    # Single file, default view, default DE model + bundled DE confusion sets:
    python add_rwse.py input.xmi

    # Directory of XMI files, specific views:
    python add_rwse.py ./xmi_dir/ --view _InitialView spelling_normalized

    # Custom model / confusion sets / sensitivity:
    python add_rwse.py input.xmi --model bert-base-german-dbmdz-uncased \\
        --confusion-sets my_sets.txt --magnitude 20

    # GPU device 0, overwrite existing RWSE annotations:
    python add_rwse.py input.xmi --gpu 0 --replace
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import cassis
import rwse_checker

from preprocessing.util import get_aslan_typesystem

# The RWSE checker (``rwse-checker`` fork) is installed editable into the
# active venv, so a plain import resolves to the checked-out fork and
# picks up its extended ``check_multi``.
from rwse_checker.rwse import RWSE_Checker, MASK

logger = logging.getLogger(__name__)

T_SENT = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
T_TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
T_RWSE = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.RWSE"

DEFAULT_VIEW = "_InitialView"

# Per-language defaults: a case-insensitive confusion-set file pairs with
# an *uncased* model — the checker lowercases input in case-insensitive
# mode, so an uncased model is the natural match. Confusion-set files are
# resolved relative to the installed rwse_checker package (no hardcoded
# checkout path).
DEFAULT_LANGUAGE = "de"
MODEL_BY_LANG = {
    "de": "bert-base-german-dbmdz-uncased",
    "en": "bert-base-uncased",
}
_DATA_DIR = Path(rwse_checker.__file__).parent / "data"
CONFUSION_SETS_BY_LANG = {
    "de": _DATA_DIR / "de_sets_ci.txt",
    "en": _DATA_DIR / "en_sets_ci.txt",
}


def _match_case(suggestion: str, original: str) -> str:
    """Transfer the original token's leading-letter case onto the
    suggestion, so a lowercased model output ("rede") is restored to
    the surface casing of the word it replaces ("Rede"). The checker's
    case-insensitive mode emits lowercased suggestions; German nouns in
    running text are capitalized, so without this the stored suggestion
    would be wrong-cased."""
    if not suggestion or not original:
        return suggestion
    if original[:1].isupper():
        return suggestion[:1].upper() + suggestion[1:]
    return suggestion


def annotate_rwse(
    view,
    ts,
    checker: RWSE_Checker,
    *,
    magnitude: float = 10,
    min_certainty: float = 0.0,
) -> int:
    """For each Sentence in ``view``, mask each in-confusion-set token in
    turn and ask the RWSE checker for a correction. Write an ``RWSE``
    annotation over every token the model would change.

    Returns the number of RWSE annotations added. Raises ``ValueError``
    if the view has no Sentence (or no Token) annotations.

    Args:
        magnitude: the winning candidate's probability must be at least
            ``magnitude`` times the original token's for a swap to be
            flagged (higher = more conservative).
        min_certainty: minimum log-ratio certainty for an annotation to
            be written (an extra absolute floor on top of ``magnitude``).
    """
    sentences = sorted(view.select(T_SENT), key=lambda s: s.begin)
    if not sentences:
        raise ValueError(
            "view has no Sentence annotations; run sentence segmentation first"
        )
    all_tokens = sorted(view.select(T_TOKEN), key=lambda t: t.begin)
    if not all_tokens:
        raise ValueError(
            "view has no Token annotations; run tokenisation first"
        )

    RWSE = ts.get_type(T_RWSE)
    sofa = view.sofa_string or ""
    count = 0

    for sent in sentences:
        sent_tokens = [
            t for t in all_tokens if t.begin >= sent.begin and t.end <= sent.end
        ]
        token_texts = [sofa[t.begin:t.end] for t in sent_tokens]
        for i, tok in enumerate(sent_tokens):
            surface = token_texts[i]
            if not checker.in_confusion_sets(surface):
                continue
            # Mask only this occurrence (positional), not all copies of
            # the surface form in the sentence.
            masked = " ".join(
                MASK if j == i else token_texts[j]
                for j in range(len(token_texts))
            )
            try:
                # check_multi (vs check) so multi-sub-word confusion
                # words (Gepäck, viele, …) are scored as spans rather
                # than dropped — see RWSE_Checker.check_multi.
                results = checker.check_multi(surface, masked)
            except Exception as e:
                logger.warning(
                    f"RWSE check failed for token '{surface}' "
                    f"[{tok.begin}:{tok.end}]: {e}"
                )
                continue
            if not results:
                continue

            decision = _decide_swap(surface, results, magnitude, min_certainty)
            if decision is None:
                continue
            suggestion, certainty = decision

            view.add(
                RWSE(
                    begin=tok.begin,
                    end=tok.end,
                    suggestion=_match_case(suggestion, surface),
                    certainty=float(certainty),
                    category="rwse",
                )
            )
            count += 1

    return count


def _decide_swap(
    surface: str,
    results: list,
    magnitude: float,
    min_certainty: float,
) -> tuple[str, float] | None:
    """Decide whether the model prefers a different confusion-set member
    over ``surface`` given the per-candidate scores from
    ``RWSE_Checker.check`` (a list of ``{token_str, score, ...}`` dicts).

    Returns ``(suggestion, certainty)`` when a swap is warranted, else
    ``None``. ``certainty`` is the base-10 log-ratio of the winning
    candidate's score to the original's.

    Deliberately bypasses ``RWSE_Checker.correct``, whose substring
    ``in sequence`` test misreads the original's score whenever one
    confusion-set member is a substring of another (``das`` ⊂ ``dass``,
    ``im`` ⊂ ``ihm``, …) and so never fires for those pairs. ``check``
    scores each member exactly via the HF ``targets=`` mechanism, so
    matching ``token_str`` directly is correct."""
    s_lower = surface.lower()
    orig_score = 0.0
    best_word: str | None = None
    best_score = -1.0
    for r in results:
        cand = str(r.get("token_str", ""))
        score = float(r.get("score", 0.0))
        if cand.lower() == s_lower:
            orig_score = score
            continue
        if score > best_score:
            best_word, best_score = cand, score

    if best_word is None:
        return None
    # The winner must beat the original by a factor of `magnitude`.
    if best_score < orig_score * magnitude:
        return None
    if orig_score > 0:
        certainty = math.log10(best_score) - math.log10(orig_score)
    else:
        # Original is (near-)impossible in context — treat as a strong
        # signal rather than dividing by zero.
        certainty = float(magnitude)
    if certainty < min_certainty:
        return None
    return best_word, certainty


def _existing_rwse(view) -> list:
    return list(view.select(T_RWSE))


def process_file(
    xmi_path: Path,
    ts: cassis.TypeSystem,
    views: list[str],
    checker: RWSE_Checker,
    output_path: Path,
    *,
    magnitude: float,
    min_certainty: float,
    replace: bool = False,
) -> None:
    """Load an XMI file, add RWSE annotations to specified views, save."""
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

        existing = _existing_rwse(view)
        if existing:
            if not replace:
                logger.info(
                    f"{xmi_path.name}: view '{view_name}' already has "
                    f"{len(existing)} RWSE annotations, skipping "
                    "(use --replace to overwrite)."
                )
                continue
            for a in existing:
                view.remove(a)
            logger.info(
                f"{xmi_path.name}: view '{view_name}' — removed "
                f"{len(existing)} existing RWSE annotations."
            )

        try:
            count = annotate_rwse(
                view, ts, checker,
                magnitude=magnitude, min_certainty=min_certainty,
            )
        except ValueError as e:
            logger.error(f"{xmi_path.name}: view '{view_name}': {e}")
            continue

        logger.info(
            f"{xmi_path.name}: view '{view_name}' — {count} RWSEs annotated"
        )
        any_change = True

    if any_change:
        cas.to_xmi(str(output_path), pretty_print=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add RWSE annotations to existing CAS XMI files."
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
        help=f"View name(s) to process (default: {DEFAULT_VIEW}).",
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=sorted(MODEL_BY_LANG),
        default=DEFAULT_LANGUAGE,
        help="Language; selects the default masked-LM model and confusion "
        f"sets (default: {DEFAULT_LANGUAGE}). Override either with "
        "--model / --confusion-sets.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="HF masked-LM model name (default: per --language).",
    )
    parser.add_argument(
        "--confusion-sets",
        "-c",
        type=Path,
        default=None,
        help="Path to a confusion-set file (comma-separated, one set "
        "per line). Default: the bundled set for --language.",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=10,
        help="Certainty-threshold multiplier passed to the checker "
        "(higher = more conservative; default: 10).",
    )
    parser.add_argument(
        "--min-certainty",
        type=float,
        default=0.0,
        help="Minimum log-ratio certainty for an annotation (default: 0).",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Match tokens case-sensitively (default: case-insensitive, "
        "matching the bundled de_sets_ci.txt).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device id (-1 for CPU, the default).",
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
        help="Remove existing RWSE annotations on each processed view "
        "before re-running. Default: skip views that already have them.",
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
    model = args.model or MODEL_BY_LANG[args.language]
    confusion_sets = args.confusion_sets or CONFUSION_SETS_BY_LANG[args.language]
    print(f"Loading RWSE checker (language={args.language}, model={model})...")
    checker = RWSE_Checker(
        model_name=model,
        confusion_sets=str(confusion_sets),
        case_sensitive=args.case_sensitive,
        gpu=args.gpu,
    )

    success = 0
    errors = 0
    for i, xmi_file in enumerate(xmi_files, 1):
        out_path = (output_dir / xmi_file.name) if output_dir else xmi_file
        try:
            process_file(
                xmi_file, ts, args.view, checker, out_path,
                magnitude=args.magnitude,
                min_certainty=args.min_certainty,
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
