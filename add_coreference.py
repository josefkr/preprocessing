#!/usr/bin/env python3
"""Add coreference annotations to existing CAS XMI files.

Calls a coreference resolution web service and annotates mention spans
with LexicalPhrase annotations, where the `text` feature stores the
1-based cluster ID.

If a view has Sentence and Token annotations, the pre-tokenized approach
is used (approach 1). Otherwise, the plain text sofa is sent (approach 2).

Usage:
    # Single file, default view (_InitialView):
    python add_coreference.py input.xmi

    # Directory of XMI files, specific views:
    python add_coreference.py ./xmi_dir/ --view _InitialView spelling_corrected

    # Custom output directory (default: overwrite in place):
    python add_coreference.py ./xmi_dir/ --output ./annotated/

    # Use a custom endpoint:
    python add_coreference.py input.xmi --endpoint https://my-coref-service/predict
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cassis
import requests
from dotenv import load_dotenv

from py_lift.util import get_lift_typesystem

load_dotenv()

logger = logging.getLogger(__name__)

T_SENT = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
T_TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
T_LEXICAL_PHRASE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.LexicalPhrase"

DEFAULT_VIEW = "_InitialView"

ENDPOINTS = {
    "en": "https://maverick-en.cats.fernuni-hagen.de/predict",
    # "de": "https://maverick-de.cats.fernuni-hagen.de/predict",  # uncomment when available
}

API_TOKEN = os.environ.get("MAVERICK_API_TOKEN", "")


def _is_cluster_id(value) -> bool:
    """Check if a LexicalPhrase text value looks like a coreference cluster ID."""
    if value is None:
        return False
    try:
        int(str(value))
        return True
    except (ValueError, TypeError):
        return False


def call_coref_service(endpoint: str, payload: dict) -> dict:
    """Call the coreference web service and return the JSON response."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _build_tokenized_payload(view) -> tuple[dict, list]:
    """Build a pre-tokenized payload (approach 1).

    Returns the payload dict and a flat list of (begin, end) character
    offsets for each token, in the same order as the flat token list.
    """
    sentences = sorted(view.select(T_SENT), key=lambda s: s.begin)
    all_tokens_in_view = sorted(view.select(T_TOKEN), key=lambda t: t.begin)

    token_lists = []
    char_offsets = []

    for sent in sentences:
        sent_tokens = [
            t for t in all_tokens_in_view
            if t.begin >= sent.begin and t.end <= sent.end
        ]
        sent_tokens.sort(key=lambda t: t.begin)

        sofa = view.sofa_string or ""
        token_texts = [sofa[t.begin:t.end] for t in sent_tokens]
        token_lists.append(token_texts)

        for t in sent_tokens:
            char_offsets.append((t.begin, t.end))

    return {"tokens": token_lists}, char_offsets


def _build_plain_payload(view) -> dict:
    """Build a plain text payload (approach 2)."""
    return {"tokens": view.sofa_string or ""}


def _map_char_offsets_from_plain(result: dict, sofa: str) -> list[tuple[int, int]]:
    """Map the flat token list from a plain-text response back to character offsets.

    Uses clusters_char_offsets if available, otherwise reconstructs
    positions by finding each token sequentially in the sofa.
    """
    tokens = result.get("tokens", [])

    # If char offsets are provided, build a per-token offset list from them
    char_offsets = result.get("clusters_char_offsets")
    if char_offsets is not None:
        # char_offsets are per-cluster, per-mention — we need per-token.
        # Fall through to reconstruction since per-token offsets aren't directly given.
        pass

    # Reconstruct per-token character offsets by matching tokens against sofa
    offsets = []
    pos = 0
    for token_text in tokens:
        idx = sofa.find(token_text, pos)
        if idx < 0:
            # Fallback: try case-insensitive or skip whitespace
            logger.warning(
                f"Could not find token '{token_text}' in sofa starting at pos {pos}"
            )
            offsets.append((pos, pos + len(token_text)))
            pos = pos + len(token_text)
        else:
            offsets.append((idx, idx + len(token_text)))
            pos = idx + len(token_text)

    return offsets


def annotate_coreference(view, ts, endpoint: str) -> int:
    """Call coref service for a view and add LexicalPhrase annotations.

    Returns the number of mention annotations added.
    """
    sofa = view.sofa_string or ""
    if not sofa.strip():
        logger.warning("View has empty text, skipping.")
        return 0

    # Decide approach based on available annotations
    sentences = list(view.select(T_SENT))
    tokens = list(view.select(T_TOKEN))
    has_sentences_and_tokens = bool(sentences) and bool(tokens)

    if has_sentences_and_tokens:
        logger.debug("Using tokenized approach (sentences + tokens available)")
        payload, token_char_offsets = _build_tokenized_payload(view)
    else:
        logger.debug("Using plain text approach (no sentence/token annotations)")
        payload = _build_plain_payload(view)
        token_char_offsets = None

    result = call_coref_service(endpoint, payload)

    clusters = result.get("clusters_token_offsets")
    if not clusters:
        raise ValueError("Response missing 'clusters_token_offsets'")

    # Build token char offset map
    if token_char_offsets is None:
        token_char_offsets = _map_char_offsets_from_plain(result, sofa)

    LP = ts.get_type(T_LEXICAL_PHRASE)
    count = 0

    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = str(cluster_idx + 1)  # 1-based

        for mention in cluster:
            first_token_idx, last_token_idx = mention

            if first_token_idx >= len(token_char_offsets) or last_token_idx >= len(token_char_offsets):
                logger.warning(
                    f"Token index out of range: [{first_token_idx}, {last_token_idx}] "
                    f"(max={len(token_char_offsets) - 1})"
                )
                continue

            begin = token_char_offsets[first_token_idx][0]
            end = token_char_offsets[last_token_idx][1]

            view.add(LP(begin=begin, end=end, text=cluster_id))
            count += 1
            logger.debug(
                f"  Cluster {cluster_id}: '{sofa[begin:end]}' [{begin}:{end}]"
            )

    return count


def process_file(
    xmi_path: Path,
    ts: cassis.TypeSystem,
    views: list[str],
    endpoint: str,
    output_path: Path,
) -> None:
    """Load an XMI file, add coreference annotations to specified views, and save."""
    with open(xmi_path, "rb") as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=ts)

    for view_name in views:
        try:
            view = cas.get_view(view_name)
        except Exception:
            logger.warning(f"{xmi_path.name}: view '{view_name}' not found, skipping.")
            continue

        # Only skip if coreference annotations already exist.
        # Coreference uses integer cluster IDs as the text value,
        # while other uses of LexicalPhrase (e.g. subject sharing)
        # use word strings — those should not block annotation.
        existing_coref = [
            lp for lp in view.select(T_LEXICAL_PHRASE)
            if _is_cluster_id(getattr(lp, "text", None))
        ]
        if existing_coref:
            logger.info(
                f"{xmi_path.name}: view '{view_name}' already has "
                f"{len(existing_coref)} coreference annotations, skipping."
            )
            continue

        count = annotate_coreference(view, ts, endpoint)
        logger.info(f"{xmi_path.name}: view '{view_name}' — {count} mentions annotated")

    cas.to_xmi(str(output_path), pretty_print=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add coreference annotations to existing CAS XMI files."
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
        "--language",
        "-l",
        default="en",
        choices=list(ENDPOINTS.keys()),
        help="Language for coreference resolution (default: en).",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        default=None,
        help="Custom endpoint URL. Overrides the language-based endpoint.",
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

    # Resolve endpoint
    if args.endpoint:
        endpoint = args.endpoint
    elif args.language in ENDPOINTS:
        endpoint = ENDPOINTS[args.language]
    else:
        print(f"No endpoint configured for language '{args.language}'.")
        sys.exit(1)

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
            process_file(xmi_file, ts, args.view, endpoint, out_path)
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
