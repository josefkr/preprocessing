#!/usr/bin/env python3
"""Re-evaluate the verbal-ellipsis (VPE) detector against the VPE JSON sets.

For each example it parses ``text`` with Stanza, runs the *structural*
detector (``preprocessing.detection.verbal_ellipsis.detect_verbal_ellipsis``
via the same CAS → CoNLL-U → udapi path the production CLI uses), and
scores the result against the gold ``is_vpe`` label.

Two scoring levels:

  * SENTENCE-LEVEL detection — did the detector fire *anywhere* in the
    passage? Scored as a binary classification against ``is_vpe``:
        positives = {VPE_RESOLVABLE, VPE_UNRESOLVABLE}
        negatives = {NO_VPE}
    → precision / recall / F1 / accuracy + a confusion matrix.

  * SITE-MATCHED recall — for gold positives, did a finding's surface
    form match the gold ``ellipsis_site`` (comma-split, lowercased)?
    A stricter recall: detected *and* on the right token.

The detector keys on POS=AUX with a non-aux/cop deprel, so it is
expected to miss German "absolute modal" VPE when Stanza tags the
stranded modal as VERB(root) rather than AUX — this harness quantifies
exactly that gap.

Usage:
    python evaluate_vpe_detector.py \\
        --data ../resolution/en_vpe_examples.json --lang en \\
        --output results_vpe_en.json
    python evaluate_vpe_detector.py \\
        --data ../resolution/de_vpe_examples.json --lang de \\
        --output results_vpe_de.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from preprocessing.api import T_SENT
from preprocessing.stanza import Stanza_Preprocessor
from preprocessing.detection.cas_conllu import view_to_conllu
from preprocessing.detection.verbal_ellipsis import detect_verbal_ellipsis

from udapi.core.document import Document


def load_examples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data["examples"]


def gold_site_forms(example: dict) -> set[str]:
    """The gold ellipsis site(s) as a set of lowercased surface forms."""
    site = (example.get("ellipsis_site") or "").strip()
    if not site:
        return set()
    return {part.strip().lower() for part in site.split(",") if part.strip()}


@dataclass
class ExampleResult:
    id: str
    category: str
    gold_is_vpe: bool
    predicted_is_vpe: bool
    site_matched: bool
    finding_forms: list[str]
    n_findings: int


def detect(preprocessor: Stanza_Preprocessor, lang: str, text: str) -> list:
    """Parse ``text`` and return the detector's findings (possibly empty)."""
    cas = preprocessor.run(text)
    n_sents = len(list(cas.select(T_SENT)))
    if n_sents == 0:
        return []
    conllu = view_to_conllu(cas, sentence_langs=[lang] * n_sents)
    doc = Document()
    doc.from_conllu_string(conllu)
    # Single-language run: don't filter (the trees are all `lang`).
    return detect_verbal_ellipsis(doc, restrict_to_lang=None)


def evaluate(examples: list[dict], lang: str) -> list[ExampleResult]:
    preprocessor = Stanza_Preprocessor(language=lang)
    results: list[ExampleResult] = []
    total = len(examples)
    for i, ex in enumerate(examples, 1):
        eid = ex["id"]
        gold = bool(ex.get("is_vpe"))
        print(f"  [{i}/{total}] {eid}", end="  ", flush=True)
        try:
            findings = detect(preprocessor, lang, ex["text"])
        except Exception as e:
            print(f"PARSE/DETECT ERROR: {e}")
            findings = []
        forms = [f.text for f in findings]
        predicted = len(findings) > 0
        sites = gold_site_forms(ex)
        site_matched = bool(sites) and any(
            f.text.strip().lower() in sites for f in findings
        )
        mark = "✓" if predicted == gold else "✗"
        print(f"{mark} gold={gold} pred={predicted} ({forms})", flush=True)
        results.append(
            ExampleResult(
                id=eid,
                category=ex.get("category") or "",
                gold_is_vpe=gold,
                predicted_is_vpe=predicted,
                site_matched=site_matched,
                finding_forms=forms,
                n_findings=len(findings),
            )
        )
    return results


def report(results: list[ExampleResult]) -> dict:
    tp = sum(1 for r in results if r.gold_is_vpe and r.predicted_is_vpe)
    fn = sum(1 for r in results if r.gold_is_vpe and not r.predicted_is_vpe)
    fp = sum(1 for r in results if not r.gold_is_vpe and r.predicted_is_vpe)
    tn = sum(1 for r in results if not r.gold_is_vpe and not r.predicted_is_vpe)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    n = len(results)
    accuracy = (tp + tn) / n if n else 0.0

    positives = [r for r in results if r.gold_is_vpe]
    site_hits = sum(1 for r in positives if r.site_matched)
    site_recall = site_hits / len(positives) if positives else 0.0

    # Per gold-category detection rate.
    by_cat: dict[str, dict] = {}
    for cat in ("VPE_RESOLVABLE", "VPE_UNRESOLVABLE", "NO_VPE"):
        rows = [r for r in results if r.category == cat]
        if not rows:
            continue
        fired = sum(1 for r in rows if r.predicted_is_vpe)
        by_cat[cat] = {
            "n": len(rows),
            "detector_fired": fired,
            # For positives, firing is good (recall); for NO_VPE it's a
            # false alarm.
            "rate": round(fired / len(rows), 3),
        }

    return {
        "n": n,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "site_matched_recall": round(site_recall, 3),
        "site_matched_hits": site_hits,
        "n_positives": len(positives),
        "by_category": by_cat,
    }


def print_report(rep: dict, lang: str) -> None:
    print("\n" + "=" * 60)
    print(f"VPE DETECTOR RE-EVALUATION — {lang}")
    print("=" * 60)
    c = rep["confusion"]
    print(f"  n = {rep['n']}   (positives={rep['n_positives']})")
    print(f"  Confusion: TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")
    print(f"  Precision: {rep['precision']:.1%}")
    print(f"  Recall:    {rep['recall']:.1%}")
    print(f"  F1:        {rep['f1']:.3f}")
    print(f"  Accuracy:  {rep['accuracy']:.1%}")
    print(
        f"  Site-matched recall: {rep['site_matched_recall']:.1%} "
        f"({rep['site_matched_hits']}/{rep['n_positives']})"
    )
    print("\n  Per gold category (detector fired / n):")
    for cat, m in rep["by_category"].items():
        note = "recall" if cat != "NO_VPE" else "false alarms"
        print(f"    {cat:<18} {m['detector_fired']:>3}/{m['n']:<3} "
              f"({m['rate']:.0%})  [{note}]")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate the VPE detector against a VPE JSON set."
    )
    parser.add_argument("--data", required=True, help="Path to VPE examples JSON")
    parser.add_argument(
        "--lang", required=True, choices=["en", "de"], help="Language"
    )
    parser.add_argument(
        "--output", default=None, help="Optional path for per-example results JSON"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Evaluate only the first N (0=all)"
    )
    args = parser.parse_args()

    examples = load_examples(args.data)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples from {args.data}")

    results = evaluate(examples, args.lang)
    rep = report(results)
    print_report(rep, args.lang)

    if args.output:
        out = {
            "lang": args.lang,
            "data": args.data,
            "report": rep,
            "examples": [asdict(r) for r in results],
        }
        Path(args.output).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nPer-example results written to {args.output}")


if __name__ == "__main__":
    main()
