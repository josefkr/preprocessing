#!/usr/bin/env python3
"""Fix mechanical UD level-2 issues in the .conllu test fixtures.

Three purely formal fixes, none of which change the annotation content the
detectors read (they use FORM/UPOS/FEATS-as-dict/DEPREL, never SpaceAfter, and
FEATS order is irrelevant once parsed):

1. ``SpaceAfter=No`` — added to MISC where ``# text`` shows no space between a
   surface token and the next (mostly before punctuation).
2. FEATS sorting — UD requires attributes sorted case-insensitively
   (e.g. ``Number`` before ``NumType``).
3. A trailing empty line after the last sentence.

Safety: each sentence's forms are aligned against its ``# text``. If alignment
fails (the text and the tokens genuinely disagree), the sentence is left
**untouched** and reported — so pre-existing content bugs are never papered
over or corrupted. Multiword-token range lines (``4-5``) are the surface
tokens; their sub-words consume no text and never get SpaceAfter. Empty nodes
(``3.1``) are skipped.

    python fix_conllu_level2.py [--dry-run] [paths...]
"""

from __future__ import annotations

import argparse
import glob
import os


def sort_feats(feats: str) -> str:
    if feats == "_" or "=" not in feats:
        return feats
    parts = feats.split("|")
    return "|".join(sorted(parts, key=lambda p: p.split("=", 1)[0].lower()))


def set_misc_spaceafter(misc: str, no_space: bool) -> str:
    """Add/remove SpaceAfter=No in MISC, preserving other keys (t_start/t_end)."""
    items = [] if misc == "_" else [i for i in misc.split("|") if i != "SpaceAfter=No"]
    if no_space:
        items.insert(0, "SpaceAfter=No")
    return "|".join(items) if items else "_"


def fix_sentence(lines: list[str]) -> tuple[list[str], str | None]:
    """Fix one sentence block. Returns (new_lines, error_or_None)."""
    text = None
    for ln in lines:
        if ln.startswith("# text ="):
            text = ln.split("=", 1)[1].strip()
    rows = [(i, ln.split("\t")) for i, ln in enumerate(lines)
            if ln and not ln.startswith("#") and "\t" in ln]

    # Which ids are covered by a multiword range → they consume no text.
    covered: set[int] = set()
    for _i, c in rows:
        if "-" in c[0]:
            a, b = c[0].split("-")
            covered.update(range(int(a), int(b) + 1))

    out = list(lines)
    # --- FEATS sorting (all token rows) ---
    for i, c in rows:
        if len(c) >= 6 and c[5] != sort_feats(c[5]):
            c = list(c)
            c[5] = sort_feats(c[5])
            out[i] = "\t".join(c)

    if text is None:
        return out, "no '# text ='"

    # --- SpaceAfter=No, by aligning surface tokens to the text ---
    cursor = 0
    surface = []  # (line_idx, cols) for tokens that consume text
    for i, c in rows:
        tid = c[0]
        if "." in tid:                     # empty node
            continue
        if "-" not in tid and int(tid) in covered:   # MWT sub-word
            continue
        surface.append((i, c))

    pending = []
    for n, (i, c) in enumerate(surface):
        form = c[1]
        if not text.startswith(form, cursor):
            return lines, (f"cannot align form {form!r} at offset {cursor} "
                           f"(text has {text[cursor:cursor + 12]!r})")
        cursor += len(form)
        is_last = n == len(surface) - 1
        if not is_last:
            no_space = not text.startswith(" ", cursor)
            if not no_space:
                cursor += 1
        else:
            no_space = False
        pending.append((i, c, no_space))

    if cursor != len(text):
        return lines, f"text not fully consumed: leftover {text[cursor:]!r}"

    for i, c, no_space in pending:
        c = list(out[i].split("\t"))
        if len(c) >= 10:
            new = set_misc_spaceafter(c[9], no_space)
            if new != c[9]:
                c[9] = new
                out[i] = "\t".join(c)
    return out, None


def fix_file(path: str, dry_run: bool) -> tuple[int, list[str]]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    blocks, cur = [], []
    for ln in raw.split("\n"):
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)

    problems, new_blocks = [], []
    for b in blocks:
        fixed, err = fix_sentence(b)
        if err:
            problems.append(err)
        new_blocks.append(fixed)

    # Each sentence followed by a blank line (fixes missing-empty-line).
    new_raw = "".join("\n".join(b) + "\n\n" for b in new_blocks)
    changed = new_raw != raw
    if changed and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_raw)
    return int(changed), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = args.paths or sorted(
        glob.glob(os.path.join(os.path.dirname(__file__) or ".", "*", "*.conllu")))

    changed = skipped = 0
    for p in paths:
        c, problems = fix_file(p, args.dry_run)
        changed += c
        for msg in problems:
            skipped += 1
            print(f"SKIPPED (left untouched) {os.path.relpath(p)}: {msg}")
    print(f"\n{changed} file(s) {'would be ' if args.dry_run else ''}changed; "
          f"{skipped} sentence(s) skipped as unalignable.")


if __name__ == "__main__":
    main()
