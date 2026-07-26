"""Strip revision archaeology from the submitted LaTeX, keep the provenance annotations.

evaluation3 section 3.11: the submitted .tex files carry several hundred lines of
revision-provenance comments quoting an external evaluator's critique verbatim
alongside the superseded wording. A submitted artefact should not contain its own
revision history; that belongs in version control. The `% SOURCE:` annotations
linking each number to its result file are a different matter and are a genuine
strength, so they are kept.

A comment annotation is a run of consecutive comment-only lines starting at a line
that matches one of the marker patterns. Annotations whose marker is in STRIP are
removed whole; everything else is left untouched. Inline trailing comments on a
line of real LaTeX are never touched.

    python revision/strip_revision_comments.py --dry_run
    python revision/strip_revision_comments.py --apply
"""
import argparse
import glob
import os
import re

KEEP = re.compile(r"^\s*%\s*(SOURCE|VERIFIED|SOURCE\s*/\s*VERIFIED)\b", re.I)
STRIP = re.compile(
    r"^\s*%\s*("
    r"REMOVED|COMPRESSED|PRIOR\s+WORDING|SCOPING|CUT|MOVED|REWRITTEN|ADDED|FLAG|"
    r"NOTE\s*\(|Phase\s*\d|Earlier\s+cuts"
    r")\b", re.I)
# An unmarked comment run that names the revision process is archaeology too.
STRIP_BODY = re.compile(r"(evaluation\s*\d|author\s+issue\s+list|Phase\s*\d|"
                        r"REVISION_|prior\s+version|superseded)", re.I)
COMMENT = re.compile(r"^\s*%")


def annotations(lines):
    """Yield (start, end, kind) for each run of consecutive comment-only lines."""
    i, n = 0, len(lines)
    while i < n:
        if not COMMENT.match(lines[i]):
            i += 1
            continue
        j = i
        while j < n and COMMENT.match(lines[j]):
            j += 1
        # split the run into annotations at each marker line
        bounds = [k for k in range(i, j) if KEEP.match(lines[k]) or STRIP.match(lines[k])]
        if not bounds:
            yield i, j, "unmarked"
        else:
            if bounds[0] > i:
                yield i, bounds[0], "unmarked"
            for a, b in zip(bounds, bounds[1:] + [j]):
                yield a, b, "keep" if KEEP.match(lines[a]) else "strip"
        i = j


def process(path):
    with open(path) as f:
        lines = f.readlines()
    drop = set()
    stats = {"strip": 0, "unmarked_strip": 0, "keep": 0}
    for a, b, kind in annotations(lines):
        block = "".join(lines[a:b])
        if kind == "strip":
            drop.update(range(a, b))
            stats["strip"] += b - a
        elif kind == "unmarked" and STRIP_BODY.search(block):
            drop.update(range(a, b))
            stats["unmarked_strip"] += b - a
        elif kind == "keep":
            stats["keep"] += b - a
    out = [ln for k, ln in enumerate(lines) if k not in drop]
    # collapse any blank-line runs the removal opened up
    collapsed, blanks = [], 0
    for ln in out:
        if ln.strip() == "":
            blanks += 1
            if blanks > 1:
                continue
        else:
            blanks = 0
        collapsed.append(ln)
    return lines, collapsed, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="Doc/final/thesis/chapters")
    p.add_argument("--also", nargs="*", default=["Doc/final/thesis/thesis.tex",
                                                 "Doc/final/proposal/proposal.tex"])
    p.add_argument("--apply", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.tex")))
    paths += [q for q in args.also if os.path.exists(q)]

    tot_before = tot_after = 0
    for path in paths:
        before, after, stats = process(path)
        tot_before += len(before)
        tot_after += len(after)
        if len(before) != len(after):
            print(f"{os.path.basename(path):26s} {len(before):4d} -> {len(after):4d} lines "
                  f"(archaeology {stats['strip'] + stats['unmarked_strip']:3d}, "
                  f"SOURCE kept {stats['keep']:3d})")
        if args.apply:
            with open(path, "w") as f:
                f.writelines(after)
    print(f"\ntotal {tot_before} -> {tot_after} lines "
          f"({tot_before - tot_after} removed)"
          f"{'  [APPLIED]' if args.apply else '  [dry run, use --apply]'}")


if __name__ == "__main__":
    main()
