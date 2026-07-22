#!/usr/bin/env python3
"""Re-derive Doc/final/chapters/ from Doc/chapters/.

Doc/ holds the substantive master (report class, \\chapter). Doc/final/ holds the
template-explicit deliverable (article class, \\section), which the 2026-07-22 14:27
log entry created by a purely mechanical transform of the master. That transform was
applied by hand at the time and no script was kept, so this file reconstructs it
exactly, so that the two builds can be kept in sync whenever the master changes.

The transform, per that log entry, is exactly three things:

 1. heading demotion by one level, which PRESERVES the numbering
    (article section/subsection/subsubsection == report chapter/section/subsection):
        \\chapter    -> \\section
        \\section    -> \\subsection
        \\subsection -> \\subsubsection
    Starred forms are demoted the same way.
 2. prose reference rewording, so cross-references read correctly in a document
    whose top level is a section:
        Chapter~\\ref  -> Section~\\ref
        the standalone words chapter/Chapter/chapters/Chapters -> section/...
 3. restoring the \\input paths, which step 2 would otherwise rewrite
    (\\input{chapters/...} must not become \\input{sections/...}).

Run with --verify to check the transform against the existing Doc/final/chapters/
using the committed Doc/chapters/ as input; run with no arguments to regenerate.
"""

import argparse
import os
import re
import subprocess
import sys

SRC = "Doc/chapters"
DST = "Doc/final/chapters"


def transform(text):
    # 1. heading demotion, deepest first so nothing is demoted twice
    text = re.sub(r"\\subsection(\*?)\{", r"\\subsubsection\1{", text)
    text = re.sub(r"\\section(\*?)\{", r"\\subsection\1{", text)
    text = re.sub(r"\\chapter(\*?)\{", r"\\section\1{", text)

    # 2. prose reference rewording
    text = text.replace("Chapter~\\ref", "Section~\\ref")
    text = re.sub(r"\bChapters\b", "Sections", text)
    text = re.sub(r"\bChapter\b", "Section", text)
    text = re.sub(r"\bchapters\b", "sections", text)
    text = re.sub(r"\bchapter\b", "section", text)

    # 3. restore the input paths that step 2 rewrote
    text = text.replace("\\input{sections/", "\\input{chapters/")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check the transform reproduces the existing Doc/final/chapters "
                         "from the committed Doc/chapters, then exit")
    args = ap.parse_args()

    if args.verify:
        ok = True
        for name in sorted(os.listdir(DST)):
            if not name.endswith(".tex"):
                continue
            try:
                src = subprocess.run(
                    ["git", "show", f"HEAD:./{SRC}/{name}"],
                    capture_output=True, text=True, check=True).stdout
            except subprocess.CalledProcessError:
                print(f"  SKIP {name} (not in HEAD; verification incomplete)")
                ok = False
                continue
            got = transform(src)
            want = open(os.path.join(DST, name), encoding="utf-8").read()
            status = "OK  " if got == want else "DIFF"
            if got != want:
                ok = False
            print(f"  {status} {name}")
        print("VERIFY:", "transform is exact" if ok else "transform does NOT reproduce Doc/final")
        return 0 if ok else 1

    os.makedirs(DST, exist_ok=True)
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".tex"):
            continue
        src = open(os.path.join(SRC, name), encoding="utf-8").read()
        with open(os.path.join(DST, name), "w", encoding="utf-8") as f:
            f.write(transform(src))
        print(f"  wrote {DST}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
