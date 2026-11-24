#!/usr/bin/env python3
"""Write STATEMENTS.tsv: each numbered statement of the Part II manuscript with the checks that close it.

    python3 make_statements.py [--tex ../manuscript]

Reads the chapter sources for the theorem-like environments and the \\checks lines, and the .aux files of the last build for the
printed numbers.  A \\checks line inside a proof is attributed to the statement that the proof follows; a \\checks line outside any
statement is listed under its section.  Run after a LaTeX build of the manuscript.
"""
import argparse
import pathlib
import re

ENVS = ("theorem", "proposition", "lemma", "corollary", "definition", "hypothesis", "example", "remark")


def labels(texdir):
    out = {}
    for aux in sorted(texdir.glob("chapters/*.aux")):
        for m in re.finditer(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}\{([^}]*)\}", aux.read_text()):
            out[m.group(1)] = m.group(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(pathlib.Path(__file__).resolve().parent.parent / "manuscript"))
    args = ap.parse_args()
    texdir = pathlib.Path(args.tex)
    num = labels(texdir)
    rows = []
    for f in sorted(texdir.glob("chapters/*.tex")):
        src = f.read_text()
        section, current, last = "", None, None
        tok = re.compile(r"\\(section|begin|end)\{([^}]*)\}(\[[^\]]*\])?|\\label\{([^}]*)\}|\\checks\{")
        pos = 0
        while True:
            m = tok.search(src, pos)
            if not m:
                break
            pos = m.end()
            if m.group(1) == "section":
                section = m.group(2)
            elif m.group(1) == "begin" and m.group(2) in ENVS:
                title = (m.group(3) or "")[1:-1]
                current = {"kind": m.group(2), "title": title, "label": "", "checks": []}
            elif m.group(1) == "end" and m.group(2) in ENVS and current:
                rows.append(current)
                last, current = current, None
            elif m.group(4) and current and not current["label"]:
                current["label"] = m.group(4)
            elif m.group(0).startswith("\\checks"):
                depth, i = 1, pos
                while depth:
                    depth += {"{": 1, "}": -1}.get(src[i], 0)
                    i += 1
                text = " ".join(src[pos:i - 1].split())
                pos = i
                if current:
                    current["checks"].append(text)
                else:
                    before = src[:m.start()]
                    in_proof = before.rfind("\\begin{proof}") > before.rfind("\\end{proof}")
                    if in_proof and last:
                        last["checks"].append(text)
                    else:
                        rows.append({"kind": "paragraph", "title": section, "label": "", "checks": [text]})
    out = pathlib.Path(__file__).resolve().parent / "STATEMENTS.tsv"
    with open(out, "w") as fh:
        fh.write("number\tkind\ttitle\tlabel\tchecks\n")
        for r in rows:
            if r["kind"] in ("remark", "definition", "hypothesis") and not r["checks"]:
                continue
            fh.write("\t".join([num.get(r["label"], "-"), r["kind"], r["title"], r["label"] or "-", "; ".join(r["checks"]) or "-"]) + "\n")
    print(f"wrote {out} ({len(rows)} entries scanned)")


if __name__ == "__main__":
    main()
