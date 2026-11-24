#!/usr/bin/env python3
"""Run the verification scripts of Part II and record the results.

    python3 run_all.py [--out ../results] [--only ch02_shift_as_curl ...]

Runs every ch*.py of this directory, each in its own process and from this directory. For each script it writes <out>/<stem>.txt
(stdout and stderr) and one line of <out>/summary.tsv: script, exit status, PASS and FAIL counts, lines that are neither, wall time, and
whether the exit status agrees with the counts (exit 0 exactly when there is at least one PASS and no FAIL). It also writes
<out>/environment.txt. The runner reports; it does not judge what a PASS establishes.
"""
import argparse
import pathlib
import platform
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results"))
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(HERE.glob("ch*.py")):
        if args.only and f.stem not in args.only:
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, f.name], cwd=HERE, capture_output=True, text=True)
        dt = time.time() - t0
        (out / f"{f.stem}.txt").write_text(p.stdout + (("\n" + p.stderr) if p.stderr else ""))
        lines = [l for l in p.stdout.splitlines() if l.strip()]
        n_pass = sum(l.startswith("PASS ") for l in lines)
        n_fail = sum(l.startswith("FAIL ") for l in lines)
        other = len(lines) - n_pass - n_fail
        rows.append((f.name, p.returncode, n_pass, n_fail, other, round(dt, 1), (p.returncode == 0) == (n_pass > 0 and n_fail == 0)))
        print(f"{f.name:28s} exit {p.returncode}  PASS {n_pass:3d}  FAIL {n_fail}  {dt:6.1f} s")
    with open(out / "summary.tsv", "w") as fh:
        fh.write("script\texit\tpass\tfail\tother_lines\tseconds\tconsistent\n")
        for r in rows:
            fh.write("\t".join(str(v) for v in r) + "\n")
        fh.write(f"total\t-\t{sum(r[2] for r in rows)}\t{sum(r[3] for r in rows)}\t-\t{round(sum(r[5] for r in rows), 1)}\t{all(r[6] for r in rows)}\n")
    import numpy, scipy, sympy
    (out / "environment.txt").write_text(
        f"python {platform.python_version()} ({platform.platform()})\nnumpy {numpy.__version__}\nscipy {scipy.__version__}\nsympy {sympy.__version__}\n")
    print(f"total: PASS {sum(r[2] for r in rows)}, FAIL {sum(r[3] for r in rows)}")
    sys.exit(0 if all(r[1] == 0 for r in rows) else 1)


if __name__ == "__main__":
    main()
