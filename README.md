# The Spacetime–Electromagnetic Framework, Part II

**Field Dynamics, Clifford Transport and a One-Family Pati–Salam Construction** — Driss Bencheikh, Independent Researcher.

**Version 1.0.0 — Preprint** (24 November 2026).

**DOI of version 1.0.0:** [10.5281/zenodo.24517088](https://doi.org/10.5281/zenodo.24517088) (Zenodo record).

**Repository:** https://github.com/driss-bencheikh/The-Spacetime-Electromagnetic-Framework-Part-2

This archive contains the manuscript of Part II, its LaTeX source, and the computational verification package that checks its statements.
Part II continues Part I, *The Spacetime–Electromagnetic Framework: Mathematical Structures and Obstructions in Split-Signature Clifford Geometry* (https://github.com/driss-bencheikh/spacetime-electromagnetic-framework), which it cites by the printed numbers of its statements.

- **PDF:** [`release/part2-v1.0.0.pdf`](release/part2-v1.0.0.pdf) (73 pages; checksum in `release/SHA256SUMS`)
- **Source:** [`manuscript/main.tex`](manuscript/main.tex) (ten chapters and Appendix A; bibliography `manuscript/references.bib`)
- **Verification:** [`verification/`](verification/) (16 scripts, 4 helper modules, 312 checks)
- **Run record:** [`reproducibility/run-v1.0.0/`](reproducibility/run-v1.0.0/)

## What Part II does

Part I gives two inequivalent readings of the same real six-dimensional space R<sup>3,3</sup>, as a displacement and as an electromagnetic
field strength, with different Lorentz actions. Part II has three parts, as its title says.

- **Field dynamics (Chapters 2–6).** Replacing the constant shift of Part I's interior dynamics by G∘curl recovers Maxwell evolution in the
  zero-current sector when Gauss's laws hold, and the six-component momentum becomes the pair of spatial currents. On a constant-rate
  background a massive two-form sector has positive canonical energy; coupled to a connection, its tree amplitudes grow with energy, and an
  so(3) Yang–Mills–Higgs completion cancels the leading growth in the fixed-angle tree amplitudes examined.
- **Clifford transport (Chapters 7–8).** A transport map in the split Clifford algebra Cl(3n,3n) replaces the clock generator of a fixed
  Lorentz four-plane by the volume element of the full algebra. At n = 1 the transported temporal algebra is the whole commutant of the
  Lorentz algebra, and left multiplication by its transported volume implements electromagnetic duality on field-strength bivectors. At
  general n, with the spectator Lorentz action, the commutant is Cl(3n,3n−3) and the transported rotations form so(3n).
- **One-family construction (Chapters 8–9).** At n = 2, gauging these rotations together with the two weak factors supplied by the spatial
  Clifford algebra gives one chiral Pati–Salam family, with a right-handed neutrino, on the full 64-dimensional spinor module, with no
  projector. For the chosen Standard-Model embedding the invariant mass kernels form one complex Majorana line; with the bidoublet
  prescribed by the transported volume, a value on it leaves su(3) ⊕ u(1)<sub>Q</sub> unbroken. A three-parameter choice of neutral Dirac
  kernels gives a linear relation among four signed Dirac entries; an allowed invariant quartic has a nonzero gradient transverse to that
  choice, computed in closed form, so the imposed symmetries do not enforce the relation.

## What it does not claim

**It is not a completed unified physical theory.** The transport rules (A1)–(A4) are adopted, not derived. The vacuum, the relative
alignment of its condensates, the mass scales and the couplings are inputs; the value sin²θ<sub>W</sub> = 3/8 is a matching value under
equal couplings at an unspecified scale; the masses of Example 9.17 are fitted, not derived; the scattering results are tree-level, at the
channels and fixed angles examined; the first-rung model with a single fermion doublet has the global SU(2) anomaly; the number of light
families and the mixings are open. No empirical prediction is made. The matter content of the Pati–Salam family is established physics:
Section 1.6 places the construction among earlier ones, in particular Krasnov's, and states what this Part adds. Chapter 10 collects the
assumptions, the obstructions and the open problems.

## Layout

| path | content |
|---|---|
| `manuscript/` | `main.tex`, `preamble.tex`, `chapters/` (Chapters 1–10 and Appendix A), `references.bib` |
| `release/` | the PDF of version 1.0.0 and its SHA-256 checksum |
| `verification/` | the 16 scripts `ch*.py`, the helper modules `p2*.py`, the runner `run_all.py`, `make_statements.py`, the inventories `SCRIPTS.tsv`, `STATEMENTS.tsv` and `PART1_STATEMENTS.tsv`, and `requirements.txt` |
| `reproducibility/` | `run-v1.0.0/`: the output of every script of the recorded run, `summary.tsv` and `environment.txt` |

## Building the PDF

The PDF is built with pdfLaTeX and BibTeX (TeX Live, with the packages loaded in `manuscript/preamble.tex`: geometry, lmodern, amsmath,
amssymb, amsthm, mathtools, booktabs, array, enumitem, xcolor, TikZ, hyperref and cleveref):

```
cd manuscript
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The release PDF was built with pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian).

## Running the checks

The checks need Python 3 with NumPy, SciPy and SymPy (`verification/requirements.txt`):

```
cd verification
python3 run_all.py --out ../reproducibility/my-run
```

Each script prints one line `PASS <id>: <claim>` or `FAIL <id>: <claim>` per check and exits with status 0 only if all pass; the runner
runs every script in its own process and writes the output of each, `summary.tsv` and `environment.txt`. A single script runs alone, for
instance `python3 ch09b_potential.py`. The recorded run of this version, made inside this archive, gives 312 checks, all passing, in
about 5.1 minutes on one machine (Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0). `SCRIPTS.tsv` lists the scripts with their checks and the chapter each serves;
`STATEMENTS.tsv` maps each numbered statement to the checks that close it (`python3 make_statements.py` regenerates it after a LaTeX build);
`PART1_STATEMENTS.tsv` lists the statements of Part I that are used. Appendix A of the manuscript states what a check establishes and what
it does not.

## Use of AI assistance

The manuscript states in Section 1.8 how AI systems were used and how their output was checked.

## Licences

The manuscript, text, results and documentation are released under CC BY 4.0, the verification software under the MIT licence;
[`LICENSE.md`](LICENSE.md) states which files each licence covers.

## Citation

Driss Bencheikh, *The Spacetime–Electromagnetic Framework, Part II. Field Dynamics, Clifford Transport and a One-Family Pati–Salam Construction*, version 1.0.0, preprint, 2026, doi:10.5281/zenodo.24517088.
Machine-readable metadata: [`CITATION.cff`](CITATION.cff).
