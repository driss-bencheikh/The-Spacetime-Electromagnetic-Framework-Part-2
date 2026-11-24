#!/usr/bin/env python3
"""Part II, Chapter 7 -- the clarified hypothesis at one constituent: the whole Clifford algebra of V^+, and the generator of mass.

The rule: the temporal volume element T = e1 e2 e3 is replaced by -Omega, with J = e2 e3 unchanged; equivalently the clock
e1 = -T J is replaced by the chirality Gamma = Omega J.  sigma: Cl(V^+) -> M_8(R), e1 -> Gamma, e2 -> e2, e3 -> e3.

    python3 ch07b_generator_of_mass.py

Exact: matrices with small integer or dyadic entries held as floats (every product exact); gm-1 uses sympy over the rationals.
"""
import itertools

import numpy as np
import sympy as sp

from p2common import report, finish, cl33, comm, acomm, zero

e = cl33()
I8 = np.eye(8)
g0, ga = e[0], e[3:6]
J = e[1] @ e[2]
T = e[0] @ e[1] @ e[2]
wS = e[3] @ e[4] @ e[5]
Om = e[0] @ wS
Gam = T @ wS
R = [0.5 * ga[1] @ ga[2], 0.5 * ga[2] @ ga[0], 0.5 * ga[0] @ ga[1]]
Bo = [0.5 * g0 @ ga[i] for i in range(3)]
lint = R + Bo
eq = lambda A, B: zero(A - B) or zero(A + B)
rank = lambda mats: np.linalg.matrix_rank(np.array([M.flatten() for M in mats]), tol=1e-9)

# the commutant of the interior Lorentz algebra, exactly
Xs = sp.Matrix(8, 8, sp.symbols("x0:64"))
eqs = []
for L in lint:
    Ls = sp.Matrix(8, 8, [sp.Rational(v).limit_denominator(4) for v in L.flatten()])
    eqs += list(Xs * Ls - Ls * Xs)
sol = sp.solve(eqs, list(Xs), dict=True)[0]
dimc = len(Xs.subs(sol).free_symbols)
tri = [Gam, e[1], e[2]]
words = [I8, Gam, e[1], e[2], e[1] @ e[2], e[2] @ Gam, Gam @ e[1], Gam @ e[1] @ e[2]]
ok = all(zero(acomm(tri[a], tri[b]) - (2 * I8 if a == b else 0)) for a in range(3) for b in range(a, 3))
report("gm-1", dimc == 8 and ok and rank(words) == 8 and all(zero(comm(w, L)) for w in words for L in lint),
       f"the commutant of the interior Lorentz algebra in M_8(R) has dimension {dimc} = 8; (Gamma, e2, e3) satisfy the relations of Cl(3,0) and their eight products span it: "
       "sigma (e1 -> Gamma) is an algebra isomorphism of Cl(V^+) onto the commutant")
report("gm-2", zero(Gam @ e[1] @ e[2] + Om) and zero(Gam - Om @ J) and zero(e[0] + T @ J) and zero(J @ g0 - T),
       "sigma(T) = Gamma e2 e3 = -Omega, Gamma = Omega J, e1 = -T J and J gamma^0 = T: 'T -> -Omega with J fixed' and 'clock -> chirality' are the same rule")
img = [e[1] @ e[2], e[2] @ Gam, Gam @ e[1]]                       # images of e2e3, e3e1, e1e2
report("gm-3", zero(img[0] - J) and zero(img[1] + e[1] @ Om) and zero(img[2] + e[2] @ Om),
       "sigma maps the rotations of V^+ onto the Yang-Mills algebra: e2e3 -> J, e3e1 -> e3 Gamma = -e2 Omega, e1e2 -> Gamma e2 = -e3 Omega")
anti = [w for w in words if zero(w + w.T)]
sym = [w for w in words if zero(w - w.T)]
report("gm-4", len(anti) == 4 and len(sym) == 4 and rank(anti + [Om, J, e[1] @ Om, e[2] @ Om]) == 4 and rank(sym + [I8, Gam, e[1], e[2]]) == 4,
       "the commutant splits by grade: scalars and clock vectors (1, Gamma, e2, e3) are symmetric; rotations and volume (J, e2 Omega, e3 Omega, Omega) are antisymmetric and form u(1)_Omega + su(2)")
# -Omega acts on the field strength of the four-plane as the binding root G
X = np.array([3.0, -1.0, 2.0, 5.0, 2.0, -4.0])
F = lambda v: sum(v[i] * R[i] + v[3 + i] * Bo[i] for i in range(3))
GX = np.concatenate([-X[3:], X[:3]])
ok = all(zero(-Om @ R[i] - Bo[i]) and zero(-Om @ Bo[i] + R[i]) for i in range(3)) and zero(-Om @ F(X) - F(GX))
rot = 3 * I8 + 4 * Om                                             # 5 exp(theta Omega) with cos theta = 3/5, sin theta = 4/5: integer entries
report("gm-5", ok and zero(rot @ F(X) - F(3 * X - 4 * GX)),
       "on F_X = sum B_i R_i + E_i Boost_i, left multiplication by sigma(T) = -Omega is X -> GX, and exp(theta Omega) F_X = F_{exp(-theta G) X}: "
       "the image of the Dirac rest rotation exp(-m lambda T) is the binding rotation exp(-kappa G tau) with kappa tau = m lambda")
so3p = [e[1] @ e[2], e[2] @ e[0], e[0] @ e[1]]
report("gm-6", all(zero(comm(Y, T)) for Y in so3p) and all(zero(comm(Y, Om)) for Y in img) and [bool(zero(comm(Y, T))) for Y in img] == [True, False, False]
       and [bool(all(zero(Y.T @ K + K @ Y) for K in [-I8] + [-g0 @ ga[a] for a in range(3)])) for Y in so3p] == [True, False, False],
       "displacement reading: the rotations of V^+ commute with T and only J keeps the kinetic forms; field reading: their images commute with -Omega and with the Lorentz algebra, and only J commutes with T")
six = [e[0] @ e[1], e[0] @ e[2], T, e[1] @ wS, e[2] @ wS, J @ wS]
cand = [g0 @ c for c in (J, e[1], e[2], e[1] @ Om, e[2] @ Om, Gam)]
# kernels invariant under the Yang-Mills algebra: solve [M, Y] = 0 inside the six-dimensional kernel space
A = np.array([np.concatenate([comm(M, Y).flatten() for Y in img]) for M in six]).T
ninv = 6 - np.linalg.matrix_rank(A, tol=1e-9)
report("gm-7", rank(six + cand) == 6 and all(zero(c + c.T) for c in cand) and zero(g0 - g0.T) and zero(g0 @ Om - (g0 @ Om).T) and ninv == 0,
       f"the kernel space of Part I, Lemma 7.11, is gamma^0 times span{{J, e2, e3, e2 Omega, e3 Omega, Gamma}}; the subspace invariant under the Yang-Mills algebra has dimension {ninv}: every mass kernel breaks it")
kap = 7.0
H = kap * Gam
report("gm-8", zero(H @ H - kap**2 * I8) and zero(H @ wS - kap * T) and zero(Gam @ wS - T),
       "the Higgs vector as the image of the clock vector with its rate as length: sigma(kappa e1) = kappa Gamma, of square kappa^2, and its kernel kappa Gamma w_S is kappa T: the Dirac mass kernel with m = kappa")
ch = 3 * I8 + 4 * Om                                              # 5 exp(xi Omega)
ch2 = -7 * I8 + 24 * Om                                           # 25 exp(2 xi Omega)
report("gm-9", zero(ch.T @ T @ ch - T @ ch2) and zero(acomm(T, Om)) and all(zero(comm(Om, g0 @ ga[a])) for a in range(3)),
       "the rotation generated by Omega preserves the kinetic forms and sends the kernel T to T exp(2 xi Omega) (Part I, Prop. 7.2(iii)): the kernel has charge two under the generator of mass")
# ---- left multiplication, not conjugation: exp(theta Omega) commutes with the bivectors of the four-plane
rinv = 3 * I8 - 4 * Om                                            # 5 exp(-theta Omega)
report("gm-10", zero(rot @ F(X) @ rinv - 25 * F(X)) and all(zero(comm(Om, R[i])) and zero(comm(Om, Bo[i])) for i in range(3)),
       "conjugation by exp(theta Omega) leaves F_X unchanged, since Omega commutes with the six bivectors of the four-plane: the duality rotation of gm-5 is left "
       "multiplication, and the matching of the angles kappa tau = m lambda is kinematic")
finish()
