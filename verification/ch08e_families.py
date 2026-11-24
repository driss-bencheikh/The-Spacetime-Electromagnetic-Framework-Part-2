#!/usr/bin/env python3
"""Part II, Chapter 8 -- the third rung as a family index.

Two ways of reading the third rung, R^512 = R^64 (x) R^8:
  (a) the interaction of the second rung held on the first two constituents: su(4) + su(2) + su(2) built with Gamma_2 = E1...E12, so
      that the third constituent only adds the index R^8;
  (b) the hypothesis applied at the third rung: the same algebras built with Gamma_3 = E1...E18.
A family is (4,2,1) + (4bar,1,2) on the left-handed fields (Omega = +i), a mirror family (4,1,2) + (4bar,2,1): they are told apart by
the product of -i G' (G' the volume of (Gamma; P), which separates the 4 from the 4bar) and w_N (which separates the two su(2)): -1 on
a family, +1 on a mirror, as at the second rung.

    python3 ch08e_families.py

All checks are numerical linear algebra on integer-valued 512 x 512 matrices; multiplicities are counted from eigenvalues rounded
after a bound on the error.
"""
import itertools
from collections import Counter

import numpy as np

from p2common import report, finish, tower, prod, comm, acomm, zero

E = tower(3)
d = 512
I = np.eye(d)
Om = prod([E[0], E[3], E[4], E[5]], d)
lint = [E[a] @ E[b] for a, b in itertools.combinations([0, 3, 4, 5], 2)]
P2 = [E[1], E[2], E[6], E[7], E[8]]
N = [E[9], E[10], E[11]]
wN = N[0] @ N[1] @ N[2]
Sg = [0.5 * N[1] @ N[2], 0.5 * N[2] @ N[0], 0.5 * N[0] @ N[1]]
s2 = [0.5 * n for n in N] + Sg
w_, V_ = np.linalg.eig(Om.astype(complex))
Lq, _ = np.linalg.qr(V_[:, np.abs(w_ - 1j) < 1e-9])           # left-handed fields: Omega = +i, complex dimension 256 = 128 Weyl x 2


def R(X):
    return Lq.conj().T @ X @ Lq


e3 = E[12:18]                                                  # the third constituent
poles = [e3[i] @ e3[i + 3] for i in range(3)]
Nop = sum(0.5 * (I - p) for p in poles)                         # its Fock number
Dd = [0.5 * (e3[1] @ e3[2] - e3[4] @ e3[5]), 0.5 * (e3[2] @ e3[0] - e3[5] @ e3[3]), 0.5 * (e3[0] @ e3[1] - e3[3] @ e3[4])]   # its diagonal rotations
wNn, VNn = np.linalg.eigh((Nop + Nop.T) / 2)
Wn = [VNn[:, np.abs(wNn - k) < 1e-9] @ VNn[:, np.abs(wNn - k) < 1e-9].T for k in range(4)]   # projectors onto N = 0, 1, 2, 3


def classes(Gc, W=None):
    """complex dimensions of the family class (product -1) and the mirror class (product +1) on the left-handed fields"""
    Gp = Gc @ prod(P2, d)
    M = R(-1j * Gp) @ R(wN)
    if W is not None:
        M = R(W) @ M @ R(W)
    ev = np.linalg.eigvals(M)
    err = np.min(np.abs(np.subtract.outer(ev, [-1, 0, 1])), axis=1).max()
    return Counter(int(round(x.real)) for x in ev), err


ok = True
Gam2, Gam3 = prod(E[:12], d), prod(E, d)
g_a = [0.5 * X @ Y for X, Y in itertools.combinations([Gam2] + P2, 2)]
ok = ok and all(zero(comm(t, l)) for t in g_a + s2 for l in lint)
cl, err = classes(Gam2)
ok = ok and cl == Counter({-1: 256}) and err < 1e-9
report("fm-1", ok,
       "(numerical) with the interaction of the second rung held on the first two constituents (its algebras built with Gamma_2), the algebra commutes with l_int "
       "and the 128 left-handed Weyl fields of R^512 are eight copies of the family (4,2,1) + (4bar,1,2), with no mirror family")

ok = all(zero(comm(x, t)) for x in Dd + [Nop] for t in g_a + s2 + lint + [Om]) and all(zero(comm(Nop, x)) for x in Dd)
ok = ok and (zero(comm(Dd[0], Dd[1]) - Dd[2]) or zero(comm(Dd[0], Dd[1]) + Dd[2]))
Cd = sum(R(x) @ R(x) for x in Dd)
counts, spin_ok = [], True
for k in range(4):
    Pk = R(Wn[k])
    dimk = int(round(np.trace(Pk).real))
    counts.append(dimk // 32)                                   # copies of the family (32 complex dimensions each)
    C = Pk @ Cd @ Pk
    ev = np.linalg.eigvalsh((C + C.conj().T) / 2)
    nz = ev[np.abs(ev) > 1e-9]                                  # the Casimir on the weight-k fields (zero outside them)
    spin_ok = spin_ok and (np.allclose(nz, -2.0, atol=1e-9) and len(nz) == dimk if k in (1, 2) else len(nz) == 0)
ok = ok and counts == [1, 3, 3, 1] and spin_ok
report("fm-2", ok,
       "(numerical) the Fock number N = sum (1 - e_i e_{i+3})/2 of the third constituent and its diagonal rotations (e_i e_j - e_{i+3} e_{j+3})/2, which close as so(3), "
       "commute with the held interaction, l_int and Omega; the eight copies are 1 + 3 + 3 + 1 at N = 0, 1, 2, 3, spin one at N = 1 and 2 (Casimir -2) and spin zero "
       "at N = 0 and 3: the triplets are the octants of weight one and two")

ok = True
g_b = [0.5 * X @ Y for X, Y in itertools.combinations([Gam3] + P2, 2)]
ok = ok and all(zero(comm(t, l)) for t in g_b + s2 for l in lint)
cl, err = classes(Gam3)
ok = ok and cl == Counter({-1: 128, 1: 128}) and err < 1e-9
byweight = []
for k in range(4):
    c_, e_ = classes(Gam3, Wn[k])
    byweight.append((c_.get(-1, 0), c_.get(1, 0)))
    err = max(err, e_)
ok = ok and byweight == [(0, 32), (96, 0), (0, 96), (32, 0)] and err < 1e-9
vol3 = prod(e3, d)
par = poles[0] @ poles[1] @ poles[2]                            # (-1)^N = K_1 K_2 K_3, basis independent
ok = ok and zero(vol3 + par) and all(zero(Wn[k] @ par - (-1)**k * Wn[k], 1e-9) for k in range(4))
report("fm-3", ok,
       "(numerical) restricted to the embedded Pati-Salam algebra built with Gamma_3 (the rotations of (Gamma_3; a, b, c, d, e) and s_2), the 128 fields are four families "
       "and four mirror families: the families at N = 1 and 3, the mirrors at N = 0 and 2; the Fock parity is (-1)^N = K_1 K_2 K_3 (exact), minus the ordered volume of the "
       "third constituent")
# ---- the full third-rung algebra mixes the Fock sectors, and the Weyl multiplicities form the irreducible (16, 8)
Yc = 0.5 * E[1] @ E[12]                                         # E2 E13/2, a generator of g_3
P3 = [E[i] for i in (1, 2, 6, 7, 8, 12, 13, 14)]
g3 = [0.5 * X @ Y for X, Y in itertools.combinations([Gam3] + P3, 2)]
ok = any(zero(Yc - t) or zero(Yc + t) for t in g3)
ok = ok and zero(acomm(Yc, poles[0])) and zero(comm(Yc, poles[1])) and zero(comm(Yc, poles[2])) and zero(comm(Nop, Yc) - 0.5 * E[1] @ E[15])
ok = ok and zero(Yc @ par + par @ Yc)
NN = [E[i] for i in (9, 10, 11, 15, 16, 17)]
blN = [prod([NN[i] for i in S], d) for r in range(7) for S in itertools.combinations(range(6), r)]
s3 = [b_ for b_ in blN if zero(b_ + b_.T)]
basis = blN + [Om @ b_ for b_ in blN]                           # C_Omega (x) Cl(0,6), the commutant of g_3 + l_int
rng = np.random.default_rng(5)
idx = rng.choice(d * d, 4000, replace=False)
Mx = np.vstack([np.array([comm(t, b_).ravel()[idx] for b_ in basis]).T for t in s3])
cdim = len(basis) - np.linalg.matrix_rank(Mx, tol=1e-8)
ok = ok and len(s3) == 28 and cdim == 2
report("fm-4", ok,
       f"the generator Y = E2 E13/2 of g_3 anticommutes with K_1 and commutes with K_2, K_3, so [N, Y] = E2 E16/2 (exact) and Y reverses the Fock parity: g_3 does not keep "
       f"the Fock grading; inside the commutant C_Omega (x) Cl(0,6) of g_3 + l_int, the elements commuting with s_3 = so(8) span a space of dimension {cdim} (numerical rank "
       "on 4000 sampled entries), C_Omega: under so(9) + so(8) the Weyl multiplicities form the irreducible (16, 8)")
finish()
