#!/usr/bin/env python3
"""Part II, Chapter 8 -- the second rung in detail: triples (a, b, X), the vacuum family of Part I, and what the hypothesis's so(6) does to them.

Cl(6,6) on R^64. a = E2, b = E3 (surplus pair of constituent 1); c, d, e = E7, E8, E9 (temporal triple of constituent 2).
The hypothesis's algebra is g_2 = span{XY/2, X Omega/2 : X, Y in P}, P = {a, b, c, d, e}  (ch08_every_rung.py).

    python3 ch08b_second_rung.py

Exact (small integers and dyadic rationals held as floats; every product exact) except sr-9 (numerical, stated tolerance).
"""
import itertools

import numpy as np

from p2common import report, finish, tower, prod, comm, acomm, zero

E = tower(2)
dim = 64
I = np.eye(dim)
g0, ga = E[0], [E[3], E[4], E[5]]
rot = [0.5 * ga[1] @ ga[2], 0.5 * ga[2] @ ga[0], 0.5 * ga[0] @ ga[1]]
boost = [0.5 * g0 @ ga[i] for i in range(3)]
lint = rot + boost
Kk = [-I] + [-g0 @ ga[i] for i in range(3)]
wS = ga[0] @ ga[1] @ ga[2]
Om = g0 @ wS
a, b, c, d, e = E[1], E[2], E[6], E[7], E[8]
P = [a, b, c, d, e]
Q = 0.5 * a @ b
G1, G12 = prod(E[:6], dim), prod(E, dim)
T1 = E[0] @ E[1] @ E[2]
gens = [0.5 * X @ Y for X, Y in itertools.combinations(P, 2)] + [0.5 * X @ Om for X in P]
eq = lambda A_, B_: zero(A_ - B_) or zero(A_ + B_)
ingens = lambda Y: any(eq(Y, q) for q in gens)

cands = {"E1": E[0], "E7": E[6], "E8": E[7], "E9": E[8], "Gamma_1": G1, "Gamma": G12, "E10": E[9]}
table = {}
for nm, X in cands.items():
    alg = [Q, 0.5 * b @ X, 0.5 * X @ a]
    table[nm] = (bool(zero(X @ X - I) and zero(X - X.T) and zero(acomm(X, a)) and zero(acomm(X, b))), bool(all(zero(Y + Y.T) for Y in alg)),
                 bool(all(zero(comm(Y, L)) for Y in alg for L in lint)), bool(all(zero(Y.T @ K + K @ Y) for Y in alg for K in Kk)))
good = [nm for nm, tup in table.items() if all(tup)]
report("sr-1", good == ["E7", "E8", "E9", "Gamma_1", "Gamma"] and table["E1"] == (True, True, False, False) and table["E10"][:2] == (False, False),
       "third member X of a triple (a, b, X), rotation algebra span{ab, bX, Xa}/2: X = E7, E8, E9, Gamma_1, Gamma are symmetric involutions whose algebra is compact, commutes with the interior Lorentz algebra and keeps the kinetic forms; "
       "the own clock E1 fails the last two; the spatial E10 is not symmetric and its algebra is not compact")
for nm in ("E7", "Gamma_1"):
    X = cands[nm]
    alg = [Q, 0.5 * b @ X, 0.5 * X @ a]
    Hn = lambda v: v[0] * X + v[1] * a + v[2] * b
    n = np.array([1.0, 2.0, 2.0])
    H = Hn(n)
    U = (3 * I + 8 * alg[1]) @ (3 * I + 8 * alg[2])                                       # U U^T = 625
    M2 = U @ H @ U.T
    c2 = np.array([np.trace(M2 @ t) / dim for t in (X, a, b)])
    ok = zero(sum(Y @ Y for Y in alg) + 0.75 * I) and zero(H @ H - 9 * I) and abs(np.trace((3 * I + H) / 6) - 32) < 1e-9 and all(zero(comm(H, L)) for L in lint) \
        and zero(U @ U.T - 625 * I) and zero(M2 - Hn(c2)) and abs(c2 @ c2 - 625**2 * 9) < 0.5
    report(f"sr-2-{nm}", ok, f"X = {nm}: Casimir -3/4 on all of R^64; H(n) = n1 X + n2 a + n3 b has H^2 = |n|^2, (1 + H/|n|)/2 a projector of rank 32, Lorentz invariant, and transforms as a vector of the algebra")
    Mk, M0, Pi0 = H @ wS, X @ wS, 0.5 * (I + X)
    tang = [comm(alg[1], Pi0), comm(alg[2], Pi0)]
    mp = [float(np.trace(comm(Y, Pi0) @ comm(Y, Pi0))) for Y in alg]
    ok = zero(Mk + Mk.T) and all(zero(comm(Mk, r)) for r in rot) and all(zero(acomm(Mk, bb)) for bb in boost) and zero(comm(M0, Q)) and zero(comm(Q, Pi0)) \
        and all(zero(t - t.T) for t in tang) and all(not zero(comm(Q, t)) for t in tang) and abs(np.trace(tang[0] @ tang[1])) < 1e-9 and mp == [0.0, 16.0, 16.0] \
        and [bool(zero(Y.T @ T1 + T1 @ Y)) for Y in alg] == [True, False, False]
    report(f"sr-3-{nm}", ok, f"X = {nm}: the kernel H(n) w_S is antisymmetric and Lorentz invariant and commutes with the charge at the pole; the tangents at the pole are symmetric, charged and orthogonal; mass pattern (0, 16, 16); only the charge keeps T")
alg7 = [Q, 0.5 * b @ E[6], 0.5 * E[6] @ a]
algG = [Q, 0.5 * b @ G1, 0.5 * G1 @ a]
report("sr-4", all(ingens(Y) for Y in alg7) and all(ingens(Y) for Y in algG) and eq(algG[1], 0.5 * a @ Om) and eq(algG[2], 0.5 * b @ Om),
       "both algebras lie in the hypothesis's so(6): the triple (a, b, E7) rotates three of the five directions of P (inside the unbroken so(5)); for (a, b, Gamma_1), b Gamma_1 = +-a Omega and Gamma_1 a = +-b Omega: it is the first-rung algebra span{J, a Omega, b Omega}/2")
# the vacuum family of Part I (Def. 8.56): Pi(n) = (1 + I5 n)/2, I5 = abcde
I5 = a @ b @ c @ d @ e
V = [Om @ prod([a, b] + E[6:12], dim)] + [X @ prod([a, b] + E[6:12], dim) for X in P]
om6 = prod(V, dim)
sgn = [1 if zero(I5 @ X - om6 @ X @ Om) else (-1 if zero(I5 @ X + om6 @ X @ Om) else 0) for X in P]
report("sr-5", zero(I5 @ e - a @ b @ c @ d) and all(s != 0 for s in sgn) and len(set(sgn)) == 1,
       f"I5 X = {'+' if sgn[0] > 0 else '-'}omega_6 X Omega for every X in P: the family Pi(n) = (1 + I5 n)/2 of Part I is the spectral projector of the coset generator n Omega of the hypothesis's algebra, with omega_6 as the imaginary unit; at n = e it is (1 + abcd)/2")
Pi_e = 0.5 * (I + a @ b @ c @ d)
stab = [Y for Y in gens if zero(comm(Y, Pi_e))]
expect = [0.5 * X @ Y for X, Y in itertools.combinations([a, b, c, d], 2)] + [0.5 * e @ Om]
report("sr-6", len(stab) == 7 and all(any(eq(Y, q) for q in stab) for Y in expect),
       "the stabiliser of (1 + abcd)/2 in so(6) has dimension 7: the six rotations of (a, b, c, d) and e Omega, so(4) + so(2); eight generators are broken")
H5 = lambda v: I5 @ (v[0] * e + v[1] * a + v[2] * b)
alg_e = [Q, 0.5 * b @ e, 0.5 * e @ a]
n = np.array([1.0, 2.0, 2.0])
U = (3 * I + 8 * alg_e[1]) @ (3 * I + 8 * alg_e[2])
M2 = U @ H5(n) @ U.T
c2 = np.array([np.trace(M2 @ I5 @ t) / dim for t in (e, a, b)])
Ta, Tb = comm(0.5 * a @ e, Pi_e), comm(0.5 * b @ e, Pi_e)
ok = all(zero(comm(H5(n), xg)) for xg in [g0] + ga) and zero(M2 - H5(c2)) and abs(c2 @ c2 - 625**2 * 9) < 0.5 \
    and abs(np.trace(Ta @ Ta) - 16) < 1e-9 and abs(np.trace(Ta @ Tb)) < 1e-9 and zero(comm(Q, Ta) + Tb) and zero(comm(Q, Tb) - Ta)
mp = [float(np.trace(comm(Y, Pi_e) @ comm(Y, Pi_e))) for Y in alg_e]
report("sr-7", ok and mp == [0.0, 16.0, 16.0],
       "restricted to span{a, b, e} the family of Part I commutes with every interior gamma matrix and is a vector of the so(3) on (a, b, e); its tangents at the vacuum are the charged scalar of Part I, Thm 8.58 ([Q, T_a] = -T_b, [Q, T_b] = T_a, Tr T^2 = 16); mass pattern (0, 16, 16)")
Pi0 = 0.5 * (I + G12)
unb = [Y for Y in gens if zero(comm(Y, Pi0))]
report("sr-8", len(unb) == 10 and len([Y for Y in gens if zero(Y.T @ T1 + T1 @ Y)]) == 7 and len([Y for Y in stab if zero(comm(Y, Pi0))]) == 6,
       "three vacua, three patterns: the chirality projector keeps so(5) (10 generators); the Dirac mass form T and the projector (1 + abcd)/2 of Part I each keep so(4) + so(2) (7 generators); the last and the first share so(4)")
# a vacuum direction that would keep colour: the complex structure J3 = ab + cd + e Omega on the six directions (clock axis, a, b, c, d, e)
J3 = a @ b + c @ d + e @ Om
cen = [Y for Y in gens if zero(comm(Y, J3))]
# the centraliser need not be spanned by basis generators: compute its dimension exactly from the adjoint action on the 15 generators
adj = np.array([[np.trace(comm(J3, Yi) @ Yj.T) for Yj in gens] for Yi in gens])
rank_ad = np.linalg.matrix_rank(adj)
evJ3 = sorted(set(np.round(np.linalg.eigvals(J3).imag, 6)))
J3b = a @ b + c @ d + e @ G12                                                     # the same in the form of Definition 7.2, with Gamma for Omega
evJ3 = evJ3 if evJ3 == sorted(set(np.round(np.linalg.eigvals(J3b).imag, 6))) else []
six_ = [G12, a, b, c, d, e]
adJ = np.array([[np.trace(six_[i].T @ comm(J3b, six_[j])) / dim for j in range(6)] for i in range(6)])
cs_ = np.linalg.matrix_power(adJ / 2, 2)
S1_ = 0.5 * E[10] @ E[11]                                                         # the T_3L + T_3R part of the electric charge
orthog = max(abs(np.trace(S1_.T @ Y)) for Y in gens) < 1e-12
report("sr-10", 15 - rank_ad == 9 and zero(comm(J3, Q)) and len(cen) == 3 and evJ3 == [-3.0, -1.0, 1.0, 3.0] and np.allclose(cs_, -np.eye(6)) and orthog,
       f"the centraliser in so(6) of J3 = ab + cd + e Omega has dimension 15 - {rank_ad} = 9 = dim u(3): a vacuum along J3 would keep su(3) + u(1); J3 has the eigenvalues "
       "+-i and +-3i on the spinors, as does ab + cd + e Gamma, for which (1/2) ad is a complex structure on the six directions (Gamma; a, b, c, d, e); the charge H_1 = ab/2 "
       "of Part I commutes with J3, "
       "while the electric charge has the component T_3L + T_3R = N_2 N_3/2, orthogonal to so(6), and is not in this u(3)")
# numerical: number of doublets for the triple (a, b, E7)
A_ = np.zeros((dim * dim, dim * dim))
for Y in lint + alg7:
    ad = np.kron(Y, I) - np.kron(I, Y.T)
    A_ += ad.T @ ad
wv = np.linalg.eigvalsh(A_)
dimc = int((wv < 1e-8 * wv.max()).sum())
report("sr-9", dimc == 128, f"(numerical, relative tolerance 1e-8) the commutant of Lorentz and the so(3) of (a, b, E7) in M_64(R) has dimension {dimc} = 128 = dim M_8(C): R^64 is eight Weyl doublets of this so(3)")
finish()
