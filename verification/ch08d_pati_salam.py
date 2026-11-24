#!/usr/bin/env python3
"""Part II, Chapter 8 -- the second rung as a Pati-Salam gauge theory (by (A4) the two su(2) of the spatial directions of the
second constituent are gauged).

(A4), reworded: the spatial triple of the four-plane carries no Yang-Mills connection (its value there is the scalar m); the spatial
directions of the other constituents carry a Yang-Mills connection with values in k_n, the antisymmetric part of their Clifford algebra.
At rung two k_2 = span{N_i/2, N_j N_k/2} = su(2) + su(2), N = (E10, E11, E12), and the gauge algebra is g_2 + k_2 = su(4) + su(2) + su(2).

    python3 ch08d_pati_salam.py

ps-1 and the identities of ps-7: exact (integer matrices held as floats).  The others are numerical linear algebra on integer-valued matrices; dimensions are
integers read off as ranks with the stated tolerances.
"""
import itertools

import numpy as np

from p2common import report, finish, tower, prod, comm, zero


def rank(mats, tol=1e-9):
    return np.linalg.matrix_rank(np.array([m.flatten() for m in mats]), tol=tol)


def orth(mats, tol=1e-9):
    """an orthonormal basis (trace form) of the span of mats"""
    Mx = np.array([m.flatten() for m in mats])
    _, s, Vt = np.linalg.svd(Mx, full_matrices=False)
    r = int((s > tol * s.max()).sum())
    d = int(round(np.sqrt(Mx.shape[1])))
    return [Vt[i].reshape(d, d) for i in range(r)]


def stab_dim(gens, Ms, tol=1e-8):
    """dimension of the subalgebra of span(gens) that commutes with every M in Ms"""
    A = np.array([np.concatenate([comm(t, M).flatten() for M in Ms]) for t in gens]).T
    return len(gens) - np.linalg.matrix_rank(A, tol=tol)


E = tower(2)
dim = 64
I = np.eye(dim)
Om = prod([E[0], E[3], E[4], E[5]], dim)
Gam = prod(E, dim)
wS = E[3] @ E[4] @ E[5]
P = [E[1], E[2], E[6], E[7], E[8]]
N = [E[9], E[10], E[11]]
lor = [E[4] @ E[5], E[5] @ E[3], E[3] @ E[4]] + [E[0] @ E[3 + a] for a in range(3)]
Kk = [-I] + [-E[0] @ E[3 + a] for a in range(3)]
g2 = [0.5 * X @ Y for X, Y in itertools.combinations([Gam] + P, 2)]
wN = N[0] @ N[1] @ N[2]
Sg = [0.5 * N[1] @ N[2], 0.5 * N[2] @ N[0], 0.5 * N[0] @ N[1]]
k2 = [0.5 * n for n in N] + Sg
la = [0.5 * (I + wN) @ s for s in Sg]                  # one su(2)
lb = [0.5 * (I - wN) @ s for s in Sg]                  # the other
su2rel = lambda l: zero(comm(l[0], l[1]) - l[2]) or zero(comm(l[0], l[1]) + l[2])
ok = all(zero(t + t.T) for t in k2) and all(zero(comm(a, b)) for a in k2 for b in g2 + lor) and all(zero(t.T @ K + K @ t) for t in k2 for K in Kk)
ok = ok and rank(k2) == 6 and rank(la + lb) == 6 and rank(k2 + la + lb) == 6 and su2rel(la) and su2rel(lb) and all(zero(comm(x, y)) for x in la for y in lb)
ok = ok and all(zero(wN @ (2 * s) - n) or zero(wN @ (2 * s) + n) for s, n in zip(Sg, N))
report("ps-1", ok and rank(g2 + k2) == 21,
       "k_2, the antisymmetric part of the Clifford algebra of E10, E11, E12, is span{N_i/2, N_j N_k/2} = su(2) + su(2) (the two halves (1 +- w_N)/2 of the rotations of that triple); "
       "it commutes with g_2 and with l_int and keeps the kinetic forms: the gauge algebra g_2 + k_2 = su(4) + su(2) + su(2) has dimension 21")
G = g2 + k2
w, V = np.linalg.eig(Om.astype(complex))
Lq, _ = np.linalg.qr(V[:, np.abs(w - 1j) < 1e-9])                # left-handed fields: Omega = +i (32 complex dimensions = 16 Weyl x 2)
restr = lambda X: Lq.conj().T @ X @ Lq
rng = np.random.default_rng(1)
an, sc = 0.0, 0.0
for _ in range(6):
    Y = sum(rng.normal() * t for t in G)
    Yh = restr(1j * Y)
    an, sc = max(an, abs(np.trace(Yh @ Yh @ Yh))), max(sc, abs(np.trace(Yh @ Yh)))
doub = []
for l in (la, lb):
    C = sum(restr(t) @ restr(t) for t in l)
    ev = np.round(np.linalg.eigvalsh((C + C.conj().T) / 2) * 4) / 4
    doub.append(int((ev == -0.75).sum()) // 4)               # 2 components per doublet x 2 Lorentz components
report("ps-2", an < 1e-10 * sc and doub == [4, 4],
       f"(numerical) on the left-handed fields Tr(Y^3) = 0 for random Y in g_2 + k_2 (largest {an:.1e} against Tr(Y^2) up to {sc:.0f}): no gauge anomaly; "
       f"each su(2) has {doub[0]} and {doub[1]} doublets of left-handed Weyl fields, an even number: no global anomaly")
blades = [prod([([Gam] + P + N)[i] for i in S], dim) for r in range(10) for S in itertools.combinations(range(9), r)]
kern = [b @ wS for b in blades if zero(b @ wS + (b @ wS).T)]
Bf = np.array([M.flatten() for M in kern])
admat = lambda t: Bf @ np.array([comm(t, M).flatten() for M in kern]).T / dim
n_g2 = 272 - np.linalg.matrix_rank(np.vstack([admat(t) for t in g2]), tol=1e-9)
n_all = 272 - np.linalg.matrix_rank(np.vstack([admat(t) for t in G]), tol=1e-9)
report("ps-3", len(kern) == 272 and n_g2 == 8 and n_all == 0,
       f"(numerical) of the 272 Lorentz-invariant mass kernels, {n_g2} are invariant under su(4) and {n_all} under su(4) + su(2) + su(2): with the two su(2) gauged there is no gauge-invariant mass term, "
       "and every fermion mass must come from a field")
H = [0.5 * P[0] @ P[1], 0.5 * P[2] @ P[3], 0.5 * P[4] @ Gam]
BL = H[0] + H[1] + H[2]                                        # 3/2 (B - L)
A = np.array([comm(BL, t).flatten() for t in g2]).T
r = np.linalg.matrix_rank(A, tol=1e-9)
u3 = [sum(c * t for c, t in zip(row, g2)) for row in np.linalg.svd(A)[2][r:]]
su3 = orth([comm(a, b) for a, b in itertools.combinations(u3, 2)])
report("ps-4", len(u3) == 9 and len(su3) == 8 and rank(su3 + [BL]) == 9,
       "the centraliser of B - L = (2/3)(H_1 + H_2 + H_3) in su(4) is u(3), of dimension 9; its derived algebra su(3), of dimension 8, is the colour algebra")


def casimir(gens):
    return Bf @ np.array([sum(comm(t, comm(t, M)) for t in gens).flatten() for M in kern]).T / dim


C6, Ca, Cb = casimir(g2), casimir(la), casimir(lb)
w6, V6 = np.linalg.eigh(C6)
q6 = np.round(w6 * 4) / 4
vecpart = V6[:, q6 == -5.0]
A_su3 = np.vstack([admat(t) for t in su3])
n_vec = vecpart.shape[1] - np.linalg.matrix_rank(A_su3 @ vecpart, tol=1e-8)
hv = [X @ wS for X in [Gam] + P]                               # the Higgs vector of (A3), times w_S
report("ps-5", vecpart.shape[1] == 24 and n_vec == 0 and stab_dim(g2, [hv[0]]) == 10,
       f"(numerical) no nonzero element of the 2(6,1,1) is invariant under su(3) ({n_vec} invariant directions): every vacuum value of the Higgs vector of (A3) breaks colour; "
       "the clock direction keeps so(5), of dimension 10")
# the Pati-Salam chain inside the mass field: step one in the (-9, spin one) part
lam9 = V6[:, q6 == -9.0]
chain = {}
for lab, (gx, gy, Cx) in {"b": (lb, la, Cb), "a": (la, lb, Ca)}.items():
    wx, vx = np.linalg.eigh(lam9.T @ Cx @ lam9)
    sub = lam9 @ vx[:, np.round(wx * 4) / 4 == -2.0]
    Y = gx[0] + (1.0 / 3.0) * BL                               # T_3 of the broken su(2) + (B - L)/2
    Am = np.vstack([admat(t) for t in su3 + gy + [Y]]) @ sub
    rk = np.linalg.matrix_rank(Am, tol=1e-8)
    if sub.shape[1] - rk:
        c = sub @ np.linalg.svd(Am)[2][rk]
        M1 = sum(ci * K for ci, K in zip(c, kern))
        Q = gy[0] + Y                                          # Q = T_3 (unbroken su(2)) + Y
        G2 = Gam @ prod(P, dim)
        Mg = G2 @ wS
        chain[lab] = (sub.shape[1] - rk, stab_dim(G, [M1]), stab_dim(G, [Mg]), stab_dim(G, [M1, Mg]),
                      bool(np.abs(comm(Q, M1)).max() < 1e-8 and zero(comm(Q, Mg))), bool(zero(Mg @ Mg + I)))
ok = len(chain) == 2 and all(v == (2, 12, 18, 9, True, True) for v in chain.values())
report("ps-6", ok,
       "(numerical) the Pati-Salam chain lies in the mass field: in its (10,1,3)-type part there is one complex direction invariant under su(3), the other su(2) and Y = T_3 + (B - L)/2, "
       "with stabiliser of dimension 12 = su(3) + su(2) + u(1); the generator of mass times w_S, G_2 w_S, lies in the (1,2,2) with stabiliser 18 = su(4) + su(2)_V and (G_2 w_S)^2 = -1; "
       "the two together keep dimension 9 = su(3) + u(1), and the electric charge Q commutes with both (for either labelling of the two su(2))")
# ---------------------------------------------------------------- rung three: what the same rule gives
E3 = tower(3)
d3 = 512
I3 = np.eye(d3)
Gam3 = prod(E3, d3)
P3 = [E3[1], E3[2]] + [E3[6 * k + a] for k in (1, 2) for a in range(3)]
N3 = [E3[6 * k + 3 + a] for k in (1, 2) for a in range(3)]
lor3 = [E3[4] @ E3[5], E3[5] @ E3[3], E3[3] @ E3[4]] + [E3[0] @ E3[3 + a] for a in range(3)]
k3 = []
for rr in (1, 2, 5, 6):                                        # blades of N with b^T = -b: degrees d with d(d+1)/2 odd
    for S in itertools.combinations(range(6), rr):
        b = prod([N3[i] for i in S], d3)
        if zero(b + b.T):
            k3.append(b)
g3s = [X @ Y for X, Y in itertools.combinations(P3, 2)] + [X @ Gam3 for X in P3]          # all 36 generators of g_3 (twice)
colour = [x @ y for x, y in itertools.combinations(N3, 2)]
ok = len(k3) == 28 and all(zero(comm(a, b)) for a in k3 for b in g3s + lor3) and rank(k3 + colour) == 28
Bk = np.array([b.flatten() for b in k3])
closed = all(np.abs(comm(a, b).flatten() - Bk.T @ (Bk @ comm(a, b).flatten() / d3)).max() < 1e-9 for a, b in itertools.combinations(k3, 2))
# simplicity: the commutant of the adjoint action on the 28-dimensional algebra is one-dimensional
ad = [np.array([Bk @ comm(a, b).flatten() / d3 for b in k3]).T for a in k3]
M = np.zeros((28 * 28, 28 * 28))
for X in ad:
    T_ = np.kron(X, np.eye(28)) - np.kron(np.eye(28), X.T)
    M += T_.T @ T_
wv = np.linalg.eigvalsh(M)
n_comm = int((wv < 1e-8 * wv.max()).sum())
G3 = Gam3 @ prod(P3, d3)
wS3 = E3[3] @ E3[4] @ E3[5]
Mg3 = G3 @ wS3
inv3 = zero(Mg3 @ Mg3 + I3) and zero(Mg3 + Mg3.T) and all(zero(comm(t, Mg3)) for t in k3 + g3s) and all(zero(comm(Mg3, r)) for r in lor3[:3]) and all(zero(comm(Mg3, b) + 2 * b @ Mg3) for b in lor3[3:])
report("ps-7", ok and closed and n_comm == 1 and inv3,
       "(rung three, the same rule) k_3, the antisymmetric part of the Clifford algebra of the six spatial generators of constituents 2 and 3, has dimension 28, closes, commutes with g_3 and l_int, "
       "contains the fifteen colour bivectors of Part I, and its adjoint representation is irreducible: k_3 = so(8). G_3 w_S is a Lorentz-invariant mass kernel invariant under g_3 + k_3 "
       "with (G_3 w_S)^2 = -1: at rung three a gauge-invariant mass term gives every fermion a mass; the representation is real and admits an invariant mass")
# ---- the anomaly from the weights: all cubic monomials of the five Cartan coordinates vanish over the sixteen left-handed weights
from fractions import Fraction
Hc = [0.5 * P[0] @ P[1], 0.5 * P[2] @ P[3], 0.5 * P[4] @ Gam, la[0], lb[0]]       # H_1, H_2, H_3, T_3L, T_3R: commuting, antisymmetric
okc = all(zero(comm(x_, y_)) for x_ in Hc for y_ in Hc)
gen = sum(c_ * restr(h_) for c_, h_ in zip((1.0, np.sqrt(2), np.sqrt(3), np.sqrt(5), np.sqrt(7)), Hc))
wv, Vv = np.linalg.eig(gen)
wts, err = [], 0.0
for k_ in range(32):
    v_ = Vv[:, k_] / np.linalg.norm(Vv[:, k_])
    raw = [(v_.conj() @ restr(h_) @ v_ / 1j).real for h_ in Hc]
    rnd = [round(2 * x_) / 2 for x_ in raw]
    err = max(err, max(abs(x_ - y_) for x_, y_ in zip(raw, rnd)))
    wts.append(tuple(Fraction(int(round(2 * x_)), 2) for x_ in raw))
cubic = [sum(w_[i] * w_[j] * w_[l] for w_ in wts) for i in range(5) for j in range(i, 5) for l in range(j, 5)]
from collections import Counter as _C
mult = _C(wts)
ok = okc and err < 1e-9 and len(cubic) == 35 and all(c_ == 0 for c_ in cubic) and len(mult) == 16 and set(mult.values()) == {2}
ok = ok and sum(1 for w_ in mult if w_[3] != 0) == 8 and sum(1 for w_ in mult if w_[4] != 0) == 8 and all((w_[3] == 0) != (w_[4] == 0) for w_ in mult)
report("ps-8", ok,
       f"(numerical diagonalisation, weights rounded to halves after a bound {err:.0e}; then exact) the sixteen left-handed weights under (H_1, H_2, H_3, T_3L, T_3R), each "
       "with its two Lorentz components, make all 35 cubic monomials of the five Cartan coordinates sum to zero exactly, so Tr Y^3 = 0 on the whole algebra by conjugacy "
       "into a Cartan subalgebra; each weight has T_3L or T_3R nonzero but not both, eight of each: four doublets for each su(2)")
# ---- ps-9: kernels invariant under g_n + s_n at rungs two and three, and the basis M, M Omega at rung three
Om3 = prod([E3[0], E3[3], E3[4], E3[5]], d3)
bl3 = [prod([N3[i] for i in S], d3) for r_ in range(7) for S in itertools.combinations(range(6), r_)]
cand = [E3[0] @ b for b in bl3] + [E3[0] @ Om3 @ b for b in bl3]                  # gamma^0 C, C a blade of C_Omega x Cl(N)
skew3 = [K for K in cand if zero(K + K.T)]                                        # each blade kernel is symmetric or antisymmetric
okg = all(zero(K + K.T) or zero(K - K.T) for K in cand) and all(zero(comm(t, K)) for t in g3s[:8] for K in skew3)
Vp = np.random.default_rng(3).normal(size=(d3, 8))                                # random probe vectors: [t, K] = 0 iff [t, K] Vp = 0, almost surely
Ap = np.vstack([np.array([(comm(t, K) @ Vp).flatten() for K in skew3]).T for t in k3])
inv_dim = len(skew3) - np.linalg.matrix_rank(Ap, tol=1e-8)
MO = Mg3 @ Om3
okM = (zero(MO + MO.T) and zero(MO @ MO + I3) and zero(Mg3 @ MO + MO @ Mg3) and all(zero(comm(t, MO)) for t in k3 + g3s)
       and np.linalg.matrix_rank(np.array([Mg3.flatten(), MO.flatten()])) == 2)
report("ps-9", len(skew3) == 72 and okg and inv_dim == 2 and okM and n_all == 0,
       f"(rung three numerical, with random probe vectors; the rest exact) of the 128 blade kernels gamma^0 C of the commutant, {len(skew3)} are antisymmetric, and they span the "
       f"g_3-invariant kernels; invariance under s_3 as well leaves dimension {inv_dim}, spanned by M = G_3 w_S and M Omega, both antisymmetric, of square -1 and anticommuting, "
       "so aM + bM Omega has the singular value sqrt(a^2 + b^2) on every mode; at rung two the dimension is 0 (ps-3) and at rung one there are no invariant kernels: (0, 0, 2)")
# ---- ps-10: the weak algebra alone is not chiral
n_s2 = 272 - np.linalg.matrix_rank(np.vstack([admat(t) for t in k2]), tol=1e-9)
report("ps-10", n_s2 == 24 and n_all == 0,
       f"(numerical) the Lorentz-invariant mass kernels invariant under s_2 = su(2) + su(2) alone span {n_s2} real dimensions, the 2 C(4,2) = 12 complex invariant bilinears of "
       "4(2,1) + 4(1,2); with su(4) imposed as well none remain: the chirality is the correlation of the two su(2) with the 4 and the 4bar")
# ---- ps-11: stabilisers of the clock and of J_3, as endomorphisms and as mass kernels
a_, b_, c_, d_, e_ = P
J3 = a_ @ b_ + c_ @ d_ + e_ @ Gam
tab = {nm: (int(stab_dim(g2, [X])), int(stab_dim(k2, [X])), int(stab_dim(G, [X]))) for nm, X in
       (("Gamma", Gam), ("Gamma wS", Gam @ wS), ("J3", J3), ("J3 wS", J3 @ wS))}
diag = [x + y for x, y in zip(la, lb)]
okd = all(zero(comm(t, J3 @ wS)) for t in diag) and stab_dim(la, [J3 @ wS]) == 0 and stab_dim(lb, [J3 @ wS]) == 0
report("ps-11", tab == {"Gamma": (10, 3, 13), "Gamma wS": (10, 6, 16), "J3": (9, 6, 15), "J3 wS": (9, 3, 12)} and okd,
       f"(numerical) stabiliser dimensions (in g_2, in s_2, in g_2 + s_2): {tab}; the weak part of the stabiliser of J_3 w_S is the diagonal su(2)_V, "
       "so the J_3 w_S kernel keeps u(3) + su(2)_V: passing from an endomorphism to its mass kernel changes which weak generators survive")
finish()
