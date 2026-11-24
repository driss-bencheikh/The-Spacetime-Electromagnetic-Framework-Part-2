#!/usr/bin/env python3
"""Part II, Chapter 8 -- the clarified hypothesis at rung n: the clock is replaced by the chirality in the whole Clifford algebra of the
temporal directions.

sigma_n: Cl(V^+_n) -> M(8^n, R),  E1 -> Gamma_n,  X -> X for the internal positive generators X in P.
  g_n   = sigma_n(rotations) = span{XY/2, X Gamma/2}                       (Yang-Mills algebra)
  G_n   = sigma_n(T_n) = Gamma (product of P)                              (image of the temporal volume: the generator of mass, test)
  N     = the 3n-3 spatial generators of the other constituents            (they generate the commutant, with Omega)

    python3 ch08c_clifford_form.py

Exact (small integers and dyadic rationals held as floats; every product exact) except cf-2i and cf-2k (numerical, stated tolerances).
"""
import itertools

import numpy as np

from p2common import report, finish, tower, prod, comm, acomm, zero

eq = lambda A, B: zero(A - B) or zero(A + B)
sgn = lambda A, B: "+" if zero(A - B) else ("-" if zero(A + B) else "?")


def cent_dim(gens, dim):
    A = np.zeros((dim * dim, dim * dim))
    I = np.eye(dim)
    for Y in gens:
        ad = np.kron(Y, I) - np.kron(I, Y.T)
        A += ad.T @ ad
    w = np.linalg.eigvalsh(A)
    return int((w < 1e-8 * w.max()).sum())


for n in (1, 2, 3):
    E = tower(n)
    dim = 8 ** n
    I = np.eye(dim)
    g0, wS = E[0], E[3] @ E[4] @ E[5]
    Om = g0 @ wS
    Gam = prod(E, dim)
    rot = [E[4] @ E[5], E[5] @ E[3], E[3] @ E[4]]
    boost = [E[0] @ E[3 + a] for a in range(3)]
    lor = rot + boost
    P = [E[1], E[2]] + [E[6 * k + a] for k in range(1, n) for a in range(3)]
    N = [E[6 * k + 3 + a] for k in range(1, n) for a in range(3)]
    prodP, wN = prod(P, dim), prod(N, dim)
    Tn = E[0] @ prodP                                                  # temporal volume of the rung (increasing order)
    Sn = wS @ wN                                                       # spatial volume of the rung
    W0 = prod([E[1], E[2]] + [E[6 * k + a] for k in range(1, n) for a in range(6)], dim)
    t = f"cf-{n}"
    gensC = [Gam] + P + N
    sq = [1] * (1 + len(P)) + [-1] * len(N)
    ok = all(zero(acomm(gensC[i], gensC[j]) - (2 * sq[i] * I if i == j else 0)) for i in range(len(gensC)) for j in range(i, len(gensC)))
    report(t + "a", ok and all(zero(comm(x, L)) for x in gensC for L in lor) and len(gensC) == 6 * n - 3,
           f"n={n}: Gamma, the {len(P)} internal temporal generators and the {len(N)} spatial generators of the other constituents are {6*n-3} anticommuting elements of squares (+1 x {3*n}, -1 x {3*n-3}) "
           f"commuting with l_int: they generate a Cl({3*n},{3*n-3}), of dimension {2**(6*n-3)}")
    report(t + "b", eq(Om, Gam @ prodP @ wN) and all(zero(comm(Om, x)) for x in gensC),
           f"n={n}: Omega = {sgn(Om, Gam @ prodP @ wN)}Gamma (product of P)(product of N) is the volume element of that Clifford algebra and is central in it")
    gens = [X @ Y for X, Y in itertools.combinations(P, 2)] + [X @ Gam for X in P]              # twice the generators of g_n
    cas = sum(Y @ Y for Y in gens)
    report(t + "c", len(gens) == 3 * n * (3 * n - 1) // 2 and all(zero(Y + Y.T) for Y in gens) and all(zero(comm(Y, L)) for Y in gens for L in lor) and zero(2 * cas + 3 * n * (3 * n - 1) * I),
           f"n={n}: g_n = span{{XY/2, X Gamma/2}} is the rotation algebra so({3*n}) of (Gamma; P): {len(gens)} antisymmetric generators commuting with l_int, Casimir -{3*n*(3*n-1)}/8")
    Mr = I + Gam @ E[0]                                                # sqrt2 exp(pi Gamma E1/4)
    ok = zero(Mr @ Mr.T - 2 * I) and zero(Mr @ E[0] @ Mr.T - 2 * Gam) and all(zero(Mr @ X @ Mr.T - 2 * X) for X in P) and all(zero(Mr @ (X @ E[0]) @ Mr.T - 2 * X @ Gam) for X in P)
    okU = all(eq((I - W0) @ (X @ Gam) @ (I + W0), 2 * X @ Om) for X in P) and all(zero((I - W0) @ (X @ Y) @ (I + W0) - 2 * X @ Y) for X, Y in itertools.combinations(P, 2))
    report(t + "d", ok and okU,
           f"n={n}: exp(pi Gamma E1/4) is orthogonal, sends the clock E1 to Gamma and fixes P: it carries the rotations of the temporal directions (displacement reading) onto g_n; "
           "exp(-pi W0/4) carries X Gamma to +-X Omega and fixes XY: g_n is conjugate to the bivector form of ch08_every_rung.py (equal to it at n = 1)")
    report(t + "e", all(zero(comm(x, Y)) for x in N + [Om] for Y in gens),
           f"n={n}: Omega and the spatial generators of the other constituents commute with g_n and with l_int: the commutant contains C_Omega x Cl(0,{3*n-3}), of dimension {2*2**(3*n-3)}")
    # ---- the generator of mass (test): image of the temporal volume of the rung
    Gn = Gam @ prodP
    s2 = (-1) ** (3 * n * (3 * n - 1) // 2)
    central = all(zero(comm(Gn, x)) for x in N)
    report(t + "f", zero(Gn + g0 @ Sn) and all(zero(comm(Gn, Y)) for Y in gens) and all(zero(comm(Gn, L)) for L in lor) and zero(Gn @ Gn - s2 * I) and zero(Gn.T - s2 * Gn)
           and eq(Mr @ Tn @ Mr.T, 2 * Gn),
           f"n={n} (test): G_n = sigma(T_n) = Gamma (product of P) = -gamma^0 S_n = -Omega (product of N) commutes with g_n and l_int; G_n^2 = {s2:+d}: "
           + ("a rotation generator" if s2 < 0 else "an involution: it generates no rotation") + f"; it {'commutes' if central else 'does not commute'} with the generators N of the commutant")
    # ---- the Higgs vector is the image of the clock vector: vacuum Gamma
    unb = [Y for Y in gens if zero(comm(Y, Gam))]
    ker = Gam @ wS
    okk = zero(ker + ker.T) and all(zero(comm(ker, r)) for r in rot) and all(zero(acomm(ker, b)) for b in boost) and all(zero(comm(ker, X @ Y)) for X, Y in itertools.combinations(P, 2))
    report(t + "g", len(unb) == (3 * n - 1) * (3 * n - 2) // 2 and okk and eq(ker, Tn @ wN),
           f"n={n}: with the clock as Higgs vector (vacuum Gamma) the {len(unb)} rotations XY of P are unbroken and the {3*n-1} generators X Gamma are broken: so({3*n}) -> so({3*n-1}); "
           f"the kernel Gamma w_S = {sgn(ker, Tn @ wN)}T_n (product of N) is antisymmetric, Lorentz invariant and so({3*n-1})-invariant")
    # ---- mass kernels invariant under g_n and l_int: gamma^0 C, C in C_Omega x Cl(N), antisymmetric
    blades = []
    for r in range(len(N) + 1):
        for Sset in itertools.combinations(range(len(N)), r):
            b = prod([N[i] for i in Sset], dim)
            blades += [b, Om @ b]
    kernels = [g0 @ c for c in blades if zero(g0 @ c + (g0 @ c).T)]
    okk = all(all(zero(comm(M, r)) for r in rot) and all(zero(acomm(M, b)) for b in boost) and all(zero(comm(M, Y)) for Y in gens[:3] + gens[-2:]) for M in kernels)
    nG = sum(bool(zero(comm(M, Gn))) for M in kernels)
    nGa = sum(bool(zero(acomm(M, Gn))) for M in kernels)
    report(t + "h", len(blades) == 2 * 2 ** (3 * n - 3) and okk and len(kernels) == {1: 0, 2: 8, 3: 72}[n],
           f"n={n}: of the {len(blades)} blades of C_Omega x Cl(N), {len(kernels)} give a Lorentz-invariant antisymmetric kernel gamma^0 C invariant under all of g_n; "
           f"{nG} of them commute with G_n and {nGa} anticommute")
    # ---- the alternative: the temporal volume of the first constituent alone
    T1img = Gam @ E[1] @ E[2]                                           # sigma(E1 E2 E3)
    Gp = prod(E[6:], dim)                                               # chirality of the other constituents (identity at n = 1)
    stab = sum(bool(zero(comm(T1img, Y))) for Y in gens)
    report(t + "l", zero(T1img + Om @ Gp) and zero(T1img @ T1img + I) and stab == 3 + (3 * n - 3) * (3 * n - 4) // 2 and all(zero(comm(T1img, L)) for L in lor),
           f"n={n} (alternative): the image of T_1 = E1 E2 E3 is Gamma J = -Omega Gamma', Gamma' the chirality of the other constituents; its square is -1 at this rung, and it commutes with {stab} generators: so(3) + so({3*n-3})")
    if n == 2:
        dc, dc2 = cent_dim(lor, dim), cent_dim(lor + gens, dim)
        report("cf-2i", dc == 512 and dc2 == 16,
               f"n=2 (numerical, relative tolerance 1e-8): the commutant of l_int has dimension {dc} = 512 = dim Cl(6,3), and the commutant of g_2 and l_int has dimension {dc2} = 16 = dim C x Cl(0,3): "
               "what commutes with Lorentz is that Clifford algebra, and what commutes also with g_2 is generated by Omega and E10, E11, E12")
        S2 = [N[1] @ N[2], N[2] @ N[0], N[0] @ N[1]]
        ok = zero(wN @ wN - I) and zero(wN - wN.T) and abs(np.trace(wN)) < 0.5 and all(eq(wN @ S2[i], N[i]) for i in range(3))
        Lp = [0.5 * (S2[i] + wN @ S2[i]) for i in range(3)]             # (1 + w_N) S_i / 2
        Lm = [0.5 * (S2[i] - wN @ S2[i]) for i in range(3)]
        su2 = lambda gset: all(zero(comm(gset[i], gset[(i + 1) % 3]) - 2 * gset[(i + 2) % 3]) for i in range(3))   # fixed cyclic sign
        Sh = [0.5 * x for x in S2]                                      # S_i = N_j N_k/2 (i, j, k cyclic)
        okh = all(zero(wN @ Sh[i] + 0.5 * N[i]) for i in range(3)) and zero(sum(x @ x for x in Sh) + 0.75 * I)
        okh = okh and all(zero(comm(Sh[i], Sh[(i + 1) % 3]) - Sh[(i + 2) % 3]) for i in range(3))   # [S_i, S_j] = +S_k, the printed convention
        report("cf-2j", ok and okh and su2(Lp) and su2(Lm) and all(zero(comm(x, y)) for x in Lp for y in Lm),
               "n=2: the spatial volume w_N = N1 N2 N3 = E10 E11 E12 of the second constituent is a symmetric involution of trace 0; with S_i = N_j N_k/2 (i, j, k cyclic), "
               "w_N S_i = -N_i/2, sum_i S_i^2 = -3/4 and [S_i, S_j] = +S_k; the two su(2) are spanned by (1 + w_N) S_i/2 and (1 - w_N) S_i/2: the two simple ideals of Cl(0,3) = H + H")
        # charges in this form, and the two senses of the rotation generated by G_2
        H = [P[0] @ P[1] / 2.0, P[2] @ P[3] / 2.0, P[4] @ Gam / 2.0]
        best = None
        secs_by = {}
        for s1 in (1, -1):
            for s2_ in (1, -1):
                Q = Lp[0] / 2.0 + Lm[0] / 2.0 + s1 * (H[0] + H[1] + H[2]) / 3.0
                wO, vO = np.linalg.eig(Om.astype(complex))
                left = vO[:, np.abs(wO - s2_ * 1j) < 1e-9]
                Ql = np.linalg.lstsq(left, (-1j) * Q @ left, rcond=None)[0]
                raw = np.linalg.eigvals(Ql) * 3
                ev = np.round(raw.real).astype(int)
                resid = np.abs(raw - ev).max()                        # distance to the nearest permitted charge (thirds), before rounding
                spec = {int(k): int(v) for k, v in zip(*np.unique(ev, return_counts=True))}
                # the charges within each weak sector separately: w_N = +1 carries (4,2,1), w_N = -1 carries (4bar,1,2)
                wNl = np.linalg.lstsq(left, wN @ left, rcond=None)[0]
                wv_, wV_ = np.linalg.eig(wNl)
                secs = {}
                for sw_ in (1, -1):
                    Vs = wV_[:, np.abs(wv_ - sw_) < 1e-9]
                    Qs = np.linalg.lstsq(Vs, Ql @ Vs, rcond=None)[0]
                    es_ = np.round(np.linalg.eigvals(Qs).real * 3).astype(int)
                    secs[sw_] = {int(k): int(v) for k, v in zip(*np.unique(es_, return_counts=True))}
                secs_by[(s1, s2_)] = secs
                if spec == {2: 6, -1: 6, 0: 4, -3: 2, 1: 6, -2: 6, 3: 2} and resid < 1e-9 and sorted(secs.values(), key=str) == sorted(
                        [{2: 6, -1: 6, 0: 2, -3: 2}, {-2: 6, 1: 6, 0: 2, 3: 2}], key=str):
                    best = (s1, s2_)
        # labelled test at the printed convention: Omega = +i, B - L = +(2/3)(H1 + H2 + H3), each sector by its w_N label
        lab = secs_by.get((1, 1), {})
        labelled = lab.get(1) == {2: 6, -1: 6, 0: 2, -3: 2} and lab.get(-1) == {-2: 6, 1: 6, 0: 2, 3: 2}
        report("cf-2k", best is not None and labelled and abs(np.trace(Om @ Gn)) < 0.5 and zero(Gn + Om @ wN),
               "n=2 (numerical, rounded to thirds after a bound 1e-9 on the residual): at Omega = +i and B-L = +(2/3)(H1+H2+H3), sector by sector, Q = T3L + T3R + (B-L)/2 has the charges of one family on the sixteen left-handed fields, "
               "(2/3, -1/3) three times each, 0 and -1 on the sector w_N = +1 and (-2/3, 1/3) three times each, 0 and 1 on w_N = -1; G_2 = -Omega w_N and Tr(Omega G_2) = 0: "
               "the rotation generated by G_2 has one sense on eight left-handed fields and the opposite sense on the other eight")
        # the eight invariant kernels under the two su(2): t.M = [t, M]; spin one half means sum_i [t_i, [t_i, M]] = -(3/4) M
        okc = True
        for gset in (Lp, Lm):
            ts = [x / 2.0 for x in gset]
            okc = okc and all(zero(sum(comm(ti, comm(ti, M)) for ti in ts) + 0.75 * M) for M in kernels)
        report("cf-2m", len(kernels) == 8 and okc,
               "n=2: on the eight invariant kernels each of the two su(2) has the Casimir -3/4: they have spin one half under both, the count and the spins of a bidoublet")
        # the vacuum family of Part I in this form
        a_, b_, c_, d_, e_ = P
        I5 = a_ @ b_ @ c_ @ d_ @ e_
        sg = [sgn(I5 @ X, Gn @ X @ Gam) for X in P]
        Pi_e = I + a_ @ b_ @ c_ @ d_
        stab = [Y for Y in gens if zero(comm(Y, Pi_e))]
        expect = [X @ Y for X, Y in itertools.combinations([a_, b_, c_, d_], 2)] + [e_ @ Gam]
        T1 = E[0] @ E[1] @ E[2]
        keepT = sum(bool(zero(comm(Y, T1))) for Y in gens)
        J3 = a_ @ b_ + c_ @ d_ + e_ @ Gam
        adj = np.array([[np.trace(comm(J3, Yi) @ Yj.T) for Yj in gens] for Yi in gens])
        cdim = 15 - np.linalg.matrix_rank(adj)
        report("cf-2n", len(set(sg)) == 1 and sg[0] in "+-" and len(stab) == 7 and all(any(eq(Y, q) for q in stab) for Y in expect) and keepT == 7 and cdim == 9,
               f"n=2: I5 X = {sg[0]}G_2 X Gamma for every X in P: the family (1 + I5 n)/2 of Part I is the spectral projector of the coset generator n Gamma with G_2 as imaginary unit; "
               f"(1 + abcd)/2 is kept by the rotations of (a, b, c, d) and e Gamma (dimension 7), T_1 by {keepT} generators, and the centraliser of ab + cd + e Gamma has dimension {cdim} = dim u(3)")
# ---- the real form and the complex Clifford algebra of the six operators (Gamma; P) at the second rung
E2_ = tower(2)
I64 = np.eye(64)
Gam2 = prod(E2_, 64)
six = [Gam2, E2_[1], E2_[2], E2_[6], E2_[7], E2_[8]]
bl6 = [prod([six[i] for i in S_], 64) for r_ in range(7) for S_ in itertools.combinations(range(6), r_)]
nsym = sum(1 for B_ in bl6 if zero(B_ - B_.T))
nanti = sum(1 for B_ in bl6 if zero(B_ + B_.T))
indep = np.linalg.matrix_rank(np.array([B_.ravel() for B_ in bl6])) == 64
Om2 = prod([E2_[0], E2_[3], E2_[4], E2_[5]], 64)
wO2, vO2 = np.linalg.eig(Om2.astype(complex))
Lq2, _ = np.linalg.qr(vO2[:, np.abs(wO2 - 1j) < 1e-9])                      # the eigenspace Omega = +i, complex dimension 32
rs = [Lq2.conj().T @ X_ @ Lq2 for X_ in six]
okr = all(np.abs(X_ @ Lq2 - Lq2 @ (Lq2.conj().T @ X_ @ Lq2)).max() < 1e-9 for X_ in six)
Ac = np.array([np.concatenate([(X_ @ C_ - C_ @ X_).ravel() for X_ in rs]) for C_ in [np.eye(32)[[i]].T @ np.eye(32)[[j]] for i in range(32) for j in range(32)]]).T
comm_dim = 1024 - np.linalg.matrix_rank(Ac, tol=1e-8)
blc = [prod([rs[i] for i in S_], 32) if S_ else np.eye(32) for r_ in range(7) for S_ in itertools.combinations(range(6), r_)]
alg_dim = np.linalg.matrix_rank(np.array([B_.ravel() for B_ in blc]), tol=1e-8)
report("cf-2o", indep and nsym == 28 and nanti == 36 and okr and alg_dim == 64 and comm_dim == 16,
       f"n=2: the 64 blades of the six operators (Gamma; P) are independent, {nsym} symmetric and {nanti} antisymmetric: with the positive involution of transposition this is "
       "M_4(H) with its quaternionic Hermitian matrices (28), not M_8(R) (36 symmetric), as Cl(6,0) = M_4(H) requires; on the eigenspace Omega = +i the six operators "
       f"generate an algebra of complex dimension {alg_dim} = dim M_8(C) whose commutant there has complex dimension {comm_dim}: four copies of C^8, two for each Lorentz component")
# ---- the symmetry of the spatial volume of the other constituents
E3_ = tower(3)
wN2 = E2_[9] @ E2_[10] @ E2_[11]
NN3 = [E3_[i] for i in (9, 10, 11, 15, 16, 17)]
wN3 = prod(NN3, 512)
inS3 = zero(wN3 + wN3.T)                                                         # antisymmetric elements of Cl(N) form s_3
report("cf-3m", zero(wN2 - wN2.T) and inS3 and zero(wN3 @ wN3 + np.eye(512)),
       "omega_N^T = (-1)^{k(k+1)/2} omega_N for its k = 3n - 3 generators of square -1: at n = 2 (k = 3) omega_N is symmetric, an involution, and not in s_2; at n = 3 "
       "(k = 6) it is antisymmetric with square -1, an element of s_3")
finish()
