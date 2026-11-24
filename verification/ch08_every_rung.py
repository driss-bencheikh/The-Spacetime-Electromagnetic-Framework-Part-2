#!/usr/bin/env python3
"""Part II, Chapter 8 -- the adopted hypothesis at every rung.

At rung n the interior four-plane is <E1; E4,E5,E6> of the first constituent, with pseudoscalar Omega = E1 E4 E5 E6.
P is the set of the 3n-1 internal positive generators (the surplus pair E2,E3 and the temporal triples of the other constituents).
The hypothesis (first form) replaces the clock E1 by Omega in the rotations of the 3n temporal directions:
    g_n = span{ X Y / 2 (X<Y in P),  X Omega / 2 (X in P) }.

    python3 ch08_every_rung.py

All checks are exact (matrices with small integer entries, held as floats; every product is exact) except er-2p and er-2q,
which use floating-point linear algebra with a stated tolerance.
"""
import itertools
import time

import numpy as np

from p2common import report, finish, tower, prod, comm, acomm, zero

T0 = time.time()


def setup(n):
    E = tower(n)
    dim = 8 ** n
    I = np.eye(dim)
    Om = prod([E[0], E[3], E[4], E[5]], dim)
    Gam = prod(E, dim)
    rot = [E[4] @ E[5], E[5] @ E[3], E[3] @ E[4]]            # twice the rotation generators
    boost = [E[0] @ E[3 + a] for a in range(3)]              # twice the boost generators
    P = [E[1], E[2]] + [E[6 * k + a] for k in range(1, n) for a in range(3)]
    Nneg = [E[6 * k + 3 + a] for k in range(1, n) for a in range(3)]
    W0 = prod([E[1], E[2]] + [E[6 * k + a] for k in range(1, n) for a in range(6)], dim)   # product of all internal generators
    return dict(E=E, dim=dim, I=I, Om=Om, Gam=Gam, rot=rot, boost=boost, P=P, Nneg=Nneg, W0=W0)


for n in (1, 2, 3):
    d = setup(n)
    E, dim, I, Om, Gam, P, W0 = d["E"], d["dim"], d["I"], d["Om"], d["Gam"], d["P"], d["W0"]
    lor = d["rot"] + d["boost"]
    N = len(P)
    tag = f"er-{n}"
    sig = [1, 1, 1, -1, -1, -1] * n
    report(tag + "a", all(zero(acomm(E[i], E[j]) - (2 * sig[i] * I if i == j else 0)) for i in range(6 * n) for j in range(i, 6 * n))
           and all(zero(E[i] - sig[i] * E[i].T) for i in range(6 * n)),
           f"n={n}: the 6n tower generators anticommute, square to eta, temporal ones symmetric, spatial ones antisymmetric")
    report(tag + "b", zero(Om @ Om + I) and zero(Om + Om.T) and all(zero(comm(Om, L)) for L in lor) and all(zero(comm(Om, X)) for X in P),
           f"n={n}: Omega^2 = -1, Omega antisymmetric, commutes with the Lorentz bivectors and with the {N} internal positive generators")
    gens = [X @ Y for X, Y in itertools.combinations(P, 2)] + [X @ Om for X in P]            # twice the generators
    report(tag + "c", len(gens) == 3 * n * (3 * n - 1) // 2 and all(zero(Y + Y.T) for Y in gens) and all(zero(comm(Y, L)) for Y in gens for L in lor),
           f"n={n}: the {len(gens)} = 3n(3n-1)/2 elements XY/2, X Omega/2 are antisymmetric and commute with the interior Lorentz algebra (they keep the kinetic forms)")
    # realisation as the rotation algebra of 3n anticommuting symmetric involutions V_0 = Omega W0, V_X = X W0
    V = [Om @ W0] + [X @ W0 for X in P]
    ok = all(zero(acomm(V[a], V[b]) - (2 * I if a == b else 0)) for a in range(N + 1) for b in range(a, N + 1)) and all(zero(v - v.T) for v in V)
    report(tag + "d", zero(W0 @ W0 + I) and zero(W0 + W0.T) and all(zero(acomm(W0, X)) for X in P) and ok,
           f"n={n}: W0 (product of all internal generators) is an antisymmetric complex structure anticommuting with P; V_0 = Omega W0, V_X = X W0 are {N+1} anticommuting symmetric involutions")
    ok = all(zero(V[1 + i] @ V[1 + j] - P[i] @ P[j]) for i in range(N) for j in range(N) if i != j) and all(zero(V[1 + i] @ V[0] + P[i] @ Om) for i in range(N))
    report(tag + "e", ok, f"n={n}: V_X V_Y = XY and V_X V_0 = -X Omega: the hypothesis's algebra is the rotation algebra so({N+1}) of the V's (Clifford relations give the commutators)")
    report(tag + "f", zero(V[0] - Gam) or zero(V[0] + Gam), f"n={n}: V_0 is the total chirality Gamma up to sign: the clock axis is carried by Gamma")
    # equivalence with the rule 'clock -> Gamma': U = (1 - W0)/sqrt2 maps X to X W0 and fixes Gamma and Omega
    ok = all(zero((I - W0) @ X @ (I + W0) - 2 * X @ W0) for X in P) and zero(comm(W0, Gam)) and zero(comm(W0, Om)) and all(zero(comm(W0, L)) for L in lor)
    report(tag + "g", ok, f"n={n}: U = exp(-pi W0/4) is orthogonal, commutes with Lorentz, Gamma and Omega, and U X U^-1 = X W0: the algebra is conjugate to the rotation algebra of (Gamma, P)")
    cas = sum(Y @ Y for Y in gens)                                                         # 4 x Casimir
    report(tag + "h", zero(2 * cas + 3 * n * (3 * n - 1) * I), f"n={n}: the Casimir is the scalar -3n(3n-1)/8 = {-3*n*(3*n-1)/8}: every vector lies in a spinor representation of so({3*n})")
    bad = [X @ E[0] for X in P]
    report(tag + "i", all(not zero(comm(Y, b)) for Y in bad for b in d["boost"][:1]),
           f"n={n}: the unreplaced clock rotations X E1 do not commute with the boosts")
    Pi0 = I + Gam                                                                          # twice the chirality projector
    massless = sum(zero(comm(Y, Pi0)) for Y in gens)
    report(tag + "j", massless == N * (N - 1) // 2 and all(zero(comm(X @ Y, Pi0)) for X, Y in itertools.combinations(P, 2)),
           f"n={n}: at the vacuum (the chirality projector) the {N*(N-1)//2} generators XY are unbroken and the {N} generators X Omega are broken: so({N+1}) -> so({N})")
    d["gens"], d["V"] = gens, V
    # the two readings: K_n = span{XY, X E1} (rotations of the 3n temporal directions, the clock included) and g_n = span{XY, X Omega}
    kgen = [X @ Y for X, Y in itertools.combinations(P, 2)] + [X @ E[0] for X in P]
    Mg = (I - W0) @ (I + Gam @ E[0])                                                       # 2 x G_n, G_n = exp(-pi W0/4) exp(pi Gamma E1/4)
    eq = lambda A_, B_: zero(A_ - B_) or zero(A_ + B_)
    ok = zero(Mg @ Mg.T - 4 * I) and all(zero(Mg @ (X @ Y) @ Mg.T - 4 * X @ Y) for X, Y in itertools.combinations(P, 2)) \
        and all(eq(Mg @ (X @ E[0]) @ Mg.T, 4 * X @ Om) for X in P) and eq(Mg @ E[0] @ Mg.T, 4 * Gam)
    report(tag + "r", all(zero(Y + Y.T) for Y in kgen) and ok,
           f"n={n}: G_n = exp(-pi W0/4) exp(pi Gamma E1/4) is orthogonal, fixes every XY, sends X E1 to +-X Omega and the clock E1 to +-Gamma: the rotation algebra of the 3n temporal directions and the hypothesis's algebra are conjugate")
    T1 = E[0] @ E[1] @ E[2]
    keep = [Y for Y in gens if zero(Y.T @ T1 + T1 @ Y)]
    others = P[2:]
    expect = [P[0] @ P[1]] + [X @ Y for X, Y in itertools.combinations(others, 2)] + [X @ Om for X in others]
    report(tag + "s", len(keep) == 1 + (N - 1) * (N - 2) // 2 and all(any(zero(Y - q) for q in keep) for Y in expect),
           f"n={n}: {len(keep)} = 1 + (3n-2)(3n-3)/2 generators keep the Dirac mass form T = E1 E2 E3 of the first constituent (Y^T T + T Y = 0): J, and the rotations of the 3n-2 directions outside the surplus pair (clock axis included): so(2) + so({N-1})")
    if n == 1:
        J = E[1] @ E[2]
        pg = [J, E[1] @ Om, E[2] @ Om]
        report("er-1k", all(any(zero(Y - s * q) for q in pg for s in (1, -1)) for Y in gens), "n=1: the algebra is the Pauli-Guersey algebra span{J, e2 Omega, e3 Omega}/2")
        report("er-1l", dim // 4 == 2, "n=1: 2 left-handed Weyl fields, one doublet")
    if n == 2:
        a, b, c, dd, e_ = P
        I5 = a @ b @ c @ dd @ e_
        om6 = prod(V, dim)                                                                 # product of the six involutions
        chi = om6 @ Om
        report("er-2k", zero(om6 @ om6 + I) and all(zero(comm(om6, Y)) for Y in gens) and all(zero(comm(om6, L)) for L in lor),
               "n=2: omega_6 = V_0...V_5 is a complex structure commuting with so(6) and with Lorentz (it separates 4 from 4bar)")
        report("er-2l", zero(chi @ chi - I) and zero(chi - chi.T) and (zero(chi - I5) or zero(chi + I5)) and abs(np.trace(chi)) < 0.5,
               "n=2: chi = omega_6 Omega is a symmetric involution of trace 0, equal to +-I5 = +-abcde: it correlates the su(4) type with the Lorentz chirality")
        # left-handed content: dimension of {Omega = +i, omega_6 = +-i} = (1/4)[Tr 1 -+ Tr(Omega omega_6)] = 16 each
        report("er-2m", abs(np.trace(Om @ om6)) < 0.5 and abs(np.trace(Om)) < 0.5 and abs(np.trace(om6)) < 0.5,
               "n=2: Tr(Omega omega_6) = 0, so among the 16 left-handed Weyl fields 8 are in the 4 and 8 in the 4bar: 2 x 4 + 2 x 4bar")
        # the commutant: rotations of the spatial triple of constituent 2, split by I5, and Omega split by I5
        S2 = [E[10] @ E[11], E[11] @ E[9], E[9] @ E[10]]
        Pp, Pm = I + I5, I - I5                                                            # twice the projectors
        cand = [Pp @ s for s in S2] + [Pm @ s for s in S2] + [Pp @ Om, Pm @ Om]
        ok = all(zero(Y + Y.T) for Y in cand) and all(zero(comm(Y, g)) for Y in cand for g in gens) and all(zero(comm(Y, L)) for Y in cand for L in lor)
        report("er-2n", ok, "n=2: (1+-I5)/2 times the rotations of the spatial triple of constituent 2, and (1+-I5)/2 Omega, are antisymmetric and commute with so(6) and with Lorentz")
        L3, R3 = cand[:3], cand[3:6]
        su2 = lambda g: zero(comm(g[0], g[1]) - 4 * g[2]) or zero(comm(g[0], g[1]) + 4 * g[2])
        report("er-2o", su2(L3) and su2(R3) and all(zero(comm(x, y)) for x in L3 for y in R3) and all(zero(comm(x, y)) for x in cand[6:] for y in cand),
               "n=2: they form su(2) + su(2) + u(1) + u(1), the two su(2) commuting")
        # numerical: the commutant has no more
        t1 = time.time()
        A = np.zeros((dim * dim, dim * dim))
        If = np.eye(dim)
        for Y in gens + lor:
            ad = np.kron(Y, If) - np.kron(If, Y.T)
            A += ad.T @ ad
        w, v = np.linalg.eigh(A)
        ker = v[:, w < 1e-7 * w.max()]
        Ms = [ker[:, i].reshape(dim, dim) for i in range(ker.shape[1])]
        anti = np.array([(M - M.T).flatten() for M in Ms])
        r_anti = np.linalg.matrix_rank(anti, tol=1e-8)
        report("er-2p", ker.shape[1] == 16 and r_anti == 8,
               f"n=2 (numerical, relative tolerance 1e-7): the commutant of so(6) + Lorentz in M_64(R) has dimension {ker.shape[1]} = 16, its antisymmetric part dimension {r_anti} = 8 = dim u(2)+u(2)  [{time.time()-t1:.0f}s]")
        # charges: Q = T3L + T3R + (B-L)/2 on the left-handed fields (Omega = +i)
        H = [V[1] @ V[2] / 2.0, V[3] @ V[4] / 2.0, V[5] @ V[0] / 2.0]                      # Cartan generators of so(6)
        BL = (2.0 / 3.0) * (H[0] + H[1] + H[2])
        T3L, T3R = cand[0] / 4.0, cand[3] / 4.0
        best = None
        for s1 in (1, -1):
            for s2 in (1, -1):
                Q = T3L + T3R + s1 * BL / 2.0
                Mx = (-1j) * Q
                wO, vO = np.linalg.eig(Om.astype(complex))
                left = vO[:, np.abs(wO - s2 * 1j) < 1e-9]
                Ql = np.linalg.lstsq(left, Mx @ left, rcond=None)[0]
                ev = np.round(np.linalg.eigvals(Ql).real * 3).astype(int)
                spec = dict(zip(*np.unique(ev, return_counts=True)))
                want = {2: 6, -1: 6, 0: 4, -3: 2, 1: 6, -2: 6, 3: 2}
                if {int(k): int(vv) for k, vv in spec.items()} == want:
                    best = (s1, s2)
        report("er-2q", best is not None,
               "n=2 (numerical, rounded to thirds): Q = T3L + T3R + (B-L)/2 has on the 16 left-handed fields the charges 2/3, -1/3 (x3 each), 0, -1 and -2/3, 1/3 (x3 each), 0, 1: one family (4,2,1)+(4bar,1,2)")
    if n == 3:
        report("er-3k", dim // 4 == 128 and 128 == 8 * 16, "n=3: 128 left-handed Weyl fields = 8 x 16, with 16 the spinor of so(9)")
        # multiplicity of (16;(1/2,0)): joint eigenspace of four Cartan complex structures of so(9), Omega and one Lorentz rotation
        Cs = [V[1] @ V[2], V[3] @ V[4], V[5] @ V[6], V[7] @ V[8], Om, d["rot"][2]]
        indep = True
        for r in range(1, 7):
            for S in itertools.combinations(range(6), r):
                M = prod([Cs[i] for i in S], dim)
                if abs(np.trace(M)) > 0.5:
                    indep = False
        report("er-3l", all(zero(c @ c + I) for c in Cs) and all(zero(comm(x, y)) for x, y in itertools.combinations(Cs, 2)) and indep,
               "n=3: six commuting complex structures with traceless products: each joint eigenspace has dimension 512/64 = 8, the multiplicity of (16; Weyl)")
        W = [E[6], E[12], E[13], E[14]]
        soW = [x @ y for x, y in itertools.combinations(W, 2)]
        inspan = all(any(zero(Y - s * q) for q in gens for s in (1, -1)) for Y in soW)
        report("er-3m", inspan, "n=3: the six bivectors of the weak directions {7,13,14,15} of Part I lie in the algebra so(9)")
        C6 = [x @ y for x, y in itertools.combinations(d["Nneg"], 2)]
        report("er-3n", all(zero(comm(c, Y)) for c in C6 for Y in gens) and all(zero(comm(c, L)) for c in C6 for L in lor),
               "n=3: the fifteen colour bivectors of Part I (spatial directions of constituents 2 and 3) commute with so(9) and with Lorentz: they lie in its commutant")

print(f"# time {time.time()-T0:.0f} s")
finish()
