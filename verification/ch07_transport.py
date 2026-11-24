#!/usr/bin/env python3
"""Part II, Chapter 7 -- the adopted hypothesis at one constituent.

Two su(2) algebras act on S = R^8 and share J/2:  so(3)_+ (rotations of the temporal triple) and the Pauli-Guersey algebra.
The hypothesis (first form) replaces the clock e1 by the pseudoscalar Omega = e1 e4 e5 e6 of the four-plane <e1; e4,e5,e6>.

    python3 ch07_transport.py

Exact: matrices with entries that are small integers or dyadic rationals (held as floats, products exact); ranks and null spaces
with sympy over the rationals.
"""
import itertools

import numpy as np
import sympy as sp

from p2common import report, finish, cl33, comm, acomm, zero

e = cl33()
I8 = np.eye(8)
g0, ga = e[0], [e[3], e[4], e[5]]
J = e[1] @ e[2]
T = e[0] @ e[1] @ e[2]
Om = e[0] @ e[3] @ e[4] @ e[5]
wS = e[3] @ e[4] @ e[5]
Gam = T @ wS
rot = [0.5 * ga[1] @ ga[2], 0.5 * ga[2] @ ga[0], 0.5 * ga[0] @ ga[1]]
boost = [0.5 * g0 @ ga[a] for a in range(3)]
lint = rot + boost
Kk = [-I8] + [-g0 @ ga[a] for a in range(3)]
so3p = [0.5 * e[1] @ e[2], 0.5 * e[2] @ e[0], 0.5 * e[0] @ e[1]]        # rotations about e1, e2, e3
pg = [0.5 * J, 0.5 * e[2] @ Om, 0.5 * Om @ e[1]]                        # the same with e1 -> Omega


def rank(mats):
    return sp.Matrix([[sp.Rational(x).limit_denominator(64) for x in M.flatten()] for M in mats]).rank()


def closure_dim(gens):
    basis = []
    for M in gens:
        if rank(basis + [M]) > len(basis):
            basis.append(M)
    changed = True
    while changed:
        changed = False
        for A, B in itertools.product(list(basis), repeat=2):
            C = comm(A, B)
            if not zero(C) and rank(basis + [C]) > len(basis):
                basis.append(C)
                changed = True
    return len(basis)


def su2(g):
    return any(zero(comm(g[0], g[1]) - s * g[2]) and zero(comm(g[1], g[2]) - s * g[0]) and zero(comm(g[2], g[0]) - s * g[1]) for s in (1, -1))


report("ax-1", all(zero(Y + Y.T) for Y in so3p + pg) and su2(so3p) and su2(pg) and zero(so3p[0] - pg[0])
       and zero(sum(Y @ Y for Y in so3p) + 0.75 * I8) and zero(sum(Y @ Y for Y in pg) + 0.75 * I8),
       "so(3)_+ and the algebra with e1 -> Omega are antisymmetric, close as su(2), share J/2 and have Casimir -3/4 on S")
report("ax-2", [all(zero(comm(Y, L)) for L in lint) for Y in so3p] == [True, False, False] and all(zero(comm(Y, L)) for Y in pg for L in lint),
       "of so(3)_+ only J commutes with the interior Lorentz algebra; the replaced algebra commutes with all of it")
d1, d2 = closure_dim(so3p + lint), closure_dim(pg + lint)
report("ax-3", d1 == 15 and d2 == 9, f"so(3)_+ and Lorentz generate an algebra of dimension {d1} = 15 = dim so(3,3); the replaced algebra and Lorentz one of dimension {d2} = 9")
kin = lambda Y: all(zero(Y.T @ K + K @ Y) for K in Kk)
report("ax-4", all(kin(Y) for Y in pg) and [kin(Y) for Y in so3p] == [True, False, False],
       "the replaced algebra keeps the four kinetic forms K^mu; of so(3)_+ only J does")
mf = lambda Y: zero(Y.T @ T + T @ Y)
report("ax-5", all(mf(Y) for Y in so3p) and [mf(Y) for Y in pg] == [True, False, False],
       "so(3)_+ keeps the mass form T; of the replaced algebra only J does")
# clock covariance with an exact rotor: U5 = (3 + 4 e3e1)(3 + 4 e1e2), U5 U5^T = 625
U5 = (3 * I8 + 4 * e[2] @ e[0]) @ (3 * I8 + 4 * e[0] @ e[1])
n = np.array([1.0, 2.0, 2.0])
ne = lambda v: sum(v[i] * e[i] for i in range(3))
M = U5 @ ne(n) @ U5.T
c = np.array([np.trace(M @ e[i]) / 8 for i in range(3)])
ok = zero(U5 @ U5.T - 625 * I8) and zero(M - ne(c)) and abs(c @ c - 625 ** 2 * (n @ n)) < 0.5
ok2 = all(zero(U5.T @ (ne(c) @ ga[a]) @ U5 - 625 ** 2 * ne(n) @ ga[a]) for a in range(3)) and zero(U5.T @ T @ U5 - 625 * T)
report("ax-6", ok and ok2, "for a rotor U of Spin(V^+): U (n.e) U^-1 = (Rn).e with R a rotation, U^T K^a(Rn) U = K^a(n) and U^T T U = T: the Dirac data are covariant once the clock is a field")
al, be, gm = 1.0, 2.0, 3.0
Y = al * J + be * e[2] @ e[0] + gm * e[0] @ e[1]
Yg0 = be * e[2] - gm * e[1]
report("ax-7", zero(0.5 * comm(Y, g0) - Yg0) and all(zero(Y.T @ Kk[a + 1] + Kk[a + 1] @ Y - 2 * Yg0 @ ga[a]) for a in range(3)),
       "Part I, Prop. 9.53(i): Y^T K^a + K^a Y = 2 Y(gamma^0) gamma^a with Y(gamma^0) = beta e3 - gamma e2: the breaking is the motion of the clock")
cos = [comm(so3p[u], boost[a]) for u in (1, 2) for a in range(3)]
mixed = [e[u] @ e[3 + a] for u in (1, 2) for a in range(3)]
report("ax-8", rank(cos) == 6 and rank(cos + mixed) == 6,
       "the commutators of the two clock rotations with the boosts span the six mixed bivectors e_u e_{a+3}, u = 2,3: with them the coset is a charged four-vector of generators")
# mass kernels of the interior Lorentz algebra (Part I, Lemma 7.11)
blades = {"e1e2": e[0] @ e[1], "e1e3": e[0] @ e[2], "T": T, "e2wS": e[1] @ wS, "e3wS": e[2] @ wS, "JwS": J @ wS}
okk = all(zero(M + M.T) and all(zero(comm(M, r)) for r in rot) and all(zero(acomm(M, b)) for b in boost) for M in blades.values())
report("ax-9", okk, "the six blades e1e2, e1e3, T, e2 w_S, e3 w_S, J w_S are antisymmetric and Lorentz invariant (the kernel space of Lemma 7.11)")
act = lambda Yg, M: Yg.T @ M + M @ Yg
triv = [blades["e2wS"], blades["e3wS"], blades["T"]]
triu = [blades["e1e2"], blades["e1e3"], blades["JwS"]]
inspan = lambda M, tri: rank(tri + [M]) == 3
report("ax-10", all(inspan(act(Yg, M), triv) for Yg in pg for M in triv) and all(inspan(act(0.5 * Om, M), triu) for M in triv),
       "the replaced algebra maps the triple (e2 w_S, e3 w_S, T) into itself; Omega maps it onto (e1e2, e1e3, J w_S)")
w = [float(np.sum(act(Yg, T) ** 2)) for Yg in pg + [0.5 * Om]]
report("ax-11", w == [0.0, 8.0, 8.0, 8.0], f"squared norms of the variation of T under J/2, e3 Omega/2, e2 Omega/2, Omega/2: {w}")
# one Weyl doublet: commutant of Lorentz + replaced algebra on S, exactly
X = sp.Matrix(8, 8, sp.symbols("x0:64"))
eqs = []
for Yg in lint + pg:
    Ys = sp.Matrix(8, 8, [sp.Rational(v).limit_denominator(64) for v in Yg.flatten()])
    eqs += list(X * Ys - Ys * X)
sol = sp.solve(eqs, list(X), dict=True)[0]
dimc = len(X.subs(sol).free_symbols)
report("ax-12", dimc == 2, f"the commutant of Lorentz and the replaced algebra on S has dimension {dimc} = 2: S is one irreducible module of complex type, one Weyl doublet")
# the map g of Part I, Prop. 9.54(iv)
Pp, Pm = 0.5 * (I8 + wS), 0.5 * (I8 - wS)
g = Pp + J @ Pm
eq = lambda A, B: zero(A - B) or zero(A + B)
report("ax-13", zero(g @ g.T - I8) and all(eq(g @ so3p[i] @ g.T, pg[i]) for i in range(3)),
       "g = P_+ + J P_- is orthogonal, and g maps the rotations of V^+ about (e1, e2, e3) to +-(J/2, e3 Omega/2, Omega e2/2): the algebras agree, g so(3)_+ g^-1 = g_1, "
       "with the axes e2, e3 interchanged relative to sigma (ax-25)")
img = [g @ e[i] @ g.T for i in range(3)]
report("ax-14", zero(img[0] - Gam) and zero(img[1] + e[2]) and zero(img[2] - e[1]) and zero(g @ T @ g.T + Om),
       "g e1 g^-1 = Gamma, g e2 g^-1 = -e3, g e3 g^-1 = e2 and g T g^-1 = -Omega: the clock goes to the chirality and the temporal volume to minus the pseudoscalar; g is the rule e1 -> Gamma followed by a quarter turn of the surplus pair")
tri = [Gam, e[1], e[2]]
report("ax-15", all(zero(acomm(tri[a], tri[b]) - (2 * I8 if a == b else 0)) for a in range(3) for b in range(a, 3)) and all(zero(v - v.T) for v in tri)
       and all(zero(comm(v, L)) for v in tri for L in lint) and closure_dim([0.5 * e[1] @ e[2], 0.5 * e[2] @ Gam, 0.5 * Gam @ e[1]] + pg) == 3,
       "(Gamma, e2, e3) are anticommuting symmetric involutions commuting with Lorentz, and their rotation algebra is the replaced algebra")
Hn = lambda v: v[0] * Gam + v[1] * e[1] + v[2] * e[2]
H = Hn(n)
report("ax-16", zero(H @ H - 9 * I8) and zero(H - H.T) and abs(np.trace((3 * I8 + H) / 6) - 4) < 1e-12 and all(zero(comm(H, L)) for L in lint),
       "for n = (1,2,2): H(n)^2 = |n|^2, H symmetric, (1 + H/|n|)/2 a projector of rank 4, commuting with the Lorentz algebra")
V5 = (3 * I8 + 8 * pg[1]) @ (3 * I8 + 8 * pg[2])
M2 = V5 @ H @ V5.T
c2 = np.array([np.trace(M2 @ t) / 8 for t in tri])
report("ax-17", zero(M2 - Hn(c2)) and abs(c2 @ c2 - 625 ** 2 * 9) < 0.5 and zero(comm(J, Gam)) and not zero(comm(J, e[1])),
       "H(n) transforms as a vector under the group of the replaced algebra; J fixes only the poles n = +-(1,0,0)")
Mk = H @ wS
report("ax-18", zero(Mk + Mk.T) and all(zero(comm(Mk, r)) for r in rot) and all(zero(acomm(Mk, b)) for b in boost) and eq(Gam @ wS, T),
       "the mass kernel H(n) w_S is antisymmetric and Lorentz invariant, and at the pole it is +-T, the Dirac mass form")
Pi0 = 0.5 * (I8 + Gam)
Q = 0.5 * J
tang = [comm(pg[1], Pi0), comm(pg[2], Pi0)]
report("ax-19", zero(comm(Q, Pi0)) and all(zero(t - t.T) for t in tang) and all(not zero(comm(Q, t)) for t in tang)
       and abs(np.trace(tang[0] @ tang[1])) < 1e-12 and all(abs(np.trace(t)) < 1e-12 and abs(np.trace(t @ t) - 2) < 1e-12 for t in tang),
       "at the pole [Q, Pi_0] = 0, and the two tangents dPi are symmetric, charged and orthogonal, with Tr dPi = 0 and Tr(dPi^2) = 2 each: the criterion of Part I, Cor. 8.53, is met")
report("ax-20", all(zero(comm(e[i] @ e[j], Pi0)) for i in range(6) for j in range(i + 1, 6)),
       "no bivector of so(3,3) moves the chirality projector: Part I, Prop. 8.55, concerns bivector variations")
mp = [float(np.trace(comm(Yg, Pi0) @ comm(Yg, Pi0))) for Yg in pg]
report("ax-21", mp == [0.0, 2.0, 2.0], f"mass pattern Tr([Y, Pi_0]^2) for the components along (e1, e2, e3): {mp}")
# Omega as the Hodge star of the two-forms of the four-plane
sgn = 1 if all(zero(Om @ rot[i] - boost[i]) for i in range(3)) else -1
report("ax-22", zero(Om @ Om + I8) and all(zero(comm(Om, L)) for L in lint) and all(zero(Om @ rot[i] - sgn * boost[i]) and zero(Om @ boost[i] + sgn * rot[i]) for i in range(3)),
       f"Omega^2 = -1, Omega commutes with Lorentz, Omega R_i = {'+' if sgn > 0 else '-'}B_i and Omega B_i = {'-' if sgn > 0 else '+'}R_i: Omega is the Hodge star of the two-forms of the four-plane")
# the Lorentz-invariant quadratic forms of a frame vector
Z3, I3 = sp.zeros(3), sp.eye(3)
G6 = sp.Matrix(sp.BlockMatrix([[Z3, -I3], [I3, Z3]]))
eta6 = sp.diag(1, 1, 1, -1, -1, -1)
cr = lambda i: sp.Matrix(3, 3, lambda a, b: -sp.LeviCivita(i, a, b))
Ls = [sp.Matrix(sp.BlockMatrix([[cr(i), Z3], [Z3, cr(i)]])) for i in range(3)]
Ks = [G6 * L for L in Ls]
Ssym = sp.Matrix(6, 6, lambda a, b: sp.Symbol(f"s{min(a,b)}{max(a,b)}"))
eq6 = []
for r in Ls + Ks:
    eq6 += list(r.T * Ssym + Ssym * r)
sol6 = sp.solve(eq6, list(Ssym.free_symbols), dict=True)[0]
dim6 = len(Ssym.subs(sol6).free_symbols)
inv = lambda S: all((r.T * S + S * r) == sp.zeros(6) for r in Ls + Ks)
report("ax-23", dim6 == 2 and inv(eta6) and inv(eta6 * G6),
       f"the symmetric forms on the frame invariant under the Lorentz algebra of the field reading form a space of dimension {dim6} = 2, spanned by eta and eta G")
# ---- the third element of the triple: a circle of candidates, and the unique one in the commutant of l_int
ok = True
for al in (0.3, 0.7, 1.2, np.pi / 2):
    Xa = np.cos(al) * e[0] + np.sin(al) * Gam
    ok = ok and zero(Xa - Xa.T, 1e-12) and zero(Xa @ Xa - I8, 1e-12) and zero(acomm(Xa, e[1]), 1e-12) and zero(acomm(Xa, e[2]), 1e-12)
    ok = ok and (all(zero(comm(Xa, L), 1e-12) for L in lint) == (abs(np.cos(al)) < 1e-12))
Cs = sp.Matrix(8, 8, sp.symbols("y0:64"))
eqc = []
for Yg in lint:
    Ys = sp.Matrix(8, 8, [sp.Rational(v).limit_denominator(64) for v in Yg.flatten()])
    eqc += list(Cs * Ys - Ys * Cs)
for Xg in (e[1], e[2]):
    Xs_ = sp.Matrix(8, 8, [int(v) for v in Xg.flatten()])
    eqc += list(Cs * Xs_ + Xs_ * Cs)
eqc += list(Cs - Cs.T)
solc = sp.solve(eqc, list(Cs), dict=True)[0]
Cg = Cs.subs(solc)
freec = sorted(Cg.free_symbols, key=str)
Gs = sp.Matrix(8, 8, [int(v) for v in Gam.flatten()])
ok = ok and len(freec) == 1 and (Cg / freec[0]).applyfunc(sp.simplify) in (Gs, -Gs)
report("ax-24", ok,
       "X_alpha = cos(alpha) e1 + sin(alpha) Gamma is a symmetric involution anticommuting with e2 and e3 for every alpha, and commutes with l_int only for cos(alpha) = 0; "
       "the symmetric elements of the commutant of l_int that anticommute with e2 and e3 are the multiples of Gamma, so the involutions among them are +-Gamma")
# ---- conjugation by g is sigma followed by a quarter turn of the surplus pair; exp(-pi J/4) g implements sigma exactly
from scipy.linalg import expm
sig = {0: Gam, 1: e[1], 2: e[2]}
quarter = {0: Gam, 1: -e[2], 2: e[1]}                                       # R sigma: e1 -> Gamma, e2 -> -e3, e3 -> e2
ok = all(zero(g @ e[i] @ g.T - quarter[i], 1e-12) for i in range(3))
ok = ok and zero(g @ (0.5 * e[2] @ e[0]) @ g.T - 0.5 * e[1] @ Gam, 1e-12) and np.linalg.matrix_rank(np.array([(e[1] @ Gam).ravel(), (e[2] @ Gam).ravel()])) == 2
ok = ok and all(zero(g @ (0.5 * e[a] @ e[b]) @ g.T - 0.5 * quarter[a] @ quarter[b], 1e-12) for a, b in ((1, 2), (2, 0), (0, 1)))
h = expm(-np.pi / 4 * J) @ g
ok = ok and zero(h @ h.T - I8, 1e-12) and all(zero(h @ e[i] @ h.T - sig[i], 1e-12) for i in range(3))
report("ax-25", ok,
       "g e1 g^-1 = Gamma, g e2 g^-1 = -e3, g e3 g^-1 = e2, so conjugation by g is sigma followed by the quarter turn e2 -> -e3, e3 -> e2, also on the rotations: "
       "g (e3 e1/2) g^-1 = e2 Gamma/2 while sigma(e3 e1/2) = e3 Gamma/2, an independent generator; exp(-pi J/4) g is orthogonal and implements sigma exactly")
finish()
