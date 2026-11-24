#!/usr/bin/env python3
"""Part II, Chapter 4 -- the coupling to b.

n massive two-form fields X^a = (B^a; E^a) of the temporal half, one common rate, gauged under the copy symmetry so(n):
    L = sum_a [(m/2)|k^a|^2 - (m/2)(rho_m^a)^2 - (m kappa^2/2)(|B^a|^2 - |E^a|^2)],   k = -(D_t B - curl_D E),  rho_m = div_D B.

    python3 ch04_b_coupling.py

Symbolic and exact (sympy), except bc-5, which is numerical on random backgrounds (stated tolerances).
"""
import sympy as sp
import numpy as np

from p2common import report, finish
from p2fields import t, x, y, z, co, xs, vec3, fld, curl, div, cross, dot, EL

m, kap, g, I0 = sp.symbols("m kappa g I0", positive=True)
eps = sp.LeviCivita
zero = lambda L: all(sp.expand(v) == 0 for v in L)

# ---- what commutes with the Lorentz algebra of the field reading on n copies
Z3, I3 = sp.zeros(3), sp.eye(3)
G6 = sp.Matrix(sp.BlockMatrix([[Z3, -I3], [I3, Z3]]))
eta6 = sp.diag(1, 1, 1, -1, -1, -1)
cr = lambda i: sp.Matrix(3, 3, lambda a, b: -eps(i, a, b))
Ls = [sp.Matrix(sp.BlockMatrix([[cr(i), Z3], [Z3, cr(i)]])) for i in range(3)]
Ks = [G6 * L for L in Ls]
report("bc-1", all((eta6 * M).T == -(eta6 * M) and M * G6 == G6 * M for M in Ls + Ks), "the generators L_i, K_i = G L_i of the Lorentz algebra of the field reading lie in so(3,3) and commute with G")
dims = []
for n in (1, 2, 3):
    N = 6 * n
    Ms = sp.Matrix(N, N, sp.symbols(f"q0:{N * N}"))
    etan = sp.kronecker_product(sp.eye(n), eta6)
    eqs = list((etan * Ms).T + etan * Ms)
    for M in Ls + Ks:
        Mn = sp.kronecker_product(sp.eye(n), M)
        eqs += list(Ms * Mn - Mn * Ms)
    sol = sp.solve(eqs, list(Ms), dict=True)[0]
    Mg = Ms.subs(sol)
    free = sorted(Mg.free_symbols, key=str)
    sol2 = sp.solve(list(Mg * etan - etan * Mg), free, dict=True)[0] if free else {}
    dims.append((len(free), len(Mg.subs(sol2).free_symbols)))
report("bc-2", dims == [(0, 0), (2, 1), (6, 3)],
       f"n = 1, 2, 3 copies under a common Lorentz algebra: the commuting elements of so(3n,3n) have dimensions {[d[0] for d in dims]} = n(n-1); those keeping the two triples apart {[d[1] for d in dims]} = n(n-1)/2 = dim so(n)")


def gauged(n):
    B = [[fld(f"B{a}_{i}") for i in range(3)] for a in range(n)]
    E = [[fld(f"E{a}_{i}") for i in range(3)] for a in range(n)]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    gens = []
    for a, b in pairs:
        M = sp.zeros(n, n)
        M[a, b], M[b, a] = 1, -1
        gens.append(M)
    ng = len(gens)
    beta = [[fld(f"beta{mu}_{c}") for c in range(ng)] for mu in range(4)]
    bm = [sum((beta[mu][c] * gens[c] for c in range(ng)), sp.zeros(n, n)) for mu in range(4)]
    D = lambda mu, V: [sp.diff(V[a], xs[mu]) + g * sum(bm[mu][a, b] * V[b] for b in range(n)) for a in range(n)]
    comp = lambda V, i: [V[a][i] for a in range(n)]

    def Dt(V):
        c = [D(0, comp(V, i)) for i in range(3)]
        return [[c[i][a] for i in range(3)] for a in range(n)]

    def curlD(V):
        d = [[D(q + 1, comp(V, r)) for r in range(3)] for q in range(3)]
        return [[sum(eps(i, q, r) * d[q][r][a] for q in range(3) for r in range(3)) for i in range(3)] for a in range(n)]

    def divD(V):
        d = [D(i + 1, comp(V, i)) for i in range(3)]
        return [sum(d[i][a] for i in range(3)) for a in range(n)]

    def gradD(S):
        d = [D(i + 1, S) for i in range(3)]
        return [[d[i][a] for i in range(3)] for a in range(n)]

    flat = lambda V: [V[a][i] for a in range(n) for i in range(3)]
    Fb = [[sp.diff(bm[nu], xs[mu]) - sp.diff(bm[mu], xs[nu]) + g * (bm[mu] * bm[nu] - bm[nu] * bm[mu]) for nu in range(4)] for mu in range(4)]
    Eb = [Fb[0][i + 1] for i in range(3)]
    Bb = [sum((sp.Rational(1, 2) * eps(i, q, r) * Fb[q + 1][r + 1] for q in range(3) for r in range(3)), sp.zeros(n, n)) for i in range(3)]
    act = lambda M, S: [sum(M[a, b] * S[b] for b in range(n)) for a in range(n)]
    cE, cB, DtB, DtE = curlD(E), curlD(B), Dt(B), Dt(E)
    kD = [[-(DtB[a][i] - cE[a][i]) for i in range(3)] for a in range(n)]
    jD = [[-(DtE[a][i] + cB[a][i]) for i in range(3)] for a in range(n)]
    rm, re = divD(B), divD(E)

    def lag(rates):
        return sum(m / 2 * sum(kD[a][i]**2 for i in range(3)) - m / 2 * rm[a]**2 - m * rates[a] / 2 * (sum(B[a][i]**2 for i in range(3)) - sum(E[a][i]**2 for i in range(3))) for a in range(n))

    L = lag([kap**2] * n)
    # gauge invariance to first order
    lam = [fld(f"lam{c}") for c in range(ng)]
    om = sum((lam[c] * gens[c] for c in range(ng)), sp.zeros(n, n))
    e_ = sp.symbols("epsilon")
    sub = {}
    for a in range(n):
        for i in range(3):
            sub[B[a][i]] = B[a][i] + e_ * sum(om[a, b] * B[b][i] for b in range(n))
            sub[E[a][i]] = E[a][i] + e_ * sum(om[a, b] * E[b][i] for b in range(n))
    for mu in range(4):
        db = om * bm[mu] - bm[mu] * om - sp.diff(om, xs[mu]) / g
        for c, (a, b) in enumerate(pairs):
            sub[beta[mu][c]] = beta[mu][c] + e_ * db[a, b]
    Lv = L.subs(sub, simultaneous=True).doit()
    report(f"bc-3-{n}a", sp.expand(sp.diff(Lv, e_).subs(e_, 0)) == 0, f"n={n}: L is invariant to first order under dX = wX, db = [w, b] - dw/g")
    flds = flat(B) + flat(E)
    el = EL(L, flds)
    elB = [[el[3 * a + i] for i in range(3)] for a in range(n)]
    elE = [[el[3 * n + 3 * a + i] for i in range(3)] for a in range(n)]
    ck = curlD(kD)
    report(f"bc-3-{n}b", all(sp.diff(L, sp.diff(E[a][i], t)) == 0 for a in range(n) for i in range(3)) and zero([elE[a][i] / m - kap**2 * E[a][i] - ck[a][i] for a in range(n) for i in range(3)]),
           f"n={n}: E has no velocity and EL_E/m = kappa^2 E + curl_D k: E is auxiliary with coefficient kappa^2 for every b")
    Dtk, grm = Dt(kD), gradD(rm)
    report(f"bc-3-{n}c", zero([elB[a][i] / m - (Dtk[a][i] + grm[a][i] - kap**2 * B[a][i]) for a in range(n) for i in range(3)]), f"n={n}: EL_B/m = D_t k + grad_D rho_m - kappa^2 B (the free force law with covariant derivatives)")
    p = [[sp.diff(L, sp.diff(B[a][i], t)) for i in range(3)] for a in range(n)]
    Hc = sum(p[a][i] * sp.diff(B[a][i], t) for a in range(n) for i in range(3)) - L
    Hpos = sum(m / 2 * sum(kD[a][i]**2 for i in range(3)) + m / 2 * rm[a]**2 + m * kap**2 / 2 * sum(B[a][i]**2 + E[a][i]**2 for i in range(3)) for a in range(n))
    ExP = [cross(E[a], p[a]) for a in range(n)]
    divExP = sum(sp.diff(ExP[a][i], co[i]) for a in range(n) for i in range(3))
    gauss = g * sum(p[a][i] * bm[0][a, b] * B[b][i] for a in range(n) for b in range(n) for i in range(3))
    EdotEL = sum(E[a][i] * elE[a][i] for a in range(n) for i in range(3))
    report(f"bc-3-{n}d", zero([p[a][i] + m * kD[a][i] for a in range(n) for i in range(3)]) and sp.expand(Hc - Hpos + gauss - divExP + EdotEL) == 0,
           f"n={n}: p = -m k, and the canonical energy density is sum_a[(m/2)(|k|^2 + rho_m^2) + m kappa^2 E_6] - g b_0^{{ab}} p^a.B^b + div(E x p) - E.EL_E")
    dck = divD(ck)
    Bk = [sum(act(Bb[i], comp(kD, i))[a] for i in range(3)) for a in range(n)]
    ok = zero([dck[a] - g * Bk[a] for a in range(n)] + [divD(elE)[a] / m - kap**2 * re[a] - g * Bk[a] for a in range(n)])
    lhs = [[Dt(elE)[a][i] / m - curlD(elB)[a][i] / m + kap**2 * jD[a][i] for i in range(3)] for a in range(n)]
    Exk = [[sum(eps(i, q, r) * act(Eb[q], comp(kD, r))[a] for q in range(3) for r in range(3)) for i in range(3)] for a in range(n)]
    Brm = [[act(Bb[i], rm)[a] for i in range(3)] for a in range(n)]
    ok = ok and zero([lhs[a][i] - g * (Exk[a][i] - Brm[a][i]) for a in range(n) for i in range(3)])
    report(f"bc-3-{n}e", ok, f"n={n}: on solutions rho_e = -(g/kappa^2) B_b.k and j = (g/kappa^2)(E_b x k - B_b rho_m), with E_b, B_b the field strength of b: they vanish when F_b = 0")
    if n == 2:
        k1, k2 = sp.symbols("kappa1 kappa2", positive=True)
        L2 = lag([k1**2, k2**2])
        el2 = EL(L2, flds)
        J = [sp.diff(L2, beta[mu][0]) for mu in range(4)]
        divJ = sum(sp.diff(J[mu], xs[mu]) for mu in range(4))
        rot = [B[1][i] for i in range(3)] + [-B[0][i] for i in range(3)] + [E[1][i] for i in range(3)] + [-E[0][i] for i in range(3)]
        Cc = sum(el2[q] * rot[q] for q in range(12))
        etaX1X2 = sum(B[0][i] * B[1][i] - E[0][i] * E[1][i] for i in range(3))
        found = [(s1, s2) for s1 in (1, -1) for s2 in (1, -1) if sp.expand(divJ - s1 * g * Cc - s2 * g * m * (k1**2 - k2**2) * etaX1X2) == 0]
        report("bc-4", len(found) == 1,
               f"n=2, unequal rates: d_mu J^mu = {found[0][0]:+d} g (EL . eps X) {found[0][1]:+d} g m (kappa_1^2 - kappa_2^2) eta(X^1, X^2): the copy current is conserved on solutions only for equal rates" if found else "n=2, unequal rates: identity for d_mu J^mu")


gauged(2)
gauged(3)

# ---- characteristics (numerical): principal symbol P = xi^2 + N of the wave form of the gauged temporal half
rng = np.random.default_rng(7)
eta4 = np.diag([1., -1, -1, -1])
prs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def symbol(n, cval):
    xi = rng.normal(size=4)
    xi2 = xi @ eta4 @ xi
    F = np.zeros((4, 4, n, n))
    for mu in range(4):
        for nu in range(mu + 1, 4):
            A = rng.normal(size=(n, n))
            A = A - A.T
            F[mu, nu], F[nu, mu] = A, -A

    def to_tensor(v):
        Xt = np.zeros((n, 4, 4))
        for a in range(n):
            for q, (i, j) in enumerate(prs):
                Xt[a, i, j], Xt[a, j, i] = v[6 * a + q], -v[6 * a + q]
        return Xt

    def Nop(v):
        Xt = to_tensor(v)
        yv = np.zeros((n, 4))
        for rho in range(4):
            for mu in range(4):
                for nu in range(4):
                    yv[:, rho] += cval * F[mu, nu] @ (xi[mu] * Xt[:, nu, rho] + xi[nu] * Xt[:, rho, mu] + xi[rho] * Xt[:, mu, nu])
        out = np.zeros((n, 4, 4))
        for a in range(n):
            out[a] = np.outer(yv[a], xi) - np.outer(xi, yv[a])
        return np.array([out[a, i, j] for a in range(n) for (i, j) in prs])

    Nm = np.array([Nop(v) for v in np.eye(6 * n)]).T
    P = xi2 * np.eye(6 * n) + Nm
    return np.abs(Nm @ Nm).max() / max(1.0, np.abs(Nm).max()**2), abs(np.linalg.det(P) / xi2**(6 * n) - 1), np.abs(Nm).max()


worst = [0.0, 0.0, np.inf]
for n in (2, 3):
    for cval in (0.3, 5.0):
        for _ in range(6):
            a, b, c = symbol(n, cval)
            worst = [max(worst[0], a), max(worst[1], b), min(worst[2], c)]
flat = [symbol(n, 0.0) for n in (2, 3)]
report("bc-5", worst[0] < 1e-12 and worst[1] < 1e-8 and worst[2] > 1e-3 and all(c_ == 0 and b_ < 1e-12 for _, b_, c_ in flat),
       f"(numerical, 24 random backgrounds with nonzero field strength, n = 2, 3, weak and strong field) principal symbol P = xi^2 + N with N = e_xi C_F e_xi linear in "
       f"the field strength, N != 0 and N^2 = 0 (relative size {worst[0]:.1e}), det P = (xi^2)^(6n) (relative error {worst[1]:.1e}); for F = 0, N = 0: the "
       f"characteristics are the light cones")

# ---- the clock field in the temporal half
B, E, th = vec3("B"), vec3("E"), fld("theta")
k = [-(sp.diff(B[i], t) - curl(E)[i]) for i in range(3)]
rho_m = div(B)
Th = sp.diff(th, t)**2 - sum(sp.diff(th, c)**2 for c in co)
etaXX = dot(B, B) - dot(E, E)
L = I0 / 2 * Th + m / 2 * dot(k, k) - m / 2 * rho_m**2 - m / 2 * Th * etaXX
elE = EL(L, E)
ck = curl(k)
report("bc-6", zero([elE[i] / m - Th * E[i] - ck[i] for i in range(3)]), "clock field in the temporal half: EL_E/m = (d theta)^2 E + curl k, so E = -curl k/(d theta)^2: algebraic while (d theta)^2 > 0, degenerate where the clock stops")
elth = EL(L, [th])[0]
W = I0 - m * etaXX
cur = [W * sp.diff(th, t)] + [-W * sp.diff(th, c) for c in co]
pth = sp.diff(L, sp.diff(th, t))
report("bc-7", sp.expand(elth + sp.diff(cur[0], t) + sum(sp.diff(cur[i + 1], co[i]) for i in range(3))) == 0 and sp.expand(pth - W * sp.diff(th, t)) == 0,
       "the theta equation is d_mu[(I0 - m eta(X, X)) d^mu theta] = 0, with clock momentum p_theta = (I0 - m eta(X, X)) d_t theta")
p = [sp.diff(L, sp.diff(B[i], t)) for i in range(3)]
Hc = dot(p, [sp.diff(B[i], t) for i in range(3)]) + pth * sp.diff(th, t) - L
Hpos = dot(p, p) / (2 * m) + m / 2 * rho_m**2 + W / 2 * (sp.diff(th, t)**2 + sum(sp.diff(th, c)**2 for c in co)) + m * Th * dot(E, E)
report("bc-8", sp.expand(Hc - Hpos - div(cross(E, p)) + dot(E, elE)) == 0,
       "energy density = |p|^2/2m + (m/2) rho_m^2 + (1/2)(I0 - m eta(X,X))(theta_t^2 + |grad theta|^2) + m (d theta)^2 |E|^2 + div(E x p) - E.EL_E: positive when I0 > m eta(X,X) and (d theta)^2 > 0")
Lc = L.subs(th, kap * t).doit()
report("bc-9", sp.expand(Lc - (m / 2 * dot(k, k) - m / 2 * rho_m**2 - m * kap**2 / 2 * etaXX) - I0 * kap**2 / 2) == 0, "theta = kappa t returns the temporal half with rate kappa (plus the constant I0 kappa^2/2)")
# ---- the dynamical clock: the canonical elimination of E can be singular although (d theta)^2 > 0 and the inertia is positive
Ev = sp.symbols("E1:4", real=True)
Bv = sp.symbols("Bb1:4", real=True)
pth_, gth = sp.symbols("p_theta", positive=True), sp.symbols("g1:4", real=True)
Cin = I0 - m * sum(b_**2 for b_ in Bv) + m * sum(e_**2 for e_ in Ev)
thdot = pth_ / Cin
s_ = thdot**2 - sum(q_**2 for q_ in gth)
Fv = sp.Matrix([m * s_ * e_ for e_ in Ev])                       # m s E - curl p, the curl p term being independent of E
Jac = Fv.jacobian(Ev)
pred = m * (s_ * sp.eye(3) - 4 * m * thdot**2 / Cin * sp.Matrix(Ev) * sp.Matrix(Ev).T)
ok = (Jac - pred).applyfunc(sp.simplify) == sp.zeros(3)
pt = {Bv[0]: 0, Bv[1]: 0, Bv[2]: 0, gth[0]: 0, gth[1]: 0, gth[2]: 0, Ev[0]: sp.sqrt(I0 / (3 * m)), Ev[1]: 0, Ev[2]: 0}
Jp = Jac.subs(pt).applyfunc(sp.simplify)
ok = ok and sp.simplify(Jp[0, 0]) == 0 and sp.simplify(s_.subs(pt)) > 0 and sp.simplify(Cin.subs(pt) - 4 * I0 / 3) == 0
report("bc-10", ok,
       "with a dynamical clock (one field, b = 0) the constraint m s E = curl p, s = theta_t^2 - |grad theta|^2, theta_t = p_theta/C, C = I0 - m|B|^2 + m|E|^2, has the "
       "Jacobian m(s 1 - (4 m theta_t^2/C) E E^T) in E at fixed canonical data; at B = 0, grad theta = 0, |E|^2 = I0/(3m) its eigenvalue along E vanishes while s > 0 "
       "and C = 4 I0/3 > 0: the elimination of E is not regular there")
b_ = sp.symbols("b", positive=True)
Bh, Eh = [b_ * sp.cos(kap * t), 0, 0], [0, 0, 0]
sub_h = {B[i]: Bh[i] for i in range(3)} | {E[i]: Eh[i] for i in range(3)}
elB_c = EL(Lc, B + E)
ok = all(sp.simplify(v.subs(sub_h, simultaneous=True).doit()) == 0 for v in elB_c)
clock_eq = sp.diff((I0 - m * etaXX) * kap, t)                    # d_mu[(I0 - m eta(X,X)) d^mu theta] at theta = kappa t
ok = ok and sp.simplify(clock_eq.subs(sub_h, simultaneous=True).doit()) != 0
report("bc-11", ok,
       "the uniform field E = 0, B = (b cos kappa t, 0, 0) solves the temporal half with the prescribed rate theta = kappa t, but not the clock equation "
       "kappa d_t eta(X, X) = 0 of the dynamical clock: prescribed and dynamical clocks have different solution sets")
# ---- the copy symmetry for general n: the commutant of l_G is {A (x) 1 + B (x) G}, and so(3n,3n) requires A, B antisymmetric (numerical rank at n = 1..4)
import numpy as _np
_G6, _eta6 = _np.array(G6.tolist(), float), _np.array(eta6.tolist(), float)
_gens = [_np.array(Mx.tolist(), float) for Mx in Ls + Ks]
ok, found = True, []
for n in (1, 2, 3, 4):
    N = 6 * n
    etan = _np.kron(_np.eye(n), _eta6)
    basis = []
    for a_ in range(N):
        for b_ in range(a_ + 1, N):
            A_ = _np.zeros((N, N))
            A_[a_, b_], A_[b_, a_] = 1, -1
            basis.append(etan @ A_)                                   # a basis of so(3n,3n): eta A with A antisymmetric
    Mn = [_np.kron(_np.eye(n), g_) for g_ in _gens]
    rows = _np.array([_np.concatenate([(X_ @ M_ - M_ @ X_).ravel() for M_ in Mn]) for X_ in basis]).T
    sv = _np.linalg.svd(rows, compute_uv=False)
    dim_comm = int((sv < 1e-9 * sv.max()).sum()) if n > 0 else 0
    rows2 = _np.array([_np.concatenate([(X_ @ M_ - M_ @ X_).ravel() for M_ in Mn] + [(X_ @ etan - etan @ X_).ravel()]) for X_ in basis]).T
    sv2 = _np.linalg.svd(rows2, compute_uv=False)
    dim_keep = int((sv2 < 1e-9 * sv2.max()).sum())
    # the predicted solutions A (x) 1 + B (x) G with A, B antisymmetric commute with l_G and lie in so(3n,3n)
    pred = []
    for a_ in range(n):
        for b_ in range(a_ + 1, n):
            A_ = _np.zeros((n, n))
            A_[a_, b_], A_[b_, a_] = 1, -1
            pred += [_np.kron(A_, _np.eye(6)), _np.kron(A_, _G6)]
    okp = all(max(_np.abs(X_ @ M_ - M_ @ X_).max() for M_ in Mn) < 1e-12 and _np.abs((etan @ X_).T + etan @ X_).max() < 1e-12 for X_ in pred)
    okp = okp and (len(pred) == 0 or _np.linalg.matrix_rank(_np.array([X_.ravel() for X_ in pred])) == n * (n - 1))
    found.append((dim_comm, dim_keep))
    ok = ok and dim_comm == n * (n - 1) and dim_keep == n * (n - 1) // 2 and okp
report("bc-12", ok,
       f"(numerical rank, n = 1, 2, 3, 4) the elements of so(3n,3n) commuting with l_G have dimensions {[f[0] for f in found]} = n(n-1), and are the A (x) 1 + B (x) G "
       f"with A, B antisymmetric, the algebra so(n, C) with G as imaginary unit; those keeping the two triples apart have dimensions {[f[1] for f in found]} = dim so(n)")
finish()
