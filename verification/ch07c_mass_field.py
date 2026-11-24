#!/usr/bin/env python3
"""Part II, Chapters 7 and 8 -- the mass as a Lorentz-scalar field, and the coupled action at one constituent.

Hypothesis (A4): the Yang-Mills value in V^- is the scalar m; the mass is a Lorentz-scalar field that can still transform
nontrivially under an internal gauge group, once projected on the four-plane.  Here: the mass is a field m(x) with values in the
antisymmetric kernels invariant under the Lorentz algebra l_int of the four-plane (so that psi^T m psi / 2 is a Lorentz scalar),
on which the internal algebra acts by m -> U m U^T.

    python3 ch07c_mass_field.py

mf-*, ca-*: exact (sympy over the rationals, or matrices with integer entries held as floats).  mr-*: rung two; mr-1 exact,
mr-2 and mr-3 diagonalise integer-valued Casimir matrices numerically and round to quarters.
"""
import itertools

import numpy as np
import sympy as sp

from p2common import report, finish, cl33, tower, prod, comm, acomm, zero

eq = lambda A, B: zero(A - B) or zero(A + B)
rank = lambda mats: np.linalg.matrix_rank(np.array([M.flatten() for M in mats]), tol=1e-9)

# ------------------------------------------------------------------ rung one
e = cl33()
I8 = np.eye(8)
g0, ga = e[0], e[3:6]
J = e[1] @ e[2]
T = e[0] @ e[1] @ e[2]
wS = e[3] @ e[4] @ e[5]
Om = g0 @ wS
Gam = T @ wS
R = [0.5 * ga[1] @ ga[2], 0.5 * ga[2] @ ga[0], 0.5 * ga[0] @ ga[1]]
Bo = [0.5 * g0 @ ga[i] for i in range(3)]
lint = R + Bo
Kk = [-I8] + [-g0 @ ga[a] for a in range(3)]
biv = [e[a] @ e[b] for a, b in itertools.combinations(range(6), 2)]


def n_kernels(gens, symmetric):
    X = sp.Matrix(8, 8, sp.symbols("x0:64"))
    eqs = list(X - X.T) if symmetric else list(X + X.T)
    for Y in gens:
        Ys = sp.Matrix(8, 8, [sp.Rational(v).limit_denominator(4) for v in Y.flatten()])
        eqs += list(Ys.T * X + X * Ys)
    sol = sp.solve(eqs, list(X), dict=True)[0]
    return len(X.subs(sol).free_symbols)


inv = lambda M: all(zero(comm(M, r)) for r in R) and all(zero(acomm(M, b)) for b in Bo)       # Lorentz-invariant kernel (rotations antisymmetric, boosts symmetric)
d = [n_kernels(biv, False), n_kernels(biv, True), n_kernels(lint, False), n_kernels(lint, True)]
report("mf-1", d == [1, 1, 6, 2] and inv(T) and zero(T + T.T),
       f"invariant kernels on R^8: under all of so(3,3), {d[0]} antisymmetric (T) and {d[1]} symmetric; under the Lorentz algebra of the four-plane only, {d[2]} antisymmetric and {d[3]} symmetric: "
       "projected on the four-plane, the space of mass kernels grows from the single T to six dimensions")
sv = [Gam, e[1], e[2]]                                         # sigma(e1), sigma(e2), sigma(e3): the Higgs vectors
g1 = [J, e[2] @ Gam, Gam @ e[1]]                               # sigma(e2e3), sigma(e3e1), sigma(e1e2): the Yang-Mills algebra (twice)
A3 = [v @ wS for v in sv]
B3 = [y @ wS for y in g1]
ok = zero(wS @ wS - I8) and zero(wS - wS.T) and all(zero(comm(wS, r)) for r in R) and all(zero(acomm(wS, b)) for b in Bo)
ok = ok and all(zero(M + M.T) and inv(M) for M in A3 + B3) and rank(A3 + B3) == 6
okS = zero(-Om @ wS + g0) and inv(g0) and zero(g0 - g0.T) and inv(wS) and rank([wS, g0]) == 2
report("mf-2", ok and okS,
       "every Lorentz-invariant kernel is C w_S with C in the commutant (w_S = e4e5e6 is a symmetric involution that commutes with the rotations and anticommutes with the boosts); "
       "with C = sigma(c), c in Cl(V^+): grade 1 (Higgs vectors) and grade 2 (Yang-Mills generators) give the six antisymmetric mass kernels; grade 0 and grade 3 give the symmetric w_S and sigma(T) w_S = -gamma^0")
# equivariance: sigma is a homomorphism and w_S commutes with sigma(bivectors), anticommutes with sigma(vectors)
vec_e = [e[0], e[1], e[2]]
biv_e = [e[1] @ e[2], e[2] @ e[0], e[0] @ e[1]]
sig = {}
for c in itertools.product(range(2), repeat=3):                # sigma on the eight blades of Cl(V^+)
    blade_e = prod([vec_e[i] for i in range(3) if c[i]], 8)
    blade_s = prod([sv[i] for i in range(3) if c[i]], 8)
    sig[c] = (blade_e, blade_s)


def sigma(x):
    """sigma of an element x of Cl(V^+), given as a matrix in the realisation of e1, e2, e3, by blade coefficients"""
    out = np.zeros((8, 8))
    for c, (be, bs) in sig.items():
        coef = np.trace(np.linalg.inv(be) @ x) / 8
        out += coef * bs
    return out


ok = all(zero(comm(0.5 * y, v @ wS) - sigma(comm(0.5 * yb, vb)) @ wS) for y, yb in zip(g1, biv_e) for v, vb in zip(sv, vec_e))
ok = ok and all(zero(comm(0.5 * y, y2 @ wS) - sigma(comm(0.5 * yb, y2b)) @ wS) for y, yb in zip(g1, biv_e) for y2, y2b in zip(g1, biv_e))
ad = np.array([np.concatenate([comm(y, M).flatten() for y in g1]) for M in A3 + B3]).T
n_inv = 6 - np.linalg.matrix_rank(ad, tol=1e-9)
report("mf-3", ok and n_inv == 0,
       f"the internal algebra acts on the mass kernels as on Cl(V^+): [Y, sigma(v) w_S] = sigma([y, v]) w_S and [Y, sigma(y') w_S] = sigma([y, y']) w_S; the Higgs-valued triple is the vector, "
       f"the Yang-Mills-valued triple the adjoint of su(2); invariant kernels: {n_inv}")
rot5 = 3 * I8 + 4 * Om                                         # 5 exp(xi Omega)
rot25 = -7 * I8 + 24 * Om                                      # 25 exp(2 xi Omega)
ok = all(zero(acomm(M, Om)) and zero(rot5.T @ M @ rot5 - M @ rot25) for M in A3 + B3) and rank(B3 + [Om @ M for M in A3]) == 3
report("mf-4", ok,
       "the generator of mass -Omega acts on every mass kernel with charge two, M -> M exp(2 xi Omega), and maps the Higgs-valued kernels onto the Yang-Mills-valued ones: "
       "with Omega as imaginary unit the six kernels are the complex vector of su(2)")
report("mf-5", zero(Gam @ wS - T) and zero(J @ wS - T @ Om) and [bool(zero(comm(M, J))) for M in A3 + B3] == [True, False, False, True, False, False],
       "the kernels that commute with the charge J are T = Gamma w_S, the Higgs value at the pole (scalar Dirac mass), and T Omega = J w_S, the Yang-Mills value along the unbroken generator "
       "(pseudoscalar Dirac mass); the other four anticommute with J")
H = lambda h: h[0] * Gam + h[1] * e[1] + h[2] * e[2]
m = lambda h: H(h) @ wS
h = np.array([2.0, -1.0, 2.0])
M = m(h)
okd = zero(M.T @ M - 9 * I8) and zero(M @ M + 9 * I8) and all(zero(acomm(g0 @ ga[a], M)) for a in range(3)) \
    and all(zero(acomm(g0 @ ga[a], g0 @ ga[b]) - (2 * I8 if a == b else 0)) for a in range(3) for b in range(3))
U5 = 3 * I8 + 4 * (Gam @ e[1])                                 # 5 exp(alpha Gamma e2): an internal rotation of (Gamma, e2)
Mr = U5 @ m([1.0, 0.0, 0.0]) @ U5.T                            # 25 x the rotated mass kernel
okr = eq(Mr, -7 * T + 24 * e[1] @ wS) or eq(Mr, -7 * T - 24 * e[1] @ wS)
neutral = 0.5 * (Mr + J @ Mr @ J.T)                            # the part that commutes with J (J X J^T = X for X commuting with J, -X for X anticommuting)
report("mf-6", okd and okr and (zero(neutral + 7 * T) or zero(neutral - 7 * T)),
       "for h = (2,-1,2): m(h)^T m(h) = |h|^2 and {gamma^0 gamma^a, m(h)} = 0, so (gamma^0 gamma^a d_a + m)^2 = Laplacian - |h|^2: the fermion mass is |h| in every direction of h. "
       "An internal rotation of the Higgs direction by cos = -7/25 with the axis of J held fixed leaves |h| and turns the component neutral under J from T into -7/25 T")

# ------------------------------------------------------------------ the coupled action at one constituent (algebra)
Rm = np.array([[np.trace(U5 @ sv[j] @ U5.T @ sv[i]) / 8 for j in range(3)] for i in range(3)])            # 25 R
hs = [np.array(v) for v in ([2.0, -1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, -4.0])]
okg = zero(U5 @ U5.T - 25 * I8) and all(zero(U5.T @ K @ U5 - 25 * K) for K in Kk) and zero(Rm @ Rm.T - 625 * np.eye(3)) \
    and all(zero(U5 @ m(hh) @ U5.T - m(Rm @ hh)) for hh in hs)
Bx = g0 @ ga[0]                                                 # twice a boost generator, Bx^2 = 1
L4 = 5 * I8 + 3 * Bx                                            # 4 (cosh s + Bx sinh s), cosh s = 5/4, sinh s = 3/4: a finite boost
okl = zero(L4 - L4.T) and all(zero(L4.T @ m(hh) @ L4 - 16 * m(hh)) for hh in hs)
report("ca-1", okg and okl,
       "the mass term psi^T m(h) psi / 2 is invariant under finite internal rotations (U^T K^mu U = K^mu, U m(h) U^T = m(Rh), R in SO(3)) and under finite boosts of the four-plane (Lambda^T m Lambda = m)")
g, Z, v, lam, rho = sp.symbols("g Z v lambda rho", positive=True)
bb = sp.symbols("b1:4", real=True)                              # one Lorentz component of the connection, along (clock axis, e2, e3)
hvec = sp.Matrix([v, 0, 0])
cr = -sp.Matrix(bb).cross(hvec)                                # D h = dh - g b x h (ca-4)
mass_term = sp.expand(Z / 2 * g**2 * (cr.T * cr)[0])
hr = sp.Matrix([v + rho, 0, 0])
V = lam / 4 * ((hr.T * hr)[0] - v**2)**2
mrho2 = sp.simplify(sp.diff(V, rho, 2).subs(rho, 0) / Z)
report("ca-2", sp.simplify(mass_term - Z * g**2 * v**2 / 2 * (bb[1]**2 + bb[2]**2)) == 0 and sp.simplify(mrho2 - 2 * lam * v**2 / Z) == 0,
       "L_h = (Z/2) D_mu h . D^mu h - (lambda/4)(h.h - v^2)^2, D h = dh - g b x h: in the vacuum h = (v,0,0) the component of b along the clock axis is massless, the two along the surplus pair have "
       "M_W^2 = Z g^2 v^2, and the radial mode has M_h^2 = 2 lambda v^2 / Z; with the Yukawa term y psi^T m(h) psi / 2 the fermion mass is m_f = y v (mf-6)")
okc = all(zero((K @ t) + (K @ t).T) for K in Kk for t in g1)
qq = 0.5 * J
okq = zero(qq @ qq + 0.25 * I8) and all(zero(comm(qq, comm(qq, t)) + t) for t in g1[1:]) and zero(comm(qq, g1[0]))
report("ca-3", okc and zero(comm(J, T)) and not zero(comm(g1[1], T)) and not zero(comm(g1[2], T)) and okq,
       "the currents psi^T K^mu t psi / 2 of the internal algebra are nonzero Grassmann bilinears (K^mu t antisymmetric); on shell d_mu j^mu_t = psi^T t m psi, so a current is conserved exactly when t commutes "
       "with the mass: the current of the charge J is conserved in the vacuum m = v T, the two broken currents are not; Q = J/2 has charges +-1/2 on the spinors and +-1 on the broken pair")
okt = all(zero(comm(0.5 * g1[A], H(hh)) + H(np.cross(np.eye(3)[A], hh))) for A in range(3) for hh in hs)
report("ca-4", okt,
       "the generators t_A = (J, e3 Gamma, Gamma e2)/2 about the axes (e1, e2, e3) act on the Higgs vector as [t_A, H(h)] = -H(e_A x h), for three vectors h: "
       "covariance, H(D h) = dH(h) + g [b^A t_A, H(h)], requires D h = dh - g b x h, the convention of Chapter 6 with g replaced by -g")

# ------------------------------------------------------------------ rung two
E = tower(2)
dim = 64
I = np.eye(dim)
Om2 = prod([E[0], E[3], E[4], E[5]], dim)
Gam2 = prod(E, dim)
wS2 = E[3] @ E[4] @ E[5]
P = [E[1], E[2], E[6], E[7], E[8]]
N = [E[9], E[10], E[11]]
rot2 = [E[4] @ E[5], E[5] @ E[3], E[3] @ E[4]]
bo2 = [E[0] @ E[3 + a] for a in range(3)]
gensC = [Gam2] + P + N
blades = [prod([gensC[i] for i in S], dim) for r in range(10) for S in itertools.combinations(range(9), r)]
kern = [b @ wS2 for b in blades if zero(b @ wS2 + (b @ wS2).T)]
okk = all(all(zero(comm(Mk, r)) for r in rot2) and all(zero(acomm(Mk, b)) for b in bo2) for Mk in kern)
report("mr-1", len(blades) == 512 and len(kern) == 272 and okk,
       f"rung two: of the 512 blades C of the commutant Cl(6,3), {len(kern)} give an antisymmetric C w_S, and each is a Lorentz-invariant kernel: the mass field lives in a space of dimension 272 "
       "(= 2 x 136, the complex symmetric mass matrices of sixteen Weyl fields)")
so6 = [0.5 * X @ Y for X, Y in itertools.combinations([Gam2] + P, 2)]
wN = N[0] @ N[1] @ N[2]
Sg = [0.5 * N[1] @ N[2], 0.5 * N[2] @ N[0], 0.5 * N[0] @ N[1]]
lp = [0.5 * (I + wN) @ s for s in Sg]
lm = [0.5 * (I - wN) @ s for s in Sg]
Bf = np.array([Mk.flatten() for Mk in kern])


def casimir(gens):
    return Bf @ np.array([sum(comm(t, comm(t, Mk)) for t in gens).flatten() for Mk in kern]).T / dim


C6, CL, CR = casimir(so6), casimir(lp), casimir(lm)
w, Vv = np.linalg.eigh(C6)
q = np.round(w * 4) / 4
table = {}
for val in sorted(set(q)):
    Vs = Vv[:, q == val]
    jl = np.round(np.linalg.eigvalsh(Vs.T @ (CL + 10 * CR) @ Vs) * 4) / 4
    for x, c in zip(*np.unique(jl, return_counts=True)):
        table[(float(val), float(x))] = int(c)
want = {(-5.0, 0.0): 24, (-8.0, -8.25): 120, (-9.0, -20.0): 60, (-9.0, -2.0): 60, (0.0, -8.25): 8}
report("mr-2", np.abs(C6 @ CL - CL @ C6).max() < 1e-9 and np.abs(CL @ CR - CR @ CL).max() < 1e-9 and table == want,
       "(numerical, Casimir eigenvalues rounded to quarters) under su(4) + su(2) + su(2) the 272 mass kernels are 2(6,1,1) + (15,2,2) + (10,3,1) + (10bar,1,3) + (1,2,2), real dimensions 24, 120, 60, 60, 8: "
       "the Lorentz-scalar bilinears of one Pati-Salam family")
cas6 = lambda Mk: sum(comm(t, comm(t, Mk)) for t in so6)
vecs = [X @ wS2 for X in [Gam2] + P]
adjs = [Y @ wS2 for Y in so6]
G2 = Gam2 @ prod(P, dim)
T2 = prod([E[0], E[1], E[2], E[6], E[7], E[8]], dim)
S2 = prod([E[3], E[4], E[5], E[9], E[10], E[11]], dim)
ok = all(zero(cas6(Mk) + 5 * Mk) and all(zero(comm(l, Mk)) for l in lp + lm) for Mk in vecs)
ok = ok and all(np.abs(cas6(Mk) + 8 * Mk).max() < 1e-9 for Mk in adjs) and not all(zero(comm(l, Mk)) for l in lp + lm for Mk in adjs)
ok = ok and eq(G2 @ wS2, E[0] @ wN) and all(zero(comm(t, G2 @ wS2)) for t in so6) and all(zero(comm(t, S2)) for t in so6) and zero(cas6(T2) + 5 * T2)
ok = ok and zero(T2 + T2.T) and zero(S2 + S2.T)
report("mr-3", ok,
       "rung two: the Higgs-valued kernels sigma(V^+) w_S are a (6,1,1); the Yang-Mills-valued g_2 w_S lie in the (15,2,2) but are not invariant under the two su(2); the six-dimensional masses of Part I, "
       "Thm 9.62, project as T_2 into the (6,1,1) and S_2 into the (1,2,2); the generator of mass times w_S is G_2 w_S = +-gamma^0 w_N, also in the (1,2,2)")
# ------------------------------------------------------------------ a varying Higgs field, and a kernel of M_1 outside the Higgs-valued triple
x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
xx_ = (x1, x2, x3)
es = [sp.Matrix(M.astype(int)) for M in e]
Gs_, wSs = es[0] * es[1] * es[2] * es[3] * es[4] * es[5], es[3] * es[4] * es[5]
al_ = [es[0] * es[3 + a] for a in range(3)]
hf = [sp.Function(f"h{i}")(x1, x2, x3) for i in range(3)]
ms_ = (hf[0] * Gs_ + hf[1] * es[1] + hf[2] * es[2]) * wSs
psi = sp.Matrix([sp.Function(f"psi{i}")(x1, x2, x3) for i in range(8)])
Dop = lambda v: sum((al_[a] * v.diff(xx_[a]) for a in range(3)), sp.zeros(8, 1)) + ms_ * v
lhs = Dop(Dop(psi))
rhs = sum((psi.diff(c, 2) for c in xx_), sp.zeros(8, 1)) - (hf[0]**2 + hf[1]**2 + hf[2]**2) * psi + sum((al_[a] * ms_.diff(xx_[a]) for a in range(3)), sp.zeros(8, 8)) * psi
report("mf-7", (lhs - rhs).applyfunc(sp.expand) == sp.zeros(8, 1),
       "for a Higgs field h(x): (alpha^a d_a + m(h(x)))^2 psi = Laplacian psi - |h|^2 psi + alpha^a (d_a m(h(x))) psi, alpha^a = gamma^0 gamma^a (exact, arbitrary h and psi): "
       "the cross terms that differentiate psi cancel and those that differentiate m remain, so the single mass |h| refers to a constant background")
Mk8 = T + (e[1] @ wS) @ Om
sv8 = np.sort(np.linalg.svd(Mk8, compute_uv=False))
report("mf-8", zero(Mk8 + Mk8.T) and all(zero(L.T @ Mk8 + Mk8 @ L) for L in lint) and np.allclose(sv8, [0] * 4 + [2] * 4, atol=1e-12),
       "T + (e2 w_S) Omega lies in M_1 (antisymmetric, Lorentz invariant) and has the singular values 0 and 2, four times each: the single mass of the Higgs-valued "
       "triple does not extend to all of M_1")
# ------------------------------------------------------------------ the labels of M_2: a kernel carries the charges of the bilinear it multiplies
import p2rung2 as R2
cBL = comm(R2.BmL, R2.Phi0)
cT3 = comm(R2.T3R, R2.Phi0)
qBL = np.trace((R2.Om @ R2.Phi0).T @ cBL) / np.trace(R2.Phi0.T @ R2.Phi0)
qT3 = np.trace((R2.Om @ R2.Phi0).T @ cT3) / np.trace(R2.Phi0.T @ R2.Phi0)
wq, Vq = np.linalg.eig(R2.Om)
Lsp = Vq[:, np.abs(wq - 1j) < 1e-9]
blL = sorted(set(np.round((np.linalg.eigvals(np.linalg.pinv(Lsp) @ R2.BmL @ Lsp) / 1j).real, 6)))
ok = np.abs(cBL - qBL * R2.Om @ R2.Phi0).max() < 1e-9 and abs(qBL - 2) < 1e-9 and np.abs(cT3 - qT3 * R2.Om @ R2.Phi0).max() < 1e-9 and abs(qT3 + 1) < 1e-9
ok = ok and blL == [-1.0, -0.333333, 0.333333, 1.0]
report("mr-4", ok,
       "with Omega as the imaginary unit and the action M -> [t, M], the Majorana kernel Phi_0 has B - L = 2 and T_3R = -1, the charges of the bilinear nu^c nu^c "
       "(B - L on the left-handed fields takes the values -1, -1/3, 1/3, 1): the parts of M_2 are labelled by the bilinears, (10bar,1,3) for the right triplet")
finish()
