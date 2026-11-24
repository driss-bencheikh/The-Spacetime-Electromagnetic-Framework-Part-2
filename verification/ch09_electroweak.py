#!/usr/bin/env python3
"""Part II, Chapter 9 -- the electroweak stage at the second rung.

Setting (Chapters 7 and 8, Definition 1.2): at rung two the gauge algebra is g_2 + s_2 = su(4) + su(2)_L + su(2)_R, with su(4) on the
temporal directions (A2) and su(2)_L + su(2)_R on the spatial triple N = (E10, E11, E12) of the second constituent (A4); su(2)_L is the
half (1 + w_N)/2 of the rotations of N.  The mass field is the 272-dimensional space M_2 of Lorentz-invariant antisymmetric kernels C w_S.
(A3), resolved: the mass is carried by the generator of mass carried into V^-, G_n w_S: at rung one this is the clock -gamma^0, which
acts through its image Gamma (the Higgs vector at the clock, kernel T); at rung two it is the kernel G_2 w_S itself.

    python3 ch09_electroweak.py

ew-1 is exact (integer matrices held as floats).  The other checks are numerical linear algebra on integer-valued matrices: dimensions
are ranks with the stated tolerances, charges and eigenvalues are rounded to the stated fractions after a bound on the rounding error,
and the spectrum of ew-12 is compared with Part I, Example 7.36, to the printed digits.  The normalisations are those of Part I, Section 7.6.1: kernels with unit singular values on their support,
generators orthonormal for the trace form <X, Y> = Tr(X^T Y)/64, g_A = 1.
"""
import itertools

import numpy as np

from p2common import report, finish, tower, prod, comm, zero, cl33

TOL = 1e-8


def rank(mats, tol=TOL):
    return np.linalg.matrix_rank(np.array([m.flatten() for m in mats]), tol=tol)


# ---------------------------------------------------------------- rung one: the generator of mass carried into V^- is the clock
e = cl33()
I8 = np.eye(8)
Gam1 = prod(e, 8)
T1 = e[0] @ e[1] @ e[2]
wS1 = e[3] @ e[4] @ e[5]
Om1 = e[0] @ e[3] @ e[4] @ e[5]
J1 = e[1] @ e[2]
G1 = Gam1 @ e[1] @ e[2]                                   # sigma_1(T) = Gamma e2 e3
g1 = [0.5 * J1, 0.5 * e[1] @ Gam1, 0.5 * e[2] @ Gam1]
ok = zero(G1 + Om1) and zero(G1 @ wS1 + e[0]) and zero(Gam1 @ wS1 - T1)
# the stabiliser of T in g_1 is span{J}; with the global phase Omega the orbit gains one direction
orb_g = rank([comm(t, T1) for t in g1])
orb_gO = rank([comm(t, T1) for t in g1 + [Om1]])
ok = ok and orb_g == 2 and orb_gO == 3 and zero(comm(J1, T1))
# the fixed space of u(1)_J in M_1 = sigma(V^+) w_S + g_1 w_S is the complex line spanned by T and T Omega
M1sp = [X @ wS1 for X in [Gam1, e[1], e[2]]] + [t @ wS1 for t in g1]
fixJ = 6 - rank([comm(J1, M) for M in M1sp]) if rank(M1sp) == 6 else -1
ok = ok and fixJ == 2 and zero(comm(J1, T1 @ Om1)) and zero(T1 @ Om1 - J1 @ wS1)
report("ew-1", ok,
       "rung one: G_1 = sigma_1(T) = -Omega and G_1 w_S = -gamma^0 = -e1, the clock itself; its image under (A1) gives the kernel sigma_1(e1) w_S = Gamma w_S = T. "
       "In g_1 = su(2) the stabiliser of T is u(1)_J (two massive vectors, one photon); the global phase Omega adds one orbit direction; "
       "the fixed space of u(1)_J in the six-dimensional mass field is the complex line {T, T Omega = J w_S}")

# ---------------------------------------------------------------- rung two: the gauge algebra, the family
E = tower(2)
dim = 64
I = np.eye(dim)
ip = lambda X, Y: np.trace(X.T @ Y) / dim
Om = prod([E[0], E[3], E[4], E[5]], dim)
Gam = prod(E, dim)
wS = E[3] @ E[4] @ E[5]
P = [E[1], E[2], E[6], E[7], E[8]]
N = [E[9], E[10], E[11]]
g2 = [0.5 * X @ Y for X, Y in itertools.combinations([Gam] + P, 2)]
wN = N[0] @ N[1] @ N[2]
Sg = [0.5 * N[1] @ N[2], 0.5 * N[2] @ N[0], 0.5 * N[0] @ N[1]]
la = [0.5 * (I + wN) @ s for s in Sg]                     # su(2)_L
lb = [0.5 * (I - wN) @ s for s in Sg]                     # su(2)_R
G21 = g2 + la + lb
Gn = [t / np.sqrt(ip(t, t)) for t in G21]                 # orthonormal for the trace form (the 21 generators are orthogonal)
G2 = Gam @ prod(P, dim)
Gl = [Om, G2]                                             # the two global phases of the commutant (Chapter 8): Omega, G_2 = -Omega w_N
Gln = [t / np.sqrt(ip(t, t)) for t in Gl]
H = [0.5 * P[0] @ P[1], 0.5 * P[2] @ P[3], 0.5 * P[4] @ Gam]
BL = H[0] + H[1] + H[2]                                   # (3/2)(B - L)
A = np.array([comm(BL, t).flatten() for t in g2]).T
r = np.linalg.matrix_rank(A, tol=1e-9)
u3 = [sum(c * t for c, t in zip(row, g2)) for row in np.linalg.svd(A)[2][r:]]
Mx = np.array([comm(a, b).flatten() for a, b in itertools.combinations(u3, 2)])
_, s_, Vt = np.linalg.svd(Mx, full_matrices=False)
su3 = [Vt[i].reshape(dim, dim) for i in range(int((s_ > 1e-9 * s_.max()).sum()))]
T3L, T3R = la[0], lb[0]
Y = T3R + BL / 3.0                                        # T_3R + (B - L)/2
Q = T3L + Y
w_, V_ = np.linalg.eig(Om.astype(complex))
Lq, _ = np.linalg.qr(V_[:, np.abs(w_ - 1j) < 1e-9])       # left-handed fields: Omega = +i
restr = lambda X: Lq.conj().T @ X @ Lq
Cs = sum(restr(t) @ restr(t) for t in su3)
ops = [restr(-1j * T3L), restr(-1j * Y), Cs, restr(-1j * (2.0 / 3.0) * BL)]
Xg = ops[0] + np.pi * ops[1] + np.e * ops[2] + np.sqrt(5) * ops[3]
_, U = np.linalg.eigh((Xg + Xg.conj().T) / 2)
joint, err = {}, 0.0
for k in range(32):
    v = U[:, k]
    vals = [(v.conj() @ op @ v).real for op in ops]
    rnd = (round(vals[0] * 2) / 2, round(vals[1] * 6) / 6, round(vals[2] * 6) / 6, round(vals[3] * 3) / 3)
    err = max(err, *(abs(v - r_) for v, r_ in zip(vals, rnd)))
    col = "1" if rnd[2] == 0 else ("3" if rnd[3] > 0 else "3b")    # a triplet with B - L = +1/3 is a 3, with -1/3 a 3bar
    key = (col, rnd[0], round(rnd[1] * 6), round(rnd[3] * 3))
    joint[key] = joint.get(key, 0) + 1
sm = {("3", 0.5, 1, 1): 6, ("3", -0.5, 1, 1): 6, ("3b", 0.0, -4, -1): 6, ("3b", 0.0, 2, -1): 6, ("1", 0.5, -3, -3): 2, ("1", -0.5, -3, -3): 2,
      ("1", 0.0, 6, 3): 2, ("1", 0.0, 0, 3): 2}                # (colour, T_3L, 6Y, 3(B - L)): counts include the two Lorentz components
report("ew-2", joint == sm and err < 1e-9,
       "(numerical) with Y = T_3R + (B - L)/2 the sixteen left-handed Weyl fields of R^64 are one family of the Standard Model: "
       "(3,2)_{1/6} + (3bar,1)_{-2/3} + (3bar,1)_{1/3} + (1,2)_{-1/2} + (1,1)_{1} + (1,1)_{0}, with B - L = 1/3, -1/3, -1/3, -1, 1, 1 "
       f"(colour from the su(3) Casimir, 3 or 3bar from the sign of B - L; rounding error {err:.1e})")
y, t3, c = restr(-1j * Y), restr(-1j * T3L), restr(-1j * su3[0])
q = restr(-1j * Q)
anom = [np.trace(y).real, np.trace(y @ y @ y).real, np.trace(t3 @ t3 @ y).real, np.trace(c @ c @ y).real, np.trace(q).real]
report("ew-3", max(abs(a) for a in anom) < 1e-12 and abs(np.trace(y @ y).real - 20 / 3) < 1e-12,
       "(numerical) on the left-handed fields Tr Y = Tr Y^3 = Tr T_3L^2 Y = Tr C^2 Y = 0 (C a colour generator): the gravitational, Y^3, "
       "SU(2)^2 Y and SU(3)^2 Y anomalies vanish; Tr Y^2 = 20/3 (10/3 per Lorentz component)")

# ---------------------------------------------------------------- the mass field and its isotypic parts
blades = [prod([([Gam] + P + N)[i] for i in S], dim) for r_ in range(10) for S in itertools.combinations(range(9), r_)]
kern = [b @ wS for b in blades if zero(b @ wS + (b @ wS).T)]
Bf = np.array([M.flatten() for M in kern])
tomat = lambda cvec: (Bf.T @ cvec).reshape(dim, dim)
admat = lambda t: Bf @ np.array([comm(t, M).flatten() for M in kern]).T / dim


def casimir(gens):
    return Bf @ np.array([sum(comm(t, comm(t, M)) for t in gens).flatten() for M in kern]).T / dim


C6, Ca, Cb = casimir(g2), casimir(la), casimir(lb)
Cmix = C6 + np.sqrt(2) * Ca + np.sqrt(3) * Cb
_, Vm = np.linalg.eigh((Cmix + Cmix.T) / 2)
groups = {}
for k in range(len(kern)):
    v = Vm[:, k]
    key = tuple(round(v @ C @ v * 4) / 4 for C in (C6, Ca, Cb))
    groups.setdefault(key, []).append(k)
iso = {key: Vm[:, ks] for key, ks in groups.items()}
name = {(0.0, -0.75, -0.75): "(1,2,2)", (-8.0, -0.75, -0.75): "(15,2,2)", (-9.0, 0.0, -2.0): "(10b,1,3)", (-9.0, -2.0, 0.0): "(10,3,1)",
        (-5.0, 0.0, 0.0): "2(6,1,1)"}
dims_ok = len(kern) == 272 and {name.get(k, k): v.shape[1] for k, v in iso.items()} == {"(1,2,2)": 8, "(15,2,2)": 120, "(10b,1,3)": 60, "(10,3,1)": 60, "2(6,1,1)": 24}
Bpart = {name[k]: v for k, v in iso.items()}


def fix(gens, basis):
    Am = np.vstack([admat(t) for t in gens]) @ basis
    rk = np.linalg.matrix_rank(Am, tol=TOL)
    return basis @ np.linalg.svd(Am)[2][rk:].T


SM = su3 + la + [Y]
fixSM = {k: fix(SM, B).shape[1] for k, B in Bpart.items()}
report("ew-4", dims_ok and fixSM == {"(1,2,2)": 0, "(15,2,2)": 0, "(10b,1,3)": 2, "(10,3,1)": 0, "2(6,1,1)": 0},
       "(numerical) the mass field 272 = 2(6,1,1) + (15,2,2) + (10,3,1) + (10b,1,3) + (1,2,2) (real dimensions 24, 120, 60, 60, 8, read from the Casimirs); "
       "the kernels invariant under su(3) + su(2)_L + u(1)_Y form one complex line Phi_0, and it lies in the (10b,1,3): "
       "the Pati-Salam direction is the only one in the mass field that keeps the algebra of the Standard Model")

# ---------------------------------------------------------------- the Pati-Salam stage
F10b = fix(SM, Bpart["(10b,1,3)"])
M1 = tomat(F10b[:, 0])
s1 = np.linalg.svd(M1, compute_uv=False)
Phi0 = M1 / s1.max()                                      # unit singular values on its support (Part I: |Phi_0|^2 = 4)


def stab(Phi, gens):
    return len(gens) - rank([comm(t, Phi) for t in gens])


def massmat(Phi, gens=Gn):
    Cm = [comm(t, Phi) for t in gens]
    return np.array([[np.trace(a.T @ b) for b in Cm] for a in Cm])


ev0 = np.sort(np.linalg.eigvalsh(massmat(Phi0))) / np.trace(Phi0.T @ Phi0)
pat0 = np.unique(np.round(ev0, 6), return_counts=True)
annul = max(np.abs(comm(t, Phi0)).max() for t in SM)
Mm = massmat(Phi0)
wv, Vv = np.linalg.eigh(Mm)
h8 = Vv[:, np.abs(wv / np.trace(Phi0.T @ Phi0) - 4) < 1e-6]
adY = np.array([[ip(a, comm(Y, b)) for b in Gn] for a in Gn])
ych = np.sort(np.linalg.eigvals(h8.T @ adY @ h8).imag)
ok = stab(Phi0, Gn) == 12 and rank(SM) == 12 and annul < 1e-12 and list(pat0[0]) == [0.0, 4.0, 20.0] and list(pat0[1]) == [12, 8, 1]
ok = ok and np.allclose(ych, [-1, -2 / 3, -2 / 3, -2 / 3, 2 / 3, 2 / 3, 2 / 3, 1], atol=1e-9)
ok = ok and max(np.abs(comm(t, Om @ Phi0)).max() for t in SM) < 1e-12 and zero(Om @ Phi0 + (Om @ Phi0).T, 1e-12) and rank([Phi0, Om @ Phi0]) == 2
report("ew-5", ok,
       "(numerical) the fixed line is spanned by Phi_0 and Omega Phi_0; at Phi_0 the stabiliser in the 21 gauge generators has dimension 12 and is su(3) + su(2)_L + u(1)_Y; the squared vector masses, "
       "relative to |Phi_0|^2, are 0 (12 times), 4 (8 times) and 20 (once): six leptoquarks with Y = +-2/3 and W_R^+- with Y = +-1 at one value, a neutral Z' "
       "five times heavier -- the pattern of Part I, Thm 7.34(b)")
kerY = np.linalg.svd(Y)[2][-4:]                           # the four real directions annihilated by Y on R^64
rangeP = np.linalg.svd(Phi0)[0][:, :4]
ok = list(np.round(s1, 6)).count(0.0) == 60 and np.linalg.matrix_rank(Y, tol=1e-9) == 60
ok = ok and np.abs(rangeP @ rangeP.T - kerY.T @ kerY).max() < 1e-9
report("ew-6", ok,
       "(numerical) Phi_0 has rank 4 and its range is the kernel of Y on R^64: it is a Majorana mass for the right-handed neutrino nu^c alone")
o21, o23 = 21 - stab(Phi0, Gn), 23 - stab(Phi0, Gn + Gln)
ok = o21 == 9 and o23 == 9 and zero(comm(Om, Phi0) + 2 * comm(T3R, Phi0), 1e-12) and zero(comm(G2, Phi0) - comm(Om, Phi0), 1e-12)
report("ew-7", ok,
       "(numerical; no Majoron) at Phi_0 the orbits of the 21 gauge generators and of the 23 generators with the global phases Omega and G_2 both have dimension 9; "
       "[Omega, Phi_0] = [G_2, Phi_0] = -2 [T_3R, Phi_0]: the fermion-number rotation of the nu^c nu^c condensate is a gauge rotation (Part I, Thm 7.41)")

# ---------------------------------------------------------------- the electroweak direction: the generator of mass carried into V^-
Mg = G2 @ wS
cg = Bf @ Mg.flatten() / dim
in122 = np.linalg.norm(Bpart["(1,2,2)"] @ (Bpart["(1,2,2)"].T @ cg)) / np.linalg.norm(cg)
stMg = stab(Mg, Gn)
stab_basis = np.linalg.svd(np.array([comm(t, Mg).flatten() for t in Gn]).T)[2][21 - stMg:]
fixMg = fix([sum(cf * t for cf, t in zip(row, Gn)) for row in stab_basis], np.eye(272)).shape[1]
lor_ok = zero(Mg + Mg.T) and all(zero(comm(Mg, E[3 + i] @ E[3 + j])) for i, j in ((0, 1), (1, 2), (0, 2))) and all(zero(Mg @ E[0] @ E[3 + a] + E[0] @ E[3 + a] @ Mg) for a in range(3))
contains = max(np.abs(comm(t, Mg)).max() for t in g2 + [T3L + T3R]) < 1e-12
others = {"clock Gamma w_S (6,1,1)": stab(Gam @ wS, Gn), "Phi_0 (10b,1,3)": stab(Phi0, Gn)}
report("ew-8", lor_ok and abs(in122 - 1) < 1e-12 and stMg == 18 and contains and fixMg == 2 and others["clock Gamma w_S (6,1,1)"] == 16 and others["Phi_0 (10b,1,3)"] == 12,
       f"(numerical) the generator of mass carried into V^-, G_2 w_S = +-gamma^0 w_N, is a Lorentz-invariant mass kernel in the (1,2,2); its stabiliser has dimension {stMg}, "
       "su(4) + su(2)_V with T_3L + T_3R in su(2)_V, against 16 for the image of the clock vector, Gamma w_S, and 12 for Phi_0; the kernels fixed by its stabiliser "
       "form the complex line through G_2 w_S, so that it is a critical direction of every invariant potential (Michel)")

# ---------------------------------------------------------------- the fermion masses
q2 = -(Q @ Q)
wq, Vq = np.linalg.eigh((q2 + q2.T) / 2)
sect = {}
for val, nm in ((0.0, "nu"), (1 / 9, "d"), (4 / 9, "u"), (1.0, "e")):
    Vs = Vq[:, np.abs(wq - val) < 1e-9]
    sect[nm] = Vs @ Vs.T
sect_dims = {nm: int(round(np.trace(Pp))) for nm, Pp in sect.items()}
EM = su3 + [Q]
fixEM = {k: fix(EM, B) for k, B in Bpart.items()}


def subtype(F, kill):
    Pk = sect[kill[0]] + sect[kill[1]]
    Am = np.array([(tomat(F[:, j]) @ Pk).flatten() for j in range(F.shape[1])]).T
    rk = np.linalg.matrix_rank(Am, tol=TOL)
    return F @ np.linalg.svd(Am)[2][rk:].T


U1, D1 = subtype(fixEM["(1,2,2)"], ("d", "e")), subtype(fixEM["(1,2,2)"], ("u", "nu"))
U15, D15 = subtype(fixEM["(15,2,2)"], ("d", "e")), subtype(fixEM["(15,2,2)"], ("u", "nu"))
ok = sect_dims == {"nu": 8, "d": 24, "u": 24, "e": 8} and {k: v.shape[1] for k, v in fixEM.items()} == {"(1,2,2)": 4, "(15,2,2)": 4, "(10b,1,3)": 2, "(10,3,1)": 2, "2(6,1,1)": 0}
ok = ok and [U1.shape[1], D1.shape[1], U15.shape[1], D15.shape[1]] == [2, 2, 2, 2]
report("ew-9", ok,
       "(numerical) the kernels invariant under su(3) + u(1)_Q form a 12-dimensional space: two each in the (10b,1,3) and (10,3,1), four each in the (15,2,2) and (1,2,2); "
       "in each bidoublet the up-type kernels (zero on the d and e sectors) and the down-type kernels (zero on u and nu) form two-dimensional spaces "
       "(Part I, Prop. 7.35(i)); the charge sectors u, d, nu, e have real dimensions 24, 24, 8, 8")


def sv(M, Pp):
    s = np.linalg.svd(M @ Pp, compute_uv=False)
    return sorted(set(np.round(s[s > 1e-9], 9))), int((s > 1e-9).sum())


Ku, Kd = tomat(U1[:, 0]), tomat(D1[:, 0])
K1u, K1d = Ku / np.linalg.svd(Ku, compute_uv=False).max(), Kd / np.linalg.svd(Kd, compute_uv=False).max()
ok = sv(K1u, sect["u"]) == ([1.0], 24) and sv(K1u, sect["nu"]) == ([1.0], 8) and sv(K1u, sect["d"] + sect["e"]) == ([], 0)
ok = ok and sv(K1d, sect["d"]) == ([1.0], 24) and sv(K1d, sect["e"]) == ([1.0], 8) and sv(K1d, sect["u"] + sect["nu"]) == ([], 0)
upn, dn = np.linalg.norm(U1.T @ cg) ** 2, np.linalg.norm(D1.T @ cg) ** 2
ok = ok and abs(upn - 0.5) < 1e-12 and abs(dn - 0.5) < 1e-12 and zero(Mg @ Mg + I) and sv(Mg, I) == ([1.0], 64)
Ks = [tomat(U15[:, j]) for j in range(2)]
Pu, Pn = sect["u"], sect["nu"]
Al = np.array([(Pu @ K @ Pu).flatten() for K in Ks]).T
coef = np.linalg.lstsq(Al, (Pu @ K1u @ Pu).flatten(), rcond=None)[0]
K15 = sum(cf * K for cf, K in zip(coef, Ks))
ok = ok and np.abs(Al @ coef - (Pu @ K1u @ Pu).flatten()).max() < 1e-12 and np.abs(Pn @ K15 @ Pn + 3 * Pn @ K1u @ Pn).max() < 1e-12
for k1, k15 in ((1.0, 0.3), (2.0, -0.5)):
    Mk = k1 * K1u + k15 * K15
    ok = ok and abs(sv(Mk, Pu)[0][0] - abs(k1 + k15)) < 1e-9 and abs(sv(Mk, Pn)[0][0] - abs(k1 - 3 * k15)) < 1e-9
report("ew-10", ok,
       "(numerical) the (1,2,2) gives m_u = m_nu^D and m_d = m_e; the generator-of-mass kernel G_2 w_S has equal up-type and down-type parts and all 64 singular values 1: "
       "with it alone every fermion of the family has the same Dirac mass; a (15,2,2) kernel aligned with the (1,2,2) gives m_quark = |k_1 + k_15|, "
       "m_lepton = |k_1 - 3 k_15| (the Georgi-Jarlskog factor, Part I, Prop. 7.35(ii))")
ok = True
for vR, kp in ((20.0, 1.0), (3.0, 2.0)):
    Phi = vR * Phi0 + kp * K1u
    s = np.linalg.svd(Phi @ sect["nu"], compute_uv=False)
    s = np.sort(s[s > 1e-9])
    exp_ = [0.5 * (np.sqrt(vR ** 2 + 4 * kp ** 2) - vR)] * 4 + [0.5 * (np.sqrt(vR ** 2 + 4 * kp ** 2) + vR)] * 4
    ok = ok and len(s) == 8 and np.allclose(s, exp_, atol=1e-9) and sv(Phi, sect["u"]) == ([kp], 24)
Phi = 20 * Phi0 + 3 * K1u + 1 * K1d                       # Part I, Example 7.36, second point
s = np.sort(np.linalg.svd(Phi @ sect["nu"], compute_uv=False))
s = s[s > 1e-9]
ok = ok and len(s) == 8 and np.allclose(s, [(np.sqrt(436) - 20) / 2] * 4 + [(np.sqrt(436) + 20) / 2] * 4, atol=1e-9)
ok = ok and sv(Phi, sect["u"]) == ([3.0], 24) and sv(Phi, sect["d"]) == ([1.0], 24) and sv(Phi, sect["e"]) == ([1.0], 8)
report("ew-11", ok,
       "(numerical) for Phi = v_R Phi_0 + k K_1u the neutral sector has singular values (sqrt(v_R^2 + 4k^2) -+ v_R)/2, four times each: the seesaw pair "
       "(Part I, Prop. 7.35(iii)), at (v_R, k) = (20, 1) and (3, 2); at (v_R, v_u, v_d) = (20, 3, 1) the masses are (sqrt(436) -+ 20)/2 for the neutral pair, "
       "3 for the u quarks and 1 for the d quarks and the electron (Part I, Example 7.36)")

# ---------------------------------------------------------------- the vector bosons
ok = True
Phi = 20 * Phi0 + 1.0 * K1u + 0.5 * K1d
ev = np.sort(np.linalg.eigvalsh(massmat(Phi)))
partI = [0] * 9 + [79.360] * 2 + [127.872] + [6400.0] * 6 + [6480.64] * 2 + [32032.1]
ok = ok and np.allclose(ev, partI, atol=0.051) and np.allclose(ev[9:12], [79.360, 79.360, 127.872], atol=5e-4)
ratio20 = ev[9] / ev[11]
Phi200 = 200 * Phi0 + 1.0 * K1u + 0.5 * K1d
ev200 = np.sort(np.linalg.eigvalsh(massmat(Phi200)))
dev200 = ev200[9] / ev200[11] - 5 / 8
ok = ok and stab(Phi, Gn) == 9 and max(np.abs(comm(t, Phi)).max() for t in su3 + [Q]) < 1e-9 and abs(ratio20 - 0.62062) < 5e-6 and abs(dev200 + 4.4e-5) < 5e-7
# the limit v_R -> infinity: the mass matrix of G_2 w_S restricted to the stabiliser of Phi_0
sb = np.linalg.svd(np.array([comm(t, Phi0).flatten() for t in Gn]).T)[2][9:]
Ml = sb @ massmat(Mg) @ sb.T / np.trace(Mg.T @ Mg)
evl = np.sort(np.linalg.eigvalsh(Ml))
ok = ok and np.allclose(evl, [0] * 9 + [2, 2, 3.2], atol=1e-9)
sin2 = ip(T3L, T3L) / (ip(T3L, T3L) + ip(Y, Y))
ok = ok and abs(sin2 - 3 / 8) < 1e-12 and abs(ip(T3L, T3L) - 1 / 8) < 1e-12 and abs(ip(Y, Y) - 5 / 24) < 1e-12
evA3 = np.sort(np.linalg.eigvalsh(massmat(20 * Phi0 + Mg)))
S3w_ = (T3L + T3R) @ wS
ratioA3 = evA3[9] / evA3[11]
ok = ok and abs(ratioA3 - 0.61349) < 5e-6 and zero(Mg - (0.5 * Mg - S3w_) - (0.5 * Mg + S3w_), 1e-12)   # G_2 w_S = K_1u + K_1d: the vacuum (20, 1, 1), printed 0.61349
report("ew-12", ok,
       f"(numerical) at Phi = v_R Phi_0 + v_u K_1u + v_d K_1d, (20, 1, 1/2), the squared vector masses are {', '.join(f'{abs(x):.3f}' for x in sorted(set(np.round(ev, 3))))} "
       f"with multiplicities 9, 2, 1, 6, 2, 1 -- the numbers of Part I, Example 7.36; m_W^2/m_Z^2 = {ratio20:.5f}, and at v_R = 200 it differs from 5/8 by {dev200:.2e}; "
       "the stabiliser is su(3) + u(1)_Q and Q annihilates Phi. Restricted to the stabiliser of Phi_0 the mass matrix of G_2 w_S has relative eigenvalues 0 (9 times), 2, 2, 16/5: "
       f"m_W^2/m_Z^2 = 5/8 = 1 - sin^2 theta_W with sin^2 theta_W = |T_3L|^2/(|T_3L|^2 + |Y|^2) = 3/8. With the generator of mass, Phi = 20 Phi_0 + G_2 w_S = (20, 1, 1): ratio {ratioA3:.10f}, the printed 0.61349")

# ---------------------------------------------------------------- the uneaten direction (Part I, Thm 7.42)
rows = [("generic", (20, 1, 0.5), (9, 10, 12, 13, 1)), ("one zero", (20, 1, 0), (9, 11, 12, 12, 0)), ("v_u = v_d = 0", (20, 0, 0), (12, 14, 9, 9, 0)),
        ("v_R = 0, |v_u| != |v_d|", (0, 1, 0.5), (16, 17, 5, 6, 1)), ("v_R = 0, |v_u| = |v_d|", (0, 1, 1), (18, 19, 3, 4, 1)),
        ("v_R = 0, one zero", (0, 1, 0), (16, 18, 5, 5, 0))]
ok = True
for _, (vR, vu, vd), expct in rows:
    Phi = vR * Phi0 + vu * K1u + vd * K1d
    s21, s23 = stab(Phi, Gn), stab(Phi, Gn + Gln)
    ok = ok and (s21, s23, 21 - s21, 23 - s23, (23 - s23) - (21 - s21)) == expct
PhiA3 = 20 * Phi0 + Mg
ok = ok and (stab(PhiA3, Gn), stab(PhiA3, Gn + Gln)) == (9, 10) and (stab(Mg, Gn), stab(Mg, Gn + Gln)) == (18, 19)
report("ew-13", ok,
       "(numerical) the table of Part I, Thm 7.42, row by row: stabilisers among the 21 gauge and the 23 generators, massive vectors, 23-orbit and uneaten directions "
       "(9,10,12,13,1), (9,11,12,12,0), (12,14,9,9,0), (16,17,5,6,1), (18,19,3,4,1), (16,18,5,5,0); the generator-of-mass vacuum 20 Phi_0 + G_2 w_S is in the generic row, "
       "and G_2 w_S alone in the row v_R = 0, |v_u| = |v_d|")
Phi = 20 * Phi0 + 1.0 * K1u + 0.5 * K1d
B4 = -G2 + (2.0 / 3.0) * BL                               # 4B = (charge +1 on (4,2,1), -1 on (4bar,1,2)) + (B - L); BL = (3/2)(B - L)
X = Om - G2                                               # charge 2 on (4,2,1), 0 on (4bar,1,2)
b4 = np.round(np.sort(np.linalg.eigvals(restr(-1j * B4)).real) * 3) / 3
xx = np.round(np.sort(np.linalg.eigvals(restr(-1j * X)).real) * 2) / 2
cn = restr(-1j * su3[0])
cn = cn / np.sqrt(np.trace(cn @ cn).real / 4)             # on the 8 colour triplets x 2 Lorentz components: Tr_3 c^2 = 1/2
xs = restr(-1j * X)
an = [np.trace(xs @ cn @ cn).real / 2, np.trace(xs @ t3 @ t3).real / 2, np.trace(xs @ y @ y).real / 2, np.trace(xs @ q @ q).real / 2, np.trace(xs).real / 2]
bs = restr(-1j * B4)
anB = [np.trace(bs @ cn @ cn).real / 2, np.trace(bs @ t3 @ t3).real / 2, np.trace(bs @ y @ y).real / 2, np.trace(bs).real / 2]
ok = zero(comm(B4, Phi), 1e-12) and not zero(comm(X, Phi), 1e-6)
ok = ok and list(b4) == [-4 / 3] * 12 + [0.0] * 8 + [4 / 3] * 12 and list(xx) == [0.0] * 16 + [2.0] * 16
ok = ok and np.allclose(an, [2, 4, 4 / 3, 16 / 3, 16], atol=1e-9) and np.allclose(anB, [0, 2, -2, 0], atol=1e-9)
report("ew-14", ok,
       "(numerical) at the generic electroweak vacuum the global charge 4B = -G_2 + (B - L), with values 4/3, 0, -4/3 (baryon number times four), annihilates Phi; "
       "the broken global charge Omega - G_2 is 2 on the (4,2,1) and 0 on the (4bar,1,2), with anomalies SU(3)^2: 2, SU(2)_L^2: 4, Y^2: 4/3, Q^2: 16/3, "
       "gravitational: 16, and 4B has 0, 2, -2, 0 -- the table of Part I after Thm 7.42, with 1 - Gamma there in the place of Omega - G_2 here")
# ---------------------------------------------------------------- the charge of the first rung at the second
H1 = H[0]
ok = zero(H1 - 0.5 * E[1] @ E[2])                         # J/2 of the first constituent (J = E2 E3 lies in P, fixed by sigma_2)
nH = ip(H1, H1)
Bs = np.array([t.flatten() for t in su3])
cc = np.linalg.solve(Bs @ Bs.T / dim, Bs @ H1.flatten() / dim)
Hc = sum(ci * t for ci, t in zip(cc, su3))
fc, fq = ip(Hc, Hc) / nH, ip(H1, Q) ** 2 / ip(Q, Q) / nH
w0, V0 = np.linalg.eigh(massmat(Phi0))
Zp = sum(ci * t for ci, t in zip(V0[:, np.argmax(w0)], Gn))        # the Z' of the Pati-Salam stage (eigenvalue 20, once)
wl, Vl = np.linalg.eigh(sb @ massmat(Mg) @ sb.T)
Zb = sum(ci * t for ci, t in zip(sb.T @ Vl[:, np.argmax(wl)], Gn))  # the Z in the limit v_R -> infinity (eigenvalue 16/5)
fz, fzp = ip(H1, Zb) ** 2 / ip(Zb, Zb) / nH, ip(H1, Zp) ** 2 / ip(Zp, Zp) / nH
zdir = T3L - 0.375 * Q
okz = abs(abs(ip(Zb, zdir)) / np.sqrt(ip(Zb, Zb) * ip(zdir, zdir)) - 1) < 1e-9
shares = [ip(Q, X) ** 2 / ip(X, X) / ip(Q, Q) for X in (T3L, T3R, H[0], H[1], H[2])]
specJ = sorted(set(np.round(np.linalg.eigvals(restr(-1j * H1)).real, 9)))
specQ = sorted(set(np.round(np.linalg.eigvals(restr(-1j * Q)).real * 3) / 3))
ok = ok and np.allclose([fc, fq, fz, fzp], [2 / 3, 1 / 12, 1 / 20, 1 / 5], atol=1e-9) and okz and np.allclose(shares, [3 / 8, 3 / 8, 1 / 12, 1 / 12, 1 / 12], atol=1e-12)
ok = ok and specJ == [-0.5, 0.5] and specQ == [-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0]
report("ew-15", ok,
       "(numerical) the charge J/2 of the first rung is, at the second, the Cartan generator H_1 = E2 E3/2 of su(4), with the values +-1/2 on every field, while Q takes 0, +-1/3, +-2/3, +-1. "
       "In the trace form and in the limit v_R -> infinity, the norm of H_1 lies 2/3 along colour, 1/12 along the photon Q, 1/20 along the Z (along T_3L - (3/8) Q) and 1/5 along the Z'; "
       "the norm of Q lies 3/8 along T_3L, 3/8 along T_3R and 1/12 along each of H_1, H_2, H_3")
# ---------------------------------------------------------------- the split of the masses (the Yang-Mills values of the neutral generators)
S3w = (T3L + T3R) @ wS                                    # the Yang-Mills value of T_3L + T_3R carried into V^-
BmL = (2.0 / 3.0) * BL                                    # B - L
Bw = BmL @ wS                                             # the Yang-Mills value of B - L carried into V^-


def zpattern(m):
    """complex values of a neutral kernel on the Dirac pairs u, d, nu, e, relative to G_2 w_S (Omega as the imaginary unit)"""
    out, res = [], 0.0
    for s_ in ("u", "d", "nu", "e"):
        Ps = sect[s_]
        R = Ps @ Mg @ Ps
        A2 = np.array([R.flatten(), (Om @ R).flatten()]).T
        c2 = np.linalg.lstsq(A2, (Ps @ m @ Ps).flatten(), rcond=None)[0]
        res = max(res, np.abs(A2 @ c2 - (Ps @ m @ Ps).flatten()).max())
        out.append(complex(c2[0], c2[1]))
    return np.array(out), res


expect = {"T3": [-0.5, 0.5, -0.5, 0.5], "B-L": [-1 / 3, -1 / 3, 1, 1], "Q": [-2 / 3, 1 / 3, 0, 1]}
ok = True
for nm, X in (("T3", T3L + T3R), ("B-L", BmL), ("Q", Q)):
    zp, res = zpattern(X @ wS)
    ok = ok and zero(X @ wS + (X @ wS).T, 1e-12) and res < 1e-9 and np.allclose(zp, expect[nm], atol=1e-9)
    zo, res = zpattern(Om @ X @ wS)
    ok = ok and res < 1e-9 and np.allclose(zo, 1j * np.array(expect[nm]), atol=1e-9)
ok = ok and zero((T3L - T3R) @ wS - ((T3L - T3R) @ wS).T, 1e-12)        # T_3L - T_3R gives a symmetric kernel: no mass
S2v = prod([E[3], E[4], E[5], E[9], E[10], E[11]], dim)                # the spatial volume of the rung
ok = ok and zero(Om @ Mg + S2v)                                     # Omega G_2 w_S = -S_2 (exact)
Kup, Kdn = 0.5 * Mg - S3w, 0.5 * Mg + S3w
ok = ok and sv(Kup, sect["u"]) == ([1.0], 24) and sv(Kup, sect["nu"]) == ([1.0], 8) and sv(Kup, sect["d"] + sect["e"]) == ([], 0)
ok = ok and sv(Kdn, sect["d"]) == ([1.0], 24) and sv(Kdn, sect["e"]) == ([1.0], 8) and sv(Kdn, sect["u"] + sect["nu"]) == ([], 0)
cS, cB = Bf @ S3w.flatten() / dim, Bf @ Bw.flatten() / dim
ok = ok and abs(np.linalg.norm(Bpart["(1,2,2)"].T @ cS) / np.linalg.norm(cS) - 1) < 1e-12
ok = ok and abs(np.linalg.norm(Bpart["(15,2,2)"].T @ cB) / np.linalg.norm(cB) - 1) < 1e-12
span6 = [Mg, S3w, Bw, Om @ Mg, Om @ S3w, Om @ Bw]
N8 = np.hstack([fixEM["(1,2,2)"], fixEM["(15,2,2)"]])                # the neutral bidoublet kernels, 8 real dimensions
C6 = np.array([Bf @ M.flatten() / dim for M in span6]).T
inN8 = np.abs(N8 @ (N8.T @ C6) - C6).max() < 1e-9
Pc = N8 - C6 @ np.linalg.lstsq(C6, N8, rcond=None)[0]
rest = np.linalg.svd(Pc, full_matrices=False)[0][:, :2]
zr, res = zpattern(tomat(rest[:, 0]))
ok = ok and rank(span6) == 6 and inN8 and res < 1e-9 and np.allclose(zr / zr[0], [1, -1, -3, 3], atol=1e-9)
report("ew-16", ok,
       "(numerical) the Yang-Mills value of a neutral generator X carried into V^-, X w_S, is a mass kernel that gives each Dirac pair (u, d, nu, e) the value -X(f_L) "
       "relative to G_2 w_S, minus the charge of the left-handed member: (-1/2, 1/2, -1/2, 1/2) for T_3L + T_3R, (-1/3, -1/3, 1, 1) for B - L, (-2/3, 1/3, 0, 1) for Q; "
       "(T_3L - T_3R) w_S is symmetric; Omega multiplies the values by i, and Omega G_2 w_S = -S_2, the spatial volume of the rung (exact). G_2 w_S/2 -+ (T_3L + T_3R) w_S are the up-type and down-type kernels (unit singular values on u, nu "
       "and on d, e); (T_3L + T_3R) w_S lies in the (1,2,2) and (B - L) w_S in the (15,2,2). With their Omega-partners they span 6 of the 8 neutral bidoublet dimensions; "
       "the other two give the values (1, -1, -3, 3)")
ok = True
for k_, x3, xB in ((1.0, 0.6, 0.0), (1.0, 0.0, 0.3), (1.0, -1.98, 0.004), (0.7, 0.5, -0.2)):
    m_ = k_ * Mg + x3 * S3w + xB * Bw
    zz = [k_ - x3 / 2 - xB / 3, k_ + x3 / 2 - xB / 3, k_ - x3 / 2 + xB, k_ + x3 / 2 + xB]
    for s_, z_ in zip(("u", "d", "nu", "e"), zz):
        got = sv(m_, sect[s_])
        ok = ok and len(got[0]) == 1 and abs(got[0][0] - abs(z_)) < 1e-9
    ok = ok and abs((zz[0] - zz[1]) - (zz[2] - zz[3])) < 1e-12
    ok = ok and np.allclose([(2 * zz[0] + zz[1] + zz[3]) / 4, zz[1] - zz[0], 0.75 * (zz[3] - zz[1])], [k_, x3, xB], atol=1e-12)
report("ew-17", ok,
       "(numerical) for the vacuum m = k G_2 w_S + x_3 (T_3L + T_3R) w_S + x_B (B - L) w_S each Dirac pair has the mass |k - X(f_L)|: "
       "m_u = |k - x_3/2 - x_B/3|, m_d = |k + x_3/2 - x_B/3|, m_nu^D = |k - x_3/2 + x_B|, m_e = |k + x_3/2 + x_B|, at four points including (1, -1.98, 0.004); "
       "the signed values obey z_u - z_d = z_nu - z_e, and (k, x_3, x_B) = ((2 z_u + z_d + z_e)/4, z_d - z_u, 3 (z_e - z_d)/4)")
ok = True
for (k_, x3, xB), expct in (((1, 0, 0), (9, 10)), ((1, -2, 0), (9, 11)), ((1, 2, 0), (9, 11)), ((1, -2, 0.3), (9, 10)), ((1, 0.7, 0.3), (9, 10)),
                            ((1, -1.98, 0.004), (9, 10))):
    Phi = 20 * Phi0 + k_ * Mg + x3 * S3w + xB * Bw
    ok = ok and (stab(Phi, Gn), stab(Phi, Gn + Gln)) == expct
Phi = 20 * Phi0 + 2 * Kup
Aq = np.array([comm(t, Phi).flatten() for t in Gn + Gln]).T
rk = np.linalg.matrix_rank(Aq, tol=1e-8)
Hs = [sum(cf * t for cf, t in zip(row, Gn + Gln)) for row in np.linalg.svd(Aq)[2][rk:]]
Ah = np.vstack([admat(h) for h in Hs])
fdim = len(kern) - np.linalg.matrix_rank(Ah, tol=1e-8)
FB = np.linalg.svd(Ah)[2][len(kern) - fdim:]
dist = {k: int(round(np.linalg.norm(B.T @ FB.T) ** 2)) for k, B in Bpart.items()}
ok = ok and fdim == 8 and dist == {"(1,2,2)": 2, "(15,2,2)": 2, "(10b,1,3)": 2, "(10,3,1)": 2, "2(6,1,1)": 0}
Pdn, Pup_ = sect["d"] + sect["e"], sect["u"] + sect["nu"]
bid = [tomat(Bpart["(1,2,2)"] @ (Bpart["(1,2,2)"].T @ fb) + Bpart["(15,2,2)"] @ (Bpart["(15,2,2)"].T @ fb)) for fb in FB]
ok = ok and max(np.linalg.norm(Pdn @ M_ @ Pdn) for M_ in bid) < 1e-9 and rank(bid) == 4 and min(np.linalg.norm(Pup_ @ M_ @ Pup_) for M_ in bid if np.linalg.norm(M_) > 1e-6) > 1e-3
Aq9 = np.array([comm(t, Phi).flatten() for t in Gn]).T
rk9 = np.linalg.matrix_rank(Aq9, tol=1e-8)
Hs9 = [sum(cf * t for cf, t in zip(row, Gn)) for row in np.linalg.svd(Aq9)[2][rk9:]]
fix9 = len(kern) - np.linalg.matrix_rank(np.vstack([admat(h) for h in Hs9]), tol=1e-8)
ok = ok and len(Hs9) == 9 and fix9 == 12
report("ew-18", ok,
       "(numerical) with 20 Phi_0 added the stabiliser of the split vacuum is su(3) + u(1)_Q (9) among the 21 gauge generators; among the 23 with the global phases it is 11 "
       "at x_3 = -+2k, x_B = 0 -- the pure up-type and pure down-type kernels, no uneaten direction -- and 10 otherwise. At the pure up-type vacuum the kernels fixed by "
       "the extended stabiliser span the lines of Phi_0, of the neutral (10,3,1) and of the up-type kernels of the (1,2,2) and (15,2,2) (8 dimensions): each fixed "
       "vector has no down-sector bidoublet part, and their bidoublet parts span four dimensions; the gauge stabiliser alone (9) fixes all twelve neutral directions")
# ---------------------------------------------------------------- the Gram matrix in closed form
Kup2, Kdn2 = 0.5 * Mg - S3w, 0.5 * Mg + S3w
ok, worst = True, 0.0
for vR, vu, vd in ((20, 1, 0.5), (20, 1, 1), (7, 0.3, 1.9)):
    evK = np.sort(np.linalg.eigvalsh(massmat(vR * Phi0 + vu * Kup2 + vd * Kdn2)))
    r_, C_ = vR**2, 64 * (vu**2 + vd**2)
    ch = np.linalg.eigvalsh(np.array([[C_, -128 * vu * vd], [-128 * vu * vd, C_ + 16 * r_]]))
    ne = np.linalg.eigvalsh(np.array([[C_, -C_, 0], [-C_, C_ + 32 * r_, -16 * np.sqrt(6) * r_], [0, -16 * np.sqrt(6) * r_, 48 * r_]]))
    pred = np.sort(np.concatenate([[0.0] * 8, ch, ch, [16.0 * r_] * 6, ne]))
    worst = max(worst, np.abs(evK - pred).max() / max(1.0, evK.max()))
    rts = np.sort(np.roots([1, -(2 * C_ + 80 * r_), 128 * C_ * r_]))
    ok = ok and abs(np.sort(ne)[0]) < 1e-9 * evK.max() and np.allclose(np.sort(ne)[1:], rts, rtol=1e-12)
# the entries, with their signs, in the named bases: neutral (T_3L, T_3R, B - L) and charged (the second generators of su(2)_L and su(2)_R)
nb_ = [t / np.sqrt(ip(t, t)) for t in (T3L, T3R, BmL)]
cb_ = [t / np.sqrt(ip(t, t)) for t in (la[1], lb[1])]
went = 0.0
for vR, vu, vd in ((20, 1, 0.5), (20, 1, 1), (7, 0.3, 1.9)):
    Ph_ = vR * Phi0 + vu * Kup2 + vd * Kdn2
    r_, C_ = vR**2, 64 * (vu**2 + vd**2)
    Kn_ = np.array([[np.trace(comm(x, Ph_).T @ comm(y, Ph_)) for y in nb_] for x in nb_])
    Kc_ = np.array([[np.trace(comm(x, Ph_).T @ comm(y, Ph_)) for y in cb_] for x in cb_])
    pn_ = np.array([[C_, -C_, 0], [-C_, C_ + 32 * r_, -16 * np.sqrt(6) * r_], [0, -16 * np.sqrt(6) * r_, 48 * r_]])
    pc_ = np.array([[C_, -128 * vu * vd], [-128 * vu * vd, C_ + 16 * r_]])
    went = max(went, np.abs(Kn_ - pn_).max() / pn_.__abs__().max(), np.abs(Kc_ - pc_).max() / pc_.__abs__().max())
ok = ok and went < 1e-12
evL = np.sort(np.linalg.eigvalsh(massmat(2000 * Phi0 + 1.0 * Kup2 + 0.5 * Kdn2)))
CL = 64 * 1.25
ok = ok and worst < 1e-12 and abs(evL[9] / CL - 1) < 1e-4 and abs(evL[11] / (8 * CL / 5) - 1) < 1e-4
report("ew-19", ok,
       f"(numerical, against a closed form) at Phi = v_R Phi_0 + v_u K_1u + v_d K_1d, with K_1u, K_1d = G_2 w_S/2 -+ (T_3L + T_3R) w_S, r = v_R^2 and C = 64(v_u^2 + v_d^2), "
       f"the Gram matrix has the eigenvalues 0 (8, gluons), 16 r (6, leptoquarks), those of [[C, -128 v_u v_d], [-128 v_u v_d, C + 16 r]] twice, and those of the neutral "
       f"block [[C, -C, 0], [-C, C + 32 r, -16 sqrt6 r], [0, -16 sqrt6 r, 48 r]], one zero and the roots of l^2 - (2C + 80r) l + 128 C r, at three vacua (relative "
       f"deviation {worst:.0e}); the entries themselves, with the signs of the off-diagonal terms, agree in the named bases (normalised T_3L, T_3R, B - L; second generators of "
       f"su(2)_L and su(2)_R) to {went:.0e}; at v_R = 2000 the light roots are C and 8C/5 to 1e-4, so m_W^2/m_Z^2 -> 5/8")
# ---------------------------------------------------------------- the uneaten direction: gauge-invariant quadratic forms on the bidoublet
B122 = Bpart["(1,2,2)"]
actp = lambda t: B122.T @ admat(t) @ B122
idx_ = [(i, j) for i in range(8) for j in range(i, 8)]


def symm8(v):
    Mq = np.zeros((8, 8))
    for k_, (i, j) in enumerate(idx_):
        Mq[i, j] = Mq[j, i] = v[k_]
    return Mq


rows = np.vstack([np.array([(Ag.T @ symm8(u_) + symm8(u_) @ Ag).ravel() for u_ in np.eye(len(idx_))]).T for Ag in [actp(t) for t in Gn]])
_, sv_, Vt_ = np.linalg.svd(rows)
forms = [symm8(v_) for v_ in Vt_[(sv_ > 1e-9 * sv_.max()).sum():]]
Xb = actp(Om - G2)
rowsb = np.array([(Xb.T @ Fm + Fm @ Xb).ravel() for Fm in forms]).T
inv_dim = len(forms) - np.linalg.matrix_rank(rowsb, tol=1e-9)
leak = max(np.abs(admat(t) @ B122 - B122 @ actp(t)).max() for t in Gn + [Om, G2])
from scipy.linalg import expm as _expm


def curv(vu, vd, Fm):
    c_ = B122.T @ (Bf @ (vu * Kup2 + vd * Kdn2).flatten() / dim)
    q_ = lambda e_: (_expm(e_ * Xb) @ c_) @ Fm @ (_expm(e_ * Xb) @ c_)
    h_ = 1e-3
    return (q_(h_) - 2 * q_(0) + q_(-h_)) / h_**2


brk = [Fm for Fm in forms if np.abs(Xb.T @ Fm + Fm @ Xb).max() > 1e-6]
c_gen = max(abs(curv(1.0, 0.5, Fm)) for Fm in brk)
c_one = max(abs(curv(1.0, 0.0, Fm)) for Fm in brk)


def slope(vu, vd, Fm):
    c_ = B122.T @ (Bf @ (vu * Kup2 + vd * Kdn2).flatten() / dim)
    q_ = lambda e_: (_expm(e_ * Xb) @ c_) @ Fm @ (_expm(e_ * Xb) @ c_)
    h_ = 1e-4
    return (q_(h_) - q_(-h_)) / (2 * h_)


# at the real diagonal background the slope and the curvature are independent functionals on the forms: one breaking form (Re det F)
# has zero slope and nonzero curvature, the other (Im det F) a slope, as -2q mu^2 Im det F and -4q^2 mu^2 Re det F require when det F is real
SC = np.array([[slope(1.0, 0.5, Fm) for Fm in forms], [curv(1.0, 0.5, Fm) for Fm in forms]])
split_ok = np.linalg.matrix_rank(SC, tol=1e-6 * np.abs(SC).max()) == 2
report("ew-20", leak < 1e-12 and len(forms) == 3 and inv_dim == 1 and c_gen > 0.1 and c_one < 1e-6 and split_ok,
       f"(numerical) on the (1,2,2) the gauge-invariant quadratic forms span a space of dimension {len(forms)}, of which {inv_dim} (the norm) is invariant under the broken "
       f"phase Omega - G_2: the other two, Re det F and Im det F, break it; along the orbit of Omega - G_2 a breaking form has second derivative up to {c_gen:.2f} at "
       f"(v_u, v_d) = (1, 1/2) and none ({c_one:.0e}) at (1, 0), at the real diagonal backgrounds v_u K_1u + v_d K_1d, where det F = v_u v_d is real; there the slope and the "
       "curvature along the orbit are independent functionals on the forms, one breaking form (Re det F) having zero slope and the other (Im det F) a slope: an explicit term "
       "mu^2 Re det F gives the uneaten direction a curvature at tree level when v_u v_d != 0")
# ---------------------------------------------------------------- the Majorana line is not transported
Sd = [0.5 * N[1] @ N[2], 0.5 * N[2] @ N[0], 0.5 * N[0] @ N[1]]
tblades = [prod([([Gam] + P)[i] for i in S_], dim) for r_ in range(7) for S_ in itertools.combinations(range(6), r_)]
okt = all(zero(comm(si, b_ @ wS)) for si in Sd for b_ in tblades) and all(zero(si - la[i] - lb[i]) for i, si in enumerate(Sd))
nr = [np.linalg.norm(comm(si, Phi0)) / np.linalg.norm(Phi0) for si in Sd]
nro = [np.linalg.norm(comm(si, Om @ Phi0)) / np.linalg.norm(Phi0) for si in Sd]
report("ew-21", okt and np.allclose(sorted(nr), [1 / np.sqrt(2), 1 / np.sqrt(2), 1.0], atol=1e-12) and np.allclose(sorted(nro), sorted(nr), atol=1e-12),
       "every kernel C w_S with C a blade of the transported temporal algebra (64 blades of (Gamma; P)) commutes with the diagonal weak rotations S_i = T_i^L + T_i^R (exact), "
       f"while ||[S_i, Phi_0]||/||Phi_0|| = {', '.join(f'{x:.6f}' for x in nr)}, and the same for Omega Phi_0: no nonzero kernel on the Majorana line is sigma_2(a) w_S")
# ---------------------------------------------------------------- the Yang-Mills values as backgrounds: X -> X w_S is not weak-equivariant
okq = zero(comm(T3L, BmL)) and not zero(comm(T3L, Bw), 1e-9) and all(zero(comm(t, wS)) for t in g2) and all(zero(comm(si, wS)) for si in Sd)
okq = okq and all(zero(N[i] @ wS + wS @ N[i]) for i in range(3))
report("ew-22", okq,
       "X -> X w_S is equivariant under su(4) (g_2 commutes with w_S) and under the diagonal su(2)_V (S_i commute with w_S), not under su(2)_L + su(2)_R: the generators N_i/2 "
       "anticommute with w_S, and B - L commutes with T_3L while (B - L) w_S does not (exact): the Yang-Mills values are backgrounds built from the neutral generators")
# ---------------------------------------------------------------- relative alignment of the bidoublet and the Majorana condensate
from scipy.linalg import expm as _expm2
qr_ = lambda M_: np.trace((M_ @ M_.T) @ (M_ @ M_.T)) / np.trace(M_ @ M_.T) ** 2
okA, rowsA = True, []
for th in (np.pi / 16, np.pi / 8, np.pi / 4):                 # 2 lb[1] has eigenvalues +-i on the right-handed doublets
    U_ = _expm2(th * 2 * lb[1])
    F_ = U_ @ Kup2 @ U_.T
    Ph_ = 20 * Phi0 + F_
    st_, sc_ = stab(Ph_, Gn), stab(Ph_, su3)
    qok_ = np.abs(comm(Q, Ph_)).max() < 1e-9
    G_ = U_ @ Mg @ U_.T
    stg_ = stab(20 * Phi0 + G_, Gn)
    rowsA.append((round(th, 4), st_, sc_, qok_, stg_))
    okA = okA and abs(qr_(F_) - 1 / 32) < 1e-12 and abs(qr_(Kup2) - 1 / 32) < 1e-12 and st_ == 8 and sc_ == 8 and not qok_ and stg_ == 9
okA = okA and stab(20 * Phi0 + Kup2, Gn) == 9 and abs(qr_(Phi0) - 1 / 4) < 1e-12
report("ew-23", okA,
       f"(numerical) with 20 Phi_0 held fixed, rotating only the rank-one up-type kernel K_1u by SU(2)_R, exp(theta 2 T_R) with T_R the second generator of su(2)_R, keeps its quartic "
       f"ratio 1/32 (and 1/4 for Phi_0) but at theta = pi/16, pi/8, pi/4 the joint gauge stabiliser has dimension 8, all of it colour, and Q no longer annihilates the vacuum "
       f"(theta, stab, colour, Q kept, stab with G_2 w_S rotated instead: {rowsA}); aligned, the stabiliser is 9. The same rotation of G_2 w_S keeps dimension 9: G_2 w_S is fixed by "
       "su(2)_V, so its SU(2)_R rotation equals an SU(2)_L rotation, which fixes Phi_0. The relative orientation is physical for a rank-one bidoublet")
finish()
