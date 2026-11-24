#!/usr/bin/env python3
"""Part II, Chapter 9 -- what a gauge-invariant potential of the mass field does at the electroweak stage.

The vacuum of Chapter 9 is Phi = v_R Phi_0 + (the split of the values of the hypothesis); pt-7 checks the split in its general form, with the value along D,
and pt-8 the gradient of the trace quartic in closed form.  No potential is given by the hypothesis.  Here: what every
gauge-invariant potential must do there (symmetric criticality), and what four explicit gauge- and phase-invariant quartics do, in
particular along the neutral direction D with the values (1, -1, -3, 3) that the split excludes, on which the sum rule rests.

The quartics are traces of products of the equivariant projections of m onto the parts of the mass field (S = (15,2,2), F = (1,2,2),
R = (10b,1,3), L = (10,3,1)): under m -> U m U^T with U orthogonal and commuting with the gauge algebra and the global phases,
Tr(A B^T C D^T) is invariant.  Their gradients are computed exactly (the derivative of a multilinear trace).

    python3 ch09b_potential.py

All checks are numerical linear algebra on integer-valued matrices; ratios are compared to the stated rationals within 1e-9.
"""
import numpy as np

from p2common import report, finish, comm
from p2rung2 import (Gn, Gln, Om, Mg, S3w, Bw, Phi0, Dmiss, DL, kern, coords, tomat, admat, Proj, Bpart, neutral12, NB,
                     neutral_components, zpattern, sect, stab, rank, dim)

ID = None                                                  # marks a slot that takes m itself (no projection)
S_, F_, R_, L_ = "(15,2,2)", "(1,2,2)", "(10b,1,3)", "(10,3,1)"
INV = {"I1": (ID, ID, ID, ID),                             # Tr((m m^T)^2), invariant under all of O(64)
       "I2": (S_, F_, R_, R_),                             # Tr(S F^T R R^T)
       "I3": (F_, F_, R_, R_),                             # Tr(F F^T R R^T)
       "I4": (L_, F_, R_, F_)}                             # Tr(L F^T R F^T)


def slots(name, M):
    return [M if r is ID else tomat(Proj[r] @ coords(M)) for r in INV[name]]


def pj(r, c):
    return c if r is ID else Proj[r] @ c


def grad(name, Phi):
    """gradient of Tr(A B^T C D^T) at Phi, in the coordinates of the mass field"""
    A, B, C, D = slots(name, Phi)
    ra, rb, rc, rd = INV[name]
    return pj(ra, coords(D @ C.T @ B)) + pj(rb, coords(C @ D.T @ A)) + pj(rc, coords(B @ A.T @ D)) + pj(rd, coords(A @ B.T @ C))


def vac(vR, k):
    return vR * Phi0 + k * Mg


# ---------------------------------------------------------------- pt-1: what the symmetry of the vacuum forces
Phi = vac(20, 1)
ok = (stab(Phi, Gn), stab(Phi, Gn + Gln)) == (9, 10) and rank(list(neutral12.values())) == 12
for gens in (Gn, Gn + Gln):
    Aq = np.array([comm(t, Phi).flatten() for t in gens]).T
    rk = np.linalg.matrix_rank(Aq, tol=1e-8)
    Hs = [sum(cf * t for cf, t in zip(row, gens)) for row in np.linalg.svd(Aq)[2][rk:]]
    Ah = np.vstack([admat(h) for h in Hs])
    fd = len(kern) - np.linalg.matrix_rank(Ah, tol=1e-8)
    FB = np.linalg.svd(Ah)[2][len(kern) - fd:]
    ok = ok and fd == 12 and np.abs(FB.T @ (FB @ NB) - NB).max() < 1e-9          # Fix(stabiliser) = the twelve neutral directions
report("pt-1", ok,
       "(numerical) at the electroweak vacuum 20 Phi_0 + G_2 w_S the kernels fixed by its stabiliser, among the 21 gauge generators (9) and among the 23 with the global "
       "phases (10), are the same twelve neutral directions: by symmetric criticality the gradient of every gauge-invariant potential lies there, so its tadpoles keep "
       "su(3) + u(1)_Q; and they include the direction D (1, -1, -3, 3) that the split excludes: no continuous symmetry forbids a tadpole along D")

# ---------------------------------------------------------------- pt-2: the tadpoles of four quartics lie in the neutral space, with real values
ok = True
comp = {}
for nm in INV:
    g = grad(nm, Phi)
    co, res = neutral_components(g)
    comp[nm] = co
    scale = np.linalg.norm(g)
    ok = ok and scale > 0 and res < 1e-9 * scale and all(abs(co[k]) < 1e-9 * scale for k in ("OG", "OS3", "OB", "OD", "OPhi0"))
report("pt-2", ok,
       "(numerical) the gradients of I1 = Tr((m m^T)^2), I2 = Tr(S F^T R R^T), I3 = Tr(F F^T R R^T) and I4 = Tr(L F^T R F^T) at 20 Phi_0 + G_2 w_S lie in the twelve "
       "neutral directions, and have no component along the products by Omega of the bidoublet directions and of Phi_0 (the phase of the neutral (10,3,1) "
       "is a convention here)")


# ---------------------------------------------------------------- pt-3: what each quartic induces, and how it scales
def dirac(co):
    m_ = sum(co[k] * neutral12[k] for k in ("G", "S3", "B", "D"))
    return zpattern(m_).real


ok = True
c20, c10, c22 = [{nm: neutral_components(grad(nm, vac(vR, k)))[0] for nm in INV} for vR, k in ((20, 1), (10, 1), (20, 2))]
# I1: a common shift plus a shift of the neutrino Dirac mass alone, the latter proportional to v_R^2 k; and a (10,3,1) part ~ v_R k^2
p = dirac(c20["I1"])
ok = ok and abs(p[0] - p[1]) < 1e-9 * abs(p[2]) and abs(p[0] - p[3]) < 1e-9 * abs(p[2]) and abs(p[2] - p[0]) > 1e-6 * abs(p[2])
nu20, nu10, nu22 = [dirac(c["I1"])[2] - dirac(c["I1"])[0] for c in (c20, c10, c22)]
ok = ok and abs(nu20 / nu10 - 4) < 1e-9 and abs(nu22 / nu20 - 2) < 1e-9 and c20["I1"]["D"] != 0
ok = ok and abs(c20["I1"]["DL"] / c10["I1"]["DL"] - 2) < 1e-9 and abs(c22["I1"]["DL"] / c20["I1"]["DL"] - 4) < 1e-9
# I2: the up-type (15,2,2) direction (1, 0, -3, 0), ~ v_R^2 k
p = dirac(c20["I2"])
ok = ok and np.allclose(p / p[0], [1, 0, -3, 0], atol=1e-9) and abs(c20["I2"]["D"] / c10["I2"]["D"] - 4) < 1e-9 and abs(c22["I2"]["D"] / c20["I2"]["D"] - 2) < 1e-9
ok = ok and abs(c20["I2"]["G"]) < 1e-9 * abs(c20["I2"]["D"]) and abs(c20["I2"]["S3"]) < 1e-9 * abs(c20["I2"]["D"])
# I3: the pure up-type (1,2,2) direction (1, 0, 1, 0), ~ v_R^2 k, nothing along D
p = dirac(c20["I3"])
ok = ok and np.allclose(p / p[0], [1, 0, 1, 0], atol=1e-9) and abs(c20["I3"]["D"]) < 1e-9 * abs(p[0]) and abs(c20["I3"]["G"] / c10["I3"]["G"] - 4) < 1e-9
# I4: only the neutral (10,3,1), ~ v_R k^2
ok = ok and all(abs(c20["I4"][k]) < 1e-9 * abs(c20["I4"]["DL"]) for k in ("G", "S3", "B", "D", "Phi0"))
ok = ok and abs(c20["I4"]["DL"] / c10["I4"]["DL"] - 2) < 1e-9 and abs(c22["I4"]["DL"] / c20["I4"]["DL"] - 4) < 1e-9
report("pt-3", ok,
       "(numerical) at v_R Phi_0 + k G_2 w_S, checked at (v_R, k) = (20, 1), (10, 1), (20, 2): I1 shifts all four Dirac masses alike (k^3) and the neutrino Dirac mass alone "
       "(~ v_R^2 k), and induces the neutral (10,3,1) (~ v_R k^2); I2 induces the up-type (15,2,2) direction (1, 0, -3, 0) (~ v_R^2 k); I3 the pure up-type (1,2,2) "
       "direction (1, 0, 1, 0) (~ v_R^2 k); I4 only the neutral (10,3,1) (~ v_R k^2). The tadpoles of I1 and I2 have a component along D; those of I3 and I4 do not")

# ---------------------------------------------------------------- pt-4: nothing is induced without the Pati-Salam condensate
ok = True
for nm in INV:
    co = neutral_components(grad(nm, vac(0, 1)))[0]
    big = max(abs(v) for v in co.values()) or 1.0
    ok = ok and all(abs(co[k]) < 1e-9 * big for k in co if k not in ("G", "OG"))
report("pt-4", ok,
       "(numerical) at v_R = 0, at k G_2 w_S, the gradients of the four quartics lie along G_2 w_S: the tadpoles off the generator-of-mass line, the one along D "
       "included, are sourced by the Pati-Salam condensate")


# ---------------------------------------------------------------- pt-5: what the trace quartic selects at fixed length
def ratio(M):
    A = M @ M.T
    return np.trace(A @ A) / np.trace(A) ** 2


B10 = Bpart[R_]
rng = np.random.default_rng(7)
best = (0.0, None)
for _ in range(16):
    c = rng.normal(size=B10.shape[1])
    c /= np.linalg.norm(c)
    for _ in range(400):
        M = tomat(B10 @ c)
        cn = B10.T @ coords(M @ M.T @ M)
        cn /= np.linalg.norm(cn)
        if np.linalg.norm(cn - c) < 1e-13:
            c = cn
            break
        c = cn
    r = ratio(tomat(B10 @ c))
    if r > best[0]:
        best = (r, c)
Mb = tomat(B10 @ best[1])
svb = np.linalg.svd(Mb, compute_uv=False)
ok = abs(best[0] - 0.25) < 1e-9 and int((svb > 1e-6 * svb.max()).sum()) == 4 and stab(Mb, Gn) == 12 and abs(ratio(Phi0) - 0.25) < 1e-12
Kup, Kdn = 0.5 * Mg - S3w, 0.5 * Mg + S3w
ok = ok and abs(ratio(Mg) - 1 / 64) < 1e-12 and abs(ratio(Kup) - 1 / 32) < 1e-12 and abs(ratio(Kdn) - 1 / 32) < 1e-12
for vu, vd in ((1.0, 0.5), (2.0, 1.0), (1.0, 0.1)):
    ok = ok and abs(ratio(vu * Kup + vd * Kdn) - (vu ** 4 + vd ** 4) / (32 * (vu ** 2 + vd ** 2) ** 2)) < 1e-12
B122 = Bpart[F_]
vals = [ratio(tomat(B122 @ rng.normal(size=B122.shape[1]))) for _ in range(500)]
ok = ok and min(vals) >= 1 / 64 - 1e-12 and max(vals) <= 1 / 32 + 1e-12
report("pt-5", ok,
       "(numerical) at fixed length the trace quartic Tr((m m^T)^2) / Tr(m m^T)^2 is largest within the (10b,1,3) at 1/4, at kernels of rank four with stabiliser 12 -- the "
       "orbit of Phi_0 (16 ascents) -- and on the (1,2,2) it lies between 1/64, at G_2 w_S (all 64 singular values equal), and 1/32, at the pure up-type and down-type "
       "kernels, with the value (v_u^4 + v_d^4) / (32 (v_u^2 + v_d^2)^2) at v_u K_1u + v_d K_1d")


# ---------------------------------------------------------------- pt-6: the sum rule, its inequality, and what D does to them
def masses(m_):
    out = []
    for s_ in ("u", "d", "nu", "e"):
        sv_ = np.linalg.svd(m_ @ sect[s_], compute_uv=False)
        out.append(sv_.max() if sv_.max() > 1e-12 else 0.0)
    return out


ok = True
for _ in range(60):
    a = rng.normal(size=3)
    b = rng.normal(size=3) if _ % 2 else np.zeros(3)                     # half with products by Omega (phases), half real
    m_ = sum((a[i] * np.eye(dim) + b[i] * Om) @ X for i, X in enumerate((Mg, S3w, Bw)))
    mu, md, mn, me = masses(m_)
    ok = ok and abs(mn - mu) <= md + me + 1e-9
k_, x3, xB, d = -0.25, 2.0, -0.75, 1.0                                     # z = (-1, 1, -2, 0) before D
z = np.array([k_ - x3 / 2 - xB / 3, k_ + x3 / 2 - xB / 3, k_ - x3 / 2 + xB, k_ + x3 / 2 + xB])
zD = z + d * np.array([1, -1, -3, 3])
mu, md, mn, me = masses(k_ * Mg + x3 * S3w + xB * Bw + d * Dmiss)
ok = ok and np.allclose(z, [-1, 1, -2, 0]) and np.allclose([mu, md, mn, me], np.abs(zD), atol=1e-9)
ok = ok and abs((zD[0] - zD[1] - zD[2] + zD[3]) - 8 * d) < 1e-12 and abs(mn - mu) > md + me + 1
report("pt-6", ok,
       "(numerical) on 60 vacua of the split, real or with products by Omega, |m_nu^D - m_u| <= m_d + m_e; a value d along D changes z_u - z_d - z_nu + z_e by 8 d "
       "and can break the inequality: at (k, x_3, x_B, d) = (-1/4, 2, -3/4, 1) the masses (u, d, nu, e) are (0, 0, 5, 3)")
# ---------------------------------------------------------------- pt-7: the split in general (with the value along D)
TL = np.array([0.5, -0.5, 0.5, -0.5])                      # T_3L of the left-handed members u_L, d_L, nu_L, e_L
BL_ = np.array([1 / 3, 1 / 3, -1.0, -1.0])                 # their B - L
ok = np.allclose(zpattern(Dmiss), 6 * TL * BL_, atol=1e-9)
for _ in range(20):
    z = rng.normal(size=4) * rng.choice([0.01, 1.0, 100.0], size=4)
    k_ = (3 * (z[0] + z[1]) + z[2] + z[3]) / 8
    x3 = (3 * (z[1] - z[0]) + z[3] - z[2]) / 4
    xB = 3 * ((z[2] + z[3]) - (z[0] + z[1])) / 8
    d = (z[0] - z[1] - z[2] + z[3]) / 8
    ok = ok and np.allclose(k_ - x3 * TL - xB * BL_ + 6 * d * TL * BL_, z, atol=1e-12)
    ok = ok and np.allclose(masses(k_ * Mg + x3 * S3w + xB * Bw + d * Dmiss), np.abs(z), atol=1e-9 * max(1.0, np.abs(z).max()))
report("pt-7", ok,
       "(numerical) the values of D on (u, d, nu, e) are 6 T_3L(f_L) (B - L)(f_L), the product of the two charges; with it the split m = k G_2 w_S + X w_S + d D gives "
       "z_f = k - X(f_L) + 6 d T_3L(f_L) (B - L)(f_L), and the map (k, x_3, x_B, d) -> (z_u, z_d, z_nu, z_e) is inverted by k = (3(z_u + z_d) + z_nu + z_e)/8, "
       "x_3 = (3(z_d - z_u) + z_e - z_nu)/4, x_B = 3((z_nu + z_e) - (z_u + z_d))/8, d = (z_u - z_d - z_nu + z_e)/8: twenty random signed spectra, over four orders "
       "of magnitude, are reached. The four Dirac masses are independent; the sum rule holds exactly when d = 0")
# ---------------------------------------------------------------- pt-8: the gradient of the trace quartic in closed form
# With K = sum_f z_f G P_f (G = G_2 w_S, P_f the charge sectors) and Phi_0^T Phi_0 = P_N (N = ker Y, inside the nu sector), m = v_R Phi_0 + K has
# m m^T m = -m^3 = v_R^3 Phi_0 + v_R^2 z_nu G P_nu + v_R z_nu^2 (2 Phi_0 - G Phi_0 G) + sum_f z_f^3 G P_f.
G_ = Mg
PN = Phi0.T @ Phi0
ok = np.allclose(G_.T @ G_, np.eye(dim), atol=1e-12) and np.allclose(PN @ PN, PN, atol=1e-12) and abs(np.trace(PN) - 4) < 1e-9
ok = ok and np.allclose(sect["nu"] @ PN, PN, atol=1e-12) and np.allclose(PN @ G_ @ PN, 0, atol=1e-12) and np.allclose(Phi0 @ G_ @ Phi0, 0, atol=1e-12)
ok = ok and all(np.allclose(G_ @ sect[f] - sect[f] @ G_, 0, atol=1e-12) for f in sect)
nublock = G_ @ sect["nu"]
ok = ok and np.allclose(nublock, (G_ - 2 * S3w + 3 * Bw - Dmiss) / 8, atol=1e-12)            # the neutrino block of the generator of mass
GPG = G_ @ Phi0 @ G_
cGPG, resGPG = neutral_components(coords(GPG))
ok = ok and resGPG < 1e-9 and np.hypot(cGPG["DL"], cGPG["ODL"]) > 0.5 and all(abs(cGPG[k_]) < 1e-9 for k_ in cGPG if k_ not in ("DL", "ODL"))
ok = ok and np.allclose(np.linalg.svd(GPG, compute_uv=False)[:4], 1, atol=1e-12) and np.allclose(GPG @ sect["nu"], GPG, atol=1e-12)
worst = 0.0
for trial in range(12):
    vR = rng.choice([0.0, 1.0, 20.0, 300.0])
    k_, x3, xB = rng.normal(size=3)
    d = 0.0 if trial % 2 == 0 else rng.normal()
    K = k_ * Mg + x3 * S3w + xB * Bw + d * Dmiss
    zu, zd, zn, ze = k_ - x3 / 2 - xB / 3 + d, k_ + x3 / 2 - xB / 3 - d, k_ - x3 / 2 + xB - 3 * d, k_ + x3 / 2 + xB + 3 * d
    m_ = vR * Phi0 + K
    closed = vR ** 3 * Phi0 + vR ** 2 * zn * nublock + vR * zn ** 2 * (2 * Phi0 - GPG) + sum(zf ** 3 * G_ @ sect[f] for zf, f in zip((zu, zd, zn, ze), ("u", "d", "nu", "e")))
    worst = max(worst, np.abs(m_ @ m_.T @ m_ - closed).max() / max(1.0, np.abs(closed).max()))
    g = grad("I1", m_)
    c_, res = neutral_components(g)
    dcoef = 0.5 * (zu ** 3 - zd ** 3 - zn ** 3 + ze ** 3 - vR ** 2 * zn)
    scale = max(1.0, np.abs(g).max())
    ok = ok and res < 1e-8 * scale and abs(c_["D"] - dcoef) < 1e-8 * scale and abs(c_["OD"]) < 1e-8 * scale
    if d == 0.0:                                                                                   # under the sum rule the cubic part factorises
        ok = ok and abs((zu ** 3 - zd ** 3 - zn ** 3 + ze ** 3) + 3 * (zu - zd) * (ze - zd) * (zu + ze)) < 1e-9 * max(1.0, abs(zu) + abs(zd) + abs(ze)) ** 3
ok = ok and worst < 1e-12
report("pt-8", ok,
       "(numerical, against a closed form) for m = v_R Phi_0 + K with K = k G_2 w_S + x_3 S_3 w_S + x_B (B - L) w_S + d D, whose values on (u, d, nu, e) are z_f: "
       "m m^T m = v_R^3 Phi_0 + v_R^2 z_nu G P_nu + v_R z_nu^2 (2 Phi_0 - G Phi_0 G) + sum_f z_f^3 G P_f, with G = G_2 w_S, P_f the charge sectors, Phi_0^T Phi_0 the "
       "projector onto ker Y (rank 4, inside the nu sector), G Phi_0 G a Majorana kernel of nu_L in the neutral (10,3,1); the neutrino block of the generator of mass is "
       "G P_nu = (G - 2 S_3 w_S + 3 (B - L) w_S - D)/8; so the gradient 4 m m^T m of Tr((m m^T)^2) has the coefficient (z_u^3 - z_d^3 - z_nu^3 + z_e^3 - v_R^2 z_nu)/2 "
       "along D, which under the sum rule (d = 0) is -[3 (z_u - z_d)(z_e - z_d)(z_u + z_e) + v_R^2 z_nu]/2; twelve random vacua, half with d = 0")
# ---------------------------------------------------------------- pt-9: the trace quartic on the real slice, and the coefficient along D
rng9 = np.random.default_rng(11)
ok, worst = True, 0.0
for _ in range(6):
    vR = rng9.uniform(0.5, 5.0)
    z = rng9.normal(size=4)
    m_ = vR * Phi0 + sum(zf * Mg @ sect[f] for zf, f in zip(z, ("u", "d", "nu", "e")))
    A_ = m_ @ m_.T
    zu, zd, zn, ze = z
    I1s = 24 * zu**4 + 24 * zd**4 + 8 * ze**4 + 8 * zn**4 + 16 * vR**2 * zn**2 + 4 * vR**4
    cDs = 0.5 * (zu**3 - zd**3 - zn**3 + ze**3 - vR**2 * zn)
    cDn = np.trace(Dmiss.T @ (4 * m_ @ m_.T @ m_)) / np.trace(Dmiss.T @ Dmiss)
    worst = max(worst, abs(np.trace(A_ @ A_) - I1s) / I1s, abs(cDn - cDs) / max(1.0, abs(cDs)))
ok = worst < 1e-12 and abs(np.trace(Dmiss.T @ Dmiss) - 192) < 1e-9 and max(abs(np.trace(Dmiss.T @ X_)) for X_ in (Mg, S3w, Bw)) < 1e-9
zD = zpattern(Dmiss)
ok = ok and np.allclose(zD, [1, -1, -3, 3], atol=1e-9)
report("pt-9", ok,
       f"(numerical) on the real slice m = v_R Phi_0 + sum_f z_f G P_f the trace quartic is 24 z_u^4 + 24 z_d^4 + 8 z_e^4 + 8 z_nu^4 + 16 v_R^2 z_nu^2 + 4 v_R^4; D has the "
       f"values (1, -1, -3, 3), Tr(D^T D) = 192, and is trace-orthogonal to G_2 w_S, (T_3L + T_3R) w_S, (B - L) w_S; the component of 4 m m^T m along D equals "
       f"(1/192) dI_1/dd = (z_u^3 - z_d^3 - z_nu^3 + z_e^3 - v_R^2 z_nu)/2 at six random points (relative {worst:.0e})")
# ---------------------------------------------------------------- pt-10: the bound 1/4 on all of M_2, and the left triplet
vals10 = [ratio(tomat(rng9.normal(size=len(kern)))) for _ in range(300)]
svL = np.linalg.svd(DL, compute_uv=False)
rkL = int((svL > 1e-9 * svL.max()).sum())
BL10 = Bpart[L_]
cL = rng9.normal(size=BL10.shape[1])
cL /= np.linalg.norm(cL)
for _ in range(400):
    cn = BL10.T @ coords(tomat(BL10 @ cL) @ tomat(BL10 @ cL).T @ tomat(BL10 @ cL))
    cn /= np.linalg.norm(cn)
    if np.linalg.norm(cn - cL) < 1e-13:
        cL = cn
        break
    cL = cn
ok = max(vals10) <= 0.25 + 1e-12 and abs(ratio(DL) - 0.25) < 1e-12 and rkL == 4 and abs(ratio(tomat(BL10 @ cL)) - 0.25) < 1e-9
report("pt-10", ok,
       f"(numerical) the ratio Tr((m m^T)^2)/Tr(m m^T)^2 is at most 1/4 on 300 random kernels of M_2 (largest {max(vals10):.4f}); the neutral left-triplet kernel of the "
       "(10,3,1) has real rank four and ratio 1/4, and an ascent within the (10,3,1) reaches 1/4: the trace quartic does not distinguish the right triplet from the left")
# ---------------------------------------------------------------- pt-11: the right-triplet part of the gradient of I_3 rescales Phi_0
kap11, vR11 = 0.8, 3.0
g3c = grad("I3", vR11 * Phi0 + kap11 * Mg)
gR = tomat(Proj[R_] @ g3c)
coefR = np.trace(Phi0.T @ gR) / np.trace(Phi0.T @ Phi0)
ok = np.linalg.norm(gR - coefR * Phi0) < 1e-9 * np.linalg.norm(gR) and abs(coefR - 2 * kap11**2 * vR11) < 1e-9
report("pt-11", ok,
       f"(numerical) at v_R Phi_0 + kappa G_2 w_S the part of the gradient of I_3 = Tr(F F^T R R^T) in the (10b,1,3) is 2 kappa^2 v_R Phi_0 (coefficient {coefR:.6f} at "
       "kappa = 0.8, v_R = 3): it rescales the condensate; the new directions that I_3 induces are in the bidoublet (pt-3)")
# ---------------------------------------------------------------- pt-12: at general split vacua the stabiliser fixes the twelve neutral directions, D among them
ok = True
for _ in range(3):
    k_, x3, xB = rng9.normal(size=3)
    Phi = 20 * Phi0 + k_ * Mg + x3 * S3w + xB * Bw
    for gens, sdim in ((Gn, 9), (Gn + Gln, 10)):
        Aq = np.array([comm(t, Phi).flatten() for t in gens]).T
        rk = np.linalg.matrix_rank(Aq, tol=1e-8)
        Hs = [sum(cf * t for cf, t in zip(row, gens)) for row in np.linalg.svd(Aq)[2][rk:]]
        Ah = np.vstack([admat(h) for h in Hs])
        fd = len(kern) - np.linalg.matrix_rank(Ah, tol=1e-8)
        FB = np.linalg.svd(Ah)[2][len(kern) - fd:]
        cD = coords(Dmiss)
        ok = ok and len(Hs) == sdim and fd == 12 and np.linalg.norm(cD - FB.T @ (FB @ cD)) < 1e-8 * np.linalg.norm(cD)
report("pt-12", ok,
       "(numerical) at three random vacua v_R Phi_0 + kappa G_2 w_S + x_3 (T_3L + T_3R) w_S + x_B (B - L) w_S the gauge stabiliser has dimension 9 and the stabiliser with "
       "the global phases dimension 10; each fixes exactly the twelve neutral directions, D among them: no gauge or global symmetry of these vacua forbids a tadpole along D")
finish()
