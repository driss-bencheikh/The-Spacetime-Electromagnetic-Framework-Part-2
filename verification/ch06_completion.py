#!/usr/bin/env python3
"""Part II, Chapter 6 -- one connection: so(3) Yang-Mills with a Higgs vector n, in frame variables, and its tree amplitudes.

    L = -(1/4) F^A_{mu nu} F^{A mu nu} + (1/2) D_mu n . D^mu n - (lambda/4)(n.n - v^2)^2,   D_mu n = d_mu n + g b_mu x n.

    python3 ch06_completion.py

co-1..5, co-11: symbolic and exact. co-6..10: numerical (g = v = 1), with the tolerances stated in each line.
"""
import itertools

import numpy as np
import sympy as sp

import p2amplitudes as A
from p2common import report, finish
from p2fields import t, x, y, z, co, xs, sg, fld, cross, EL

g, v = sp.symbols("g v", positive=True)
eps = sp.LeviCivita
zero = lambda L: all(sp.expand(q) == 0 for q in L)
beta = [[fld(f"b{C}_{mu}") for mu in range(4)] for C in range(3)]
nH = [fld(f"n{C}") for C in range(3)]
bvec = lambda mu: [beta[C][mu] for C in range(3)]


def D(mu, V):
    c = cross(bvec(mu), V)
    return [sp.diff(V[C], xs[mu]) + g * c[C] for C in range(3)]


Fmn = [[[sp.diff(beta[C][nu], xs[mu]) - sp.diff(beta[C][mu], xs[nu]) + g * cross(bvec(mu), bvec(nu))[C] for nu in range(4)] for mu in range(4)] for C in range(3)]
E = [[Fmn[C][0][i + 1] for i in range(3)] for C in range(3)]
B = [[sum(sp.Rational(1, 2) * eps(i, q, r) * Fmn[C][q + 1][r + 1] for q in range(3) for r in range(3)) for i in range(3)] for C in range(3)]
LYM = -sp.Rational(1, 4) * sum(sg[mu] * sg[nu] * Fmn[C][mu][nu]**2 for C in range(3) for mu in range(4) for nu in range(4))
etaXX = sum(B[C][i]**2 - E[C][i]**2 for C in range(3) for i in range(3))
report("co-1", sp.expand(LYM + etaXX / 2) == 0, "L_YM = -(1/2) sum_A eta(X^A, X^A) with X^A = (B^A; E^A) the frame vector of the A-th field strength (E_i = F_0i, B_i = (1/2) eps_ijk F_jk)")
compv = lambda V, i: [V[C][i] for C in range(3)]


def Dt(V):
    c = [D(0, compv(V, i)) for i in range(3)]
    return [[c[i][C] for i in range(3)] for C in range(3)]


def curlD(V):
    d = [[D(q + 1, compv(V, r)) for r in range(3)] for q in range(3)]
    return [[sum(eps(i, q, r) * d[q][r][C] for q in range(3) for r in range(3)) for i in range(3)] for C in range(3)]


def divD(V):
    d = [D(i + 1, compv(V, i)) for i in range(3)]
    return [sum(d[i][C] for i in range(3)) for C in range(3)]


DtB, cE, dB = Dt(B), curlD(E), divD(B)
report("co-2", zero([DtB[C][i] - cE[C][i] for C in range(3) for i in range(3)] + dB), "Bianchi identities in frame form: D_t B^A = curl_D E^A and div_D B^A = 0 identically: no magnetic current")
Dn = [D(mu, nH) for mu in range(4)]
LH = sp.Rational(1, 2) * sum(sg[mu] * Dn[mu][C]**2 for mu in range(4) for C in range(3))
el = EL(LYM + LH, [beta[C][mu] for C in range(3) for mu in range(4)])
ELc = [[el[4 * C + mu] for mu in range(4)] for C in range(3)]
DtE, cB, dE = Dt(E), curlD(B), divD(E)
rho = [-g * cross(nH, Dn[0])[C] for C in range(3)]
jv = [[g * cross(nH, Dn[i + 1])[C] for i in range(3)] for C in range(3)]
found = [(s1, s2) for s1 in (1, -1) for s2 in (1, -1)
         if zero([ELc[C][0] - s1 * (dE[C] - rho[C]) for C in range(3)]) and zero([ELc[C][i + 1] - s2 * (DtE[C][i] + cB[C][i] + jv[C][i]) for C in range(3) for i in range(3)])]
report("co-3", len(found) == 1, "field equations in frame form: div_D E^A = rho^A, D_t E^A = -curl_D B^A - j^A (the system of Part I, Prop. 3.10, with covariant derivatives), with (rho, j)^A = (-g (n x D_t n)^A, g (n x D_i n)^A)")
sub = {nH[0]: 0, nH[1]: 0, nH[2]: v}
Dn0 = [[q.subs(sub).doit() for q in Dn[mu]] for mu in range(4)]
J0 = [[sp.expand(g * cross([0, 0, v], Dn0[mu])[C]) for mu in range(4)] for C in range(3)]
report("co-4", all(sp.expand(J0[C][mu] - (g**2 * v**2 * beta[C][mu] if C < 2 else 0)) == 0 for C in range(3) for mu in range(4)),
       "in the vacuum n = (0, 0, v): g (n x D_mu n)^A = g^2 v^2 (b^1_mu, b^2_mu, 0): the current of each massive component is M^2 times its own potential, M = g v (London relation, exact)")
# charged/neutral split with independent symbols for the fields and their first derivatives
bs = [[sp.Symbol(f"b{C}{mu}") for mu in range(4)] for C in range(3)]
db = [[[sp.Symbol(f"d{mu}_b{C}{nu}") for nu in range(4)] for C in range(3)] for mu in range(4)]
bv = lambda mu: [bs[C][mu] for C in range(3)]
Fs = [[[db[mu][C][nu] - db[nu][C][mu] + g * cross(bv(mu), bv(nu))[C] for nu in range(4)] for mu in range(4)] for C in range(3)]
up = lambda M: [[sg[a] * sg[b] * M[a][b] for b in range(4)] for a in range(4)]
LY = -sp.Rational(1, 4) * sum(Fs[C][a][b] * up(Fs[C])[a][b] for C in range(3) for a in range(4) for b in range(4))
I, r2 = sp.I, sp.sqrt(2)
res = {}
for sW in (1, -1):
    W = [(bs[0][mu] - I * sW * bs[1][mu]) / r2 for mu in range(4)]
    Wb = [(bs[0][mu] + I * sW * bs[1][mu]) / r2 for mu in range(4)]
    dW = [[(db[mu][0][nu] - I * sW * db[mu][1][nu]) / r2 for nu in range(4)] for mu in range(4)]
    dWb = [[(db[mu][0][nu] + I * sW * db[mu][1][nu]) / r2 for nu in range(4)] for mu in range(4)]
    a_ = [bs[2][mu] for mu in range(4)]
    f = [[db[mu][2][nu] - db[nu][2][mu] for nu in range(4)] for mu in range(4)]
    for sD in (1, -1):
        DW = [[dW[mu][nu] + I * sD * g * a_[mu] * W[nu] for nu in range(4)] for mu in range(4)]
        DWb = [[dWb[mu][nu] - I * sD * g * a_[mu] * Wb[nu] for nu in range(4)] for mu in range(4)]
        Wmn = [[DW[a][b] - DW[b][a] for b in range(4)] for a in range(4)]
        Wbmn = [[DWb[a][b] - DWb[b][a] for b in range(4)] for a in range(4)]
        Lmin = -sp.Rational(1, 4) * sum(f[a][b] * up(f)[a][b] for a in range(4) for b in range(4)) - sp.Rational(1, 2) * sum(Wbmn[a][b] * up(Wmn)[a][b] for a in range(4) for b in range(4))
        WbW = sum(sg[a] * Wb[a] * W[a] for a in range(4))
        WW = sum(sg[a] * W[a] * W[a] for a in range(4))
        WbWb = sum(sg[a] * Wb[a] * Wb[a] for a in range(4))
        quart = -g**2 / 2 * (WbW**2 - WW * WbWb)
        mom = I * g * sum(up(f)[a][b] * Wb[a] * W[b] for a in range(4) for b in range(4))
        for c in (1, -1):
            if sp.expand(LY - (Lmin + c * mom + quart)) == 0:
                res[(sW, sD)] = c
cs = sorted(c for (sW, sD), c in res.items() if sD == 1)
report("co-5", len(res) == 2 and cs == [1],
       "with W = (b^1 -+ i b^2)/sqrt2 and a = b^3: L_YM = -(1/4) f^2 - (1/2)|D^W|^2 + i g c f^{mu nu} conj(W)_mu W_nu - (g^2/2)[(conj(W).W)^2 - (W.W)(conj(W).conj(W))] with c = +1 for D = d + i g a: minimal coupling, the unit moment term, and a quartic term")

# ---- amplitudes (numerical)
model = A.Model(lam=2.0)
rng = np.random.default_rng(2)
worst = 0
for _ in range(12):
    P, th = 10**rng.uniform(-0.5, 1.5), rng.uniform(0.3, 2.8)
    p1, q1, p2, q2 = A.kin(P, th)
    b1 = rng.normal(size=3) + 1j * rng.normal(size=3)
    b2 = rng.normal(size=3) + 1j * rng.normal(size=3)
    e1, e2 = A.polV(p1, b1 / np.linalg.norm(b1)), A.polV(p2, b2 / np.linalg.norm(b2))
    ep1, ep2 = A.photon(q1, rng.choice([1, -1])), A.photon(q2, rng.choice([1, -1]))
    ref = A.amp_vector(1.0, e1, p1, ep1, q1, e2, p2, np.conj(ep2), q2)
    mine = A.amplitude(model, [A.Win(+1, p1, e1), A.Ain(q1, ep1), A.Wout(+1, p2, e2), A.Aout(q2, ep2)])
    worst = max(worst, abs(abs(mine) - abs(ref)) / max(abs(ref), 1e-12))
report("co-6", worst < 1e-8, f"(numerical) W + massless b -> W + massless b from the so(3) Lagrangian equals the vector with unit moment term (the amplitude of qs-10 with c = +1): largest relative difference of |M| on 12 random configurations {worst:.1e} < 1e-8")
Ps = [3., 10., 30., 100., 300.]


def scanWW(mod, with_h, hel=(0, 0, 0, 0)):
    vals = []
    for P in Ps:
        p1, p2, p3, p4 = A.kinWW(P, np.pi / 2)
        legs = [A.Win(+1, p1, A.polW(p1, hel[0])), A.Win(-1, p2, A.polW(p2, hel[1])), A.Wout(+1, p3, A.polW(p3, hel[2])), A.Wout(-1, p4, A.polW(p4, hel[3]))]
        vals.append(A.amplitude(mod, legs, with_h))
    return np.array(vals)


slope = lambda a: np.log(abs(a[-1]) / abs(a[-2])) / np.log(Ps[-1] / Ps[-2])
a_no = scanWW(A.Model(0.5), False)
coef = abs(a_no[-1]) / (4 * (Ps[-1]**2 + A.MW**2))
report("co-7", abs(slope(a_no) - 2) < 0.01 and abs(coef - 0.5) < 0.01,
       f"(numerical) W+W- -> W+W-, all longitudinal, 90 degrees, Higgs vector of fixed length: |M| = {['%.3g' % abs(q) for q in a_no]} at P/M = 3 ... 300; exponent {slope(a_no):.3f}, |M| = {coef:.3f} s/v^2")
p1, p2, p3, p4 = A.kinWW(100., np.pi / 2)
legs = [A.Win(+1, p1, A.polW(p1, 0)), A.Win(-1, p2, A.polW(p2, 0)), A.Wout(+1, p3, A.polW(p3, 0)), A.Wout(-1, p4, A.polW(p4, 0))]
tot, pc = A.amplitude(model, legs, True, pieces=True)
big = max(abs(q) for q in pc.values())
report("co-8", big > 1e8 and abs(tot.real + 5.0) < 0.05,
       f"(numerical) the same amplitude with the radial mode, M_h = 2M, P = 100 M: contact {pc['contact'].real:.3e}, exchanges {', '.join('%.3e' % q.real for kq, q in pc.items() if kq != 'contact')}, sum {tot.real:.3f}: the terms of order 1e8 cancel")
lims = []
for lam in (0.5, 2.0, 8.0):
    ah = scanWW(A.Model(lam), True)
    lims.append((abs(ah[-1]), slope(ah), abs(3 - 4 * lam)))                 # 3 g^2 - 2 M_h^2/v^2 with M_h^2 = 2 lam v^2
report("co-9", all(abs(a - c) < 0.02 * max(1, c) and abs(s) < 0.01 for a, s, c in lims),
       f"(numerical) with the radial mode the amplitude is bounded: |M| at P = 300 M is {', '.join('%.3f' % a for a, s, c in lims)} for M_h = M, 2M, 4M, against |3 g^2 - 2 M_h^2/v^2| = {', '.join('%.0f' % c for a, s, c in lims)}")
mx, grow = 0, 0
for hel in itertools.product((1, 0, -1), repeat=4):
    ah = scanWW(model, True, hel)
    if abs(ah[-1]) > 1e-6:
        mx, grow = max(mx, abs(ah[-1])), max(grow, slope(ah))
report("co-10", mx < 5.05 and grow < 0.01,
       f"(numerical) all 81 helicity combinations at 90 degrees with the radial mode (M_h = 2M): the largest |M| at P = 300 M is {mx:.3f} g^2; among the channels retained by "
       f"the threshold |M| > 1e-6 at P = 300 M, the largest exponent fitted between P = 100 M and 300 M is {grow:.4f}")
# ---- magnetic charge of a radial Higgs vector
xx, yy, zz_, gg = sp.symbols("x y z g", positive=True)
r = sp.sqrt(xx**2 + yy**2 + zz_**2)
c3 = (xx, yy, zz_)
nr = [xx / r, yy / r, zz_ / r]
dn = [[sp.diff(nr[C], c) for C in range(3)] for c in c3]
ff = [[-sum(nr[C] * cross(dn[i], dn[q])[C] for C in range(3)) / gg for q in range(3)] for i in range(3)]
Bk = [sp.simplify(sum(sp.Rational(1, 2) * eps(k_, i, q) * ff[i][q] for i in range(3) for q in range(3))) for k_ in range(3)]
th_, ph_, R_ = sp.symbols("theta phi R", positive=True)
flux = sp.integrate(sp.integrate((-1 / (gg * R_**2)) * R_**2 * sp.sin(th_), (th_, 0, sp.pi)), (ph_, 0, 2 * sp.pi))
grad2 = sp.simplify(sum(dn[i][C]**2 for i in range(3) for C in range(3)))
Rb = sp.symbols("R_b", positive=True)
rr_ = sp.symbols("r_", positive=True)
egrad = sp.integrate(sp.Rational(1, 2) * (2 / rr_**2) * 4 * sp.pi * rr_**2, (rr_, 0, Rb))
report("co-11", all(sp.simplify(Bk[k_] + c3[k_] / (gg * r**3)) == 0 for k_ in range(3)) and sp.simplify(flux + 4 * sp.pi / gg) == 0
       and sp.simplify(grad2 - 2 / r**2) == 0 and sp.simplify(egrad - 4 * sp.pi * Rb) == 0,
       "a radial Higgs vector n = x/r (with b = 0) has field -(1/g) n.(d_i n x d_j n) of the unbroken charge equal to B_k = -x_k/(g r^3) in the frame variable B = -B_text: "
       "flux -4 pi/g through every sphere with the outward normal, of magnitude one Dirac unit for the charge g/2 of a doublet; |d_i n|^2 = 2/r^2, so the gradient "
       "energy in a ball of radius R is 4 pi R: a test of the flux, not a monopole of finite energy")
# ---- the exact all-longitudinal amplitude at 90 degrees, and its limit at other couplings
def _llll(model_, P_, with_h=True, pieces=False):
    p1_, p2_, p3_, p4_ = A.kinWW(P_, np.pi / 2)
    legs_ = [A.Win(+1, p1_, A.polW(p1_, 0)), A.Win(-1, p2_, A.polW(p2_, 0)), A.Wout(+1, p3_, A.polW(p3_, 0)), A.Wout(-1, p4_, A.polW(p4_, 0))]
    return A.amplitude(model_, legs_, with_h, pieces=pieces)


def _closed(p_, r_):
    return 2 * p_**2 + 3 - 4 * (2 * p_**2 + 1)**2 / (4 * (p_**2 + 1) - r_) + 4 * p_**4 / (2 * p_**2 + r_)


worst = 0.0
for lam_ in (0.3, 0.5, 2.0, 8.0):                                   # M = g v = 1, r = M_h^2/M^2 = 2 lambda
    for P_ in (0.5, 1.7, 3.0, 10.0, 30.0, 100.0):
        val_ = _llll(A.Model(lam_), P_)
        worst = max(worst, abs(val_ - _closed(P_, 2 * lam_)) / max(1.0, abs(_closed(P_, 2 * lam_))))
fixed = max(abs(_llll(A.Model(0.5), P_, with_h=False) - (2 * P_**2 + 3)) / (2 * P_**2 + 3) for P_ in (0.5, 3.0, 30.0, 300.0))
_, pc0 = _llll(A.Model(2.0), 100.0, with_h=False, pieces=True)
p2_ = 1e4
okg = abs(pc0["contact"] + (p2_ + 1) * (3 * p2_ + 1)) < 1e-9 * 3e8 and abs(pc0["(13|24)"] - (3 * p2_**2 + 6 * p2_ + 4)) < 1e-9 * 3e8 and abs(pc0["(12|34)"]) < 1e-6
# exact: the limit 3 - 2r for every r, and at r = 4 the expansion -5 + 7/p^2 - 16/p^4 + O(p^-6), approached from below in magnitude
ps_, rs_, w_ = sp.symbols("p r w", positive=True)
cl_ = 2 * ps_**2 + 3 - 4 * (2 * ps_**2 + 1)**2 / (4 * (ps_**2 + 1) - rs_) + 4 * ps_**4 / (2 * ps_**2 + rs_)
lim_ = sp.simplify(sp.limit(cl_, ps_, sp.oo) - (3 - 2 * rs_))
ser_ = sp.series(cl_.subs(rs_, 4).subs(ps_, 1 / sp.sqrt(w_)), w_, 0, 3).removeO()
okg = okg and lim_ == 0 and sp.expand(ser_ - (-5 + 7 * w_ - 16 * w_**2)) == 0
mag_ = [abs(float(cl_.subs({rs_: 4, ps_: q_}))) for q_ in (100, 300)]
okg = okg and mag_[0] < mag_[1] < 5 and abs(mag_[0] - 4.999300159968) < 1e-11 and abs(mag_[1] - 4.999922224197) < 1e-11
report("co-12", worst < 1e-6 and fixed < 1e-9 and okg,
       f"(numerical, against a closed form) W+W- -> W+W-, all longitudinal, 90 degrees: M/g^2 = 2p^2 + 3 - 4(2p^2 + 1)^2/(4(p^2 + 1) - r) + 4p^4/(2p^2 + r), p = P/M, "
       f"r = M_h^2/M^2, at 24 points (largest relative deviation {worst:.1e}); with a Higgs vector of fixed length 2p^2 + 3 (deviation {fixed:.0e}); in the gauge part at "
       f"p = 100 the contact term is -(p^2 + 1)(3p^2 + 1), the t-channel exchange 3p^4 + 6p^2 + 4 and the s-channel exchange zero; exactly, the limit is 3 - 2r for every r, "
       f"and at r = 4 the expansion is -5 + 7/p^2 - 16/p^4 + O(p^-6): |M| = {mag_[0]:.12f} and {mag_[1]:.12f} g^2 at p = 100 and 300, increasing towards 5 from below")
pts = ((1.0, 1.0, 2.0), (0.5, 1.0, 2.0), (2.0, 1.0, 2.0), (1.0, 2.0, 1.0), (0.3, 1.5, 0.7))
dev = []
for gg_, vv_, lam_ in pts:
    A.g, A.v, A.MW = gg_, vv_, gg_ * vv_
    dev.append(abs(abs(_llll(A.Model(lam_), 300.0 * A.MW)) - abs(3 * gg_**2 - 4 * lam_)))      # 2 M_h^2/v^2 = 4 lambda
A.g, A.v, A.MW = 1.0, 1.0, 1.0
gcrit = np.sqrt((8 * np.pi + 3) / 2)
report("co-13", max(dev) < 2e-3 and abs(gcrit - 3.7505) < 1e-4,
       f"(numerical) at five values of (g, v, lambda) the 90-degree longitudinal amplitude at P = 300 M equals |3 g^2 - 2 M_h^2/v^2| to {max(dev):.1e}; "
       f"|3 g^2 - 2 M_h^2/v^2| < 8 pi gives M_h < sqrt((8 pi + 3 g^2)/2) v = {gcrit:.4f} v at g = 1 and sqrt(4 pi) v = {np.sqrt(4 * np.pi):.4f} v as g -> 0")
finish()
