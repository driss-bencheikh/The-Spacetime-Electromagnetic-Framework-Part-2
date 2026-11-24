#!/usr/bin/env python3
"""Part II, Chapter 3 -- kappa as a mass.

The covariant lift of the Lagrangian of Part I, Thm 4.12:  L_full = (m/2) eta(d_mu X, d^mu X) - (m kappa^2/2) eta(X, X).
It is the sum of a temporal half (B dynamical) and a spatial half (E dynamical); the rate is the gradient of a clock field.

    python3 ch03_mass.py

All checks are symbolic and exact (sympy).
"""
import itertools

import sympy as sp

from p2common import report, finish, cl33
from p2fields import t, x, y, z, co, xs, sg, vec3, fld, curl, div, grad, lap, box, cross, dot, EL, zero, currents

m, kap, kk, I0, ell = sp.symbols("m kappa k I0 ell", positive=True)
B, E = vec3("B"), vec3("E")
rho_e, j, rho_m, k = currents(B, E)
etaXX = dot(B, B) - dot(E, E)
E6 = (dot(B, B) + dot(E, E)) / 2

# ---- the lift
Lkin = m / 2 * sum(sp.diff(B[i], t)**2 - sum(sp.diff(B[i], c)**2 for c in co) - sp.diff(E[i], t)**2 + sum(sp.diff(E[i], c)**2 for c in co) for i in range(3))
tg = [-m * (box(B[i]) + kap**2 * B[i]) for i in range(3)] + [m * (box(E[i]) + kap**2 * E[i]) for i in range(3)]
report("ma-1", zero([p - q for p, q in zip(EL(Lkin - m * kap**2 / 2 * etaXX, B + E), tg)]),
       "L_full = (m/2) eta(d_mu X, d^mu X) - (m kappa^2/2) eta(X, X): Euler-Lagrange = -m eta (Box + kappa^2) X")
Lfull = m / 2 * ((rho_e**2 - dot(j, j)) - (rho_m**2 - dot(k, k))) - m * kap**2 / 2 * etaXX
LB = -m / 2 * (rho_m**2 - dot(k, k)) - m * kap**2 / 2 * etaXX            # temporal half
LA = m / 2 * (rho_e**2 - dot(j, j)) - m * kap**2 / 2 * etaXX             # spatial half
report("ma-2", sp.simplify(Lfull - LA - LB - m * kap**2 / 2 * etaXX) == 0, "L_full = L_A + L_B + (m kappa^2/2) eta(X, X), with L_B = -(m/2) J_m.J_m - (m kappa^2/2) eta(X,X) and L_A = +(m/2) J_e.J_e - (m kappa^2/2) eta(X,X)")

# ---- the temporal half
el = EL(LB, B + E)
elB, elE = el[:3], el[3:]
ok = zero([div(elE) - m * kap**2 * rho_e]) and zero([sp.diff(elE[i], t) - curl(elB)[i] + m * kap**2 * j[i] for i in range(3)])
report("ma-3", ok, "temporal half: div(EL_E) = m kappa^2 rho_e and d_t(EL_E) - curl(EL_B) = -m kappa^2 j: the electric charge and current vanish on solutions")
report("ma-4", zero([elE[i] / m - kap**2 * E[i] - curl(k)[i] for i in range(3)]) and all(sp.diff(LB, sp.diff(E[i], t)) == 0 for i in range(3)),
       "temporal half: E has no velocity and EL_E/m = kappa^2 E + curl k: E is auxiliary, E = -curl k / kappa^2")
report("ma-5", zero([elB[i] / m - (sp.diff(k[i], t) + grad(rho_m)[i] - kap**2 * B[i]) for i in range(3)]) and zero([-elB[i] / m - (box(B[i]) + kap**2 * B[i]) - curl(j)[i] for i in range(3)]),
       "temporal half: EL_B/m = d_t k + grad rho_m - kappa^2 B (force law d_t k = kappa^2 B - grad rho_m), and -EL_B/m = (Box + kappa^2)B + curl j")
p = [sp.diff(LB, sp.diff(B[i], t)) for i in range(3)]
Hc = dot(p, [sp.diff(B[i], t) for i in range(3)]) - LB
Hpos = m / 2 * (dot(k, k) + rho_m**2) + m * kap**2 * E6
report("ma-6", zero([p[i] + m * k[i] for i in range(3)]) and sp.simplify(sp.expand(Hc - Hpos - div(cross(E, p)) + dot(E, elE))) == 0,
       "temporal half: p = dL/d(d_t B) = -m k, and the canonical energy density is (m/2)(|k|^2 + rho_m^2) + m kappa^2 E_6 + div(E x p) - E.EL_E: positive on solutions")
# ---- the spatial half
elA = EL(LA, B + E)
aB = elA[:3]
pA = [sp.diff(LA, sp.diff(E[i], t)) for i in range(3)]
HA = dot(pA, [sp.diff(E[i], t) for i in range(3)]) - LA
Hneg = -(m / 2 * (dot(j, j) + rho_e**2) + m * kap**2 * E6)
ok = any(sp.simplify(sp.expand(HA - Hneg - s1 * div(cross(B, pA)) - s2 * dot(B, aB))) == 0 for s1 in (1, -1) for s2 in (1, -1))
report("ma-7", zero([div(aB) + m * kap**2 * rho_m]) and zero([pA[i] - m * j[i] for i in range(3)]) and ok,
       "spatial half: rho_m = 0 on solutions, p = dL/d(d_t E) = m j, and the energy density is -[(m/2)(|j|^2 + rho_e^2) + m kappa^2 E_6] up to a divergence and a multiple of EL_B")
Bn, En = vec3("Bn"), vec3("En")
LBn = LB.subs({B[i]: Bn[i] for i in range(3)} | {E[i]: En[i] for i in range(3)}, simultaneous=True).doit()
LBG = LBn.subs({Bn[i]: -E[i] for i in range(3)} | {En[i]: B[i] for i in range(3)}, simultaneous=True).doit()
report("ma-8", sp.simplify(sp.expand(LBG + LA)) == 0, "the binding root exchanges the halves with a sign: L_B(GX) = -L_A(X), GX = (-E; B)")
# ---- covariant form
g4 = sp.diag(1, -1, -1, -1)
Xl = sp.zeros(4, 4)
for i in range(3):
    Xl[0, i + 1] = E[i]
    Xl[i + 1, 0] = -E[i]
for i, q, r in itertools.permutations(range(3)):
    Xl[i + 1, q + 1] = sp.LeviCivita(i, q, r) * B[r]
Xu = g4 * Xl * g4
H = lambda a, b, c: sp.diff(Xl[b, c], xs[a]) + sp.diff(Xl[c, a], xs[b]) + sp.diff(Xl[a, b], xs[c])
H2 = sum(H(a, b, c)**2 * g4[a, a] * g4[b, b] * g4[c, c] for a in range(4) for b in range(4) for c in range(4))
X2 = sum(Xl[a, b] * Xu[a, b] for a in range(4) for b in range(4))
report("ma-9", sp.simplify(sp.expand(LB - m * (H2 / 12 - kap**2 * X2 / 4))) == 0,
       "temporal half = m[(1/12) H_{mnr}H^{mnr} - (kappa^2/4) X_{mn}X^{mn}] with X_{0i} = E_i, X_{ij} = eps_ijk B_k, H = dX: a massive two-form (Kalb-Ramond) field")
# ---- dual vector potential: X = G(dC)
C, C0 = vec3("C"), fld("C0")
Bc = [-sp.diff(C[i], t) + grad(C0)[i] for i in range(3)]
Ec = curl(C)
subc = {B[i]: Bc[i] for i in range(3)} | {E[i]: Ec[i] for i in range(3)}
S = lambda v: v.subs(subc, simultaneous=True).doit()
report("ma-10", zero([S(v) for v in j] + [S(rho_e)]) and zero([S(k[i]) - (box(C[i]) - grad(sp.diff(C0, t) - div(C))[i]) for i in range(3)] + [S(rho_m) - (lap(C0) - sp.diff(div(C), t))]),
       "with X = G(dC), B = -d_t C + grad C0, E = curl C: the electric current and charge vanish identically, k = Box C - grad(d_t C0 - div C), rho_m = Lap C0 - d_t div C")
om = sp.sqrt(kk**2 + kap**2)
ph = kk * z - om * t
okw = True
ens = []
for Cw, C0w in (([sp.cos(ph), 0, 0], 0), ([0, 0, sp.cos(ph)], -kk / om * sp.cos(ph))):
    sw = {C[0]: Cw[0], C[1]: Cw[1], C[2]: Cw[2], C0: C0w}
    Bw = [sp.simplify(v.subs(sw).doit()) for v in Bc]
    Ew = [sp.simplify(v.subs(sw).doit()) for v in Ec]
    s2 = {B[i]: Bw[i] for i in range(3)} | {E[i]: Ew[i] for i in range(3)}
    S2 = lambda v: sp.simplify(v.subs(s2, simultaneous=True).doit())
    okw = okw and all(S2(v) == 0 for v in el) and all(sp.simplify(Cw[i] + S2(k[i]) / kap**2) == 0 for i in range(3)) and sp.simplify(C0w - S2(rho_m) / kap**2) == 0
    ens.append(sp.simplify(sp.integrate(S2(Hpos).subs(t, 0), (z, 0, 2 * sp.pi / kk)) * kk / (2 * sp.pi)))      # mean over a wavelength
    if C0w == 0:
        okw = okw and Ew[0] == 0 and sp.simplify(Ew[1]**2 * om**2 - Bw[0]**2 * kk**2) == 0          # |E|/|B| = k/omega, E perpendicular to B
    else:
        okw = okw and all(v == 0 for v in Ew)                                                        # longitudinal: E = 0
report("ma-11", okw and sp.simplify(ens[0] - m * kap**2 * om**2 / 2) == 0 and sp.simplify(ens[1] - m * kap**4 / 2) == 0,
       "transverse and longitudinal waves (omega^2 = k^2 + kappa^2) solve the temporal half with C = -k/kappa^2, C0 = rho_m/kappa^2 (dual London relation); transverse: E perpendicular to B, |E|/|B| = k/omega, mean energy density m kappa^2 omega^2/2; longitudinal: E = 0, mean energy density m kappa^4/2")
# ---- every solution of the full theory is a sum of the two halves
Ba, Ea = curl(j), [sp.diff(j[i], t) + grad(rho_e)[i] for i in range(3)]
Bb, Eb = [sp.diff(k[i], t) + grad(rho_m)[i] for i in range(3)], [-v for v in curl(k)]
ca, cb = currents(Ba, Ea), currents(Bb, Eb)
report("ma-12", zero([box(B[i]) + Ba[i] + Bb[i] for i in range(3)] + [box(E[i]) + Ea[i] + Eb[i] for i in range(3)]) and zero(ca[3] + [ca[2]] + cb[1] + [cb[0]]),
       "-Box X = (curl j; d_t j + grad rho_e) + (d_t k + grad rho_m; -curl k), the first part without magnetic current, the second without electric current: on (Box + kappa^2)X = 0, X = X_a + X_b")


def T00(Bv, Ev):
    e2 = lambda P, Q: dot(P[0], Q[0]) - dot(P[1], Q[1])
    d = lambda v, c: [sp.diff(q, c) for q in v]
    s = e2((d(Bv, t), d(Ev, t)), (d(Bv, t), d(Ev, t))) + sum(e2((d(Bv, c), d(Ev, c)), (d(Bv, c), d(Ev, c))) for c in co) + kap**2 * e2((Bv, Ev), (Bv, Ev))
    return sp.simplify(m / 2 * s)


avg = lambda fn: sp.simplify(sp.integrate(fn.subs(t, 0), (z, 0, 2 * sp.pi / kk)) * kk / (2 * sp.pi))
Bt, Et = [-om * sp.sin(ph), 0, 0], [0, -kk * sp.sin(ph), 0]                 # temporal-half transverse wave
Bs, Es = [0, kk * sp.sin(ph), 0], [-om * sp.sin(ph), 0, 0]                  # its image under G: a spatial-half wave
cs = currents(Bs, Es)
report("ma-13", zero([box(v) + kap**2 * v for v in Bs + Es] + cs[3] + [cs[2]]) and sp.simplify(avg(T00(Bt, Et)) - m * kap**2 * om**2 / 2) == 0 and sp.simplify(avg(T00(Bs, Es)) + m * kap**2 * om**2 / 2) == 0,
       "a wave of the temporal half and its G-image (a wave of the spatial half) carry mean energy densities +m kappa^2 omega^2/2 and -m kappa^2 omega^2/2 in the full theory")
report("ma-14", T00([sp.cos(kap * t), 0, 0], [-sp.sin(kap * t), 0, 0]) == 0 and sp.simplify(T00([sp.cos(kap * t), 0, 0], [0, 0, 0]) - m * kap**2 / 2) == 0,
       "uniform fields: a duality rotation z = e^{-i kappa t} z_0 has energy 0 in the full theory; B = cos(kappa t), E = 0 (a rest state of the temporal half, Part I, Prop. 4.11(iv)) has energy density m kappa^2/2")
# ---- potentials, London relation, Bopp-Podolsky form
a3, A0 = vec3("a"), fld("A0")
Bp, Ep = curl(a3), [sp.diff(a3[i], t) - grad(A0)[i] for i in range(3)]          # conventions of Part I, Prop. 3.20
rp, jp, rmp, kp = currents(Bp, Ep)
a0 = [sp.cos(ph), 0, 0]
Bw, Ew = curl(a0), [sp.diff(v, t) for v in a0]
jw = currents(Bw, Ew)[1]
report("ma-15", zero(kp + [rmp]) and zero([box(v) + kap**2 * v for v in Bw + Ew] + [jw[i] - kap**2 * a0[i] for i in range(3)]),
       "with potentials (B = curl a, E = d_t a - grad A0) the magnetic current and charge vanish identically; on a transverse wave of the spatial half j = kappa^2 a (London relation, a = j/kappa^2)")
LBP = sp.Rational(1, 2) * (dot(Ep, Ep) - dot(Bp, Bp)) + ell**2 / 2 * (rp**2 - dot(jp, jp))
tgp = [rp + ell**2 * box(rp)] + [jp[i] + ell**2 * box(jp[i]) for i in range(3)]
report("ma-16", zero([u - v for u, v in zip(EL(LBP, [A0] + a3), tgp)]),
       "the same density varied in the potentials: L_M + (l^2/2) J_e.J_e has Euler-Lagrange expression (1 + l^2 Box)(rho_e, j): a massless branch and a massive one (Bopp-Podolsky form)")
# ---- no first-order form on the octants
e = [sp.Matrix(M.astype(int)) for M in cl33()]
I8 = sp.eye(8)
ap = [(e[i] + e[i + 3]) / 2 for i in range(3)]
am = [(e[i] - e[i + 3]) / 2 for i in range(3)]
PN = sp.eye(8)
for i in range(3):
    PN = PN * (sp.eye(8) - 2 * ap[i] * am[i])
T = e[0] * e[1] * e[2]
Ld = [(e[(i + 1) % 3 + 3] * e[(i + 2) % 3 + 3] - e[(i + 1) % 3] * e[(i + 2) % 3]) / 2 for i in range(3)]
Ms = sp.Matrix(8, 8, sp.symbols("M0:64"))
eqs = []
for i in range(3):
    eqs += list(Ms * e[i] + e[i] * Ms)
    eqs += list(Ms * Ld[i] - Ld[i] * Ms)
sol = sp.solve(eqs, list(Ms), dict=True)[0]
nfree = len(Ms.subs(sol).free_symbols)
al, be = sp.symbols("alpha beta", real=True)
Mc = al * PN + be * T * PN
ok = all(v == 0 for i in range(3) for v in list(sp.expand(Mc * e[i] + e[i] * Mc)) + list(sp.expand(Mc * Ld[i] - Ld[i] * Mc)))
report("ma-17", nfree == 2 and ok and sp.expand(Mc * Mc - (al**2 + be**2) * I8) == sp.zeros(8),
       f"the rotation-invariant matrices anticommuting with e1, e2, e3 form a space of dimension {nfree} = 2, alpha(-1)^N + beta T(-1)^N, with square (alpha^2 + beta^2) >= 0: Box + kappa^2 has no rotation-covariant first-order factor on the octants")
# ---- the clock field
th = fld("theta")
Gop = lambda P: ([-v for v in P[1]], list(P[0]))
eta = lambda P, Q: dot(P[0], Q[0]) - dot(P[1], Q[1])


def Dmu(mu, theta):
    dX = ([sp.diff(v, xs[mu]) for v in B], [sp.diff(v, xs[mu]) for v in E])
    GX = Gop((B, E))
    c = sp.diff(theta, xs[mu])
    return ([a + c * b for a, b in zip(dX[0], GX[0])], [a + c * b for a, b in zip(dX[1], GX[1])])


Lcl = lambda theta: m / 2 * sum(sg[mu] * eta(Dmu(mu, theta), Dmu(mu, theta)) for mu in range(4))
EB = dot(E, B)
dth2 = sum(sg[mu] * sp.diff(th, xs[mu])**2 for mu in range(4))
crs = sum(sg[mu] * sp.diff(th, xs[mu]) * sp.diff(EB, xs[mu]) for mu in range(4))
report("ma-18", sp.simplify(sp.expand(Lcl(th) - Lkin + m * crs + m / 2 * dth2 * etaXX)) == 0,
       "(m/2) eta(D_mu X, D^mu X) with D_mu = d_mu + (d_mu theta) G equals (m/2) eta(dX, dX) - m d_mu theta d^mu(E.B) - (m/2)(d theta)^2 eta(X, X): kappa^2 = (d theta)^2")
report("ma-19", zero([a - b for a, b in zip(EL(Lcl(kap * t), B + E), tg)]), "clamped clock theta = kappa t: Euler-Lagrange in X is -m eta (Box + kappa^2) X")
Bw_, Ew_, thw = [sp.Function(f"b{i}")(t) for i in (1, 2, 3)], [sp.Function(f"e{i}")(t) for i in (1, 2, 3)], sp.Function("vartheta")(t)
sub = {B[i]: Bw_[i] for i in range(3)} | {E[i]: Ew_[i] for i in range(3)} | {th: thw}
Lw = (I0 / 2 * dth2 + Lcl(th)).subs(sub, simultaneous=True).doit()
u = ([sp.diff(Bw_[i], t) - sp.diff(thw, t) * Ew_[i] for i in range(3)], [sp.diff(Ew_[i], t) + sp.diff(thw, t) * Bw_[i] for i in range(3)])
report("ma-20", sp.simplify(sp.expand(Lw - I0 / 2 * sp.diff(thw, t)**2 - m / 2 * eta(u, u))) == 0,
       "on fields that do not depend on x the clock Lagrangian is (I0/2) theta_t^2 + (m/2) eta(u, u), u = X_t + theta_t G X: the closure of Part I, Prop. 4.24")
elth = EL(I0 / 2 * dth2 + Lcl(th), [th])[0]
cur = [sg[mu] * ((I0 - m * etaXX) * sp.diff(th, xs[mu]) - m * sp.diff(EB, xs[mu])) for mu in range(4)]
report("ma-21", sp.simplify(elth + sum(sp.diff(cur[mu], xs[mu]) for mu in range(4))) == 0, "the theta equation is the conservation law d_mu[(I0 - m eta(X, X)) d^mu theta - m d^mu(E.B)] = 0")
# ---- a prescribed clock that varies: the X equation has a term in Box theta; the clock equation at theta = kappa t
boxth = sum(sg[mu] * sp.diff(th, xs[mu], 2) for mu in range(4))
GXv = Gop((B, E))
tgv = [-m * (box(B[i]) + dth2 * B[i] + boxth * GXv[0][i]) for i in range(3)] + [m * (box(E[i]) + dth2 * E[i] + boxth * GXv[1][i]) for i in range(3)]
elv = EL(Lcl(th), B + E)
th2 = t**2 / 2
ok = zero([a - b for a, b in zip(elv, tgv)])
ok = ok and sp.diff(th2, t).subs(t, 0) == 0 and sum(sg[mu] * sp.diff(th2, xs[mu], 2) for mu in range(4)) == 1
report("ma-22", ok,
       "for a prescribed clock theta(x), the Euler-Lagrange equation of X is (Box + d theta.d theta) X + (Box theta) G X = 0; theta = t^2/2 has d theta = 0 at t = 0 "
       "and Box theta = 1, so pointwise vanishing of the clock gradient does not give the massless equation")
clk = [((I0 - m * etaXX) * kap - m * sp.diff(EB, t))] + [m * sp.diff(EB, c) for c in co]          # the current of ma-21 at theta = kappa t
divclk = sp.diff(clk[0], t) + sum(sp.diff(clk[i + 1], co[i]) for i in range(3))
report("ma-23", sp.simplify(sp.expand(divclk + m * (kap * sp.diff(etaXX, t) + box(EB)))) == 0,
       "at theta = kappa t the clock equation of the full action is kappa d_t eta(X, X) + Box(E.B) = 0: the background theta = kappa t solves the joint system only "
       "for matter fields that satisfy it")
# ---- a positive, duality-invariant conserved quantity of the free equations, which is not the canonical energy
Xs6 = B + E
eE = sum(sp.diff(v, t)**2 + sum(sp.diff(v, c)**2 for c in co) + kap**2 * v**2 for v in Xs6) / 2
Sflux = [-sum(sp.diff(v, t) * sp.diff(v, c) for v in Xs6) for c in co]
cons = sp.diff(eE, t) + sum(sp.diff(Sflux[i], co[i]) for i in range(3)) - sum(sp.diff(v, t) * (box(v) + kap**2 * v) for v in Xs6)
GX6 = [-v for v in E] + list(B)
eG = sum(sp.diff(v, t)**2 + sum(sp.diff(v, c)**2 for c in co) + kap**2 * v**2 for v in GX6) / 2
ecan = m / 2 * (sum(sp.diff(v, t)**2 + sum(sp.diff(v, c)**2 for c in co) + kap**2 * v**2 for v in B)
                - sum(sp.diff(v, t)**2 + sum(sp.diff(v, c)**2 for c in co) + kap**2 * v**2 for v in E))
report("ma-24", sp.simplify(sp.expand(cons)) == 0 and sp.simplify(sp.expand(eG - eE)) == 0 and sp.simplify(sp.expand(ecan.subs({v: 0 for v in B}) + m * eE.subs({v: 0 for v in B}))) == 0,
       "the free equations (Box + kappa^2)X = 0 conserve e = (1/2)(|d_t X|^2 + |grad X|^2 + kappa^2|X|^2) (Euclidean norm on the six components): d_t e + div S = "
       "d_t X.(Box + kappa^2)X with S_i = -d_t X.d_i X; e(GX) = e(X), so e is positive and duality invariant, while the canonical energy of L_full is m times "
       "the same expression with the sign of eta, negative on the E components")
finish()
