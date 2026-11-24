#!/usr/bin/env python3
"""Part II, Chapter 5 -- quantum status of the temporal half with b and with a scalar clock.

    python3 ch05_quantum_status.py

qs-1..4, qs-12..14: symbolic and exact. qs-5..11, qs-15..17: numerical, with the tolerances stated in each line. qs-18: exact
(perturbation theory in sympy) and numerical.
The criterion |M| < 8 pi used in the text is an order-of-magnitude criterion; it is not tested here.
"""
import itertools

import numpy as np
import sympy as sp
from scipy.linalg import eigh_tridiagonal

import p2amplitudes as A
from p2common import report, finish

# ---- free propagators: L_k = X^dagger K(omega, k) X, X = (B; E), momentum k along e_3
w, k, m, kap = sp.symbols("omega k m kappa", positive=True)
kv = sp.Matrix([0, 0, k])
crm = lambda v: sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
I3, kx, kkT, k2 = sp.eye(3), crm(kv), kv * kv.T, k**2
KB = sp.Rational(1, 2) * m * sp.Matrix(sp.BlockMatrix([[w**2 * I3 - kkT - kap**2 * I3, w * kx], [-w * kx, kx.T * kx + kap**2 * I3]]))
KA = sp.Rational(1, 2) * m * sp.Matrix(sp.BlockMatrix([[-kx.T * kx - kap**2 * I3, -w * kx], [w * kx, -w**2 * I3 + kkT + kap**2 * I3]]))
eta6 = sp.diag(1, 1, 1, -1, -1, -1)
KF = sp.Rational(1, 2) * m * (w**2 - k2 - kap**2) * eta6
t, z = sp.symbols("t z", real=True)
a = sp.symbols("a1:7", real=True)
ph = k * z - w * t
cz = sp.symbols("x y", real=True) + (z,)
Bv, Ev = [a[i] * sp.cos(ph) for i in range(3)], [a[3 + i] * sp.cos(ph) for i in range(3)]
curl = lambda v: [sp.diff(v[2], cz[1]) - sp.diff(v[1], cz[2]), sp.diff(v[0], cz[2]) - sp.diff(v[2], cz[0]), sp.diff(v[1], cz[0]) - sp.diff(v[0], cz[1])]
div = lambda v: sum(sp.diff(v[i], cz[i]) for i in range(3))
dot = lambda p, q: sum(u * v for u, v in zip(p, q))
kq = [-(sp.diff(Bv[i], t) - curl(Ev)[i]) for i in range(3)]
jq = [-(sp.diff(Ev[i], t) + curl(Bv)[i]) for i in range(3)]
LB = m / 2 * dot(kq, kq) - m / 2 * div(Bv)**2 - m * kap**2 / 2 * (dot(Bv, Bv) - dot(Ev, Ev))
LA = m / 2 * div(Ev)**2 - m / 2 * dot(jq, jq) - m * kap**2 / 2 * (dot(Bv, Bv) - dot(Ev, Ev))
avg = lambda f: sp.simplify(sp.integrate(f.subs(t, 0), (z, 0, 2 * sp.pi / k)) * k / (2 * sp.pi))
av = sp.Matrix(a)
report("qs-1", sp.simplify(avg(LB) - (av.T * KB * av)[0] / 2) == 0 and sp.simplify(avg(LA) - (av.T * KA * av)[0] / 2) == 0,
       "the matrices K_B, K_A reproduce the mean Lagrangians of the temporal and the spatial half on plane waves")
res = {}
for name, K in (("temporal", KB), ("spatial", KA), ("full", KF)):
    d = sp.factor(K.det())
    Kinv = K.inv().applyfunc(sp.cancel)
    R = ((w**2 - k2 - kap**2) * Kinv).applyfunc(sp.cancel).subs(w, sp.sqrt(k2 + kap**2)).applyfunc(sp.simplify)
    Rn = np.array(R.subs({k: sp.Rational(3, 2), kap: 1, m: 1}).evalf(30).tolist(), dtype=float)
    evn = np.linalg.eigvalsh((Rn + Rn.T) / 2)
    lam = sp.symbols("lambda", positive=True)
    lim = Kinv.subs({w: lam * w, k: lam * k}).applyfunc(lambda e: sp.limit(sp.cancel(e), lam, sp.oo))
    poles = sp.roots(sp.Poly(sp.numer(sp.together(d)), w)).keys()
    res[name] = (R.rank(), int((evn > 1e-9).sum()), int((evn < -1e-9).sum()), lim.rank(), all(sp.simplify(r**2 - k2 - kap**2) == 0 for r in poles), lim)
report("qs-2", res["temporal"][:5] == (3, 3, 0, 3, True) and res["spatial"][:5] == (3, 0, 3, 3, True) and res["full"][:5] == (6, 3, 3, 0, True),
       "poles only at omega^2 = k^2 + kappa^2; residue: temporal half rank 3, all positive; spatial half rank 3, all negative; full theory rank 6, three of each sign (signs evaluated at k = 3/2, kappa = 1)")
lim = res["temporal"][5]
scaled = (lim * m * kap**2).applyfunc(sp.simplify)
lam2 = sp.symbols("lambda2", positive=True)
ok = lim.rank() == 3 and not scaled.free_symbols & {m, kap} and (scaled.subs({w: lam2 * w, k: lam2 * k}) - scaled).applyfunc(sp.simplify) == sp.zeros(6) and sp.simplify(lim[5, 5] - 2 / (m * kap**2)) == 0
report("qs-3", ok and res["full"][3] == 0,
       "at large momentum (omega, k scaled together) the propagator of the temporal half keeps a part of rank 3, equal to 1/(m kappa^2) times a matrix that depends only on omega/k; that of the full theory falls off")
om = sp.sqrt(k2 + kap**2)
pols = [sp.Matrix([0, 0, 1, 0, 0, 0]), sp.Matrix([om / kap, 0, 0, 0, k / kap, 0]), sp.Matrix([0, om / kap, 0, -k / kap, 0, 0])]
ok = all(sp.simplify(KB.subs(w, om) * v) == sp.zeros(6, 1) and sp.simplify((v.T * eta6 * v)[0]) == 1 for v in pols)
report("qs-4", ok and sp.limit(pols[1].norm() / (k / kap), k, sp.oo) == sp.sqrt(2),
       "polarisations at momentum k e_3, normalised to eta(X, X) = 1: helicity 0 (B along k) stays of size one; the two transverse ones grow like sqrt2 k/kappa")

# ---- X + b -> X + b (two constituents, one abelian b)
p1, q1, p2, q2 = A.kin(3.0, 1.1)
e1, e2 = A.polKR(p1, A.hel_vec(p1[1:] / 3.0, 1)), A.polKR(p2, A.hel_vec(p2[1:] / 3.0, -1))
ep1, ep2 = A.photon(q1, 1), np.conj(A.photon(q2, 1))
M = A.amp_KR(e1, p1, ep1, q1, e2, p2, ep2, q2)
shell = np.abs(A.E0(p1) @ e1).max()
ward = max(abs(A.amp_KR(e1, p1, q1, q1, e2, p2, ep2, q2)), abs(A.amp_KR(e1, p1, ep1, q1, e2, p2, q2, q2))) / abs(M)
report("qs-5", shell < 1e-10 and abs(A.ip(e1, e1) / 2 - 1) < 1e-12 and ward < 1e-10,
       f"(numerical) polarisations on shell ({shell:.0e}) and normalised; Ward identity: replacing a polarisation of b by its momentum gives |M|/|M_0| = {ward:.0e} < 1e-10")
rat = []
for wph in (1e-2, 1e-3):
    p1r, q1r = np.array([A.kap, 0, 0, 0.]), np.array([wph, 0, 0, wph])
    n2 = np.array([np.sin(0.7), 0, np.cos(0.7)])
    w2 = wph / (1 + wph / A.kap * (1 - np.cos(0.7)))
    q2r = np.concatenate([[w2], w2 * n2])
    p2r = p1r + q1r - q2r
    b1 = np.array([1, 2j, 0.5]) / np.linalg.norm([1, 2j, 0.5])
    b2 = np.array([0.3, 1j, 1]) / np.linalg.norm([0.3, 1j, 1])
    a1 = np.array([0, 1, 0.4j, 0]) / np.linalg.norm([1, 0.4j])
    a2v = np.cross(n2, [0, 1, 0])
    a2 = np.concatenate([[0], a2v + 0.5j * np.array([0, 1, 0])])
    a2 /= np.linalg.norm(a2)
    ref = -2 * (a1[1:] @ np.conj(a2[1:])) * (np.conj(b2) @ b1)
    rat.append(abs(A.amp_KR(A.polKR(p1r, b1), p1r, a1, q1r, A.polKR(p2r, b2), p2r, np.conj(a2), q2r) / ref - 1))
report("qs-6", rat[0] < 5e-2 and rat[1] < 5e-3 and rat[1] < rat[0] / 5,
       f"(numerical) Thomson limit: at energies 1e-2 and 1e-3 kappa of the b quantum, M / [-2 g^2 (eps_1.eps_2*)(beta_2*.beta_1)] - 1 = {rat[0]:.1e}, {rat[1]:.1e}, decreasing with the energy")
Ps = [10., 30., 100., 300., 1000.]


def scan(fn_pol, fn_amp):
    out = {}
    for h1, h2, l1, l2 in itertools.product((1, 0, -1), (1, 0, -1), (1, -1), (1, -1)):
        vals = []
        for P in Ps:
            p1, q1, p2, q2 = A.kin(P, np.pi / 2)
            vals.append(abs(fn_amp(fn_pol(p1, A.hel_vec(p1[1:] / P, h1)), p1, A.photon(q1, l1), q1, fn_pol(p2, A.hel_vec(p2[1:] / P, h2)), p2, np.conj(A.photon(q2, l2)), q2)))
        out[(h1, l1, h2, l2)] = vals
    return out


slope = lambda vals: np.log(vals[-1] / vals[-2]) / np.log(Ps[-1] / Ps[-2])
resKR = scan(A.polKR, A.amp_KR)
top = max(resKR.values(), key=lambda vv: vv[-1])
report("qs-7", abs(slope(top) - 2) < 0.01 and abs(top[-1] / Ps[-1]**2 - 1) < 0.01,
       f"(numerical) X + b -> X + b at 90 degrees: the largest helicity amplitude grows with exponent {slope(top):.3f} (between P = 300 and 1000 kappa), |M| = {top[-1] / Ps[-1]**2:.3f} g^2 (P/kappa)^2")
sc = []
for P in Ps:
    p1, q1, p2, q2 = A.kin(P, np.pi / 2)
    sc.append(abs(A.amp_scalar(p1, A.photon(q1, 1), q1, p2, np.conj(A.photon(q2, 1)), q2)))
report("qs-8", max(sc) <= 2.0 and abs(sc[-1] - 2) < 1e-5, f"(numerical) scalar control of the same charge, equal helicities of b: |M|/g^2 = {['%.4f' % s for s in sc]} at P/kappa = 10 ... 1000: bounded by 2")
rng = np.random.default_rng(3)
worst = 0
for _ in range(40):
    P, th = 10**rng.uniform(-1, 2.5), rng.uniform(0.2, 2.9)
    p1, q1, p2, q2 = A.kin(P, th)
    b1 = rng.normal(size=3) + 1j * rng.normal(size=3)
    b2 = rng.normal(size=3) + 1j * rng.normal(size=3)
    b1, b2 = b1 / np.linalg.norm(b1), b2 / np.linalg.norm(b2)
    ep1, ep2 = A.photon(q1, rng.choice([1, -1])), np.conj(A.photon(q2, rng.choice([1, -1])))
    x1 = A.amp_KR(A.polKR(p1, b1), p1, ep1, q1, A.polKR(p2, b2), p2, ep2, q2)
    x2 = A.amp_vector(0.0, A.polV(p1, b1), p1, ep1, q1, A.polV(p2, b2), p2, ep2, q2)
    worst = max(worst, abs(x1 - x2) / max(abs(x1), abs(x2)))
report("qs-9", worst < 1e-8, f"(numerical) the gauged two-form and the minimally coupled vector give the same amplitude: largest relative difference on 40 random configurations {worst:.1e} < 1e-8")
out = {}
for c in (0.0, 1.0, -1.0):
    best = None
    for h1, h2, l1, l2 in itertools.product((1, 0, -1), (1, 0, -1), (1, -1), (1, -1)):
        vals = []
        for P in Ps:
            p1, q1, p2, q2 = A.kin(P, np.pi / 2)
            vals.append(abs(A.amp_vector(c, A.polV(p1, A.hel_vec(p1[1:] / P, h1)), p1, A.photon(q1, l1), q1, A.polV(p2, A.hel_vec(p2[1:] / P, h2)), p2, np.conj(A.photon(q2, l2)), q2)))
        if best is None or vals[-1] > best[-1]:
            best = vals
    out[c] = best
report("qs-10", abs(slope(out[0.0]) - 2) < 0.01 and max(out[1.0]) <= 4.0 + 1e-6 and abs(slope(out[1.0])) < 0.01 and abs(slope(out[-1.0]) - 2) < 0.01 and abs(out[-1.0][-1] / out[0.0][-1] - 4) < 0.01,
       f"(numerical, at 90 degrees) moment term i g c F^(mu nu) conj(W)_mu W_nu: c = 0 grows with exponent {slope(out[0.0]):.2f}; c = +1 is bounded at this angle, largest "
       f"|M| = {max(out[1.0]):.3f} g^2 over P = 10 ... 1000 kappa; c = -1 grows with the same exponent {slope(out[-1.0]):.2f} and {out[-1.0][-1] / out[0.0][-1]:.2f} times the "
       f"coefficient of c = 0")
# ---- gyromagnetic ratio two: the bounded amplitude depends on the angle
def same_hel(c, P, th, h=1):
    p1, q1, p2, q2 = A.kin(P, th)
    return abs(A.amp_vector(c, A.polV(p1, A.hel_vec(p1[1:] / P, h)), p1, A.photon(q1, h), q1, A.polV(p2, A.hel_vec(p2[1:] / P, h)), p2, np.conj(A.photon(q2, h)), q2))


def closed(s_, th, M=1.0):                           # |M_{++;++}|/g^2 for g = 2 (gyromagnetic ratio two), mass M
    t_ = -(s_ - M**2)**2 * (1 - np.cos(th)) / (2 * s_)
    u_ = 2 * M**2 - s_ - t_
    return 2 * ((s_ - M**2)**2 + M**2 * t_)**2 / ((s_ - M**2)**3 * (M**2 - u_))


Pof = lambda s_: (s_ - 1) / (2 * np.sqrt(s_))      # centre-of-mass momentum for kappa = 1
worst = max(abs(same_hel(1.0, Pof(s_), np.radians(dg)) / closed(s_, np.radians(dg)) - 1) for s_ in (4.0, 100.0, 1e4) for dg in (40, 90, 120, 150))
v120 = same_hel(1.0, Pof(100.0), np.radians(120))
lim = [abs(same_hel(1.0, 1e4, np.radians(dg)) / (4 / (1 + np.cos(np.radians(dg)))) - 1) for dg in (60, 90, 120, 150)]
report("qs-17", worst < 1e-9 and abs(v120 - 7.65092233) < 1e-7 and max(lim) < 1e-5,
       f"(numerical, against a closed form) for c = +1 the amplitude with all helicities +1 equals 2[(s - M^2)^2 + M^2 t]^2 / ((s - M^2)^3 (M^2 - u)) g^2 at twelve "
       f"(s, angle) points (relative {worst:.0e}); at fixed angle it tends to 4/(1 + cos angle) g^2, which is 4 at 90 degrees and 8 at 120 degrees; at s = 100 M^2 and "
       f"120 degrees it is {v120:.8f} g^2 > 4 g^2: the bound 4 g^2 holds at 90 degrees, not at every angle")
# ---- X + pi -> X + pi (scalar clock, theta = kappa t + pi/f)
f = 1.0
tab = {}
for h1, h2 in itertools.product((1, 0, -1), repeat=2):
    vals = []
    for P in Ps:
        p1, q1, p2, q2 = A.kin(P, np.pi / 2)
        vals.append(abs(A.amp_clock(f, A.polKR(p1, A.hel_vec(p1[1:] / P, h1)), p1, q1, A.polKR(p2, A.hel_vec(p2[1:] / P, h2)), p2, q2)))
    tab[(h1, h2)] = vals
trans = [tab[(h1, h2)] for h1 in (1, -1) for h2 in (1, -1)]
mixed = [tab[(h1, h2)] for h1, h2 in ((1, 0), (-1, 0), (0, 1), (0, -1))]
top = max(trans, key=lambda vv: vv[-1])
l00 = tab[(0, 0)]
ok = all(abs(slope(v_) - 4) < 0.01 and abs(v_[-1] / Ps[-1]**4 - 2) < 0.02 for v_ in trans) and abs(slope(l00) - 2) < 0.01 and abs(l00[-1] / Ps[-1]**2 - 2) < 0.02
ok = ok and all(abs(slope(v_) - 1) < 0.01 for v_ in mixed) and max(tab.values(), key=lambda vv: vv[-1]) in trans
report("qs-11", ok,
       f"(numerical) X + pi -> X + pi at 90 degrees in the clock frame, channels labelled (h_in, h_out): the four transverse-to-transverse amplitudes grow with exponent "
       f"{slope(top):.3f}, |M| = {top[-1] / Ps[-1]**4:.3f} P^4/(kappa^2 f^2); (0, 0): |M| = {l00[-1] / Ps[-1]**2:.3f} P^2/f^2; the four mixed channels grow with exponent "
       f"{slope(mixed[0]):.3f} at kappa = f = 1")

# ---- symbolic identities
tt, x, y, zz = sp.symbols("t x y z", real=True)
xs = (tt, x, y, zz)
g = sp.symbols("g", positive=True)
F_ = lambda nm: sp.Function(nm)(tt, x, y, zz)
n = 3
gens = []
for a_, b_ in itertools.combinations(range(n), 2):
    Mg = sp.zeros(n, n)
    Mg[a_, b_], Mg[b_, a_] = 1, -1
    gens.append(Mg)
beta = [[F_(f"beta{mu}_{c}") for c in range(3)] for mu in range(4)]
bm = [sum((beta[mu][c] * gens[c] for c in range(3)), sp.zeros(n, n)) for mu in range(4)]
Av = [sp.Matrix([F_(f"A{mu}_{a_}") for a_ in range(n)]) for mu in range(4)]
Dc = lambda mu, V: sp.diff(V, xs[mu]) + g * bm[mu] * V
Fb = [[sp.diff(bm[nu], xs[mu]) - sp.diff(bm[mu], xs[nu]) + g * (bm[mu] * bm[nu] - bm[nu] * bm[mu]) for nu in range(4)] for mu in range(4)]
DA = [[Dc(mu, Av[nu]) - Dc(nu, Av[mu]) for nu in range(4)] for mu in range(4)]
ok = True
for mu, nu, rho in itertools.combinations(range(4), 3):
    lhs = Dc(mu, DA[nu][rho]) + Dc(nu, DA[rho][mu]) + Dc(rho, DA[mu][nu])
    rhs = g * (Fb[mu][nu] * Av[rho] + Fb[nu][rho] * Av[mu] + Fb[rho][mu] * Av[nu])
    ok = ok and (lhs - rhs).applyfunc(sp.expand) == sp.zeros(n, 1)
report("qs-12", ok, "for so(3): D^(D^A) = g F_b ^ A, so X -> X + D^A/kappa changes H = D^X by (g/kappa) F_b ^ A: the vector partner exposes interactions with the scale kappa/g")
th = F_("theta")
cvec = sp.Matrix(sp.symbols("c1:7", real=True))
Z3 = sp.zeros(3)
G = sp.Matrix(sp.BlockMatrix([[Z3, -sp.eye(3)], [sp.eye(3), Z3]]))
R = sp.cos(th) * sp.eye(6) - sp.sin(th) * G
report("qs-13", (sp.diff(R, th) + G * R).applyfunc(sp.simplify) == sp.zeros(6) and all((sp.diff(R * cvec, xs[mu]) + sp.diff(th, xs[mu]) * G * R * cvec).applyfunc(sp.simplify) == sp.zeros(6, 1) for mu in range(4)),
       "D_mu(e^{-theta G} c) = 0 for D_mu = d_mu + (d_mu theta) G: the full clock-coupled theory is invariant under the six shifts X -> X + e^{-theta G} c")
Xs = sp.Matrix(sp.symbols("B1:4 E1:4", real=True))
report("qs-14", sp.simplify(sp.expand(((Xs + R * cvec).T * eta6 * (Xs + R * cvec))[0] - (Xs.T * eta6 * Xs)[0])) != 0, "eta(X, X) is not invariant under these shifts: the mass term of the temporal half breaks them")

# ---- the uniform sector with the rotor as quantum mechanics: H = |p|^2/2m + h^2/(2(I0 - m|B|^2)), h = hbar N
hbar = mm = 1.0


def levels(N, I0, npts=40000, nlev=4):
    rmax = np.sqrt(I0 / mm) * (1 - 1e-6)
    r = np.linspace(0, rmax, npts + 2)[1:-1]
    dr = r[1] - r[0]
    V = hbar**2 * N**2 / (2 * (I0 - mm * r**2))
    return eigh_tridiagonal(hbar**2 / (mm * dr**2) + V, -hbar**2 / (2 * mm * dr**2) * np.ones(npts - 1), select="i", select_range=(0, nlev - 1))[0]


rows = []
for N, I0 in ((100, 1000.), (400, 4000.), (1600, 16000.)):
    kapN, Erot = hbar * N / I0, hbar**2 * N**2 / (2 * I0)
    ev = (levels(N, I0) - Erot) / (hbar * kapN)
    rows.append((N, ev, (ev[0] - 1.5) * N))
report("qs-15", all(np.abs(ev - np.array([1.5, 3.5, 5.5, 7.5])).max() < 60.0 / N for N, ev, _ in rows) and abs(rows[-1][2] - 15 / 8) < 0.01,
       f"(numerical) s-wave levels above the rotor energy, in units of hbar kappa_N: (1.5, 3.5, 5.5, 7.5)(1 + O(1/N)) for N = 100, 400, 1600; shift of the lowest level times N = {rows[-1][2]:.3f} at N = 1600 (15/8 = 1.875)")
I0 = 400.
Elev = [levels(N, I0, nlev=1)[0] for N in (39, 40, 41)]
zp = 1.5 * hbar / I0
d1, d2 = Elev[1] - Elev[0] - (hbar * (39 + 40) / (2 * I0) + zp), Elev[2] - Elev[1] - (hbar * (40 + 41) / (2 * I0) + zp)
report("qs-16", abs(d1) < 2e-4 and abs(d2) < 2e-4,
       f"(numerical) rate ladder at I0 = 400: E_0(N+1) - E_0(N) = hbar(kappa_N + kappa_(N+1))/2 + (3/2) hbar (kappa_(N+1) - kappa_N) to {max(abs(d1), abs(d2)):.1e} (absolute, levels of order 0.1)")
# ---- the ground-state expansion: E_0 = (hbar^2/I0)(N^2/2 + 3N/2 + 15/8 + 45/(32N) + O(1/N^2))
r_ = sp.symbols("r", positive=True)
eps0 = sp.Rational(3, 2)


def psi(n):
    f_ = sp.assoc_laguerre(n, sp.Rational(1, 2), r_**2) * sp.exp(-r_**2 / 2)
    return f_ / sp.sqrt(sp.integrate(4 * sp.pi * r_**2 * f_**2, (r_, 0, sp.oo)))


def mel(a_, op, b_):
    return sp.simplify(sp.integrate(4 * sp.pi * r_**2 * a_ * op * b_, (r_, 0, sp.oo)))


ps = [psi(n) for n in range(4)]
V1, V2 = r_**4 / 2, r_**6 / 2
first = mel(ps[0], V1, ps[0])
second = -sum(mel(ps[n], V1, ps[0])**2 / (2 * n) for n in (1, 2, 3))
c1 = sp.nsimplify(mel(ps[0], V2, ps[0]) + second)
ok = first == sp.Rational(15, 8) and mel(ps[3], V1, ps[0]) == 0 and c1 == sp.Rational(45, 32)
ok = ok and sp.nsimplify(mel(ps[0], V2, ps[0])) == sp.Rational(105, 16) and sp.nsimplify(second) == sp.Rational(-165, 32)


def ground(N, I0_, npts=200000):
    rmax = np.sqrt(I0_ / mm) * (1 - 1e-9)
    rr = np.linspace(0, rmax, npts + 2)[1:-1]
    dr = rr[1] - rr[0]
    V = hbar**2 * N**2 / (2 * (I0_ - mm * rr**2))
    return eigh_tridiagonal(hbar**2 / (mm * dr**2) + V, -hbar**2 / (2 * mm * dr**2) * np.ones(npts - 1), select="i", select_range=(0, 0))[0][0]


cN = {N: (ground(N, 10.0 * N) * 10.0 * N - (N**2 / 2 + 1.5 * N + 15 / 8)) * N for N in (50, 100)}
rich = 2 * cN[100] - cN[50]
report("qs-18", ok and abs(rich - 45 / 32) < 0.01,
       f"(exact and numerical) with B = sqrt(I0/(mN)) q the Hamiltonian is (hbar^2/I0)[N^2/2 + N(-Laplacian + |q|^2)/2 + |q|^4/2 + |q|^6/(2N) + ...]; perturbation theory in "
       f"the s-wave oscillator basis gives E_0 = (hbar^2/I0)(N^2/2 + 3N/2 + {first} + ({c1})/N + O(1/N^2)), the 1/N term being 105/16 - 165/32; the radial solver on "
       f"|B| < sqrt(I0/m) with Dirichlet conditions, extrapolated from N = 50, 100, gives {rich:.4f} for the 1/N coefficient: the ladder of qs-16 holds up to O(hbar^2/(I0 N^2))")
# ---- the backward direction at gyromagnetic ratio two
back = [same_hel(1.0, P_, np.pi * (1 - 1e-11)) for P_ in (10., 100., 1000.)]
sb = [(np.sqrt(P_**2 + 1) + P_)**2 for P_ in (10., 100., 1000.)]
pb = [2 * (s_ - 1)**2 / s_ for s_ in sb]
ok = all(abs(b_ / q_ - 1) < 1e-6 for b_, q_ in zip(back, pb)) and abs(closed(sb[1], np.pi) - pb[1]) < 1e-6 * pb[1]
report("qs-19", ok,
       f"(numerical, against the closed form) at gyromagnetic ratio two and exactly backward scattering, where t = -(s - M^2)^2/s and u = M^4/s, the all-plus amplitude is "
       f"2(s - M^2)^2/(s M^2) g^2: {', '.join(f'{b_:.4g}' for b_ in back)} at P/M = 10, 100, 1000, growing like 2s/M^2: the cancellation holds at fixed angles 0 < angle < pi")
# ---- longitudinal Compton scattering in the minimal model grows; at ratio two it is bounded
def long00(c, P_, l1, l2):
    p1_, q1_, p2_, q2_ = A.kin(P_, np.pi / 2)
    return abs(A.amp_vector(c, A.polV(p1_, A.hel_vec(p1_[1:] / P_, 0)), p1_, A.photon(q1_, l1), q1_, A.polV(p2_, A.hel_vec(p2_[1:] / P_, 0)), p2_,
                            np.conj(A.photon(q2_, l2)), q2_))


def lform(P_):
    v_ = np.sqrt(1 + P_**2) + P_
    return (v_**2 - 1) * (2 * v_**4 + 7 * v_**2 - 3) / (8 * v_**2 * (v_**2 + 1))


vals = {P_: max(long00(0.0, P_, 1, -1), long00(0.0, P_, -1, 1)) for P_ in (10., 30., 100.)}
kr = abs(A.amp_KR(A.polKR(A.kin(30., np.pi / 2)[0], A.hel_vec(np.array([0, 0, 1.]), 0)), A.kin(30., np.pi / 2)[0], A.photon(A.kin(30., np.pi / 2)[1], 1),
                  A.kin(30., np.pi / 2)[1], A.polKR(A.kin(30., np.pi / 2)[2], A.hel_vec(A.kin(30., np.pi / 2)[2][1:] / 30., 0)), A.kin(30., np.pi / 2)[2],
                  np.conj(A.photon(A.kin(30., np.pi / 2)[3], -1)), A.kin(30., np.pi / 2)[3]))
ok = all(abs(vals[P_] / lform(P_) - 1) < 1e-9 for P_ in vals) and abs(kr / lform(30.) - 1) < 1e-9
report("qs-20", ok,
       f"(numerical, against a closed form) at 90 degrees the minimally coupled amplitude with both matter helicities zero and opposite helicities of b is "
       f"(v^2 - 1)(2v^4 + 7v^2 - 3)/(8 v^2 (v^2 + 1)) g^2, v = (E + P)/M: {', '.join(f'{vals[P_]:.4f}' for P_ in vals)} at P/M = 10, 30, 100, growing like (P/M)^2 "
       f"(the two-form gives the same)")
# ---- at gyromagnetic ratio two, channel by channel: opposite helicities of b vanish, equal helicities give 2 g^2 P/E
opp = {P_: max(long00(1.0, P_, 1, -1), long00(1.0, P_, -1, 1)) for P_ in (10., 100., 1000.)}
eqh = {P_: (long00(1.0, P_, 1, 1), long00(1.0, P_, -1, -1)) for P_ in (10., 100., 1000.)}
ok = all(opp[P_] < 1e-9 * (1 + P_) for P_ in opp) and all(abs(v_ / (2 * P_ / np.sqrt(1 + P_**2)) - 1) < 1e-9 for P_ in eqh for v_ in eqh[P_])
report("qs-23", ok,
       f"(numerical) at gyromagnetic ratio two and 90 degrees, with both matter helicities zero: the channel with opposite helicities of b vanishes "
       f"(largest {max(opp.values()):.1e} g^2 at P/M = 10, 100, 1000), and the channels with equal helicities have the magnitude 2 g^2 P/E "
       f"({', '.join(f'{eqh[P_][0]:.10f}' for P_ in eqh)}), which tends to 2 g^2")
# ---- the scalar amplitudes are bounded at all energies and angles
worst, okpos = 0.0, True
for P_ in (0.3, 3.0, 30.0, 300.0):
    for th_ in (0.3, 1.2, np.pi / 2, 2.4, 3.0):
        Ep_ = np.sqrt(1 + P_**2)
        s_ = (Ep_ + P_)**2
        t_ = -(s_ - 1)**2 * (1 - np.cos(th_)) / (2 * s_)
        u_ = 2 - s_ - t_
        D_ = (s_ - 1) * (1 - u_)
        a1_, a2_ = (1 - s_ * u_) / D_, -t_ / D_
        okpos = okpos and a1_ >= -1e-12 and a2_ >= -1e-12 and abs(a1_ + a2_ - 1) < 1e-9
        p1_, q1_, p2_, q2_ = A.kin(P_, th_)
        got = sorted(abs(A.amp_scalar(p1_, A.photon(q1_, l1), q1_, p2_, np.conj(A.photon(q2_, l2)), q2_)) / 2 for l1, l2 in itertools.product((1, -1), repeat=2))
        worst = max(worst, max(abs(x_ - y_) for x_, y_ in zip(got, sorted([a1_, a1_, a2_, a2_]))))
report("qs-21", okpos and worst < 1e-10,
       f"(numerical, against a closed form) for the scalar of the same charge, with D = (s - M^2)(M^2 - u) > 0, the helicity amplitudes have the magnitudes 2 g^2 (M^4 - s u)/D "
       f"and 2 g^2 (-M^2 t)/D at twenty (P, angle) points (largest deviation {worst:.0e}); both are nonnegative and they add up to 2 g^2: |M| <= 2 g^2 at all energies and angles")
# ---- the residue as a sum of polarisations, and the propagators on the light cone
S6 = sum((v_ * v_.T for v_ in pols), sp.zeros(6))
ok = ((((w**2 - k2 - kap**2) * KB.inv()).applyfunc(sp.cancel).subs(w, om) - 2 / m * S6).applyfunc(sp.simplify) == sp.zeros(6))
nullF = KF.inv().applyfunc(sp.cancel).subs(w, k).applyfunc(sp.simplify)
ok = ok and (nullF + 2 * eta6 / (m * kap**2)).applyfunc(sp.simplify) == sp.zeros(6)
nullB = KB.inv().applyfunc(sp.cancel).subs(w, k).applyfunc(sp.simplify)
ok = ok and sp.limit(nullB[0, 0] * m * kap**4 / k**2, k, sp.oo) == -2
report("qs-22", ok,
       "the residue of the temporal half at k e_3 is (2/m) sum_h v_h v_h^T over the three eta-normalised polarisations of qs-4, hence positive semidefinite of rank 3; "
       "on the light cone omega = k the propagator of the full theory is the constant -2 eta/(m kappa^2) and that of the temporal half grows like k^2/(m kappa^4): "
       "the fall-off of the full theory holds along non-null rays")
finish()
