#!/usr/bin/env python3
"""Part II, Chapter 2 -- the shift as curl.

Chapter 4 of Part I with X read as the field (B; E)(t, x): kappa G -> G curl, alpha_+- -> A_+- = d_t +- G curl.

    python3 ch02_shift_as_curl.py

All checks are symbolic and exact (sympy). The octant checks use the matrices of Part I, eq. (1.1), in the repository convention.
"""
import sympy as sp

from p2common import report, finish, cl33
from p2fields import t, x, y, z, co, vec3, fld, curl, div, grad, lap, box, cross, dot, EL, zero, currents

m, kap, kk = sp.symbols("m kappa k", real=True)
B, E = vec3("B"), vec3("E")
X = (B, E)
eta = lambda P, Q: dot(P[0], Q[0]) - dot(P[1], Q[1])
Gop = lambda P: ([-v for v in P[1]], list(P[0]))
Curl = lambda P: (curl(P[0]), curl(P[1]))
dt = lambda P: ([sp.diff(v, t) for v in P[0]], [sp.diff(v, t) for v in P[1]])
add = lambda P, Q, s=1: ([a + s * b for a, b in zip(P[0], Q[0])], [a + s * b for a, b in zip(P[1], Q[1])])
Ap = lambda P: add(dt(P), Gop(Curl(P)))
Am = lambda P: add(dt(P), Gop(Curl(P)), -1)
flat = lambda P: list(P[0]) + list(P[1])
rho_e, j, rho_m, k = currents(B, E)

# ---- Lagrangians and their Euler-Lagrange expressions
pi = Ap(X)                                                # pi/m = A_+ X = -(k; j)
report("sc-1", zero([pi[0][i] + k[i] for i in range(3)] + [pi[1][i] + j[i] for i in range(3)]),
       "A_+X = (d_t + G curl)X = -(k; j): A_+X = 0 is the frame Maxwell system of Part I, Prop. 3.10, and the momentum m A_+X is minus the pair of currents")
Lsplit = m / 2 * eta(pi, pi)
el = EL(Lsplit, B + E)
ccB, ccE = curl(curl(B)), curl(curl(E))
tgt = [-m * (sp.diff(B[i], t, 2) + ccB[i]) for i in range(3)] + [m * (sp.diff(E[i], t, 2) + ccE[i]) for i in range(3)]
report("sc-2", zero([a - b for a, b in zip(el, tgt)]), "split form (m/2) eta(A_+X, A_+X): Euler-Lagrange = -m eta (d_t^2 + curl curl) X")
AmPi = Am(pi)
tgt = [-m * v for v in AmPi[0]] + [m * v for v in AmPi[1]]
r1 = [sp.diff(k[i], t) + curl(j)[i] + AmPi[0][i] for i in range(3)]      # A_-pi = -(d_t k + curl j; d_t j - curl k)
r2 = [sp.diff(j[i], t) - curl(k)[i] + AmPi[1][i] for i in range(3)]
report("sc-3", zero([a - b for a, b in zip(el, tgt)]) and zero(r1 + r2),
       "the same expression is -m eta A_-(A_+X): the force law is A_- pi = 0, in currents d_t k = -curl j, d_t j = curl k")
Lpos = m / 2 * (dot(pi[0], pi[0]) + dot(pi[1], pi[1]))
A2 = Ap(pi)
report("sc-4", zero([a + m * b for a, b in zip(EL(Lpos, B + E), flat(A2))]), "positive form (m/2)|A_+X|^2: Euler-Lagrange = -m A_+^2 X")
Lcov = Lsplit - m / 2 * (rho_m**2 - rho_e**2)
el_cov = EL(Lcov, B + E)
tgt = [-m * box(B[i]) for i in range(3)] + [m * box(E[i]) for i in range(3)]
report("sc-5", zero([a - b for a, b in zip(el_cov, tgt)]), "completed form L_cov = split form + (m/2)(rho_e^2 - rho_m^2): Euler-Lagrange = -m eta Box X on all six components")
Lkin = m / 2 * sum(sp.diff(B[i], t)**2 - sum(sp.diff(B[i], c)**2 for c in co) - sp.diff(E[i], t)**2 + sum(sp.diff(E[i], c)**2 for c in co) for i in range(3))
report("sc-6", zero([a - b for a, b in zip(EL(Lkin, B + E), el_cov)]) and sp.simplify(sp.expand(Lcov - m / 2 * ((rho_e**2 - dot(j, j)) - (rho_m**2 - dot(k, k))))) == 0,
       "L_cov = (m/2)(J_e.J_e - J_m.J_m) (Minkowski squares of the two four-currents) has the Euler-Lagrange expression of the free kinetic term (m/2) eta(d_mu X, d^mu X)")
report("sc-7", zero([sp.diff(rho_e, t) + div(j), sp.diff(rho_m, t) + div(k)]), "both charges are conserved identically: d_t rho_e + div j = 0, d_t rho_m + div k = 0")
report("sc-8", zero([box(B[i]) + curl(j)[i] + sp.diff(k[i], t) + grad(rho_m)[i] for i in range(3)] + [box(E[i]) + sp.diff(j[i], t) + grad(rho_e)[i] - curl(k)[i] for i in range(3)]),
       "for every field: -Box B = curl j + d_t k + grad rho_m and -Box E = d_t j + grad rho_e - curl k (sourced Maxwell implies the wave equations)")
report("sc-9", zero(EL(eta(dt(X), Gop(Curl(X))), B + E)), "the cross term eta(d_t X, G curl X) has vanishing Euler-Lagrange expression (field form of the total derivative of Part I, Thm 4.12)")
Hd = m / 2 * (eta(dt(X), dt(X)) + eta(Curl(X), Curl(X)))
report("sc-10", sp.simplify(sp.expand(Hd - m / 2 * eta(Ap(X), Am(X)))) == 0, "energy density (m/2)[eta(d_t X, d_t X) + eta(curl X, curl X)] = (m/2) eta(A_+X, A_-X)")
W = [a + b for a, b in zip(cross(B, pi[1]), cross(E, pi[0]))]
vir = sp.diff(eta(X, pi), t) - eta(pi, pi) - eta(X, AmPi) - div(W)
report("sc-11", sp.simplify(sp.expand(vir)) == 0 and sp.simplify(sp.expand(eta(X, pi) - (dot(E, j) - dot(B, k)))) == 0,
       "virial identity d_t eta(X, pi) = eta(pi, pi) + eta(X, A_-pi) + div(B x pi_E + E x pi_B), with eta(X, pi) = E.j - B.k (pi in units of m)")
L2 = m / 2 * eta(pi, Gop(pi))
GA = Gop(AmPi)
tgt2 = [-m * v for v in GA[0]] + [m * v for v in GA[1]]
report("sc-12", zero([a - b for a, b in zip(EL(L2, B + E), tgt2)]) and sp.simplify(sp.expand(eta(pi, Gop(pi)) + 2 * dot(j, k))) == 0,
       "second Lorentz-class form (m/2) eta(A_+X, G A_+X) = -m j.k: Euler-Lagrange = -m eta G A_-A_+X, the same equations of motion")
# external conserved current
rx, jx = fld("rx"), vec3("jx")
Lx = m / 2 * ((rho_e - rx)**2 - dot([a - b for a, b in zip(j, jx)], [a - b for a, b in zip(j, jx)]) + dot(k, k) - rho_m**2)
tg = [-m * (box(B[i]) + curl(jx)[i]) for i in range(3)] + [m * (box(E[i]) + sp.diff(jx[i], t) + grad(rx)[i]) for i in range(3)]
report("sc-13", zero([a - b for a, b in zip(EL(Lx, B + E), tg)]), "with an external current J_x, L = (m/2)[(J_e - J_x)^2 - J_m^2] gives Box B = -curl j_x, Box E = -(d_t j_x + grad rho_x)")

# ---- zero momentum does not fix the Gauss constraints; the residual of all eight components propagates
phi = sp.Function("phi")(x, y, z)                                  # static potential
Xg = ([0, 0, 0], grad(phi))                                          # B = 0, E = grad phi, static
pig = Ap(Xg)
report("sc-26", zero(pig[0] + pig[1]) and sp.simplify(div(Xg[1]) - lap(phi)) == 0,
       "the static field B = 0, E = grad phi has A_+X = 0 (zero momentum) and rho_e = Laplacian phi: the two Gauss constraints are not fixed by the "
       "momentum; zero momentum gives the source-free evolution equations, and the source-free Maxwell sector also needs rho_e = rho_m = 0")
R0, Rv, Kv, R3 = rho_e - rx, [a - b for a, b in zip(j, jx)], k, rho_m
eq0 = sp.diff(R0, t) + div(Rv)
eqR = [sp.diff(Rv[i], t) + grad(R0)[i] - curl(Kv)[i] for i in range(3)]
eqK = [sp.diff(Kv[i], t) + grad(R3)[i] + curl(Rv)[i] for i in range(3)]
eq3 = sp.diff(R3, t) + div(Kv)
elx = EL(Lx, B + E)
conserved = sp.diff(rx, t) + div(jx)
flux = [R0 * Rv[i] + R3 * Kv[i] - cross(Kv, Rv)[i] for i in range(3)]
energy = (R0**2 + dot(Rv, Rv) + dot(Kv, Kv) + R3**2) / 2
ok = zero([elx[i] - m * eqK[i] for i in range(3)] + [elx[3 + i] + m * eqR[i] for i in range(3)])
ok = ok and sp.simplify(sp.expand(eq0 + conserved)) == 0 and sp.simplify(sp.expand(eq3)) == 0
ok = ok and sp.simplify(sp.expand(sp.diff(energy, t) + div(flux) - (R0 * eq0 + dot(Rv, eqR) + dot(Kv, eqK) + R3 * eq3))) == 0
report("sc-27", ok,
       "with a conserved external current J_x, the Euler-Lagrange equations of (m/2)[(J_e - J_x)^2 - J_m^2] are the first-order system (d_t + D)R = 0 for the residual "
       "R = (rho_e - rho_x, j - j_x, k, rho_m), whose two continuity equations hold identically; (1/2)|R|^2 obeys a local conservation law, so R = 0 at one time "
       "gives R = 0 at all times: the Maxwell sector with source J_x is R = 0, all eight components")
dtj = [-curl([sp.diff(v, t) for v in B])[i] - sp.diff(E[i], t, 2) for i in range(3)]
on_shell = {sp.diff(E[i], t, 2): lap(E[i]) for i in range(3)}
init = {**{sp.diff(B[i], t): 0 for i in range(3)}, **{E[i]: grad(phi)[i] for i in range(3)}}
dtj0 = [sp.simplify(sp.expand(v.subs(on_shell).subs(init).doit())) for v in dtj]
report("sc-28", zero([dtj0[i] + grad(lap(phi))[i] for i in range(3)]),
       "with J_x = 0, data B = 0, E = grad phi, d_t X = 0 have k = j = 0 at t = 0, but on solutions of Box X = 0 the current starts with d_t j = -grad Laplacian phi: "
       "the vanishing of the six spatial residuals alone does not propagate")

# ---- explicit complex solutions, z = B + iE
I = sp.I
maxres = lambda zv, sign: [sp.simplify(sp.diff(zv[i], t) - sign * I * curl(zv)[i]) for i in range(3)]
al, be = sp.symbols("alpha beta")
a = [al * sp.exp(I * kk * z), al * I * sp.exp(I * kk * z), 0]
b = [be * sp.exp(-I * kk * z), -be * I * sp.exp(-I * kk * z), 0]
zM = [v * sp.exp(-I * kk * t) for v in a]
zW = [v * sp.exp(I * kk * t) for v in b]
zz = [p + q for p, q in zip(zM, zW)]
qd = lambda v: sum(c * c for c in v)
Hc = m / 2 * (qd([sp.diff(c, t) for c in zz]) + qd(curl(zz)))
report("sc-14", zero([curl(a)[i] - kk * a[i] for i in range(3)] + [curl(b)[i] - kk * b[i] for i in range(3)]) and zero(maxres(zM, -1) + maxres(zW, +1))
       and sp.simplify(Hc - 2 * m * kk**2 * dot(a, b)) == 0,
       "a Maxwell part a e^{-ikt} plus a mirror part b e^{+ikt} (curl a = k a, curl b = k b) has energy density Re 2 m k^2 a.b, constant in time: Part I, Prop. 4.11(iii) "
       "with c_- = a, c_+ = b (for these circular plane waves a.b = 2 alpha beta is also constant in space)")
# the energy density is constant in time only: a counterexample to constancy in space
lam = -1
a2 = [sp.cos(z), sp.sin(z), 0]
b2 = [0, sp.cos(x), sp.sin(x)]
zz2 = [p * sp.exp(-I * lam * t) + q * sp.exp(I * lam * t) for p, q in zip(a2, b2)]
H2 = sp.simplify(m / 2 * (qd([sp.diff(c, t) for c in zz2]) + qd(curl(zz2))))
report("sc-25", zero([curl(a2)[i] - lam * a2[i] for i in range(3)] + [curl(b2)[i] - lam * b2[i] for i in range(3)])
       and zero(maxres([p * sp.exp(-I * lam * t) for p in a2], -1) + maxres([q * sp.exp(I * lam * t) for q in b2], +1))
       and sp.simplify(H2 - 2 * m * sp.sin(z) * sp.cos(x)) == 0 and sp.diff(H2, t) == 0 and sp.simplify(sp.diff(H2, x)) != 0,
       "the energy density 2 m lambda^2 Re(a.b) is constant in time and not in space in general: a = (cos z, sin z, 0) and b = (0, cos x, sin x) have "
       "curl a = -a, curl b = -b, and the field a e^{it} + b e^{-it} has the density 2 m sin z cos x")
w = [sp.exp(I * kk * (z + t)), I * sp.exp(I * kk * (z + t)), 0]
S = [sp.simplify(I / 2 * c) for c in cross([sp.conjugate(c) for c in w], w)]
piw = [sp.simplify(sp.diff(w[i], t) + I * curl(w)[i]) for i in range(3)]
report("sc-15", zero(maxres(w, +1)) and S == [0, 0, -1] and zero([piw[i] - 2 * sp.diff(w[i], t) for i in range(3)]),
       "the mirror wave (1, i, 0)e^{ik(z+t)} has E x B = -e_3 in frame variables (E x B_text = +e_3) while its phase moves along -e_3; its momentum field is 2 d_t z")
c_ = sp.symbols("c")
zr = [(1 + c_ * t) * sp.exp(I * kk * (z - t)), (1 + c_ * t) * I * sp.exp(I * kk * (z - t)), 0]
p1 = [sp.simplify(sp.diff(zr[i], t) + I * curl(zr)[i]) for i in range(3)]
p2 = [sp.simplify(sp.diff(p1[i], t) + I * curl(p1)[i]) for i in range(3)]
report("sc-16", zero([p1[0] - c_ * sp.exp(I * kk * (z - t))] + p2), "positive form: (1 + ct)(1, i, 0)e^{ik(z-t)} has A_+z a Maxwell field and A_+^2 z = 0 (resonant drift, Part I, Thm 4.5 in field form)")
zc = [sp.exp(-I * kap * t) * v for v in zM]
report("sc-17", zero([sp.diff(zc[i], t) + I * curl(zc)[i] + I * kap * zc[i] for i in range(3)]),
       "both shifts at first order: e^{-i kappa t} x (Maxwell field) solves (d_t + G curl + kappa G)X = 0, with frequencies kappa + h|k|")
A1, B1, C1, A2_, B2, C2 = sp.symbols("A1 B1 C1 A2 B2 C2", real=True)
abc = lambda A, Bc, C: [A * sp.sin(kap * z) + C * sp.cos(kap * y), Bc * sp.sin(kap * x) + A * sp.cos(kap * z), C * sp.sin(kap * y) + Bc * sp.cos(kap * x)]
Z1, Z2 = abc(A1, B1, C1), abc(A2_, B2, C2)
zb = [sp.exp(-I * kap * t) * (p + I * q) for p, q in zip(Z1, Z2)]
Bb = [sp.cos(kap * t) * v for v in Z1]
Eb = [-sp.sin(kap * t) * v for v in Z1]
report("sc-18", zero([curl(Z1)[i] - kap * Z1[i] for i in range(3)] + [div(Z1)]) and zero(maxres(zb, -1) + [sp.simplify(div(zb))])
       and zero([sp.diff(Bb[i], t) - curl(Eb)[i] for i in range(3)] + [sp.diff(Eb[i], t) + curl(Bb)[i] for i in range(3)] + cross(Eb, Bb)),
       "curl eigenfields (ABC family, curl Z = kappa Z): z = e^{-i kappa t}(Z1 + iZ2) is a source-free Maxwell field, at each point the orbit e^{-kappa G t}X_0 of Part I, Chapter 4; for Z2 = 0, E x B = 0")
cm, cp = sp.symbols("c_m c_p")
zh = [cm * sp.exp(I * kap * (z - t)) + cp * sp.exp(-I * kap * (z - t)), I * cm * sp.exp(I * kap * (z - t)) + I * cp * sp.exp(-I * kap * (z - t)), 0]
ch4 = [sp.simplify(sp.diff(zh[i], t) + I * kap * zh[i]) for i in range(3)]
report("sc-19", zero(maxres(zh, -1)) and zero([ch4[0] - 2 * I * kap * cp * sp.exp(-I * kap * (z - t))]),
       "two helicities at one frequency: a Maxwell field (currents zero) whose pointwise Chapter-4 momentum d_t z + i kappa z is 2 i kappa times the negative-helicity part")

# ---- the octant form (Part I, Thm 3.53), repository convention
e = [sp.Matrix(M.astype(int)) for M in cl33()]
I8 = sp.eye(8)
ap = [(e[i] + e[i + 3]) / 2 for i in range(3)]
am = [(e[i] - e[i + 3]) / 2 for i in range(3)]
ns = sp.Matrix.vstack(*am).nullspace()
vac = ns[0] / sp.sqrt((ns[0].T * ns[0])[0])
f = [ap[i] * vac for i in range(3)]
g = [ap[1] * ap[2] * vac, ap[2] * ap[0] * vac, ap[0] * ap[1] * vac]
full = ap[0] * ap[1] * ap[2] * vac
basis = [vac] + f + g + [full]
Pm = sp.Matrix.hstack(*basis)
PN = sp.eye(8)
for i in range(3):
    PN = PN * (sp.eye(8) - 2 * ap[i] * am[i])
par = sp.diag(1, -1, -1, -1, 1, 1, 1, -1)
report("sc-20", len(ns) == 1 and sp.simplify(Pm.T * Pm - I8) == sp.zeros(8) and sp.simplify(Pm.T * PN * Pm - par) == sp.zeros(8),
       "octant basis (vac, f_k = a^+_k vac, g_m, full) is orthonormal and (-1)^N = diag(+,-,-,-,+,+,+,-) on it")
D = lambda psi: sum((e[i] * psi.diff(co[i]) for i in range(3)), sp.zeros(8, 1))
comp = lambda psi: [sp.expand(v) for v in (Pm.T * psi)]
Phi = sum((E[i] * f[i] + B[i] * g[i] for i in range(3)), sp.zeros(8, 1))
cPi = comp(Phi.diff(t) - D(Phi))
tgt = [-rho_e] + [-v for v in j] + [-v for v in k] + [-rho_m]
report("sc-21", all(sp.simplify(u - v) == 0 for u, v in zip(cPi, tgt)),
       "Part I, eq. (3.53): Pi = (d_t - D) Phi_oct(X) = -(rho_e vac + j.f + k.g + rho_m full): charges on the two poles, currents on the six field octants")
quad = sum(par[i, i] * cPi[i]**2 for i in range(8))
report("sc-22", sp.simplify(quad - (rho_e**2 - dot(j, j) + dot(k, k) - rho_m**2)) == 0, "Pi^T (-1)^N Pi = J_e.J_e - J_m.J_m, so L_cov = (m/2) Pi^T (-1)^N Pi")
p0, p3, av, bv = fld("p0"), fld("p3"), vec3("a"), vec3("b")
psi = p0 * vac + sum((av[i] * f[i] + bv[i] * g[i] for i in range(3)), sp.zeros(8, 1)) + p3 * full
cp_ = comp(psi.diff(t) + D(psi))
tg = [sp.diff(p0, t) + div(av)] + [sp.diff(av[i], t) - curl(bv)[i] + grad(p0)[i] for i in range(3)] + [sp.diff(bv[i], t) + curl(av)[i] + grad(p3)[i] for i in range(3)] + [sp.diff(p3, t) + div(bv)]
ok = all(e[i].T == e[i] and e[i] * PN + PN * e[i] == sp.zeros(8) for i in range(3)) and all((e[i] * e[q] + e[q] * e[i]) == (2 * I8 if i == q else sp.zeros(8)) for i in range(3) for q in range(3))
report("sc-23", all(sp.simplify(u - v) == 0 for u, v in zip(cp_, tg)) and ok,
       "(d_t + D) on (p0, a, b, p3): p0_t + div a | a_t - curl b + grad p0 | b_t + curl a + grad p3 | p3_t + div b; D is self-adjoint for (-1)^N and D^2 is the Laplacian: the force law (d_t + D)Pi = 0 is the Box equation")
T = e[0] * e[1] * e[2]
PhiG = sum((B[i] * f[i] - E[i] * g[i] for i in range(3)), sp.zeros(8, 1))        # Phi(GX), GX = (-E; B)
report("sc-24", sp.simplify(T * Phi + PhiG) == sp.zeros(8, 1) and T * T == -I8, "T Phi_oct(X) = -Phi_oct(GX) with T = e1 e2 e3 (Part I, Prop. 3.45): on the octants the binding root is -T")
# ---- a standing mirror field, and the energy as a function of the pair (X, pi)
zs = [sp.exp(I * kap * t) * v for v in Z1]                                    # mirror time dependence, real curl eigenfield
Bs_ = [sp.cos(kap * t) * v for v in Z1]
Es_ = [sp.sin(kap * t) * v for v in Z1]
_, js_, _, ks_ = currents(Bs_, Es_)
report("sc-29", zero(maxres(zs, +1)) and zero([sp.re(sp.expand_complex(zs[i])) - Bs_[i] for i in range(3)] + [sp.im(sp.expand_complex(zs[i])) - Es_[i] for i in range(3)])
       and zero(cross(Es_, Bs_)) and zero([ks_[i] - 2 * kap * sp.sin(kap * t) * Z1[i] for i in range(3)] + [js_[i] + 2 * kap * sp.cos(kap * t) * Z1[i] for i in range(3)]),
       "a standing mirror field z = e^{+i kappa t} Z (Z a real ABC curl eigenfield) solves the mirror equation, has E parallel to B and zero Poynting vector, "
       "and carries the currents k = 2 kappa sin(kappa t) Z, j = -2 kappa cos(kappa t) Z: the Poynting vector opposes the phase velocity only for travelling mirror waves")
amax = [0, sp.cos(x), sp.sin(x)]                                             # a Maxwell part with lambda = -1: curl = -1 times itself
zadd = [p_ + q_ * sp.exp(-I * lam * t) for p_, q_ in zip(zz2, amax)]
mom = lambda zv: [sp.simplify(sp.diff(zv[i], t) + I * curl(zv)[i]) for i in range(3)]
H3 = sp.simplify(m / 2 * (qd([sp.diff(c, t) for c in zadd]) + qd(curl(zadd))))
report("sc-30", zero([curl(amax)[i] + amax[i] for i in range(3)]) and zero([u - v for u, v in zip(mom(zadd), mom(zz2))]) and sp.simplify(H3 - H2 - 2 * m) == 0,
       "adding the Maxwell solution (0, cos x, sin x) e^{it} to the field of sc-25 leaves the momentum d_t z + i curl z unchanged and changes the energy density "
       "from 2 m sin z cos x by 2 m: the energy is a function of the pair (X, pi), not of pi alone")
# ---- lifts exist for every value: the curl eigenfield Z_lambda through a prescribed vector u at a point
lam_ = sp.symbols("lambda", nonzero=True, real=True)
x0, y0, z0_ = sp.symbols("x0 y0 z0", real=True)
ok31 = True
a1, b1, a2, b2 = sp.symbols("a1 b1 a2 b2", real=True)
cases = [((0, 0, 1), [a1 + I * b1, a2 + I * b2, 0])]                     # n = e_z, u general in the plane orthogonal to n
nn = [sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)]                # a rotated case: n = (1, 2, 2)/3, u orthogonal to n with real and imaginary parts
cases.append((nn, [2 + 0 * I, -1 + I, -I]))
for n_, u_ in cases:
    assert sp.simplify(dot(n_, u_)) == 0
    s_ = n_[0] * (x - x0) + n_[1] * (y - y0) + n_[2] * (z - z0_)
    nxu = cross(n_, u_)
    Zl = [u_[i] * sp.cos(lam_ * s_) - nxu[i] * sp.sin(lam_ * s_) for i in range(3)]
    ok31 = ok31 and zero([curl(Zl)[i] - lam_ * Zl[i] for i in range(3)]) and zero([div(Zl)])
    ok31 = ok31 and zero([Zl[i].subs({x: x0, y: y0, z: z0_}) - u_[i] for i in range(3)])
report("sc-31", ok31,
       "for a real unit n orthogonal to the real and imaginary parts of u and s = n.(x - x0), Z = u cos(lambda s) - (n x u) sin(lambda s) has curl Z = lambda Z, div Z = 0 "
       "and Z(x0) = u, for symbolic lambda, x0 and u in the plane orthogonal to n = e_z and for n = (1, 2, 2)/3: every value has a curl eigenfield through it")
finish()
