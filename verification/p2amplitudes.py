"""Tree amplitudes for Part II, Chapters 5 and 6 (numerical, numpy). Units hbar = c = 1, rate kappa = 1, metric (+,-,-,-).

(a) Two-form.  Z = (X^1 + i X^2)/sqrt2 with charge g under one abelian b:  L = (1/6) conj(H).H - (kappa^2/2) conj(Z).Z, H = D^Z,
    D = d + i g b.  Action S = <Z, O Z>, O = -(1/2) EE, EE Z = D_mu H^{mu nu rho} + kappa^2 Z^{nu rho}.
    Amplitude at second order in g:  M = <e2| O2 - O1 O0^-1 O1 |e1>  (s and u channels).
(b) Controls: a scalar and a vector (Proca) with minimal coupling; the vector with an added moment term i g c F^{mu nu} conj(W)_mu W_nu.
(c) The completed theory: so(3) Yang-Mills with a Higgs vector in unitary gauge. Vertices are extracted from the Lagrangian by
    polarisation (inclusion-exclusion); M = V4 - sum over channels of J K^-1 J.
Amplitudes are quoted in units of g^2 (g = 1).
"""
import itertools

import numpy as np

eta = np.diag([1., -1, -1, -1])
kap = 1.0
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
low = lambda V: eta @ V


def v2t(v):
    Z = np.zeros((4, 4), complex)
    for q, (i, j) in enumerate(pairs):
        Z[i, j], Z[j, i] = v[q], -v[q]
    return Z


t2v = lambda Z: np.array([Z[i, j] for (i, j) in pairs])


def dop(c, Zup):                                     # c lower; returns H lower
    Zl = eta @ Zup @ eta
    return np.einsum("m,nr->mnr", c, Zl) + np.einsum("n,rm->mnr", c, Zl) + np.einsum("r,mn->mnr", c, Zl)


def delop(c, Hl):                                    # c lower; returns an upper two-index tensor
    Hu = np.einsum("ma,nb,rc,abc->mnr", eta, eta, eta, Hl)
    return np.einsum("m,mnr->nr", c, Hu)


def mat(op):
    return np.array([t2v(op(v2t(e))) for e in np.eye(6)]).T


def E0(P):
    c = -1j * low(P)
    return mat(lambda Z: delop(c, dop(c, Z)) + kap**2 * Z)


def E1(eps, Pin, Pout):
    ce, ci, co = 1j * low(eps), -1j * low(Pin), -1j * low(Pout)
    return mat(lambda Z: delop(co, dop(ce, Z)) + delop(ce, dop(ci, Z)))


def E2(ea, eb):
    ca, cb = 1j * low(ea), 1j * low(eb)
    return mat(lambda Z: delop(ca, dop(cb, Z)) + delop(cb, dop(ca, Z)))


G6 = np.diag([2 * eta[i, i] * eta[j, j] for (i, j) in pairs])       # <Z, W> = conj(Z)_{nu rho} W^{nu rho}
ip = lambda a, b: np.conj(a) @ G6 @ b


def amp_KR(e1, p1, eps1, q1, e2, p2, eps2c, q2):
    s = E1(eps2c, p1 + q1, p2) @ np.linalg.solve(E0(p1 + q1), E1(eps1, p1, p1 + q1) @ e1)
    u = E1(eps1, p1 - q2, p2) @ np.linalg.solve(E0(p1 - q2), E1(eps2c, p1, p1 - q2) @ e1)
    return -0.5 * ip(e2, E2(eps1, eps2c) @ e1 - s - u)


def amp_scalar(p1, eps1, q1, p2, eps2c, q2):           # L = |D phi|^2 - kappa^2 |phi|^2
    md = lambda a, b: a @ eta @ b
    O0 = lambda P: md(P, P) - kap**2
    O1 = lambda eps, Pin, Pout: -(md(Pout, eps) + md(eps, Pin))
    O2 = 2 * md(eps1, eps2c)
    return O2 - O1(eps2c, p1 + q1, p2) * O1(eps1, p1, p1 + q1) / O0(p1 + q1) - O1(eps1, p1 - q2, p2) * O1(eps2c, p1, p1 - q2) / O0(p1 - q2)


def P0(P):                                           # Proca: O W^nu = D_mu(D^mu W^nu - D^nu W^mu) + kappa^2 W^nu
    c = -1j * low(P)
    return np.array([(c @ eta @ c) * e - (eta @ c) * (c @ e) + kap**2 * e for e in np.eye(4)]).T


_f = lambda c1, c2, e: (c1 @ eta @ c2) * e - (eta @ c2) * (c1 @ e)


def P1(eps, Pin, Pout):
    ce, ci, co = 1j * low(eps), -1j * low(Pin), -1j * low(Pout)
    return np.array([_f(co, ce, e) + _f(ce, ci, e) for e in np.eye(4)]).T


def P2(ea, eb):
    ca, cb = 1j * low(ea), 1j * low(eb)
    return np.array([_f(ca, cb, e) + _f(cb, ca, e) for e in np.eye(4)]).T


ipv = lambda a, b: np.conj(a) @ eta @ b


def Fmat(q, eps, incoming):                          # F^{mu}{}_{nu} of a plane wave of b, acting on W^nu
    s = -1j if incoming else 1j
    return s * (np.outer(q, eps) - np.outer(eps, q)) @ eta


def amp_vector(c, e1, p1, eps1, q1, e2, p2, eps2c, q2):
    """vector with minimal coupling and moment term of coefficient c (c = 0: gyromagnetic ratio 1; c = 1: ratio 2)"""
    O1a = P1(eps1, p1, p1 + q1) + 1j * c * Fmat(q1, eps1, True)
    O1b = P1(eps2c, p1 + q1, p2) + 1j * c * Fmat(q2, eps2c, False)
    O1c = P1(eps2c, p1, p1 - q2) + 1j * c * Fmat(q2, eps2c, False)
    O1d = P1(eps1, p1 - q2, p2) + 1j * c * Fmat(q1, eps1, True)
    s = O1b @ np.linalg.solve(P0(p1 + q1), O1a @ e1)
    u = O1d @ np.linalg.solve(P0(p1 - q2), O1c @ e1)
    return ipv(e2, P2(eps1, eps2c) @ e1 - s - u)


def amp_clock(f, e1, p1, q1, e2, p2, q2):
    """X + pi -> X + pi for theta = kappa t + pi/f: (d theta)^2 = kappa^2 + (2 kappa/f) d_t pi + (d pi)^2/f^2"""
    w1, w2 = q1[0], q2[0]
    s = np.linalg.solve(E0(p1 + q1), e1)
    u = np.linalg.solve(E0(p1 - q2), e1)
    return -0.5 * ip(e2, (2 * (q1 @ eta @ q2) / f**2) * e1 - (2 * kap / f)**2 * w1 * w2 * (s + u))


# ---- kinematics and polarisations
def boost(p, mass=kap):
    En, pv = p[0], p[1:]
    P = np.linalg.norm(pv)
    L = np.eye(4)
    if P < 1e-14:
        return L
    n = pv / P
    ga = En / mass
    L[0, 0] = ga
    L[0, 1:] = L[1:, 0] = ga * pv / En
    L[1:, 1:] = np.eye(3) + (ga - 1) * np.outer(n, n)
    return L


def frame(n):
    a = np.array([0, 1., 0]) if abs(n[1]) < 0.9 else np.array([1., 0, 0])
    e1 = np.cross(a, n)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def hel_vec(n, h):
    e1, e2 = frame(n)
    return n.astype(complex) if h == 0 else (e1 + 1j * h * e2) / np.sqrt(2)


def polKR(p, beta):                                  # rest frame Z^{ij} = eps_ijk beta_k, boosted; <e, e> = 2
    Z = np.zeros((4, 4), complex)
    for i, j, k_ in itertools.permutations(range(3)):
        Z[i + 1, j + 1] += ((i - j) * (j - k_) * (k_ - i)) / 2 * beta[k_]
    L = boost(p)
    return t2v(L @ Z @ L.T)


def polV(p, beta):                                   # rest frame W = (0, beta), boosted
    return boost(p) @ np.concatenate([[0], beta])


def photon(q, lam):
    e1, e2 = frame(q[1:] / np.linalg.norm(q[1:]))
    return np.concatenate([[0], (e1 + 1j * lam * e2) / np.sqrt(2)])


def kin(P, th):
    Ep = np.sqrt(P**2 + kap**2)
    n2 = np.array([np.sin(th), 0, np.cos(th)])
    return np.array([Ep, 0, 0, P]), np.array([P, 0, 0, -P]), np.concatenate([[Ep], P * n2]), np.concatenate([[P], -P * n2])


# ---- the completed theory
eps3 = np.zeros((3, 3, 3))
for i, j, k_ in itertools.permutations(range(3)):
    eps3[i, j, k_] = np.linalg.det(np.eye(3)[[i, j, k_]])
g = 1.0
v = 1.0
MW = g * v


class Model:
    """so(3) Yang-Mills + Higgs vector n = (0, 0, v + h); fields beta^C_mu (C = 1, 2 massive, C = 3 massless) and h, M_h^2 = 2 lam v^2"""

    def __init__(self, lam):
        self.lam = lam

    def parts(self, beta, d, h, dh):
        F1 = np.array([d[:, C, :] - d[:, C, :].T for C in range(3)])
        F2 = g * np.einsum("cde,dm,en->cmn", eps3, beta, beta)
        up = lambda F: np.einsum("ma,nb,cab->cmn", eta, eta, F)
        sq = lambda u, w: np.einsum("m,m->", eta @ u, w)
        b1b1 = sq(beta[0], beta[0]) + sq(beta[1], beta[1])
        gf = np.einsum("mn,mn->", eta, d[:, 2, :])
        L2 = -0.25 * np.sum(F1 * up(F1)) + 0.5 * sq(dh, dh) + 0.5 * g**2 * v**2 * b1b1 - self.lam * v**2 * h**2 - 0.5 * gf**2
        L3 = -0.5 * np.sum(F1 * up(F2)) + g**2 * v * h * b1b1 - self.lam * v * h**3
        L4 = -0.25 * np.sum(F2 * up(F2)) + 0.5 * g**2 * h**2 * b1b1 - self.lam / 4 * h**4
        return L2, L3, L4

    def multi(self, legs, deg):
        tot = 0
        for r in range(1, len(legs) + 1):
            for S in itertools.combinations(legs, r):
                beta = sum(l["beta"] for l in S)
                h = sum(l["h"] for l in S)
                d = sum(np.einsum("m,cn->mcn", -1j * l["k"], l["beta"]) for l in S)
                dh = sum(-1j * l["k"] * l["h"] for l in S)
                tot += (-1)**(len(legs) - r) * self.parts(beta, d, h, dh)[deg - 2]
        return tot


def leg(beta=None, h=0.0, k=None):
    return {"beta": np.zeros((3, 4), complex) if beta is None else np.array(beta, complex), "h": complex(h), "k": np.array(k, complex)}


def _basis(I, k):
    b = np.zeros((3, 4), complex)
    hh = 0.0
    if I < 12:
        b[I // 4, I % 4] = 1
    else:
        hh = 1.0
    return leg(b, hh, k)


def amplitude(model, legs, with_h=True, pieces=False):
    idx = list(range(13 if with_h else 12))
    V4 = model.multi(legs, 4)
    out = {"contact": V4}
    total = V4
    for (a, b_), (c, d_) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        q = legs[a]["k"] + legs[b_]["k"]
        K = np.array([[model.multi([_basis(I, -q), _basis(Jx, q)], 2) for Jx in idx] for I in idx])
        J1 = np.array([model.multi([legs[a], legs[b_], _basis(I, -q)], 3) for I in idx])
        J2 = np.array([model.multi([legs[c], legs[d_], _basis(I, q)], 3) for I in idx])
        ex = 0 if (np.abs(J1).max() < 1e-14 or np.abs(J2).max() < 1e-14) else -J1 @ np.linalg.solve(K, J2)
        out[f"({a+1}{b_+1}|{c+1}{d_+1})"] = ex
        total += ex
    return (total, out) if pieces else total


isoP, isoM, isoA = np.array([1, 1j, 0]) / np.sqrt(2), np.array([1, -1j, 0]) / np.sqrt(2), np.array([0, 0, 1.])
Win = lambda charge, p, pol: leg(np.outer(isoP if charge > 0 else isoM, low(pol)), 0, low(p))
Wout = lambda charge, p, pol: leg(np.outer(np.conj(isoP if charge > 0 else isoM), np.conj(low(pol))), 0, -low(p))
Ain = lambda q, pol: leg(np.outer(isoA, low(pol)), 0, low(q))
Aout = lambda q, pol: leg(np.outer(isoA, np.conj(low(pol))), 0, -low(q))


def polW(p, h):
    P = np.linalg.norm(p[1:])
    n = p[1:] / P
    if h == 0:
        return np.concatenate([[P], p[0] * n]) / MW
    e1, e2 = frame(n)
    return np.concatenate([[0], (e1 + 1j * h * e2) / np.sqrt(2)])


def kinWW(P, th):
    Ee = np.sqrt(P**2 + MW**2)
    n = np.array([np.sin(th), 0, np.cos(th)])
    return np.array([Ee, 0, 0, P]), np.array([Ee, 0, 0, -P]), np.concatenate([[Ee], P * n]), np.concatenate([[Ee], -P * n])
