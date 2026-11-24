"""Shared setup of the second rung for ch09b_potential.py (the same objects as ch09_electroweak.py builds).

The gauge algebra g_2 + s_2 = su(4) + su(2)_L + su(2)_R, the global phases Omega and G_2, the charges T_3L, T_3R, B - L, Y, Q,
the 272 Lorentz-invariant mass kernels with their isotypic parts, the Pati-Salam direction Phi_0, the generator-of-mass kernel G_2 w_S,
the charge sectors u, d, nu, e, and the neutral directions.  Importing this module computes them (a few seconds); it prints nothing.
"""
import itertools

import numpy as np

from p2common import tower, prod, comm, zero

TOL = 1e-8
E = tower(2)
dim = 64
I = np.eye(dim)


def ip(X, Y):
    return np.trace(X.T @ Y) / dim


def rank(mats, tol=TOL):
    return np.linalg.matrix_rank(np.array([m.flatten() for m in mats]), tol=tol)


def stab(Phi, gens):
    """dimension of the stabiliser of the kernel Phi in span(gens)"""
    return len(gens) - rank([comm(t, Phi) for t in gens])


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
Gn = [t / np.sqrt(ip(t, t)) for t in g2 + la + lb]        # the 21 gauge generators, orthonormal for the trace form
G2 = Gam @ prod(P, dim)
Gln = [t / np.sqrt(ip(t, t)) for t in (Om, G2)]           # the two global phases
H = [0.5 * P[0] @ P[1], 0.5 * P[2] @ P[3], 0.5 * P[4] @ Gam]
BL = H[0] + H[1] + H[2]                                   # (3/2)(B - L)
_A = np.array([comm(BL, t).flatten() for t in g2]).T
_r = np.linalg.matrix_rank(_A, tol=1e-9)
_u3 = [sum(c * t for c, t in zip(row, g2)) for row in np.linalg.svd(_A)[2][_r:]]
_Mx = np.array([comm(a, b).flatten() for a, b in itertools.combinations(_u3, 2)])
_, _s, _Vt = np.linalg.svd(_Mx, full_matrices=False)
su3 = [_Vt[i].reshape(dim, dim) for i in range(int((_s > 1e-9 * _s.max()).sum()))]
T3L, T3R = la[0], lb[0]
BmL = (2.0 / 3.0) * BL                                    # B - L
Y = T3R + BL / 3.0
Q = T3L + Y

blades = [prod([([Gam] + P + N)[i] for i in S], dim) for r_ in range(10) for S in itertools.combinations(range(9), r_)]
kern = [b @ wS for b in blades if zero(b @ wS + (b @ wS).T)]
Bf = np.array([M.flatten() for M in kern])              # orthonormal coordinates of the mass field (272 kernels)


def coords(M):
    return Bf @ M.flatten() / dim


def tomat(c):
    return (Bf.T @ c).reshape(dim, dim)


def admat(t):
    return Bf @ np.array([comm(t, M).flatten() for M in kern]).T / dim


def _casimir(gens):
    return Bf @ np.array([sum(comm(t, comm(t, M)) for t in gens).flatten() for M in kern]).T / dim


_C6, _Ca, _Cb = _casimir(g2), _casimir(la), _casimir(lb)
_Cm = _C6 + np.sqrt(2) * _Ca + np.sqrt(3) * _Cb
_, _Vm = np.linalg.eigh((_Cm + _Cm.T) / 2)
_groups = {}
for _k in range(len(kern)):
    _v = _Vm[:, _k]
    _groups.setdefault(tuple(round(_v @ C @ _v * 4) / 4 for C in (_C6, _Ca, _Cb)), []).append(_k)
_name = {(0.0, -0.75, -0.75): "(1,2,2)", (-8.0, -0.75, -0.75): "(15,2,2)", (-9.0, 0.0, -2.0): "(10b,1,3)", (-9.0, -2.0, 0.0): "(10,3,1)",
         (-5.0, 0.0, 0.0): "2(6,1,1)"}
Bpart = {_name[k]: _Vm[:, ks] for k, ks in _groups.items()}
Proj = {k: B @ B.T for k, B in Bpart.items()}          # the equivariant projections onto the isotypic parts


def part(r, M):
    return tomat(Proj[r] @ coords(M))


def fix(gens, basis):
    """the kernels in span(basis) fixed by every generator in gens (basis: columns of coordinates)"""
    Am = np.vstack([admat(t) for t in gens]) @ basis
    rk = np.linalg.matrix_rank(Am, tol=TOL)
    return basis @ np.linalg.svd(Am)[2][rk:].T


Phi0 = tomat(fix(su3 + la + [Y], Bpart["(10b,1,3)"])[:, 0])
Phi0 = Phi0 / np.linalg.svd(Phi0, compute_uv=False).max()   # unit singular values on its support
Mg = G2 @ wS                                                # the generator of mass carried into V^-
S3w = (T3L + T3R) @ wS
Bw = BmL @ wS
_q2 = -(Q @ Q)
_wq, _Vq = np.linalg.eigh((_q2 + _q2.T) / 2)
sect = {}
for _val, _nm in ((4 / 9, "u"), (1 / 9, "d"), (0.0, "nu"), (1.0, "e")):
    _Vs = _Vq[:, np.abs(_wq - _val) < 1e-9]
    sect[_nm] = _Vs @ _Vs.T
fixEM = {k: fix(su3 + [Q], B) for k, B in Bpart.items()}  # the neutral kernels of each part (2, 2, 4, 4, 0)


def zpattern(m):
    """complex values of a neutral kernel on the Dirac pairs u, d, nu, e, relative to G_2 w_S (Omega as the imaginary unit)"""
    out = []
    for s_ in ("u", "d", "nu", "e"):
        Ps = sect[s_]
        R = Ps @ Mg @ Ps
        A2 = np.array([R.flatten(), (Om @ R).flatten()]).T
        c2 = np.linalg.lstsq(A2, (Ps @ m @ Ps).flatten(), rcond=None)[0]
        out.append(complex(c2[0], c2[1]))
    return np.array(out)


# the neutral direction that the split of Definition 9.9 excludes, with the real values (1, -1, -3, 3)
_span6 = [Mg, S3w, Bw, Om @ Mg, Om @ S3w, Om @ Bw]
_N8 = np.hstack([fixEM["(1,2,2)"], fixEM["(15,2,2)"]])
_C6m = np.array([coords(M) for M in _span6]).T
_Pc = _N8 - _C6m @ np.linalg.lstsq(_C6m, _N8, rcond=None)[0]
_M0 = tomat(np.linalg.svd(_Pc, full_matrices=False)[0][:, 0])
_w = 1 / zpattern(_M0)[0]
Dmiss = _w.real * _M0 + _w.imag * (Om @ _M0)
DL = tomat(fixEM["(10,3,1)"][:, 0])
DL = DL / np.linalg.svd(DL, compute_uv=False).max()        # the neutral (10,3,1): a Majorana mass for nu_L
neutral12 = {"G": Mg, "OG": Om @ Mg, "S3": S3w, "OS3": Om @ S3w, "B": Bw, "OB": Om @ Bw, "D": Dmiss, "OD": Om @ Dmiss,
             "Phi0": Phi0, "OPhi0": Om @ Phi0, "DL": DL, "ODL": Om @ DL}
NB = np.array([coords(M) for M in neutral12.values()]).T  # 272 x 12


def neutral_components(c):
    """coefficients of a vector of kernel coordinates in the basis neutral12, and the norm of what lies outside it"""
    co = np.linalg.solve(NB.T @ NB, NB.T @ c)
    return dict(zip(neutral12, co)), np.linalg.norm(NB @ co - c)
