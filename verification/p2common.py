"""Shared helpers for the verification scripts of Part II.

Conventions of Part I (Appendix A): on R^8,
  e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1, e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep,  ep = [[0,-1],[1,0]],
with eta = diag(1,1,1,-1,-1,-1).  Tower: E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1^{(x)(n-k)}, omega = e1...e6, numbered A = 6(k-1)+a.
The temporal generators (a <= 3) are symmetric matrices, the spatial ones (a >= 4) antisymmetric.

Every script prints one line 'PASS <id>: <claim>' or 'FAIL <id>: <claim>' per check and exits with status 0 only if all pass.
"""
import sys
from functools import reduce

import numpy as np

FAILED = []


def report(cid, ok, text):
    print(f"{'PASS' if bool(ok) else 'FAIL'} {cid}: {text}")
    if not bool(ok):
        FAILED.append(cid)


def finish():
    if FAILED:
        print("# failed:", " ".join(FAILED))
    sys.exit(1 if FAILED else 0)


# float64 arrays holding small integers: every product below is exact (entries far below 2^53), and BLAS makes the 512 x 512 products fast
sx = np.array([[0, 1], [1, 0]], dtype=np.float64)
sz = np.array([[1, 0], [0, -1]], dtype=np.float64)
ep = np.array([[0, -1], [1, 0]], dtype=np.float64)
I2 = np.eye(2, dtype=np.float64)


def kron(*ms):
    return reduce(np.kron, ms)


def cl33():
    """the six generators of Cl(3,3) on R^8 (integer matrices), as a list e[0..5] = e1..e6"""
    return [kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]


def tower(n):
    """the 6n generators E[0..6n-1] of Cl(3n,3n) on R^(8^n); E[6(k-1)+a-1] = E_(k,a)"""
    e = cl33()
    om = reduce(lambda a, b: a @ b, e)
    I8 = np.eye(8, dtype=np.float64)
    E = []
    for k in range(n):
        for a in range(6):
            E.append(kron(*([om] * k + [e[a]] + [I8] * (n - k - 1))))
    return E


def prod(ms, dim):
    return reduce(lambda a, b: a @ b, ms, np.eye(dim, dtype=np.float64))


def comm(A, B):
    return A @ B - B @ A


def acomm(A, B):
    return A @ B + B @ A


def zero(M, tol=0):
    return np.abs(M).max() <= tol
