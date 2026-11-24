"""Symbolic helpers for the field-theory checks of Part II (sympy).

Frame variables of Part I: X = (B; E), G(B; E) = (-E; B), eta(X, X) = |B|^2 - |E|^2, c = 1.
Sources: d_t B = curl E - k, d_t E = -curl B - j, rho_e = div E, rho_m = div B.
"""
import sympy as sp
from sympy.calculus.euler import euler_equations

t, x, y, z = sp.symbols("t x y z", real=True)
co = (x, y, z)
xs = (t, x, y, z)
sg = (1, -1, -1, -1)


def fld(name):
    return sp.Function(name)(t, x, y, z)


def vec3(name):
    return [fld(f"{name}{i}") for i in (1, 2, 3)]


def curl(v):
    return [sp.diff(v[2], y) - sp.diff(v[1], z), sp.diff(v[0], z) - sp.diff(v[2], x), sp.diff(v[1], x) - sp.diff(v[0], y)]


def div(v):
    return sum(sp.diff(v[i], co[i]) for i in range(3))


def grad(f):
    return [sp.diff(f, c) for c in co]


def lap(f):
    return sum(sp.diff(f, c, 2) for c in co)


def box(f):
    return sp.diff(f, t, 2) - lap(f)


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(p * q for p, q in zip(a, b))


def EL(L, fields):
    """Euler-Lagrange expressions dL/df - d_mu dL/d(d_mu f) + ..., one per field (0 where sympy returns True)"""
    return [sp.expand(e.lhs) if hasattr(e, "lhs") else sp.Integer(0) for e in euler_equations(L, fields, [t, x, y, z])]


def zero(L):
    return all(sp.simplify(sp.expand(v)) == 0 for v in L)


def currents(B, E):
    """(rho_e, j, rho_m, k) of a field X = (B; E)"""
    j = [-(sp.diff(E[i], t) + curl(B)[i]) for i in range(3)]
    k = [-(sp.diff(B[i], t) - curl(E)[i]) for i in range(3)]
    return div(E), j, div(B), k
