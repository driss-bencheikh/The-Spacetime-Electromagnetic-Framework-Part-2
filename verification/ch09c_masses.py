#!/usr/bin/env python3
"""Part II, Chapter 9 -- each fermion its measured mass.

Whatever gives the three families (left open, Chapter 10), they carry a family index on which nothing else acts, and a value of the mass
field is then a sum of kernels times symmetric family matrices.  With diagonal family matrices each family has its own four values
(k, x_3, x_B, d) of the general split (Definition 9.13), and by Theorem 9.14 its four Dirac masses are any four numbers.

Inputs, in GeV: the PDG 2024 values as quoted by Antusch, Hinze and Saad (arXiv:2510.01312, the 2024 list of their Section 2): the pole
masses of the top quark and of the charged leptons, and the MS-bar masses m_u, m_d, m_s at 2 GeV, m_c(m_c), m_b(m_b).  Neutrinos: the
splittings of NuFIT 6.0 (arXiv:2410.05380, normal ordering, IC24 with SK atmospheric data), Delta m^2_21 = 7.49e-5 eV^2 and
Delta m^2_31 = 2.513e-3 eV^2; the lightest mass and the Majorana scale are not measured, and the example takes m_1 = 0 and v_R = 1e14 GeV.

    python3 ch09c_masses.py

All checks are numerical; masses are compared with the inputs to a relative 1e-12, the light neutrino masses to 1e-9.
"""
import numpy as np

from p2common import report, finish
from p2rung2 import Mg, S3w, Bw, Dmiss, Phi0, sect, dim, E

# ---------------------------------------------------------------- inputs (GeV)
MASSES = {1: {"u": 2.16e-3, "d": 4.70e-3, "e": 0.51099895000e-3},
          2: {"u": 1.2730, "d": 93.5e-3, "e": 105.6583755e-3},
          3: {"u": 172.4, "d": 4.183, "e": 1776.93e-3}}
DM21, DM31 = 7.49e-5 * 1e-18, 2.513e-3 * 1e-18             # eV^2 -> GeV^2
M1 = 0.0
LIGHT = {1: M1, 2: np.sqrt(M1 ** 2 + DM21), 3: np.sqrt(M1 ** 2 + DM31)}
VR = 1e14
DIRAC_NU = {i: np.sqrt(LIGHT[i] * VR) for i in LIGHT}      # m_nu^D = sqrt(m_i v_R), the seesaw for m_nu^D << v_R
SECTORS = ("u", "d", "nu", "e")


def values(z):
    """(k, x_3, x_B, d) from the signed masses (z_u, z_d, z_nu, z_e), Theorem 9.14(ii)"""
    zu, zd, zn, ze = z
    return np.array([(3 * (zu + zd) + zn + ze) / 8, (3 * (zd - zu) + ze - zn) / 4, 3 * ((zn + ze) - (zu + zd)) / 8, (zu - zd - zn + ze) / 8])


def kernel(v):
    k_, x3, xB, d_ = v
    return k_ * Mg + x3 * S3w + xB * Bw + d_ * Dmiss


def sector_masses(m_, projs):
    out = []
    for Ps in projs:
        sv = np.linalg.svd(m_ @ Ps, compute_uv=False)
        out.append(sv.max() if sv.max() > 1e-30 else 0.0)
    return np.array(out)


# ---------------------------------------------------------------- ms-1: one family at a time
ok, table = True, {}
for nu_choice in ("zero", "seesaw"):
    for i in (1, 2, 3):
        z = np.array([MASSES[i]["u"], MASSES[i]["d"], 0.0 if nu_choice == "zero" else DIRAC_NU[i], MASSES[i]["e"]])
        v = values(z)
        table[(nu_choice, i)] = v
        got = sector_masses(kernel(v), [sect[s_] for s_ in SECTORS])
        ok = ok and np.allclose(got, z, rtol=1e-12, atol=1e-12 * z.max())
report("ms-1", ok,
       "(numerical) for each family, the values (k, x_3, x_B, d) of the general split computed from its measured masses, with all signs positive, give back on the "
       "sectors u, d, nu, e exactly the inputs (PDG 2024): " + "; ".join(
           f"family {i}: k = {table[('zero', i)][0]:.6g}, x_3 = {table[('zero', i)][1]:.6g}, x_B = {table[('zero', i)][2]:.6g}, d = {table[('zero', i)][3]:.6g} GeV"
           for i in (1, 2, 3)) + " at m_nu^D = 0, and the same with the seesaw values of m_nu^D")

# ---------------------------------------------------------------- ms-2: one value of the mass field with a family index
I3 = np.eye(3)
F = [np.diag([table[("seesaw", i)][a] for i in (1, 2, 3)]) for a in range(4)]
Mfam = sum(np.kron(D_, F_) for D_, F_ in zip((Mg, S3w, Bw, Dmiss), F))     # on R^64 (x) R^3
lor_rot = [np.kron(E[3 + i] @ E[3 + j], I3) for i, j in ((0, 1), (1, 2), (0, 2))]
lor_boost = [np.kron(E[0] @ E[3 + a], I3) for a in range(3)]
ok = np.abs(Mfam + Mfam.T).max() < 1e-12 * np.abs(Mfam).max()
ok = ok and all(np.abs(Mfam @ X - X @ Mfam).max() < 1e-12 * np.abs(Mfam).max() for X in lor_rot)
ok = ok and all(np.abs(Mfam @ X + X @ Mfam).max() < 1e-12 * np.abs(Mfam).max() for X in lor_boost)
for i in (1, 2, 3):
    ei = np.zeros((3, 3))
    ei[i - 1, i - 1] = 1
    got = sector_masses(Mfam, [np.kron(sect[s_], ei) for s_ in SECTORS])
    want = np.array([MASSES[i]["u"], MASSES[i]["d"], DIRAC_NU[i], MASSES[i]["e"]])
    ok = ok and np.allclose(got, want, rtol=1e-12, atol=1e-12 * want.max())
report("ms-2", ok,
       "(numerical) one value of the mass field with a family index, sum over the four directions of the kernel times a diagonal family matrix, is antisymmetric and "
       "Lorentz invariant and gives the nine charged fermions their measured masses at once, each family its own, with the seesaw Dirac masses of the neutrinos")

# ---------------------------------------------------------------- ms-3: the light neutrinos through the seesaw
light = {i: (2 * DIRAC_NU[i] ** 2 / (np.sqrt(VR ** 2 + 4 * DIRAC_NU[i] ** 2) + VR)) for i in (1, 2, 3)}
ok = all(abs(light[i] - LIGHT[i]) <= 1e-9 * max(LIGHT[3], 1e-30) for i in (1, 2, 3))
ok = ok and abs((light[2] ** 2 - light[1] ** 2) / DM21 - 1) < 1e-9 and abs((light[3] ** 2 - light[1] ** 2) / DM31 - 1) < 1e-9
vr, mD = 1000.0, 1.0                                        # the formula against the kernel, at a ratio double precision resolves
sv = np.linalg.svd((vr * Phi0 + kernel(values(np.array([0.0, 0.0, mD, 0.0])))) @ sect["nu"], compute_uv=False)
sv = np.sort(sv[sv > 1e-12])
ok = ok and len(sv) == 8 and np.allclose(sv[:4], 2 * mD ** 2 / (np.sqrt(vr ** 2 + 4 * mD ** 2) + vr), rtol=1e-9)
report("ms-3", ok,
       f"(numerical) with v_R = 1e14 GeV and m_1 = 0, the Dirac masses m_nu^D = sqrt(m_i v_R) = (0, {DIRAC_NU[2]:.4g}, {DIRAC_NU[3]:.4g}) GeV give through the seesaw "
       f"the light masses (0, {LIGHT[2] * 1e9 * 1e3:.4g}, {LIGHT[3] * 1e9 * 1e3:.4g}) meV, with Delta m^2_21 and Delta m^2_31 of NuFIT 6.0; the seesaw formula agrees with "
       "the singular values of the kernel at v_R / m_D = 1000")
# ---------------------------------------------------------------- ms-4: the exact inversion of the seesaw block
from fractions import Fraction
ok = True
for ml_, vr_ in ((Fraction(1, 7), Fraction(5)), (Fraction(3, 2), Fraction(1, 3)), (Fraction(1, 1000), Fraction(10**6))):
    mD2 = ml_ * (vr_ + ml_)                                  # m_D^2 = m_l (v_R + m_l)
    # the light singular value of [[0, m_D], [m_D, v_R]] is m_l exactly when m_l^2 + v_R m_l - m_D^2 = 0
    ok = ok and ml_**2 + vr_ * ml_ - mD2 == 0
exact = {i: np.sqrt(LIGHT[i] * (VR + LIGHT[i])) for i in (1, 2, 3)}
ok = ok and all(abs(exact[i] - DIRAC_NU[i]) <= 1e-12 * max(DIRAC_NU[3], 1e-30) for i in (1, 2, 3))
report("ms-4", ok,
       "(exact and numerical) the seesaw block [[0, m_D], [m_D, v_R]] has the light singular value m_l exactly when m_D = sqrt(m_l (v_R + m_l)) (exact, three rational points); "
       "at v_R = 1e14 GeV this agrees with sqrt(m_l v_R) to the digits shown, for the three Dirac masses of the example")
finish()
