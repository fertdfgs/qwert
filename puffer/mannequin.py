"""Female mannequin 1.68 m built from an SDF (smooth union of primitives) -> mannequin.npz"""
import math, os
import numpy as np
from skimage.measure import marching_cubes

# Keep in sync with build.py (arm pose)
P0X, P0Z, ABD, ARM_DY = 0.175, 1.335, math.radians(22), -0.03

V = 0.005
xs = np.arange(-0.52, 0.52, V, dtype=np.float32)
ys = np.arange(-0.22, 0.22, V, dtype=np.float32)
zs = np.arange(-0.01, 1.71, V, dtype=np.float32)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")


def ell(c, r, basis=None):
    px, py, pz = X - c[0], Y - c[1], Z - c[2]
    if basis is not None:
        e = basis
        px, py, pz = (px * e[0][0] + py * e[0][1] + pz * e[0][2],
                      px * e[1][0] + py * e[1][1] + pz * e[1][2],
                      px * e[2][0] + py * e[2][1] + pz * e[2][2])
    k0 = np.sqrt((px / r[0]) ** 2 + (py / r[1]) ** 2 + (pz / r[2]) ** 2)
    k1 = np.sqrt((px / r[0] ** 2) ** 2 + (py / r[1] ** 2) ** 2 + (pz / r[2] ** 2) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def cone(a, b, ra, rb):
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    ba = b - a
    pax, pay, paz = X - a[0], Y - a[1], Z - a[2]
    h = np.clip((pax * ba[0] + pay * ba[1] + paz * ba[2]) / float(ba @ ba), 0, 1)
    return np.sqrt((pax - ba[0] * h) ** 2 + (pay - ba[1] * h) ** 2 + (paz - ba[2] * h) ** 2) - (ra + (rb - ra) * h)


def smin(a, b, k=0.03):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def norm(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


prims = [
    ell((0, 0.005, 1.565), (0.074, 0.093, 0.108)),
    ell((0, -0.025, 1.50), (0.052, 0.058, 0.05)),
    cone((0, 0.012, 1.38), (0, 0.005, 1.50), 0.05, 0.046),
    ell((0, 0.005, 1.24), (0.135, 0.095, 0.16)),
    cone((-0.15, 0.01, 1.37), (0.15, 0.01, 1.37), 0.045, 0.045),
    ell((0, 0.01, 1.06), (0.11, 0.08, 0.13)),
    ell((0, 0.01, 0.92), (0.16, 0.105, 0.12)),
]
for s in (-1, 1):
    prims += [
        ell((0.06 * s, -0.06, 1.215), (0.07, 0.055, 0.06)),
        ell((0.065 * s, 0.045, 0.86), (0.085, 0.075, 0.10)),
        cone((0.085 * s, 0.0, 0.88), (0.09 * s, -0.005, 0.49), 0.088, 0.055),
        ell((0.09 * s, -0.015, 0.48), (0.05, 0.05, 0.05)),
        cone((0.09 * s, -0.005, 0.49), (0.095 * s, 0.02, 0.08), 0.05, 0.032),
        ell((0.09 * s, 0.03, 0.36), (0.05, 0.05, 0.11)),
        ell((0.095 * s, -0.05, 0.035), (0.042, 0.115, 0.035)),
    ]
    p0 = np.array((P0X * s, 0, P0Z))
    d = norm((math.sin(ABD) * s, ARM_DY, -math.cos(ABD)))
    u = norm(np.cross((0, 1, 0), d)); v = np.cross(d, u)
    prims += [
        ell(p0, (0.052, 0.052, 0.052)),
        cone(p0, p0 + d * 0.29, 0.045, 0.036),
        cone(p0 + d * 0.29, p0 + d * 0.54, 0.037, 0.025),
        ell(p0 + d * 0.625, (0.02, 0.042, 0.085), basis=(u, v, d)),
    ]

F = prims[0]
for p in prims[1:]:
    F = smin(F, p)
verts, faces, _, _ = marching_cubes(F, 0.0, spacing=(V, V, V))
verts += np.array([xs[0], ys[0], zs[0]])
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mannequin.npz"), v=verts.astype(np.float32), f=faces.astype(np.int32))
print("mannequin", verts.shape, faces.shape, verts.min(0), verts.max(0))
