"""Two-dimensional linear sail theory for an extensible membrane wing.

This is the independent reference the three-dimensional coupled model is
checked against. It imports nothing from this repository and shares no code
with the lifting surface or the membrane elements.

A membrane of chord c is pinned at its leading and trailing edges, which lie
on a chord line at incidence alpha to a stream U. In the linear theory of
Thwaites (1961) and Nielsen (1963) the camber z(x), measured from the chord
line towards the suction side, and the bound vorticity gamma(x) satisfy

    tangency   (1 / 2 pi) int gamma(xi) / (x - xi) dxi = U (alpha - z'(x))
    membrane   T z'' = - rho U gamma(x)        (pressure jump rho U gamma)
    edges      z(0) = z(c) = 0, and the Kutta condition at x = c

with a uniform tension T, uniform because the load is normal to the cloth.
The cloth is extensible: it is pre-strained by e0 in both directions and
held in plane strain by the pinned edges, so to first order

    T = E t / (1 - nu^2) (dL / c + (1 + nu) e0),   dL = int z'^2 dx / 2

which is the mid-span state of a long three-dimensional membrane wing pinned
on its luff and leech. Given T the problem is linear; T itself is the root of
a monotone scalar equation and is found by bisection.

The discretisation is the lumped-vortex method on cosine-spaced panels for
the flow and second differences for the membrane. It converges to the
continuous theory as the panel count grows; `solve` is the discrete problem
and `extrapolate` its Richardson limit.
"""

import numpy as np


def _discrete(alpha, C_T, n, spacing="cosine"):
    """Camber and circulation for a given tension coefficient C_T = T / (q c).

    Lengths are in chords and velocities in U. Returns (x, z, gamma_panels).
    """
    if spacing == "cosine":
        x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n + 1)))
    elif spacing == "uniform":
        x = np.linspace(0.0, 1.0, n + 1)
    else:
        raise ValueError(f"unknown spacing '{spacing}'")
    h = np.diff(x)
    xv = x[:-1] + 0.25 * h
    xc = x[:-1] + 0.75 * h
    # unknowns: z_1 .. z_{n-1} (interior nodes), then G_0 .. G_{n-1}
    nz = n - 1
    A = np.zeros((nz + n, nz + n))
    b = np.zeros(nz + n)
    # tangency at every collocation point:
    #   sum_j G_j / (2 pi (xc_m - xv_j)) + (z_{m+1} - z_m) / h_m = alpha
    A[:n, nz:] = 1.0 / (2.0 * np.pi * (xc[:, None] - xv[None, :]))
    for m in range(n):
        if m + 1 <= n - 1:
            A[m, m] += 1.0 / h[m]              # z_{m+1}, column m
        if m >= 1:
            A[m, m - 1] -= 1.0 / h[m]          # z_m, column m - 1
    b[:n] = alpha
    # membrane at interior node k (1 .. n-1), with T = C_T q c and the load
    # rho U G lumped 3/4 and 1/4 to the ends of each panel; in units of q c,
    # rho U G = 2 G:
    #   C_T [(z_{k+1} - z_k) / h_k - (z_k - z_{k-1}) / h_{k-1}] + 2 F_k = 0
    for k in range(1, n):
        row = n + k - 1
        for node, coef in ((k + 1, 1.0 / h[k]),
                           (k, -1.0 / h[k] - 1.0 / h[k - 1]),
                           (k - 1, 1.0 / h[k - 1])):
            if 1 <= node <= n - 1:
                A[row, node - 1] += C_T * coef
        A[row, nz + k - 1] += 2.0 * 0.25     # panel k-1, aft quarter
        A[row, nz + k] += 2.0 * 0.75         # panel k, forward three quarters
    sol = np.linalg.solve(A, b)
    z = np.concatenate([[0.0], sol[:nz], [0.0]])
    return x, z, sol[nz:]


def solve(alpha_rad, et_over_qc, prestrain, nu=0.3, n=256, spacing="cosine"):
    """Discrete linear sail with elastic tension. Returns a dict.

    et_over_qc  the elasticity number E t / (q c)
    prestrain   the pre-strain e0 of the cloth
    spacing     "cosine", or "uniform" to match a uniform chordwise mesh
    """
    eb = et_over_qc / (1.0 - nu * nu)

    def excess(C_T):
        x, z, g = _discrete(alpha_rad, C_T, n, spacing)
        dz = np.diff(z)
        return 0.5 * np.sum(dz * dz / np.diff(x)), (x, z, g)

    def mismatch(C_T):
        return C_T - eb * (excess(C_T)[0] + (1.0 + nu) * prestrain)

    lo = eb * (1.0 + nu) * prestrain
    if lo <= 0.0:
        lo = 1e-8
    hi = 2.0 * lo
    while mismatch(hi) < 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mismatch(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * hi:
            break
    C_T = 0.5 * (lo + hi)
    dl, (x, z, g) = excess(C_T)
    k = int(np.argmax(z))
    return {"C_T": C_T, "CL": 2.0 * np.sum(g), "depth": z[k],
            "draft": x[k], "excess": dl, "x": x, "z": z, "gamma": g}


def extrapolate(alpha_rad, et_over_qc, prestrain, nu=0.3, n=128):
    """Richardson limit from n, 2n and 4n cosine panels, at the observed order.

    The order is close to one, not two: the load is singular as x^-1/2 at the
    leading edge, and the lumped vortex resolves that singularity to first
    order only. Returns the limits, the observed orders and the spreads.
    """
    runs = [solve(alpha_rad, et_over_qc, prestrain, nu, m)
            for m in (n, 2 * n, 4 * n)]
    out = {}
    for key in ("C_T", "CL", "depth"):
        a, b, c = (r[key] for r in runs)
        p = np.log2(abs((a - b) / (c - b))) if c != b else 2.0
        out[key] = c + (c - b) / (2.0 ** p - 1.0)
        out[key + "_order"] = p
        out[key + "_spread"] = abs(c - b)
    return out


def main():
    alpha = np.radians(6.0)
    print(" E t / q c    C_T       CL      depth")
    for et in (50.0, 100.0, 200.0, 400.0, 1e4):
        r = solve(alpha, et, 0.002)
        print(f"{et:9.0f} {r['C_T']:8.3f} {r['CL']:8.4f} {r['depth']:8.4f}")


if __name__ == "__main__":
    main()
