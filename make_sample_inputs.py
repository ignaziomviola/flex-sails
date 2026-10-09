"""Generate sample input files for panel_wing.py.

Creates:
- wing_mesh.npz          tapered, swept, twisted wing with parabolic camber
- velocity_profile.csv   linear shear profile U(z) = 1 + 0.3 z
- rect_mesh.npz          flat rectangular wing (AR 8), for regression checks
- uniform_profile.csv    uniform profile U = 1
"""

import numpy as np


def generic_wing_mesh(nchord=6, nspan=20, span=8.0, root_chord=4.0 / 3.0,
                      taper=0.5, sweep_deg=15.0, twist_deg=-3.0, camber=0.02):
    """Mean camber surface of a tapered, swept, linearly twisted wing.

    Twist rotates each section about its quarter-chord point (washout for
    negative twist_deg at the tips); camber is a parabolic arc of height
    `camber` * local chord.
    """
    points = np.zeros((nchord + 1, nspan + 1, 3))
    tan_sweep = np.tan(np.radians(sweep_deg))
    xi = np.linspace(0.0, 1.0, nchord + 1)
    for j, y in enumerate(np.linspace(-span / 2.0, span / 2.0, nspan + 1)):
        eta = abs(2.0 * y / span)
        chord = root_chord * (1.0 + (taper - 1.0) * eta)
        x_le = tan_sweep * abs(y)
        theta = np.radians(twist_deg) * eta
        x_sec = (xi - 0.25) * chord
        z_sec = 4.0 * camber * chord * xi * (1.0 - xi)
        x_rot = x_sec * np.cos(theta) + z_sec * np.sin(theta)
        z_rot = -x_sec * np.sin(theta) + z_sec * np.cos(theta)
        points[:, j, 0] = x_le + 0.25 * chord + x_rot
        points[:, j, 1] = y
        points[:, j, 2] = z_rot
    return points


def rectangular_mesh(nchord=4, nspan=12, span=8.0, chord=1.0):
    """Flat rectangular wing in the z = 0 plane."""
    x = np.linspace(0.0, chord, nchord + 1)
    y = np.linspace(-span / 2.0, span / 2.0, nspan + 1)
    points = np.zeros((nchord + 1, nspan + 1, 3))
    points[..., 0] = x[:, None]
    points[..., 1] = y[None, :]
    return points


def main():
    np.savez("wing_mesh.npz", points=generic_wing_mesh())
    print("wrote wing_mesh.npz (6 x 20 panels, taper 0.5, sweep 15 deg, "
          "twist -3 deg, camber 2%)")

    z = np.linspace(-4.0, 4.0, 17)
    u = np.maximum(1.0 + 0.3 * z, 0.1)
    np.savetxt("velocity_profile.csv", np.column_stack([z, u]),
               delimiter=",", header="z, U", comments="# ")
    print("wrote velocity_profile.csv (linear shear U = 1 + 0.3 z)")

    np.savez("rect_mesh.npz", points=rectangular_mesh())
    print("wrote rect_mesh.npz (flat rectangular, AR 8, 4 x 12 panels)")

    np.savetxt("uniform_profile.csv",
               np.column_stack([[-10.0, 10.0], [1.0, 1.0]]),
               delimiter=",", header="z, U", comments="# ")
    print("wrote uniform_profile.csv (U = 1)")


if __name__ == "__main__":
    main()
