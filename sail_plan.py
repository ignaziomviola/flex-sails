"""A two-sail plan upwind: rig, apparent wind, interaction and stresses.

The vendored lifting surface takes one structured surface, so the genoa and
the mainsail are meshed separately. They are nonetheless solved together:
each sail's converged vortex system enters the other's boundary condition and
loads through `sail_fluid.induced_field`, and the two flying shapes are
iterated to a joint fixed point (`solve_sail_plan`). This is the frozen-wake
solution of both lifting surfaces at once; case C7 of verify_sail.py shows
that a wing split in two and solved this way reproduces the whole wing to
1e-10 in lift and induced drag.

Frames. Boat coordinates are X aft from the tack, y up from the sheer line
and L to leeward. The solver frame is the vendored one: x along the apparent
wind at the reference height, y up, z to leeward. The boat's aft axis is at
the apparent wind angle A to x, aft = (cos A, 0, -sin A), and its leeward
axis is lee = (sin A, 0, cos A).

The apparent wind is the vector difference of a true wind with a power-law
gradient over the sea and the boat's velocity, so both its speed and its angle
grow with height.

Data. The rig is the J/80 one-design (I, J, P, E from the class measurements).
The cloth is woven polyester with E = 873 MPa, t = 0.25 mm and nu = 0.4, the
values of Blicblau et al. (2008) for a woven polyester sail. The wind
gradient exponent 0.11 is the value Hsu et al. (1994) recommend over the
sea. The boat speed, true wind angle, trim, moulded shapes, pre-strains,
leech tapes and battens are stated choices, listed in J80_UPWIND.

Run as a program it solves the J/80 sail plan upwind, prints a summary, and
writes the deformation and stress figures to docs/figures/.
"""

import sys
import time

import numpy as np

import sail_fluid as sf
import sail_fsi as fsi
import sail_membrane as sm

FT = 0.3048
KNOT = 1852.0 / 3600.0

J80 = {"I": 31.5 * FT, "J": 9.5 * FT, "P": 30.0 * FT, "E": 12.5 * FT}

J80_UPWIND = {
    # true wind and boat
    "tws_kn": 10.0, "z_ref": 10.0, "shear": 0.11, "twa_deg": 42.0,
    "boat_kn": 5.5, "freeboard": 0.6, "y_ref": 5.0,
    # cloth: woven polyester (Blicblau et al. 2008)
    "E_cloth": 873e6, "t_cloth": 0.25e-3, "nu": 0.4, "prestrain": 0.001,
    # genoa: 135% on the J/80 forestay
    "lp": 1.35, "genoa_tack_y": 0.15, "genoa_head_drop": 0.3,
    "genoa_head": 0.15, "genoa_depth": 0.13, "genoa_draft": 0.40,
    "genoa_sheet_deg": (10.0, 18.0), "genoa_tape_EA": 1e5,
    "genoa_tape_e0": 0.01,
    # mainsail: boom above sheer, headboard, three full battens
    "boom_y": 0.9, "main_head": 0.15, "main_depth": 0.11,
    "main_draft": 0.45, "main_sheet_deg": (2.0, 10.0), "main_tape_EA": 1e5,
    "main_tape_e0": 0.01, "batten_EI": 50.0,
    # meshes
    "nchord": 10, "nspan": 16,
}


# ------------------------------------------------------------ apparent wind

def apparent_wind(tws, twa_deg, boat, z_ref=10.0, shear=0.11):
    """Apparent wind speed and angle [rad] as functions of height h above
    the water: true wind tws (h / z_ref)^shear at twa to the bow, minus the
    boat velocity along the bow."""
    twa = np.radians(twa_deg)

    def wind(h):
        t = tws * (np.maximum(h, 0.05) / z_ref) ** shear
        aws = np.hypot(t * np.sin(twa), t * np.cos(twa) + boat)
        awa = np.arctan2(t * np.sin(twa), t * np.cos(twa) + boat)
        return aws, awa
    return wind


def apparent_onset(case):
    """Onset callable in the solver frame, and the reference speed and angle.

    x is the apparent wind at y_ref above the sheer line. Higher up the wind
    is stronger and further aft; the extra angle beta = AWA(h) - AWA_ref
    rotates the onset towards +z, which raises the incidence aloft.
    """
    wind = apparent_wind(case["tws_kn"] * KNOT, case["twa_deg"],
                         case["boat_kn"] * KNOT, case["z_ref"],
                         case["shear"])
    u_ref, awa_ref = wind(case["y_ref"] + case["freeboard"])

    def onset(points):
        pts = np.atleast_2d(points)
        aws, awa = wind(pts[:, 1] + case["freeboard"])
        beta = awa - awa_ref
        v = np.zeros((len(pts), 3))
        v[:, 0] = aws * np.cos(beta)
        v[:, 2] = aws * np.sin(beta)
        return v
    return onset, float(u_ref), float(awa_ref), wind


# ----------------------------------------------------------------- geometry

def boat_axes(awa):
    """Aft and leeward unit vectors of the boat in the solver frame."""
    return (np.array([np.cos(awa), 0.0, -np.sin(awa)]),
            np.array([np.sin(awa), 0.0, np.cos(awa)]))


def sail_mesh(luff0, luff1, chord0, chord1, sheet_deg, depth, draft, awa,
              nchord, nspan):
    """Moulded sail with horizontal chords, in the solver frame.

    luff0, luff1  (X, y) of the luff at the foot and the head, boat axes
    chord0, chord1  foot and head chords [m], varying linearly with height
    sheet_deg  (foot, head) sheeting angle of the chord to the centreline,
               towards leeward, varying linearly: the trimmed twist
    depth, draft  moulded camber depth and its position, fractions of chord
    """
    aft, lee = boat_axes(awa)
    eta = np.linspace(0.0, 1.0, nspan + 1)
    xi = np.linspace(0.0, 1.0, nchord + 1)
    cam = fsi.camber_line(xi, draft)
    pts = np.zeros((nchord + 1, nspan + 1, 3))
    for j, e in enumerate(eta):
        X = luff0[0] + e * (luff1[0] - luff0[0])
        y = luff0[1] + e * (luff1[1] - luff0[1])
        c = chord0 + e * (chord1 - chord0)
        d = np.radians(sheet_deg[0] + e * (sheet_deg[1] - sheet_deg[0]))
        along = np.cos(d) * aft + np.sin(d) * lee
        normal = -np.sin(d) * aft + np.cos(d) * lee
        le = X * aft + np.array([0.0, y, 0.0])
        pts[:, j] = (le[None, :] + (xi * c)[:, None] * along[None, :]
                     + (depth * c * cam)[:, None] * normal[None, :])
    return pts


def j80_sails(case=J80_UPWIND, rig=J80, nchord=None, nspan=None):
    """Moulded genoa and mainsail meshes and their structural models."""
    nc = nchord or case["nchord"]
    ns = nspan or case["nspan"]
    _, _, awa, _ = apparent_onset(case)
    I, J, P, E = rig["I"], rig["J"], rig["P"], rig["E"]
    Et = case["E_cloth"] * case["t_cloth"]

    # genoa: luff on the forestay from the stem (0, 0) to the hounds (J, I);
    # the foot chord gives the luff perpendicular LP = lp J
    y0, y1 = case["genoa_tack_y"], I - case["genoa_head_drop"]
    foot = case["lp"] * J / np.sin(np.arctan2(I, J))
    genoa = sail_mesh((J * y0 / I, y0), (J * y1 / I, y1), foot,
                      case["genoa_head"], case["genoa_sheet_deg"],
                      case["genoa_depth"], case["genoa_draft"], awa, nc, ns)
    genoa_model = fsi.build_sail(
        genoa, Et, case["nu"], case["prestrain"], foot="clew", leech="free",
        head="pinned", leech_EA=case["genoa_tape_EA"],
        foot_EA=case["genoa_tape_EA"], cable_prestrain=case["genoa_tape_e0"])

    # mainsail: luff on the mast at X = J from the boom to the head
    main = sail_mesh((J, case["boom_y"]), (J, case["boom_y"] + P), E,
                     case["main_head"], case["main_sheet_deg"],
                     case["main_depth"], case["main_draft"], awa, nc, ns)
    battens = [(ns * q // 4, case["batten_EI"]) for q in (1, 2, 3)]
    main_model = fsi.build_sail(
        main, Et, case["nu"], case["prestrain"], foot="pinned", leech="free",
        head="pinned", leech_EA=case["main_tape_EA"],
        cable_prestrain=case["main_tape_e0"], battens=battens)
    return {"genoa": (genoa, genoa_model), "main": (main, main_model)}


# ---------------------------------------------------------------- the plan

def solve_sail_plan(sails, onset, u_ref, tol=1e-4, max_outer=12, **kw):
    """Flying shapes of several sails that see one another.

    sails  {name: (points0, model)}. Each outer iteration solves every sail's
    static aeroelastic problem in turn (Gauss-Seidel), with the induced field
    of the others' latest solutions as `extra`, warm-started from its last
    displacement. Stops when no sail's CL changes by more than tol. Returns
    {name: result} with the history of CL in result["outer_CL"].
    """
    names = list(sails)
    res = {n: None for n in names}
    field = {n: None for n in names}
    history = {n: [] for n in names}
    converged = False
    for outer in range(max_outer):
        change = 0.0
        for n in names:
            pts0, model = sails[n]
            others = [field[m] for m in names if m != n and field[m]]

            def extra(p, use, fs=tuple(others)):
                return sum(f(p, use) for f in fs) if fs else 0.0
            u0 = None if res[n] is None else res[n]["u"]
            res[n] = fsi.static_aeroelastic(model, pts0, onset, u_ref,
                                            u0=u0, extra=extra, **kw)
            field[n] = sf.induced_field(res[n]["fluid"])
            cl = res[n]["fluid"]["CL"]
            if history[n]:
                change = max(change, abs(cl - history[n][-1]))
            history[n].append(cl)
        if outer > 0 and change < tol:
            converged = True
            break
    for n in names:
        res[n]["outer_CL"] = history[n]
        res[n]["plan_converged"] = converged
    return res


def plan_coefficients(res, u_ref, rho=sf.RHO_AIR):
    """Lift, drag and centre of effort of the whole plan, on the total area."""
    F = sum(r["fluid"]["F"] for r in res.values())
    heel = sum(r["fluid"]["heel"] for r in res.values())
    area = sum(r["fluid"]["area"] for r in res.values())
    q = 0.5 * rho * u_ref ** 2
    return {"F": F, "area": area, "CL": F[2] / (q * area),
            "CD": F[0] / (q * area), "ce_height": heel / F[2]}


def cloth_stress(model, points, t):
    """Principal tensions [N/m], stresses [MPa], directions and states."""
    N, d, state = sm.membrane_tension(model, points.reshape(-1, 3))
    return {"N": N, "sigma": N / t / 1e6, "direction": d, "state": state}


# ------------------------------------------------------ stress trajectories

def nodal_tension(model, points):
    """Cauchy tension tensor averaged to the nodes, (nchord+1, nspan+1, 3, 3).

    The Cartesian components are averaged with the triangle areas as
    weights, which does not depend on any surface basis; each query point
    then projects the interpolated tensor on its own tangent plane.
    """
    nc, ns = points.shape[0] - 1, points.shape[1] - 1
    x = points.reshape(-1, 3)
    tris = model["tris"]
    N = sm.tension_tensor(model, x)
    xe = x[tris]
    area = 0.5 * np.linalg.norm(np.cross(xe[:, 1] - xe[:, 0],
                                         xe[:, 2] - xe[:, 0]), axis=1)
    acc = np.zeros((len(x), 3, 3))
    wsum = np.zeros(len(x))
    for k in range(3):
        np.add.at(acc, tris[:, k], area[:, None, None] * N)
        np.add.at(wsum, tris[:, k], area)
    return (acc / wsum[:, None, None]).reshape(nc + 1, ns + 1, 3, 3)


def _cell(points, u):
    """Bilinear position, tangent basis (3, 2) and cell weights at u."""
    nc, ns = points.shape[0] - 1, points.shape[1] - 1
    i = min(int(np.floor(u[0])), nc - 1)
    j = min(int(np.floor(u[1])), ns - 1)
    s, t = u[0] - i, u[1] - j
    p00, p10 = points[i, j], points[i + 1, j]
    p01, p11 = points[i, j + 1], points[i + 1, j + 1]
    x = ((1 - s) * (1 - t) * p00 + s * (1 - t) * p10 + (1 - s) * t * p01
         + s * t * p11)
    a1 = (1 - t) * (p10 - p00) + t * (p11 - p01)
    a2 = (1 - s) * (p01 - p00) + s * (p11 - p10)
    w = (i, j, s, t)
    return x, np.stack([a1, a2], axis=1), w


def principal_direction(points, tensor, u, prev=None):
    """Unit tangent of the larger principal tension at grid point u.

    The grid indices (i, j) are the surface coordinates and the bilinear
    surface through the nodes has the tangent basis a = dx/du (3, 2). With
    the bilinearly interpolated nodal tension N, the principal directions
    solve a^T N a p = lambda a^T a p. Returns the step direction in grid
    coordinates scaled to unit length in space, with the sign that continues
    `prev`. Returns None where the two principal tensions coincide.
    """
    _, a, (i, j, s, t) = _cell(points, u)
    N = ((1 - s) * (1 - t) * tensor[i, j] + s * (1 - t) * tensor[i + 1, j]
         + (1 - s) * t * tensor[i, j + 1] + s * t * tensor[i + 1, j + 1])
    T = a.T @ N @ a
    g = a.T @ a
    L = np.linalg.cholesky(g)
    Li = np.linalg.inv(L)
    w, v = np.linalg.eigh(Li @ T @ Li.T)
    if w[1] - w[0] <= 1e-9 * max(abs(w[1]), 1e-300):
        return None
    p = Li.T @ v[:, 1]
    p = p / np.linalg.norm(a @ p)
    if prev is not None and p @ g @ prev < 0.0:
        p = -p
    return p


def stress_trajectories(model, points, spacing, step=None, max_steps=2000):
    """Curves on the sail tangent everywhere to the larger principal tension.

    Evenly spaced tensor lines in the manner of Jobard and Lefer (1997):
    seeds are tried at every cell centre in order of decreasing tension, and
    a seed farther than `spacing` [m] from every line drawn so far starts a
    line, integrated both ways by the midpoint rule with steps of `step` [m]
    in space, cut short at every cell edge so that no step straddles the
    kink of the bilinear surface there, and stopped at the edge of the sail, at a degenerate point, or
    within spacing / 2 of another line. Returns a list of (n, 3) polylines
    on the surface, in the frame of `points`.
    """
    nc, ns = points.shape[0] - 1, points.shape[1] - 1
    step = step or 0.25 * spacing
    tensor = nodal_tension(model, points)
    centres = [(i + 0.5, j + 0.5) for i in range(nc) for j in range(ns)]
    # seeds in order of decreasing mean tension, the trace of N
    mag = [np.trace(tensor[int(c[0]), int(c[1])]) for c in centres]
    order = np.argsort(mag)[::-1]
    lines, cloud = [], np.zeros((0, 3))

    def near(x, dist):
        return len(cloud) and np.min(np.sum((cloud - x) ** 2, axis=1)) \
            < dist * dist

    def inside(u):
        return 0.0 <= u[0] <= nc and 0.0 <= u[1] <= ns

    def to_edge(u, p):
        """Distance in space along p to the edge of the cell u is in."""
        d = np.inf
        for k in (0, 1):
            if p[k] > 0.0:
                d = min(d, (np.floor(u[k]) + 1.0 - u[k]) / p[k])
            elif p[k] < 0.0:
                d = min(d, (u[k] - np.floor(u[k])) / -p[k])
        return d

    def march(u0, p0):
        u, prev, out = np.array(u0, float), p0, []
        for _ in range(max_steps):
            k1 = principal_direction(points, tensor, u, prev)
            if k1 is None:
                break
            # stop at the cell edge, where the bilinear surface has a kink,
            # and step just across it, so that every step lies in one cell
            h = min(step, to_edge(u, k1))
            cross = h < step
            um = u + 0.5 * h * k1
            k2 = principal_direction(points, tensor, um, k1)
            if k2 is None:
                break
            un = u + h * k2
            if cross:
                un = un + 1e-9 * k2
            if not inside(un):
                break
            xn = _cell(points, un)[0]
            if near(xn, 0.5 * spacing):
                break
            out.append(xn)
            u, prev = un, k2
        return out

    for k in order:
        u0 = np.array(centres[k])
        x0 = _cell(points, u0)[0]
        if near(x0, spacing):
            continue
        p0 = principal_direction(points, tensor, u0)
        if p0 is None:
            continue
        fwd = march(u0, p0)
        bwd = march(u0, -p0)
        line = np.array(bwd[::-1] + [x0] + fwd)
        if len(line) < 3:
            continue
        lines.append(line)
        cloud = np.vstack([cloud, line])
    return lines


# ------------------------------------------------------------------ output

def to_boat(p, awa):
    """Solver-frame points (..., 3) to boat axes (X aft, L leeward, y up)."""
    aft, lee = boat_axes(awa)
    return np.stack([p @ aft, p @ lee, p[..., 1]], axis=-1)


def element_values(field, model, pts0, pts, t):
    """Per-triangle values: "displacement" |u| [mm] or "stress" sigma_1 [MPa],
    and for the stress the directions of sigma_1."""
    if field == "displacement":
        u = np.linalg.norm(pts - pts0, axis=2).reshape(-1)
        return 1e3 * u[model["tris"]].mean(axis=1), None
    st = cloth_stress(model, pts, t)
    return st["sigma"][:, 0], st["direction"]


def plot_plan(sails, res, case, field, awa, save=None):
    """One 3D panel per sail, coloured by `field` on the flying shape, with
    the moulded shape as a grey wireframe; for the stress, the stress
    trajectories: black curves tangent everywhere to sigma_1. One colour scale for both
    sails; the stress scale saturates at the 99th percentile, because the
    pinned corners carry stresses that grow with mesh refinement."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 10})
    vals = {n: element_values(field, m, p0, res[n]["points"],
                              case["t_cloth"])
            for n, (p0, m) in sails.items()}
    allv = np.concatenate([v for v, _ in vals.values()])
    vmax = allv.max() if field == "displacement" else np.percentile(allv, 99)
    norm = colors.Normalize(0.0, vmax)
    cmap = plt.get_cmap("Blues" if field == "displacement" else "Oranges")
    fig = plt.figure(figsize=(9.0, 6.4))
    for k, (n, (pts0, model)) in enumerate(sails.items()):
        ax = fig.add_subplot(1, len(sails), k + 1, projection="3d",
                             computed_zorder=False)
        P = to_boat(res[n]["points"], awa)
        P0 = to_boat(pts0, awa)
        tri = P.reshape(-1, 3)[model["tris"]]
        val, direction = vals[n]
        col = Poly3DCollection(tri, cmap=cmap, norm=norm, edgecolor="none")
        col.set_array(np.minimum(val, vmax))
        ax.add_collection3d(col)
        for j in range(0, P0.shape[1], 2):
            ax.plot(*P0[:, j].T, color="0.6", lw=0.5)
        for i in (0, -1):
            ax.plot(*P0[i, :].T, color="0.6", lw=0.5)
        if direction is not None:
            spacing = 0.04 * np.ptp(P[..., 2])
            lines = [to_boat(c, awa)
                     for c in stress_trajectories(model, res[n]["points"],
                                                  spacing)]
            ax.add_collection3d(Line3DCollection(lines, colors="k",
                                                 linewidths=0.6))
        # each panel on its own sail, at equal scale in all three axes
        plo, phi = P.reshape(-1, 3).min(axis=0), P.reshape(-1, 3).max(axis=0)
        mid = 0.5 * (plo + phi)
        ax.set_xlim(plo[0] - 0.2, phi[0] + 0.2)
        ax.set_ylim(mid[1] - 1.0, mid[1] + 1.0)
        ax.set_zlim(0.0, phi[2])
        ax.set_box_aspect((phi[0] - plo[0] + 0.4, 2.0, phi[2]))
        ax.set_yticks([np.ceil(mid[1] - 1.0), np.floor(mid[1] + 1.0)])
        ax.set_xlabel(r"$X$ [m]")
        ax.set_ylabel(r"$L$ [m]")
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$y$ [m]", rotation=90)
        ax.view_init(elev=18, azim=-130)
        ax.set_title(f"({'ab'[k]}) {n}", loc="left")
    sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    label = (r"$|\boldsymbol{u}|$ [mm]" if field == "displacement"
             else r"$\sigma_1$ [MPa]")
    fig.colorbar(sm_, ax=fig.axes, shrink=0.5, pad=0.04,
                 extend="neither" if field == "displacement" else "max",
                 label=label)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sections(sails, res, awa, save=None):
    """Moulded (grey) and flying (black) sections at the foot and at a
    quarter, half and three quarters of the height, in boat axes."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, len(sails), figsize=(10.0, 4.0))
    for ax, (n, (pts0, model)) in zip(np.atleast_1d(axes), sails.items()):
        pts = to_boat(res[n]["points"], awa)
        p0 = to_boat(pts0, awa)
        ns = pts.shape[1] - 1
        styles = ("-", "--", "-.", ":")
        for k, q in enumerate((0, 1, 2, 3)):
            j = ns * q // 4
            ax.plot(p0[:, j, 0], p0[:, j, 1], color="0.6", ls=styles[k],
                    lw=0.8)
            ax.plot(pts[:, j, 0], pts[:, j, 1], color="k", ls=styles[k],
                    lw=1.2, label=rf"$y = {pts[0, j, 2]:.1f}$ m")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(r"$X$ [m]")
        ax.set_ylabel(r"$L$ [m]")
        ax.set_title(f"({'ab'[list(sails).index(n)]}) {n}", loc="left")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200)
        plt.close(fig)
    return fig


def report(sails, res, case, u_ref, awa, wind):
    """Printed summary of the solved plan."""
    print(f"apparent wind at {case['y_ref']:.1f} m above the sheer: "
          f"{u_ref:.2f} m/s at {np.degrees(awa):.1f} deg; at the masthead "
          f"{wind(case['boom_y'] + J80['P'] + case['freeboard'])[0]:.2f} m/s"
          f" at {np.degrees(wind(case['boom_y'] + J80['P'] + case['freeboard'])[1]):.1f}"
          f" deg; at the boom {wind(case['boom_y'] + case['freeboard'])[0]:.2f}"
          f" m/s at {np.degrees(wind(case['boom_y'] + case['freeboard'])[1]):.1f} deg")
    for n, (pts0, model) in sails.items():
        r = res[n]
        fl = r["fluid"]
        st = cloth_stress(model, r["points"], case["t_cloth"])
        u = np.linalg.norm(r["points"] - pts0, axis=2)
        rows = fsi.flying_shape_report(r)
        ns = len(rows) - 1
        states = np.bincount(st["state"], minlength=3) / len(st["state"])
        print(f"\n{n}: area {fl['area']:.2f} m^2, CL {fl['CL']:.3f}, "
              f"CD {fl['CD']:.3f}, force {np.linalg.norm(fl['F']):.0f} N, "
              f"CE {fl['ce_height']:.2f} m above the sheer")
        print(f"  outer CL history {np.round(r['outer_CL'], 4)}")
        print(f"  displacement max {1e3 * u.max():.0f} mm, mean "
              f"{1e3 * u.mean():.0f} mm")
        print(f"  sigma_1 max {st['sigma'][:, 0].max():.1f} MPa "
              f"({st['N'][:, 0].max():.0f} N/m), median "
              f"{np.median(st['sigma'][:, 0]):.1f} MPa; cloth "
              f"{states[0]:.0%} taut, {states[1]:.0%} wrinkled, "
              f"{states[2]:.0%} slack")
        for c in model["cables"]:
            N = sm.cable_forces(c, r["points"].reshape(-1, 3), False)[2]
            print(f"  edge tape tension {N.min():.0f} to {N.max():.0f} N")
        print("        y   depth   draft   incidence to x [deg]")
        for j in range(0, ns + 1, ns // 4):
            row = rows[j]
            print(f"  {row['y']:7.2f} {row['depth']:7.3f} {row['draft']:7.3f}"
                  f" {row['incidence_deg']:9.2f}")


def main():
    case = dict(J80_UPWIND)
    onset, u_ref, awa, wind = apparent_onset(case)
    sails = j80_sails(case)
    t0 = time.time()
    res = solve_sail_plan(sails, onset, u_ref)
    print(f"solved in {time.time() - t0:.0f} s, converged "
          f"{res['main']['plan_converged']}")
    report(sails, res, case, u_ref, awa, wind)
    tot = plan_coefficients(res, u_ref)
    print(f"\nsail plan: CL {tot['CL']:.3f}, CD {tot['CD']:.3f}, "
          f"CE {tot['ce_height']:.2f} m above the sheer, force "
          f"{np.linalg.norm(tot['F']):.0f} N")
    if "--no-figures" not in sys.argv:
        import os
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "docs", "figures")
        os.makedirs(out, exist_ok=True)
        plot_plan(sails, res, case, "displacement", awa,
                  os.path.join(out, "j80_displacement.png"))
        plot_plan(sails, res, case, "stress", awa,
                  os.path.join(out, "j80_stress.png"))
        plot_sections(sails, res, awa,
                      os.path.join(out, "j80_sections.png"))
        print(f"figures written to {out}")


if __name__ == "__main__":
    main()
