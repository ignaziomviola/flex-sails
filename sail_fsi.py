"""Static aeroelastic flying shape of a flexible sail.

A sail is a membrane whose shape is set by the load it carries, and the load
by the shape. This module couples the membrane of `sail_membrane.py` with the
lifting surface of `sail_fluid.py` (the vendored free-wake method) by a
partitioned fixed point on the structural displacement:

    u_k  ->  fluid on X + u_k  ->  nodal forces f(u_k)
         ->  structural equilibrium under f(u_k), warm-started from u_k: u~
         ->  residual r_k = u~ - u_k  ->  u_{k+1} from the accelerator

with the update from IQN-ILS (the default), Aitken's Delta^2 process, or a
constant relaxation factor. The structural and fluid meshes are the SAME grid
of nodes, so the transfer is the bilinear lumping of `sail_fluid.nodal_forces` and the fluid sees the displaced nodes
directly; nothing is interpolated, and force and moment pass exactly.

Run as a program it prompts for a sail and prints the flying shape.
"""

import numpy as np

import panel_wing as pw
import sail_fluid as sf
import sail_membrane as sm

TOL_COUPLING = 1.0e-4      # ||u~ - u|| / ||u~|| at convergence
MAX_COUPLING = 40
OMEGA_START = 0.5
OMEGA_MAX = 1.0e3          # blow-up guard on the Aitken factor, not a limiter
IQN_REUSE = 8              # columns kept in the IQN-ILS least-squares system


# ------------------------------------------------------------- geometry

def sail_planform(luff=10.0, foot=3.5, head=1.0, nchord=8, nspan=12,
                  camber=0.0, draft=0.5, alpha_deg=10.0, cosine=False):
    """Mesh of a quadrilateral sail, luff on the mast (x = 0) and foot at y = 0.

    The chord varies linearly from `foot` to `head`; the head chord must be
    finite because the lifting surface needs a non-degenerate tip panel.
    `camber` is the depth of the stress-free (moulded) shape as a fraction of
    the local chord, with its maximum at `draft` of the chord; zero gives a
    flat sail. The sail is then sheeted to an incidence `alpha_deg` about the
    mast, trailing edge to windward (-z), so that it lifts towards +z.
    Returns (nchord+1, nspan+1, 3).
    """
    if head <= 0.0:
        raise ValueError("the head chord must be positive")
    if cosine:
        xi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nchord + 1)))
    else:
        xi = np.linspace(0.0, 1.0, nchord + 1)
    eta = np.linspace(0.0, 1.0, nspan + 1)
    chord = foot + (head - foot) * eta
    pts = np.zeros((nchord + 1, nspan + 1, 3))
    pts[..., 0] = xi[:, None] * chord[None, :]
    pts[..., 1] = luff * eta[None, :]
    if camber:
        pts[..., 2] = camber * chord[None, :] * camber_line(xi, draft)[:, None]
    return pw.pitch_mesh(pts, np.radians(alpha_deg))


def camber_line(xi, draft=0.5):
    """Unit-depth camber line with its maximum at `draft`: two parabolas."""
    xi = np.asarray(xi, dtype=float)
    fwd = 1.0 - ((draft - xi) / draft) ** 2
    aft = 1.0 - ((xi - draft) / (1.0 - draft)) ** 2
    return np.where(xi <= draft, fwd, aft)


def build_sail(points, Et, nu=0.3, prestrain=0.0, foot="pinned",
               leech="free", head="free", leech_EA=0.0, head_EA=0.0, foot_EA=0.0,
               cable_prestrain=0.0, deck=False, wrinkling=True, battens=()):
    """Structural model of the sail on the fluid mesh `points`.

    The luff (i = 0) is always pinned to a rigid mast or forestay.
    foot   "pinned" pins it to a rigid boom; "clew" pins the clew only;
           "free" leaves it free. With `deck` an unpinned foot slides on the
           deck plane, so that the deck image stays exact.
    leech  "free" or "pinned"; a sail pinned on luff and leech is the
           membrane wing of two-dimensional sail theory.
    head   "free" or "pinned"; a pinned head is a headboard fixed at the
           masthead, which with a short head chord is a triangular mainsail
           whose leech runs between two fixed points, the clew and the head.
    Free edges may carry a tension-only cable of axial stiffness EA [N] with
    pre-strain `cable_prestrain`, which is how a sheet or leech-line tension
    enters. `battens` is a list of (j, EI): a full-length batten of bending
    stiffness EI [N m^2] along the chordwise grid line j, stress free in the
    moulded shape. The cloth itself has no bending stiffness.
    """
    nchord, nspan = points.shape[0] - 1, points.shape[1] - 1
    X = points.reshape(-1, 3)
    nid = lambda i, j: sm.node_id(i, j, nspan)
    fixed = np.zeros((len(X), 3), dtype=bool)
    luff = [nid(0, j) for j in range(nspan + 1)]
    leech_n = [nid(nchord, j) for j in range(nspan + 1)]
    foot_n = [nid(i, 0) for i in range(nchord + 1)]
    head_n = [nid(i, nspan) for i in range(nchord + 1)]
    fixed[luff] = True
    if foot not in ("pinned", "clew", "free"):
        raise ValueError(f"unknown foot '{foot}'; use 'pinned', 'clew' or "
                         f"'free'")
    if leech not in ("free", "pinned"):
        raise ValueError(f"unknown leech '{leech}'; use 'free' or 'pinned'")
    if foot == "pinned":
        fixed[foot_n] = True
    else:
        if foot == "clew":
            fixed[nid(nchord, 0)] = True
        if deck:
            fixed[foot_n, 1] = True
    if leech == "pinned":
        fixed[leech_n] = True
    if head not in ("free", "pinned"):
        raise ValueError(f"unknown head '{head}'; use 'free' or 'pinned'")
    if head == "pinned":
        fixed[head_n] = True

    cables = []
    for nodes, EA, free_edge in ((foot_n, foot_EA, foot != "pinned"),
                                 (leech_n, leech_EA, leech == "free"),
                                 (head_n, head_EA, head == "free")):
        if EA > 0.0 and free_edge:
            cables.append({"pairs": sm.edge_chain(nodes), "EA": EA,
                           "prestrain": cable_prestrain})
    hinges = batten_hinges(points, battens) if battens else None
    model = sm.build_model(X, sm.grid_triangles(nchord, nspan), Et, nu,
                           prestrain, wrinkling, cables, fixed, hinges)
    model["shape"] = points.shape
    model["battens"] = list(battens)
    return model


def batten_hinges(points, battens):
    """Hinges of full-length battens along chordwise grid lines.

    A batten of bending stiffness EI along line j has one hinge at each
    interior station i: a = (i, j), c and d its chordwise neighbours, and the
    axis b = (i, j +/- 1) its spanwise neighbour. The hinge angle is then the
    turn kappa h of the batten at a, h the mean of the two adjacent panel
    chords, so k = EI / h stores EI kappa^2 h / 2 per station: the
    Euler-Bernoulli energy. Returns {"nodes", "k"} for build_model.
    """
    nchord, nspan = points.shape[0] - 1, points.shape[1] - 1
    nid = lambda i, j: sm.node_id(i, j, nspan)
    nodes, k = [], []
    for j, EI in battens:
        if not 0 <= j <= nspan:
            raise ValueError(f"batten line {j} outside 0..{nspan}")
        for i in range(1, nchord):
            a, c, d = nid(i, j), nid(i - 1, j), nid(i + 1, j)
            # both virtual triangles counter-clockwise, as in the mesh
            if j < nspan:
                nodes.append((a, nid(i, j + 1), c, d))
            else:
                nodes.append((a, nid(i, j - 1), d, c))
            h = 0.5 * (np.linalg.norm(points[i + 1, j] - points[i, j])
                       + np.linalg.norm(points[i, j] - points[i - 1, j]))
            k.append(EI / h)
    return {"nodes": np.array(nodes, dtype=int), "k": np.array(k)}


# ------------------------------------------------------------- coupling

class FixedPointAccelerator:
    """Aitken relaxation or IQN-ILS over a sequence of coupling residuals.

    Both take the residual r = u~ - u of the fixed point u -> u~ and return
    the next iterate; this is the accelerator of bem-fem-fsi's fsi_driver.
    Aitken carries one scalar, IQN-ILS (Degroote's least-squares form) a small
    system built from the differences of successive residuals and outputs.
    On the battened mainsail of verify_sail.py they take four to five and
    five to six iterations, and constant relaxation by one half 14 to 15
    (case C4).
    """

    def __init__(self, method="iqn", omega=OMEGA_START, reuse=IQN_REUSE):
        if method not in ("aitken", "iqn", "constant"):
            raise ValueError(f"unknown accelerator '{method}'; use 'aitken', "
                             f"'iqn' or 'constant'")
        self.method, self.omega0, self.reuse = method, float(omega), int(reuse)
        self.omega = self.omega0
        self.r_prev = self.u_prev = None
        self.cols_v, self.cols_w = [], []

    def step(self, u, u_tilde):
        """One accelerated iterate from flat arrays u and u~."""
        r = u_tilde - u
        if self.method == "constant":
            return u + self.omega0 * r
        if self.method == "aitken":
            if self.r_prev is not None:
                d_r = r - self.r_prev
                den = float(d_r @ d_r)
                if den > 0.0:
                    self.omega = -self.omega * float(self.r_prev @ d_r) / den
                    # a guard against a degenerate denominator, not a limiter
                    self.omega = float(np.clip(self.omega, -OMEGA_MAX,
                                               OMEGA_MAX))
                    if abs(self.omega) < 1e-6 or not np.isfinite(self.omega):
                        self.omega = self.omega0
            self.r_prev = r
            return u + self.omega * r
        if self.r_prev is not None:
            self.cols_v.append(r - self.r_prev)
            self.cols_w.append(u_tilde - self.u_prev)
            self.cols_v = self.cols_v[-self.reuse:]
            self.cols_w = self.cols_w[-self.reuse:]
        self.r_prev, self.u_prev = r, u_tilde
        if not self.cols_v:
            return u + self.omega0 * r
        v_mat = np.stack(self.cols_v, axis=1)
        w_mat = np.stack(self.cols_w, axis=1)
        coeff, *_ = np.linalg.lstsq(v_mat, -r, rcond=None)
        return u + w_mat @ coeff + r


def static_aeroelastic(model, points0, onset, u_ref, rho=sf.RHO_AIR,
                       wake="frozen", deck=False, load="pressure",
                       tol=TOL_COUPLING, max_iter=MAX_COUPLING,
                       accel="iqn", omega=OMEGA_START, nsteps_first=1,
                       u0=None):
    """Flying shape: the displacement at which the sail carries its own load.

    Returns a dict with the displacement u (nn, 3), the deformed mesh, the
    fluid solution on it, the structural solution, and the coupling history
    (residual, omega, structural iterations per coupling iteration).
    """
    shape = points0.shape
    nn = shape[0] * shape[1]
    u = np.zeros((nn, 3)) if u0 is None else np.array(u0, dtype=float)
    acc = FixedPointAccelerator(accel, omega)
    hist = {"residual": [], "omega": [], "newton": [], "CL": []}
    converged = False
    fluid = struct = last = None
    for k in range(max_iter):
        pts = points0 + u.reshape(shape)
        fluid = sf.solve_fluid(pts, onset, u_ref, rho, wake, deck)
        f = sf.nodal_forces(fluid, load).reshape(nn, 3)
        first = k == 0 and u0 is None
        struct = sm.solve_static(model, f_ext=f, u0=u,
                                 nsteps=nsteps_first if first else 1)
        # an accelerated iterate need not be a configuration the cloth can
        # reach smoothly; the last equilibrium and the moulded shape are
        if not struct["converged"] and last is not None:
            struct = sm.solve_static(model, f_ext=f, u0=last)
        if not struct["converged"]:
            struct = sm.solve_static(model, f_ext=f)
        if not struct["converged"]:
            raise RuntimeError(f"structural solve failed at coupling "
                               f"iteration {k}")
        last = struct["u"]
        r = (struct["u"] - u).reshape(-1)
        res = np.linalg.norm(r) / max(np.linalg.norm(struct["u"]), 1e-300)
        hist["residual"].append(res)
        hist["omega"].append(acc.omega)
        hist["newton"].append(struct["iterations"])
        hist["CL"].append(fluid["CL"])
        if res < tol:
            converged = True
            break
        u = acc.step(u.reshape(-1), struct["u"].reshape(-1)).reshape(nn, 3)

    u_final = struct["u"]
    return {"u": u_final, "points0": points0,
            "points": points0 + u_final.reshape(shape),
            "points_fluid": points0 + u.reshape(shape),
            "fluid": fluid, "struct": struct, "history": hist,
            "converged": converged, "iterations": len(hist["residual"])}


# ----------------------------------------------------------- diagnostics

def section_shape(points, j):
    """Chord, maximum depth ratio, draft position and incidence of section j.

    Measured in the section's own chord line from luff to leech, projected on
    the plane y = const; the incidence is the chord-line angle to the x axis,
    positive with the leech to windward (-z). Its fall with height is the
    twist of the sail.
    """
    sec = points[:, j]
    le, te = sec[0, [0, 2]], sec[-1, [0, 2]]
    d = te - le
    c = np.linalg.norm(d)
    t = d / c
    nrm = np.array([-t[1], t[0]])          # towards +z for a positive chord
    rel = sec[:, [0, 2]] - le
    depth = rel @ nrm
    along = rel @ t
    k = int(np.argmax(depth))
    d_max, x_max = depth[k], along[k]
    if 0 < k < len(depth) - 1:
        # vertex of the parabola through the three nodes about the maximum
        cf = np.polyfit(along[k - 1:k + 2], depth[k - 1:k + 2], 2)
        if cf[0] < 0.0:
            x_max = -cf[1] / (2.0 * cf[0])
            d_max = np.polyval(cf, x_max)
    return {"chord": c, "depth": d_max / c, "draft": x_max / c,
            "incidence_deg": np.degrees(np.arctan2(-d[1], d[0]))}


def flying_shape_report(result):
    """Per-section chord, depth, draft and incidence of the flying shape."""
    pts = result["points"]
    rows = []
    for j in range(pts.shape[1]):
        row = section_shape(pts, j)
        row["y"] = pts[0, j, 1]
        rows.append(row)
    return rows


def sectional_cl(fluid):
    """Lift coefficient of each spanwise strip, (nspan,)."""
    f, a = fluid["force"], fluid["areas"]
    return f[..., 2].sum(axis=0) / (fluid["q"] * a.sum(axis=0))


def cloth_state(result):
    """Fractions of the triangles that are taut, wrinkled and slack."""
    st = result["struct"]["state"]
    return np.bincount(st, minlength=3) / len(st)


def transfer_report(fluid, load="pressure"):
    """Force and moment of the panel loads against those of the nodal loads.

    With load = "pressure" the panel side is the normal component, which is
    what the transfer is asked to conserve.
    """
    f = fluid["force"]
    if load == "pressure":
        n = fluid["normals"]
        f = np.einsum("ijk,ijk->ij", f, n)[..., None] * n
    nchord, nspan = f.shape[:2]
    lat = fluid["lat"]
    j0 = lat["nspan"] - nspan if fluid["deck"] else 0
    sel = (np.arange(nchord)[:, None] * lat["nspan"]
           + j0 + np.arange(nspan)[None, :])
    mid = 0.5 * (lat["bound_a"] + lat["bound_b"])[sel]
    lat_pts = lat["points"][:, j0:j0 + nspan + 1]
    fn = sf.nodal_forces(fluid, load)
    F_p, F_n = f.reshape(-1, 3).sum(0), fn.reshape(-1, 3).sum(0)
    M_p = np.cross(mid.reshape(-1, 3), f.reshape(-1, 3)).sum(0)
    M_n = np.cross(lat_pts.reshape(-1, 3), fn.reshape(-1, 3)).sum(0)
    scale_f = max(np.linalg.norm(F_p), 1e-300)
    scale_m = max(np.linalg.norm(M_p), 1e-300)
    return {"force_defect": np.linalg.norm(F_n - F_p) / scale_f,
            "moment_defect": np.linalg.norm(M_n - M_p) / scale_m}


# ------------------------------------------------------------------ output

def plot_flying_shape(result, save=None):
    import matplotlib.pyplot as plt
    pts, p0 = result["points"], result["points0"]
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for P, col in ((p0, "0.6"), (pts, "k")):
        for i in range(P.shape[0]):
            ax.plot(P[i, :, 0], P[i, :, 1], P[i, :, 2], color=col, lw=0.6)
        for j in range(P.shape[1]):
            ax.plot(P[:, j, 0], P[:, j, 1], P[:, j, 2], color=col, lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Moulded (grey) and flying (black) shape")
    ax.set_box_aspect((np.ptp(pts[..., 0]), np.ptp(pts[..., 1]),
                       max(np.ptp(pts[..., 2]), 0.1 * np.ptp(pts[..., 1]))))

    ax = fig.add_subplot(1, 3, 2)
    for j in range(0, pts.shape[1], max(1, pts.shape[1] // 4)):
        ax.plot(pts[:, j, 0], pts[:, j, 2], "o-", ms=3,
                label=f"y = {pts[0, j, 1]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Flying sections")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    ax.semilogy(result["history"]["residual"], "o-")
    ax.set_xlabel("coupling iteration")
    ax.set_ylabel(r"$\|\tilde u - u\| / \|\tilde u\|$")
    ax.set_title("Coupling convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    else:
        plt.show()


def prompt(text, default):
    reply = input(f"{text} [{default}]: ").strip()
    return reply if reply else default


def main():
    luff = float(prompt("Luff length [m]", "10.0"))
    foot = float(prompt("Foot chord [m]", "3.5"))
    head = float(prompt("Headboard chord [m]", "0.5"))
    camber = float(prompt("Moulded depth ratio", "0.08"))
    alpha = float(prompt("Incidence at the foot [deg]", "10.0"))
    u_ref = float(prompt("Apparent wind speed at the head [m/s]", "6.0"))
    twist = float(prompt("Apparent wind twist, foot to head [deg]", "0.0"))
    Et = float(prompt("Cloth membrane stiffness E t [N/m]", "5.0e5"))
    e0 = float(prompt("Cloth pre-strain", "0.002"))
    leech = float(prompt("Leech line EA [N]", "5.0e4"))
    leech_e0 = float(prompt("Leech line pre-strain (sheet tension)", "0.01"))
    EI = float(prompt("EI of three full battens [N m^2], 0 for none", "50.0"))
    wake = prompt("Wake (frozen/free)", "frozen")
    deck = prompt("Foot sealed on the deck (y/n)", "n").lower().startswith("y")

    pts0 = sail_planform(luff, foot, head, camber=camber, alpha_deg=alpha)
    ns = pts0.shape[1] - 1
    battens = [(ns * q // 4, EI) for q in (1, 2, 3)] if EI > 0.0 else []
    model = build_sail(pts0, Et, prestrain=e0, foot="pinned", head="pinned",
                       leech_EA=leech, cable_prestrain=leech_e0, deck=deck,
                       battens=battens)
    onset = sf.sail_onset(u_ref, luff, twist_deg=twist)
    rigid = sf.solve_fluid(pts0, onset, u_ref, wake=wake, deck=deck)
    res = static_aeroelastic(model, pts0, onset, u_ref, wake=wake, deck=deck)
    fl = res["fluid"]
    print(f"\nCoupling: {res['iterations']} iterations, converged "
          f"{res['converged']}, final residual "
          f"{res['history']['residual'][-1]:.2e}")
    print(f"Moulded shape held rigid: CL = {rigid['CL']:.4f}, "
          f"CD = {rigid['CD']:.4f}")
    print(f"Flying shape:             CL = {fl['CL']:.4f}, "
          f"CD = {fl['CD']:.4f}, centre of effort at "
          f"{fl['ce_height']:.3f} m")
    print(f"Aerodynamic force [N]: {np.array2string(fl['F'], precision=2)}")
    taut, wr, sl = cloth_state(res)
    print(f"Cloth: {taut:.0%} taut, {wr:.0%} wrinkled, {sl:.0%} slack")
    print("\n      y   chord   depth   draft   incidence [deg]")
    for row in flying_shape_report(res):
        print(f"{row['y']:7.3f} {row['chord']:7.3f} {row['depth']:7.4f} "
              f"{row['draft']:7.3f} {row['incidence_deg']:9.3f}")
    plot_flying_shape(res)


if __name__ == "__main__":
    main()
