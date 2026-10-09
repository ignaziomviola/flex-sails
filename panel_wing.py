"""Thin-lifting-surface panel code with free-wake relaxation.

Generalises vortex_lattice_wing.py to a generic wing geometry and a
non-uniform onset velocity:

- The wing is supplied as a structured mesh of its mean camber surface in a
  NumPy .npz file with key 'points', array shape (nchord+1, nspan+1, 3).
  Index i runs from the leading edge (i=0) to the trailing edge (i=nchord),
  index j runs across the span. The onset flow is nominally along +x, z up.
  Camber, twist, taper, sweep and dihedral are all encoded in the mesh.
- The onset velocity is supplied as a CSV shear profile with two columns
  z, U (a header line is allowed). The onset velocity at a point is
  (U(z), 0, 0), linearly interpolated and clamped to the table end values.

Each panel carries a horseshoe vortex: a bound segment at the panel quarter
chord, rigid legs that follow the curved surface side edges to the trailing
edge, and force-free filaments beyond. At the wing tips each chordwise
station releases its own free filament from the bound-vortex end at the side
edge, so the tip vortex develops along the chord. All free filaments are
relaxed iteratively until aligned with the local streamlines.

Forces are computed with the vector Kutta-Joukowski theorem using the local
(onset + induced) velocity at the bound segments, and normalised with a
user-supplied reference speed and the total lattice area.
"""

import numpy as np
import matplotlib.pyplot as plt

RHO = 1.0
NWAKE = 50                    # free-wake segments per interior trailing filament
WAKE_LENGTH_SPANS = 2.0       # free-wake length in span extents
TIP_FINE_STEP_CHORDS = 0.1    # step of tip filaments near the wing [mean chords]
TIP_FINE_LENGTH_CHORDS = 1.5  # finely-resolved tip length beyond the TE [mean chords]
FAR_FIELD = 1.0e4             # length of the closing straight segment
MAX_ITER = 30                 # wake relaxation iterations
OMEGA = 0.7                   # under-relaxation factor for node positions
TOL = 1.0e-3                  # convergence: max near-field displacement [mean chords]
CORE_WING = 1.0e-6            # vortex core radius for influence coefficients
RC_WAKE_FRACTION = 0.2        # wake core radius as fraction of spanwise spacing


# ---------------------------------------------------------------- kernels

def segment_velocities(points, p1, p2, core):
    """Unit-strength velocity induced by S straight vortex segments at M points.

    Returns an (M, S, 3) array. `core` is a per-segment absolute core radius
    (van Garrel regularisation) so the velocity vanishes smoothly on the axis.
    """
    r1 = points[:, None, :] - p1[None, :, :]
    r2 = points[:, None, :] - p2[None, :, :]
    r0 = p2 - p1
    cr = np.cross(r1, r2)
    cr2 = np.sum(cr * cr, axis=2)
    r0n2 = np.sum(r0 * r0, axis=1)[None, :]
    r1n = np.sqrt(np.sum(r1 * r1, axis=2))
    r2n = np.sqrt(np.sum(r2 * r2, axis=2))
    dot1 = np.sum(r0[None, :, :] * r1, axis=2)
    dot2 = np.sum(r0[None, :, :] * r2, axis=2)
    denom = 4.0 * np.pi * (cr2 + (core[None, :] ** 2) * r0n2)
    k = (dot1 / np.maximum(r1n, 1e-12) - dot2 / np.maximum(r2n, 1e-12))
    k /= np.maximum(denom, 1e-30)
    return k[:, :, None] * cr


def induced_velocity(points, p1, p2, strengths, core, chunk=300):
    """Total induced velocity at M points, chunked to bound memory use."""
    v = np.zeros_like(points)
    for s in range(0, len(points), chunk):
        block = segment_velocities(points[s:s + chunk], p1, p2, core)
        v[s:s + chunk] = np.einsum("msk,s->mk", block, strengths)
    return v


# ------------------------------------------------------------------ inputs

def load_mesh(path):
    data = np.load(path)
    if "mesh_format" in data and str(data["mesh_format"]).startswith("thick-wrap"):
        raise ValueError(
            f"{path} is a thick-wing wrap mesh for the source-doublet code; see "
            f"https://github.com/ignaziomviola/free-wake-thick-panel-method")
    if "points" not in data:
        raise ValueError(f"{path} must contain an array under key 'points'")
    points = np.asarray(data["points"], dtype=float)
    if points.ndim != 3 or points.shape[2] != 3 or min(points.shape[:2]) < 2:
        raise ValueError(
            f"'points' must have shape (nchord+1, nspan+1, 3); got {points.shape}")
    return points


def load_velocity_profile(path):
    data = np.genfromtxt(path, delimiter=",", comments="#")
    data = np.atleast_2d(data)
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] < 1 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain rows of 'z, U' values")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def make_onset(z_table, u_table):
    """Onset-velocity function: (M,3) points -> (M,3) velocities."""
    def onset(points):
        pts = np.atleast_2d(points)
        v = np.zeros((len(pts), 3))
        v[:, 0] = np.interp(pts[:, 2], z_table, u_table)
        return v
    return onset


def pitch_mesh(points, alpha_rad):
    """Rotate the mesh nose-up by alpha about the y axis (through the origin)."""
    c, s = np.cos(alpha_rad), np.sin(alpha_rad)
    out = points.copy()
    out[..., 0] = c * points[..., 0] + s * points[..., 2]
    out[..., 2] = -s * points[..., 0] + c * points[..., 2]
    return out


# ---------------------------------------------------------------- lattice

def build_lattice(points):
    """Horseshoe lattice on the mesh: collocation points, normals, rigid segments.

    Bound segments sit at each panel quarter chord. Rigid legs follow the
    surface side edges to the trailing edge along interior spanwise edges
    only; the tip edges shed free filaments instead (see init_filaments).
    """
    nchord = points.shape[0] - 1
    nspan = points.shape[1] - 1
    n = nchord * nspan

    colloc = np.zeros((n, 3))
    normals = np.zeros((n, 3))
    areas = np.zeros(n)
    bound_a = np.zeros((n, 3))
    bound_b = np.zeros((n, 3))
    seg_p1, seg_p2, seg_owner = [], [], []

    # Chordwise tangents of the smooth camber surface at the mesh nodes of
    # each mid-strip curve; interpolated to the 3/4-chord collocation points
    # below. Using the local surface slope there (rather than the flat panel
    # normal) preserves the accuracy of the 1/4-3/4 rule on cambered wings.
    mids = 0.5 * (points[:, :-1, :] + points[:, 1:, :])  # (nchord+1, nspan, 3)
    tangents = np.empty_like(mids)
    tangents[0] = mids[1] - mids[0]
    tangents[-1] = mids[-1] - mids[-2]
    if nchord > 1:
        tangents[1:-1] = mids[2:] - mids[:-2]
    tangents /= np.linalg.norm(tangents, axis=2, keepdims=True)

    for i in range(nchord):
        for j in range(nspan):
            h = i * nspan + j
            p00, p10 = points[i, j], points[i + 1, j]
            p01, p11 = points[i, j + 1], points[i + 1, j + 1]
            a = p00 + 0.25 * (p10 - p00)
            b = p01 + 0.25 * (p11 - p01)
            bound_a[h], bound_b[h] = a, b
            tq_l = p00 + 0.75 * (p10 - p00)
            tq_r = p01 + 0.75 * (p11 - p01)
            colloc[h] = 0.5 * (tq_l + tq_r)
            areas[h] = 0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p10))
            t_chord = 0.25 * tangents[i, j] + 0.75 * tangents[i + 1, j]
            nvec = np.cross(t_chord, tq_r - tq_l)
            nvec = nvec / max(np.linalg.norm(nvec), 1e-14)
            normals[h] = nvec if nvec[2] >= 0.0 else -nvec

            seg_p1.append(a)
            seg_p2.append(b)
            seg_owner.append(h)
            if j > 0:  # left leg rigid: TE -> ... -> a along side edge j
                path = [points[k, j] for k in range(nchord, i, -1)] + [a]
                for p, q in zip(path[:-1], path[1:]):
                    seg_p1.append(p)
                    seg_p2.append(q)
                    seg_owner.append(h)
            if j < nspan - 1:  # right leg rigid: b -> ... -> TE along edge j+1
                path = [b] + [points[k, j + 1] for k in range(i + 1, nchord + 1)]
                for p, q in zip(path[:-1], path[1:]):
                    seg_p1.append(p)
                    seg_p2.append(q)
                    seg_owner.append(h)

    return {
        "nchord": nchord, "nspan": nspan, "points": points,
        "colloc": colloc, "normals": normals, "areas": areas,
        "bound_a": bound_a, "bound_b": bound_b,
        "seg_p1": np.array(seg_p1), "seg_p2": np.array(seg_p2),
        "seg_owner": np.array(seg_owner),
    }


# -------------------------------------------------------------- filaments

def initial_nodes(start, waypoints, steps, onset):
    """Initial filament: along the given surface path, then along the onset flow."""
    nodes = [start.copy()]
    p = start.copy()
    path = [w.copy() for w in waypoints]
    for length in steps:
        remaining = length
        while path and remaining > 1e-14:
            d = path[0] - p
            dist = np.linalg.norm(d)
            if dist > remaining:
                p = p + d * (remaining / dist)
                remaining = 0.0
            else:
                p = path.pop(0)
                remaining -= dist
        if remaining > 1e-14:
            v = onset(p[None, :])[0]
            p = p + v / max(np.linalg.norm(v), 1e-12) * remaining
        nodes.append(p.copy())
    return np.array(nodes)


def init_filaments(lat, onset, chord_mean, span_extent):
    """Free filaments: interior wake shed at the TE, plus per-station tip legs."""
    points = lat["points"]
    nchord, nspan = lat["nchord"], lat["nspan"]
    filaments = []
    step_w = WAKE_LENGTH_SPANS * span_extent / NWAKE

    for j in range(1, nspan):
        steps = np.full(NWAKE, step_w)
        filaments.append({
            "nodes": initial_nodes(points[nchord, j], [], steps, onset),
            "steps": steps,
            "cols_plus": [i * nspan + j - 1 for i in range(nchord)],
            "cols_minus": [i * nspan + j for i in range(nchord)],
            "is_tip": False,
        })

    step_fine = TIP_FINE_STEP_CHORDS * chord_mean
    for i in range(nchord):
        for j_edge, j_col, sign in ((nspan, nspan - 1, +1), (0, 0, -1)):
            h = i * nspan + j_col
            anchor = lat["bound_b"][h] if sign > 0 else lat["bound_a"][h]
            waypoints = [points[k, j_edge] for k in range(i + 1, nchord + 1)]
            path_len = np.linalg.norm(waypoints[0] - anchor)
            path_len += sum(np.linalg.norm(waypoints[k + 1] - waypoints[k])
                            for k in range(len(waypoints) - 1))
            fine_len = path_len + TIP_FINE_LENGTH_CHORDS * chord_mean
            n_fine = int(np.ceil(fine_len / step_fine))
            step_w_tip = WAKE_LENGTH_SPANS * span_extent / NWAKE
            n_coarse = max(int(np.ceil(
                (WAKE_LENGTH_SPANS * span_extent - fine_len) / step_w_tip)), 0)
            steps = np.concatenate([np.full(n_fine, step_fine),
                                    np.full(n_coarse, step_w_tip)])
            filaments.append({
                "nodes": initial_nodes(anchor, waypoints, steps, onset),
                "steps": steps,
                "cols_plus": [h] if sign > 0 else [],
                "cols_minus": [] if sign > 0 else [h],
                "is_tip": True,
            })
    return filaments


def filament_segments(fil, onset):
    """Downstream-oriented segments of one filament, plus far-field closure."""
    nodes = fil["nodes"]
    v_end = onset(nodes[-1:])[0]
    d_end = v_end / max(np.linalg.norm(v_end), 1e-12)
    p1 = np.vstack([nodes[:-1], nodes[-1:]])
    p2 = np.vstack([nodes[1:], nodes[-1:] + FAR_FIELD * d_end])
    return p1, p2


# ------------------------------------------------------------------ solver

def assemble_matrix(lat, filaments, onset):
    """Normal-velocity influence matrix for the current filament geometry."""
    colloc, normals = lat["colloc"], lat["normals"]
    n = len(colloc)
    core = np.full(len(lat["seg_p1"]), CORE_WING)
    v = segment_velocities(colloc, lat["seg_p1"], lat["seg_p2"], core)
    vn = np.einsum("msk,mk->ms", v, normals)
    a_mat = np.zeros((n, n))
    for s, owner in enumerate(lat["seg_owner"]):
        a_mat[:, owner] += vn[:, s]

    for fil in filaments:
        p1, p2 = filament_segments(fil, onset)
        vf = segment_velocities(
            colloc, p1, p2, np.full(len(p1), CORE_WING)).sum(axis=1)
        vnf = np.einsum("mk,mk->m", vf, normals)
        for col in fil["cols_plus"]:
            a_mat[:, col] += vnf
        for col in fil["cols_minus"]:
            a_mat[:, col] -= vnf
    return a_mat


def all_segments(gamma, lat, filaments, onset, rc_wake):
    """All vortex segments of the system with their strengths and core radii."""
    p1s, p2s = [lat["seg_p1"]], [lat["seg_p2"]]
    strengths = [gamma[lat["seg_owner"]]]
    cores = [np.full(len(lat["seg_p1"]), CORE_WING)]
    for fil in filaments:
        g = sum(gamma[c] for c in fil["cols_plus"])
        g -= sum(gamma[c] for c in fil["cols_minus"])
        p1, p2 = filament_segments(fil, onset)
        p1s.append(p1)
        p2s.append(p2)
        strengths.append(np.full(len(p1), g))
        cores.append(np.full(len(p1), rc_wake))
    return (np.vstack(p1s), np.vstack(p2s),
            np.concatenate(strengths), np.concatenate(cores))


def relax_filaments(filaments, gamma, lat, onset, rc_wake, x_near):
    """One relaxation step: advect every filament along the local velocity.

    Returns the maximum node displacement within the near field (x < x_near);
    far downstream the discrete filaments keep orbiting the rolled-up vortex
    core indefinitely without affecting the loads.
    """
    p1, p2, strengths, cores = all_segments(gamma, lat, filaments, onset, rc_wake)
    nodes_all = np.vstack([fil["nodes"] for fil in filaments])
    v = onset(nodes_all) + induced_velocity(nodes_all, p1, p2, strengths, cores)

    delta = 0.0
    offset = 0
    for fil in filaments:
        nodes = fil["nodes"]
        nn = len(nodes)
        v_fil = v[offset:offset + nn]
        offset += nn
        new = nodes.copy()
        for k, length in enumerate(fil["steps"]):
            vk = v_fil[k]
            new[k + 1] = new[k] + length * vk / max(np.linalg.norm(vk), 1e-12)
        new = nodes + OMEGA * (new - nodes)
        near = nodes[:, 0] < x_near
        if np.any(near):
            delta = max(delta, np.max(
                np.linalg.norm((new - nodes)[near], axis=1)))
        fil["nodes"] = new
    return delta


def compute_loads(gamma, lat, filaments, onset, rc_wake, u_ref, s_ref):
    """Vector Kutta-Joukowski forces on the bound segments -> CL, CDi."""
    mid = 0.5 * (lat["bound_a"] + lat["bound_b"])
    dl = lat["bound_b"] - lat["bound_a"]
    p1, p2, strengths, cores = all_segments(gamma, lat, filaments, onset, rc_wake)
    v = onset(mid) + induced_velocity(mid, p1, p2, strengths, cores)
    force = RHO * np.sum(gamma[:, None] * np.cross(v, dl), axis=0)
    q = 0.5 * RHO * u_ref ** 2 * s_ref
    return force[2] / q, force[0] / q


def solve(lat, onset, u_ref):
    points = lat["points"]
    nchord, nspan = lat["nchord"], lat["nspan"]
    chords = np.linalg.norm(np.diff(points, axis=0), axis=2).sum(axis=0)
    chord_mean = chords.mean()
    span_extent = np.ptp(points[..., 1])
    rc_wake = RC_WAKE_FRACTION * span_extent / nspan
    s_ref = lat["areas"].sum()
    x_near = points[..., 0].max() + span_extent

    rhs = -np.einsum("mk,mk->m", onset(lat["colloc"]), lat["normals"])
    filaments = init_filaments(lat, onset, chord_mean, span_extent)

    cl_frozen = None
    for it in range(MAX_ITER):
        a_mat = assemble_matrix(lat, filaments, onset)
        gamma = np.linalg.solve(a_mat, rhs)
        if it == 0:
            cl_frozen, _ = compute_loads(gamma, lat, filaments, onset,
                                         rc_wake, u_ref, s_ref)
        delta = relax_filaments(filaments, gamma, lat, onset, rc_wake, x_near)
        print(f"  wake iteration {it + 1:2d}: "
              f"max near-field node displacement {delta / chord_mean:.2e} c")
        if delta < TOL * chord_mean:
            break

    a_mat = assemble_matrix(lat, filaments, onset)
    gamma = np.linalg.solve(a_mat, rhs)
    cl, cdi = compute_loads(gamma, lat, filaments, onset, rc_wake, u_ref, s_ref)
    return gamma, filaments, cl, cdi, cl_frozen, s_ref


def spanwise_loading(gamma, lat, u_ref):
    """Section lift coefficient cl(y) from chordwise-summed circulation."""
    points = lat["points"]
    nchord, nspan = lat["nchord"], lat["nspan"]
    gamma_sec = gamma.reshape(nchord, nspan).sum(axis=0)
    edge_chords = np.linalg.norm(np.diff(points, axis=0), axis=2).sum(axis=0)
    chords = 0.5 * (edge_chords[:-1] + edge_chords[1:])
    y_mid = points[..., 1].mean(axis=0)
    y_sec = 0.5 * (y_mid[:-1] + y_mid[1:])
    return y_sec, 2.0 * gamma_sec / (u_ref * chords)


# ------------------------------------------------------------------ output

def plot_results(lat, filaments, y_sec, cl_span, z_table, u_table):
    points = lat["points"]
    span_extent = np.ptp(points[..., 1])
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(2.0 * y_sec / span_extent, cl_span, "o-")
    ax.set_xlabel(r"non-dimensional span $2y/b$")
    ax.set_ylabel(r"section lift coefficient $c_l(y)$")
    ax.set_title("Spanwise loading")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    all_nodes = np.vstack([fil["nodes"] for fil in filaments])
    for fil in filaments:
        color, lw = ("tab:red", 1.2) if fil["is_tip"] else ("tab:blue", 0.6)
        w = fil["nodes"]
        ax.plot(w[:, 0], w[:, 1], w[:, 2], color=color, lw=lw)
    for i in range(points.shape[0]):
        ax.plot(points[i, :, 0], points[i, :, 1], points[i, :, 2],
                color="k", lw=0.6)
    for j in range(points.shape[1]):
        ax.plot(points[:, j, 0], points[:, j, 1], points[:, j, 2],
                color="k", lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Relaxed (force-free) wake, tip filaments in red")
    x_range = np.ptp(all_nodes[:, 0])
    y_range = max(np.ptp(all_nodes[:, 1]), span_extent)
    z_range = max(np.ptp(all_nodes[:, 2]), 0.15 * y_range)
    ax.set_box_aspect((x_range, y_range, z_range))
    ax.view_init(elev=22, azim=-140)

    ax = fig.add_subplot(gs[0, 2])
    z_lo = min(z_table.min(), all_nodes[:, 2].min())
    z_hi = max(z_table.max(), all_nodes[:, 2].max())
    z_plot = np.linspace(z_lo, z_hi, 200)
    ax.plot(np.interp(z_plot, z_table, u_table), z_plot, "-")
    ax.plot(u_table, z_table, "o", ms=4, label="profile data")
    ax.axhspan(points[..., 2].min(), points[..., 2].max(),
               color="0.85", label="wing z extent")
    ax.set_xlabel("U(z)")
    ax.set_ylabel("z")
    ax.set_title("Onset velocity profile")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for fil in filaments:
        color, lw = ("tab:red", 1.2) if fil["is_tip"] else ("tab:blue", 0.6)
        w = fil["nodes"]
        ax.plot(w[:, 1], w[:, 2], color=color, lw=lw)
    ends = np.vstack([fil["nodes"][-1] for fil in filaments])
    ax.plot(ends[:, 1], ends[:, 2], "k.", ms=4, label="filament end points")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title("Wake, rear view (tip roll-up)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    x_te_max = points[-1, :, 0].max()
    for fil in filaments:
        if not fil["is_tip"]:
            continue
        w = fil["nodes"]
        if w[0, 1] > points[..., 1].mean():  # right tip only
            ax.plot(w[:, 0], w[:, 2], color="tab:red", lw=1.2)
            ax.plot(w[0, 0], w[0, 2], "ko", ms=4)
    j_tip = points.shape[1] - 1
    ax.plot(points[:, j_tip, 0], points[:, j_tip, 2], color="0.4", lw=2.0,
            label="tip section")
    ax.set_xlim(points[..., 0].min() - 0.2, x_te_max + 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Tip-vortex development along the chord (side view)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    for fil in filaments:
        color, lw = ("tab:red", 1.2) if fil["is_tip"] else ("tab:blue", 0.6)
        w = fil["nodes"]
        ax.plot(w[:, 0], w[:, 1], color=color, lw=lw)
    outline = np.vstack([points[0, :], points[1:, -1], points[-1, ::-1],
                         points[::-1, 0]])
    ax.plot(outline[:, 0], outline[:, 1], color="k", lw=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Top view (planform and wake)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def prompt(text, default):
    reply = input(f"{text} [{default}]: ").strip()
    return reply if reply else default


def main():
    mesh_path = prompt("Wing mesh (.npz with 'points')", "wing_mesh.npz")
    profile_path = prompt("Velocity profile CSV (z, U)", "velocity_profile.csv")
    u_ref = float(prompt("Reference speed U_ref", "1.0"))
    alpha_deg = float(prompt("Additional pitch angle [deg]", "0.0"))

    points = load_mesh(mesh_path)
    if abs(alpha_deg) > 0.0:
        points = pitch_mesh(points, np.radians(alpha_deg))
    z_table, u_table = load_velocity_profile(profile_path)
    onset = make_onset(z_table, u_table)

    lat = build_lattice(points)
    print(f"\nMesh: {lat['nchord']} chordwise x {lat['nspan']} spanwise panels, "
          f"span extent {np.ptp(points[..., 1]):.3f}, "
          f"lattice area {lat['areas'].sum():.3f}")

    print("Relaxing the wake and tip filaments...")
    gamma, filaments, cl, cdi, cl_frozen, s_ref = solve(lat, onset, u_ref)

    print(f"\nReference: U_ref = {u_ref}, S_ref = {s_ref:.4f}")
    print(f"CL (frozen wake): {cl_frozen:.4f}")
    print(f"CL (free wake and tip filaments): {cl:.4f}")
    print(f"CDi (induced drag): {cdi:.5f}")

    y_sec, cl_span = spanwise_loading(gamma, lat, u_ref)
    plot_results(lat, filaments, y_sec, cl_span, z_table, u_table)


if __name__ == "__main__":
    main()
