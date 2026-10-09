"""The sail's aerodynamics: an adapter over the vendored `panel_wing.py`.

`panel_wing.py` is the free-wake thin lifting-surface method of
https://github.com/ignaziomviola/free-wake-lifting-surface, carried here
verbatim (see docs/FLUID.md). It is not modified; everything a sail needs on
top of it lives here and reaches it only through its public functions:

- the sail frame. The vendored code takes the span along y, the lift along +z
  and the onset nominally along +x. A sail is therefore set up with its mast
  along y, the apparent wind along +x, and its leeward (suction) side along
  +z. Heeling is about the x axis, and heights are y.
- the onset. The vendored `solve` accepts any onset callable; the one built
  here has a speed that grows with height (the wind gradient) and a direction
  that rotates about the mast with height (the twist of the apparent wind).
  The vendored `make_onset` varies the speed with z only and cannot do either.
- the deck. A sail whose foot seals on the deck sees its mirror image in the
  deck plane. The image is realised by MESH DOUBLING: the mesh is mirrored in
  y = 0 and joined at the foot, which is exact for an onset that is even in
  y, as the one built here is.
- per-panel loads. The vendored `compute_loads` returns the totals only. The
  per-panel vector Kutta-Joukowski force is recomputed here with the vendored
  segment and velocity functions, at the same points and with the same
  velocities, so that the panel forces sum to the vendored coefficients.
- quiet solves and a frozen wake. The vendored `solve` prints each wake
  iteration, so it is called under a redirected stdout; a frozen-wake solve
  sets its `MAX_ITER` to zero for the call and restores it.

Panel forces are lumped to the mesh nodes by the bilinear weights of the
point they act at, the midpoint of the bound vortex: a partition of unity that
reproduces the point, so the nodal forces conserve force and moment exactly.
"""

import contextlib
import io

import numpy as np

import panel_wing as pw

RHO_AIR = 1.225        # kg/m^3

# bilinear weights of the bound-vortex midpoint, at a quarter of the panel
# chord and half its span, on the corners (i, j), (i+1, j), (i, j+1), (i+1, j+1)
W00, W10, W01, W11 = 0.375, 0.125, 0.375, 0.125


# ------------------------------------------------------------------ onset

def sail_onset(u_ref, y_ref, shear=0.0, twist_deg=0.0, y_twist=None,
               y_offset=0.0):
    """Apparent wind with a gradient and a twist, as an onset callable.

    Speed      U(y) = u_ref ((|y| + y_offset) / (y_ref + y_offset))^shear
    Direction  rotated about the mast (y) by beta(y) = twist * |y| / y_twist,
               towards +z, so a positive twist raises the incidence aloft

    y_offset is the height of the foot above the water, so that the power law
    is measured from the sea. The onset depends on |y| so that it is even in
    y, which the deck image requires; below the foot it is used only by wake
    filaments that dip under it.
    """
    y_twist = y_ref if y_twist is None else y_twist
    beta_top = np.radians(twist_deg)

    def onset(points):
        pts = np.atleast_2d(points)
        h = np.abs(pts[:, 1])
        if shear:
            ratio = np.maximum(h + y_offset, 1e-6) / (y_ref + y_offset)
            speed = u_ref * ratio ** shear
        else:
            speed = np.full(len(pts), float(u_ref))
        beta = beta_top * h / y_twist
        v = np.zeros((len(pts), 3))
        v[:, 0] = speed * np.cos(beta)
        v[:, 2] = speed * np.sin(beta)
        return v
    return onset


# -------------------------------------------------------------- geometry

def mirror_in_deck(points):
    """Mesh doubled by its image in y = 0, joined at the foot (j = 0).

    Returns (nchord+1, 2 nspan+1, 3) with the image first, so that the real
    sail occupies spanwise panels nspan .. 2 nspan - 1.
    """
    foot = points[:, 0, 1]
    if np.max(np.abs(foot)) > 1e-12 * max(np.ptp(points[..., 1]), 1.0):
        raise ValueError("a deck image needs the foot on the deck plane y = 0")
    image = points[:, :0:-1].copy()
    image[..., 1] *= -1.0
    return np.concatenate([image, points], axis=1)


# ------------------------------------------------------------------ solve

def solve_fluid(points, onset, u_ref, rho=RHO_AIR, wake="frozen",
                deck=False, extra=None):
    """Steady free- or frozen-wake solution on the sail mesh `points`.

    points  (nchord+1, nspan+1, 3) sail surface, luff at i = 0, foot at j = 0
    wake    "frozen" (the default, as in the coupling) keeps the initial
            wake along the onset streamlines; "free" relaxes it with the
            vendored solver, at one to two orders of magnitude more cost
    deck    mirror the sail in the deck plane y = 0
    extra   optional velocity field of other lifting surfaces, a callable
            extra(points, use) such as `induced_field` of another sail, with
            use "matrix" at the collocation points and "loads" at the bound
            vortices. It enters the boundary condition and the loads but not
            the wake, which stays on the streamlines of `onset`: the
            frozen-wake statement of several surfaces solved together. It
            reaches the boundary condition through the one onset call the
            vendored `solve` makes on the collocation array itself,
            recognised by identity.

    Returns a dict with the per-panel forces of the sail itself (not of its
    image), the vendored lattice and solution, and the coefficients.
    """
    if wake not in ("free", "frozen"):
        raise ValueError(f"unknown wake '{wake}'; use 'free' or 'frozen'")
    nchord, nspan = points.shape[0] - 1, points.shape[1] - 1
    full = mirror_in_deck(points) if deck else points
    lat = pw.build_lattice(full)

    colloc = lat["colloc"]

    def flow(p):
        v = onset(p)
        return v + extra(p, "matrix") if extra is not None and p is colloc \
            else v

    saved = pw.MAX_ITER
    try:
        if wake == "frozen":
            pw.MAX_ITER = 0
        with contextlib.redirect_stdout(io.StringIO()):
            gamma, filaments, cl_v, cdi_v, _, s_ref_v = pw.solve(lat, flow,
                                                                 u_ref)
    finally:
        pw.MAX_ITER = saved

    force = panel_forces(gamma, lat, filaments, onset, rho, extra)
    full_span = lat["nspan"]
    j0 = full_span - nspan if deck else 0
    sel = (np.arange(nchord)[:, None] * full_span
           + j0 + np.arange(nspan)[None, :])
    out = {
        "lat": lat, "gamma": gamma, "filaments": filaments,
        "force": force[sel],                       # (nchord, nspan, 3)
        "normals": lat["normals"][sel],
        "areas": lat["areas"][sel],
        "gamma_sail": gamma[sel],
        "cl_vendored": cl_v, "cdi_vendored": cdi_v, "s_ref_vendored": s_ref_v,
        "u_ref": u_ref, "rho": rho, "deck": deck, "wake": wake,
        "onset": onset,
    }
    out.update(coefficients(out, points))
    return out


def panel_forces(gamma, lat, filaments, onset, rho=RHO_AIR, extra=None):
    """Per-panel vector Kutta-Joukowski force, (n, 3), as the vendored totals.

    The same bound-segment midpoints, the same total velocity and the same
    regularised kernel as `panel_wing.compute_loads`, so that the sum over
    panels divided by q S is the vendored (CL, CDi) for rho = pw.RHO.
    With `extra` the velocity of other surfaces is added at the bound
    vortices, and the vendored totals no longer include it.
    """
    nspan = lat["nspan"]
    span_extent = np.ptp(lat["points"][..., 1])
    rc_wake = pw.RC_WAKE_FRACTION * span_extent / nspan
    mid = 0.5 * (lat["bound_a"] + lat["bound_b"])
    dl = lat["bound_b"] - lat["bound_a"]
    p1, p2, strengths, cores = pw.all_segments(gamma, lat, filaments, onset,
                                               rc_wake)
    v = onset(mid) + pw.induced_velocity(mid, p1, p2, strengths, cores)
    if extra is not None:
        v = v + extra(mid, "loads")
    return rho * gamma[:, None] * np.cross(v, dl)


def wake_segments(sol, use="loads"):
    """Every vortex segment of a solved sail, its image and its wake.

    Returns (p1, p2, strength, core). The cores are those the vendored code
    uses for the same purpose: use = "matrix" gives the influence-matrix core
    CORE_WING on every segment, wake included (panel_wing.assemble_matrix);
    use = "loads" gives CORE_WING on the lattice and the wake core on the
    wake (panel_wing.compute_loads).
    """
    lat = sol["lat"]
    rc_wake = (pw.RC_WAKE_FRACTION * np.ptp(lat["points"][..., 1])
               / lat["nspan"])
    p1, p2, strength, core = pw.all_segments(sol["gamma"], lat,
                                             sol["filaments"], sol["onset"],
                                             rc_wake)
    if use == "matrix":
        core = np.full_like(core, pw.CORE_WING)
    elif use != "loads":
        raise ValueError(f"unknown use '{use}'; use 'matrix' or 'loads'")
    return p1, p2, strength, core


def induced_field(sol):
    """The velocity a solved sail induces elsewhere, as an `extra` field.

    Called as field(points, use), with use = "matrix" for a boundary
    condition and "loads" for a load evaluation, so that another surface
    sees this one exactly as the vendored code lets a surface see itself.
    Iterating two sails to a fixed point is then the joint frozen-wake
    solution of both (sail_plan.solve_sail_plan).
    """
    segs = {use: wake_segments(sol, use) for use in ("matrix", "loads")}

    def field(points, use):
        p1, p2, strength, core = segs[use]
        return pw.induced_velocity(np.atleast_2d(points), p1, p2, strength,
                                   core)
    return field


def coefficients(sol, points):
    """Resultants of the sail panels in the sail frame, and their coefficients.

    Lift is +z (to leeward), drag +x (along the apparent wind at u_ref), the
    heeling moment is about the x axis through the foot, positive to leeward.
    """
    f = sol["force"].reshape(-1, 3)
    q = 0.5 * sol["rho"] * sol["u_ref"] ** 2
    area = sol["areas"].sum()
    mids = 0.25 * (points[:-1, :-1] + points[1:, :-1]
                   + points[:-1, 1:] + points[1:, 1:]).reshape(-1, 3)
    total = f.sum(axis=0)
    heel = np.sum(f[:, 2] * mids[:, 1])
    return {
        "F": total, "area": area, "q": q,
        "CL": total[2] / (q * area), "CD": total[0] / (q * area),
        "heel": heel,
        "ce_height": heel / total[2] if total[2] != 0.0 else np.nan,
    }


# ---------------------------------------------------------------- lumping

def nodal_forces(sol, load="pressure"):
    """Panel forces lumped to the mesh nodes, (nchord+1, nspan+1, 3).

    load = "pressure" passes the component of each panel force normal to the
    panel, which is the pressure jump the cloth carries; the in-plane rest is
    the leading-edge suction, which a sail's luff support carries, and it is
    reported by `suction`. load = "kj" passes the full vector.
    """
    f = sol["force"]
    if load == "pressure":
        n = sol["normals"]
        f = np.einsum("ijk,ijk->ij", f, n)[..., None] * n
    elif load != "kj":
        raise ValueError(f"unknown load '{load}'; use 'pressure' or 'kj'")
    nchord, nspan = f.shape[:2]
    out = np.zeros((nchord + 1, nspan + 1, 3))
    out[:-1, :-1] += W00 * f
    out[1:, :-1] += W10 * f
    out[:-1, 1:] += W01 * f
    out[1:, 1:] += W11 * f
    return out


def suction(sol):
    """Resultant of the in-plane panel forces, which the cloth does not take."""
    f, n = sol["force"], sol["normals"]
    fn = np.einsum("ijk,ijk->ij", f, n)[..., None] * n
    return (f - fn).reshape(-1, 3).sum(axis=0)


def pressure_jump(sol):
    """Panel pressure jump, leeward suction positive, (nchord, nspan) [Pa]."""
    fn = np.einsum("ijk,ijk->ij", sol["force"], sol["normals"])
    return fn / sol["areas"]
