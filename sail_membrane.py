"""Geometrically nonlinear membrane finite elements for a sail.

The sail cloth is a membrane: it carries in-plane tension and no bending. It is
discretised with three-node constant-strain triangles in a total Lagrangian
formulation, so displacements and rotations may be large. The strain is the
Green-Lagrange strain, the stress the second Piola-Kirchhoff stress, and the
material is a plane-stress Saint Venant-Kirchhoff law written per unit
reference width, with the membrane stiffness E t as its modulus. A membrane
cannot carry compression; it wrinkles. The wrinkling model is the tension-field
theory of the relaxed strain energy, in the mixed stress-strain form:

    taut      the smaller principal stress of the elastic law is positive
    slack     the larger principal strain is not positive: no stress
    wrinkled  otherwise: uniaxial tension E t e1 along the major principal
              strain direction n1, free contraction across it

Edge cables (luff, leech and foot lines) are two-node bars with the same
Green-Lagrange kinematics, optionally tension-only. A uniform pressure can be
applied as a follower load, normal to the current surface, with its
consistent (non-symmetric) load stiffness; the aerodynamic load arrives
instead as nodal force vectors from the fluid. Tension is introduced by a
pre-strain, which shifts the strain at which each element or cable is stress
free: a pre-strain e0 is a cloth cut smaller than the reference geometry by
the stretch sqrt(1 + 2 e0).

The static solution is a Newton iteration on the residual with load stepping,
a backtracking line search, and Levenberg-Marquardt damping of the tangent.
The damping is not cosmetic: a membrane that is slack, flat or wrinkled has a
singular tangent, so the undamped Newton step does not exist at the very
configurations a sail starts from.

Every block is a pure function over explicit arguments. `build_model` returns
the topology, materials and the reference-configuration cache; `solve_static`
returns a solution dict and mutates nothing. Nothing here prints.
"""

import numpy as np

# strain state labels returned per element by membrane_stress
TAUT, WRINKLED, SLACK = 0, 1, 2

TOL = 1.0e-9           # residual reduction against the scale of the call
MAX_ITER = 300         # Newton iterations per load step
LINE_SEARCH = 12       # halvings of the step before the damping is raised
MU_START = 1.0e-8      # first Levenberg-Marquardt damping, relative to diag K
TOL_STALL = 1.0e-6     # a residual that stops falling below this is converged
STALL_ITER = 10        # iterations without halving the best residual


# ------------------------------------------------------------------- mesh

def node_id(i, j, nspan):
    """Node of the structured grid (i chordwise, j spanwise), as in the fluid."""
    return i * (nspan + 1) + j


def grid_triangles(nchord, nspan, alternate=True):
    """Two triangles per grid quadrilateral, the diagonal alternating.

    Alternating the diagonal removes the directional bias of a single
    diagonal family; `alternate=False` keeps the single family, whose nodes
    are all equivalent. Every triangle is ordered so that its normal follows the
    grid orientation (i, j) -> (x, y) -> +z.
    """
    tris = []
    for i in range(nchord):
        for j in range(nspan):
            a, b = node_id(i, j, nspan), node_id(i + 1, j, nspan)
            c, d = node_id(i + 1, j + 1, nspan), node_id(i, j + 1, nspan)
            if not alternate or (i + j) % 2 == 0:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]
    return np.array(tris, dtype=int)


def edge_chain(nodes):
    """Consecutive node pairs of a polyline, as cable elements."""
    nodes = list(nodes)
    return np.array(list(zip(nodes[:-1], nodes[1:])), dtype=int)


# ------------------------------------------------------------------ model

def build_model(X, tris, Et, nu=0.3, prestrain=0.0, wrinkling=True,
                cables=None, fixed=None, hinges=None):
    """Topology, materials and the reference-configuration cache.

    X          (nn, 3) reference (stress-free apart from pre-strain) positions
    tris       (ne, 3) triangle connectivity
    Et         membrane stiffness E t [N/m], scalar or per element
    nu         Poisson ratio
    prestrain  pre-strain e0, scalar or per element
    cables     list of dicts {"pairs": (nc, 2), "EA": [N], "prestrain": e0,
               "tension_only": bool}
    fixed      (nn, 3) boolean mask of constrained degrees of freedom
    hinges     optional {"nodes": (nh, 4), "k": (nh,)}: rotation-free bending
               hinges, stress free in the reference, which is how battens
               enter (see hinge_forces)
    """
    X = np.asarray(X, dtype=float)
    tris = np.asarray(tris, dtype=int)
    ne = len(tris)
    X1, X2, X3 = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    a, b = X2 - X1, X3 - X1
    nvec = np.cross(a, b)
    two_area = np.linalg.norm(nvec, axis=1)
    if np.any(two_area <= 1e-14 * np.max(two_area)):
        raise ValueError("degenerate triangle in the reference mesh")
    e1 = a / np.linalg.norm(a, axis=1, keepdims=True)
    e3 = nvec / two_area[:, None]
    e2 = np.cross(e3, e1)
    # local 2D coordinates of nodes 2 and 3 (node 1 at the origin)
    J = np.empty((ne, 2, 2))
    J[:, 0, 0] = np.einsum("ek,ek->e", a, e1)
    J[:, 1, 0] = 0.0
    J[:, 0, 1] = np.einsum("ek,ek->e", b, e1)
    J[:, 1, 1] = np.einsum("ek,ek->e", b, e2)
    dn_drs = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    G = np.einsum("ar,erJ->eaJ", dn_drs, np.linalg.inv(J))   # (ne, 3, 2)

    nn = len(X)
    if fixed is None:
        fixed = np.zeros((nn, 3), dtype=bool)
    fixed = np.asarray(fixed, dtype=bool).reshape(nn, 3)

    cab = []
    for c in cables or []:
        pairs = np.asarray(c["pairs"], dtype=int).reshape(-1, 2)
        L0 = np.linalg.norm(X[pairs[:, 1]] - X[pairs[:, 0]], axis=1)
        cab.append({"pairs": pairs, "L0": L0, "EA": float(c["EA"]),
                    "prestrain": float(c.get("prestrain", 0.0)),
                    "tension_only": bool(c.get("tension_only", True))})

    if hinges is not None:
        nodes = np.asarray(hinges["nodes"], dtype=int).reshape(-1, 4)
        hinges = {"nodes": nodes,
                  "k": np.broadcast_to(np.asarray(hinges["k"], dtype=float),
                                       (len(nodes),)).copy(),
                  "theta0": dihedral(X[nodes])}

    return {
        "X": X, "tris": tris, "G": G, "A0": 0.5 * two_area,
        "hinges": hinges,
        "Et": np.broadcast_to(np.asarray(Et, dtype=float), (ne,)).copy(),
        "nu": float(nu),
        "prestrain": np.broadcast_to(
            np.asarray(prestrain, dtype=float), (ne,)).copy(),
        "wrinkling": bool(wrinkling), "cables": cab, "fixed": fixed,
        "edofs": (3 * tris[:, :, None] + np.arange(3)).reshape(ne, 9),
    }


# --------------------------------------------------------------- material

def plane_stress_matrix(Et, nu):
    """Voigt plane-stress stiffness per element, (ne, 3, 3)."""
    Eb = Et / (1.0 - nu * nu)
    D = np.zeros((len(Et), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = Eb
    D[:, 0, 1] = D[:, 1, 0] = Eb * nu
    D[:, 2, 2] = Eb * 0.5 * (1.0 - nu)
    return D


def membrane_stress(E, Et, nu, wrinkling=True):
    """Second Piola-Kirchhoff stress, tangent and state from Green strain.

    E is (ne, 2, 2), already shifted by the pre-strain. Returns S (ne, 2, 2),
    the Voigt tangent D (ne, 3, 3) with engineering shear strain, and the
    per-element state TAUT, WRINKLED or SLACK.
    """
    ne = len(E)
    D = plane_stress_matrix(Et, nu)
    Ev = np.stack([E[:, 0, 0], E[:, 1, 1], 2.0 * E[:, 0, 1]], axis=1)
    Sv = np.einsum("eij,ej->ei", D, Ev)
    S = np.empty((ne, 2, 2))
    S[:, 0, 0], S[:, 1, 1] = Sv[:, 0], Sv[:, 1]
    S[:, 0, 1] = S[:, 1, 0] = Sv[:, 2]
    state = np.full(ne, TAUT)
    if not wrinkling:
        return S, D, state

    s_min = np.linalg.eigvalsh(S)[:, 0]
    w, v = np.linalg.eigh(E)                    # ascending: e2, e1
    e2, e1 = w[:, 0], w[:, 1]
    n1, n2 = v[:, :, 1], v[:, :, 0]
    slack = e1 <= 0.0
    wrinkled = (s_min <= 0.0) & ~slack
    state[wrinkled] = WRINKLED
    state[slack] = SLACK

    S[slack] = 0.0
    D[slack] = 0.0
    if np.any(wrinkled):
        k = wrinkled
        m1, m2 = n1[k], n2[k]
        S[k] = (Et[k] * e1[k])[:, None, None] * np.einsum("ei,ej->eij", m1, m1)
        # d(e1 n1 n1): the eigenvalue varies as n1.dE.n1 and the direction
        # as (n2.dE.n1) / (e1 - e2); in Voigt form both are rank one
        av = np.stack([m1[:, 0] ** 2, m1[:, 1] ** 2, m1[:, 0] * m1[:, 1]], 1)
        bv = np.stack([m1[:, 0] * m2[:, 0], m1[:, 1] * m2[:, 1],
                       0.5 * (m1[:, 0] * m2[:, 1] + m1[:, 1] * m2[:, 0])], 1)
        gap = np.maximum(e1[k] - e2[k], 1e-14 * np.maximum(np.abs(e1[k]), 1.0))
        D[k] = Et[k][:, None, None] * (
            np.einsum("ei,ej->eij", av, av)
            + (2.0 * e1[k] / gap)[:, None, None]
            * np.einsum("ei,ej->eij", bv, bv))
    return S, D, state


# ----------------------------------------------------------- element level

def membrane_kinematics(model, x):
    """Deformation gradient (ne, 3, 2) and Green strain (ne, 2, 2)."""
    xe = x[model["tris"]]                                     # (ne, 3, 3)
    F = np.einsum("eak,eaJ->ekJ", xe, model["G"])
    C = np.einsum("ekI,ekJ->eIJ", F, F)
    E = 0.5 * (C - np.eye(2))
    E = E + model["prestrain"][:, None, None] * np.eye(2)
    return F, E


def membrane_forces(model, x, tangent=True):
    """Internal nodal forces of the membrane and, optionally, element tangents.

    Returns (fe (ne, 9), Ke (ne, 9, 9) or None, S, state). The internal force
    is the integral of B^T S over the reference area.
    """
    F, E = membrane_kinematics(model, x)
    S, D, state = membrane_stress(E, model["Et"], model["nu"],
                                  model["wrinkling"])
    G, A0 = model["G"], model["A0"]
    P = np.einsum("ekI,eIJ->ekJ", F, S)                       # first PK
    fe = (A0[:, None, None] * np.einsum("ekJ,eaJ->eak", P, G)).reshape(-1, 9)
    if not tangent:
        return fe, None, S, state
    ne = len(G)
    B = np.empty((ne, 3, 3, 3))                               # (e, voigt, a, k)
    B[:, 0] = np.einsum("ek,ea->eak", F[:, :, 0], G[:, :, 0])
    B[:, 1] = np.einsum("ek,ea->eak", F[:, :, 1], G[:, :, 1])
    B[:, 2] = (np.einsum("ek,ea->eak", F[:, :, 0], G[:, :, 1])
               + np.einsum("ek,ea->eak", F[:, :, 1], G[:, :, 0]))
    B = B.reshape(ne, 3, 9)
    Km = np.einsum("evi,evw,ewj->eij", B, D, B)
    gsg = np.einsum("eaI,eIJ,ebJ->eab", G, S, G)              # (ne, 3, 3)
    Kg = np.einsum("eab,kl->eakbl", gsg, np.eye(3)).reshape(ne, 9, 9)
    Ke = A0[:, None, None] * (Km + Kg)
    return fe, Ke, S, state


def cable_forces(cable, x, tangent=True):
    """Internal forces and tangents of one family of two-node cables."""
    pairs, L0 = cable["pairs"], cable["L0"]
    d = x[pairs[:, 1]] - x[pairs[:, 0]]
    l2 = np.einsum("ek,ek->e", d, d)
    Etot = 0.5 * (l2 / L0 ** 2 - 1.0) + cable["prestrain"]
    N = cable["EA"] * Etot                     # second PK force
    if cable["tension_only"]:
        on = Etot > 0.0
        N = np.where(on, N, 0.0)
    else:
        on = np.ones(len(L0), dtype=bool)
    f2 = (N / L0)[:, None] * d
    fe = np.concatenate([-f2, f2], axis=1)
    if not tangent:
        return fe, None, N
    k = ((N / L0)[:, None, None] * np.eye(3)
         + (cable["EA"] * on / L0 ** 3)[:, None, None]
         * np.einsum("ei,ej->eij", d, d))
    Ke = np.zeros((len(L0), 6, 6))
    Ke[:, :3, :3] = Ke[:, 3:, 3:] = k
    Ke[:, :3, 3:] = Ke[:, 3:, :3] = -k
    return fe, Ke, N


def skew(v):
    """Cross-product matrices [v]x, (n, 3, 3)."""
    s = np.zeros(v.shape[:-1] + (3, 3))
    s[..., 0, 1], s[..., 0, 2] = -v[..., 2], v[..., 1]
    s[..., 1, 0], s[..., 1, 2] = v[..., 2], -v[..., 0]
    s[..., 2, 0], s[..., 2, 1] = -v[..., 1], v[..., 0]
    return s


def pressure_forces(model, x, p, tangent=True):
    """Follower pressure p on every triangle, along its current normal.

    Each node takes a third of p times the current area vector. Returns the
    external element forces and their derivative with respect to x.
    """
    xe = x[model["tris"]]
    a, b = xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0]
    c = np.cross(a, b)
    pe = np.broadcast_to(np.asarray(p, dtype=float), (len(c),))
    fe = np.repeat((pe / 6.0)[:, None, None] * c[:, None, :], 3, axis=1)
    fe = fe.reshape(-1, 9)
    if not tangent:
        return fe, None
    sa, sb = skew(a), skew(b)
    dc = np.concatenate([sb - sa, -sb, sa], axis=2)          # (ne, 3, 9)
    Ke = np.repeat((pe / 6.0)[:, None, None, None] * dc[:, None], 3, axis=1)
    return fe, Ke.reshape(-1, 9, 9)


# ---------------------------------------------------------------- bending

CSTEP = 1.0e-30        # complex step for the hinge-angle gradient


def dihedral(xh):
    """Signed angle between triangles (a, b, c) and (b, a, d), (nh,).

    xh is (nh, 4, 3) with the nodes in the order (a, b, c, d); a-b is the
    hinge axis, and the angle is zero when the four nodes are coplanar with
    c and d on opposite sides of the axis.

    Written with complex-safe operations only (no conjugate, no arctan2), so
    that a complex step through it gives the gradient to round-off.
    """
    xa, xb, xc, xd = xh[:, 0], xh[:, 1], xh[:, 2], xh[:, 3]
    n1 = np.cross(xb - xa, xc - xa)
    n2 = np.cross(xa - xb, xd - xb)
    n1 = n1 / np.sqrt(np.sum(n1 * n1, axis=1))[:, None]
    n2 = n2 / np.sqrt(np.sum(n2 * n2, axis=1))[:, None]
    e = xb - xa
    e = e / np.sqrt(np.sum(e * e, axis=1))[:, None]
    sin = np.sum(np.cross(n1, n2) * e, axis=1)
    cos = np.sum(n1 * n2, axis=1)
    return 2.0 * np.arctan(sin / (1.0 + cos))


def dihedral_gradient(xh):
    """Gradient of the hinge angles, (nh, 12), by complex step."""
    nh = len(xh)
    flat = xh.reshape(nh, 12).astype(complex)
    g = np.empty((nh, 12))
    for k in range(12):
        z = flat.copy()
        z[:, k] += 1j * CSTEP
        g[:, k] = dihedral(z.reshape(nh, 4, 3)).imag / CSTEP
    return g


def hinge_forces(model, x, tangent=True):
    """Internal forces (nh, 12), tangents (nh, 12, 12) and energy of hinges.

    A hinge is a rotation-free bending element over four nodes: the angle
    between the virtual triangles (a, b, c) and (b, a, d) about the axis a-b.
    With c and d the neighbours of a along a line and b a neighbour across
    it, the angle is the turn of the line at a, so a line of hinges is a
    beam that resists bending about the axis direction and nothing else.
    W = k (theta - theta0)^2 / 2, so f = k (theta - theta0) grad theta and
    K = k grad theta grad theta^T + k (theta - theta0) Hess theta. The first
    term is exact; the second is a central difference of the complex-step
    gradient, and it is scaled by the angle change, which is small.
    """
    hg = model["hinges"]
    xh = x[hg["nodes"]]
    dth = dihedral(xh) - hg["theta0"]
    g = dihedral_gradient(xh)
    k = hg["k"]
    fe = (k * dth)[:, None] * g
    energy = 0.5 * np.sum(k * dth * dth)
    if not tangent:
        return fe, None, energy
    nh = len(xh)
    flat = xh.reshape(nh, 12)
    step = 1e-6 * np.sqrt(np.sum((xh[:, 1] - xh[:, 0]) ** 2, axis=1))
    H = np.empty((nh, 12, 12))
    for m in range(12):
        zp, zm = flat.copy(), flat.copy()
        zp[:, m] += step
        zm[:, m] -= step
        H[:, :, m] = ((dihedral_gradient(zp.reshape(nh, 4, 3))
                       - dihedral_gradient(zm.reshape(nh, 4, 3)))
                      / (2.0 * step)[:, None])
    H = 0.5 * (H + np.transpose(H, (0, 2, 1)))
    Ke = (k[:, None, None] * np.einsum("ei,ej->eij", g, g)
          + (k * dth)[:, None, None] * H)
    return fe, Ke, energy


# --------------------------------------------------------------- assembly

def assemble(model, u, f_ext=None, pressure=0.0, tangent=True):
    """Residual R = f_int - f_ext - f_pressure and its tangent dR/du."""
    x = model["X"] + u
    ndof = 3 * len(x)
    R = np.zeros(ndof)
    K = np.zeros((ndof, ndof)) if tangent else None

    fe, Ke, _, _ = membrane_forces(model, x, tangent)
    ed = model["edofs"]
    np.add.at(R, ed, fe)
    if tangent:
        np.add.at(K, (ed[:, :, None], ed[:, None, :]), Ke)

    for cab in model["cables"]:
        fc, Kc, _ = cable_forces(cab, x, tangent)
        cd = (3 * cab["pairs"][:, :, None] + np.arange(3)).reshape(-1, 6)
        np.add.at(R, cd, fc)
        if tangent:
            np.add.at(K, (cd[:, :, None], cd[:, None, :]), Kc)

    if model["hinges"] is not None:
        fh, Kh, _ = hinge_forces(model, x, tangent)
        hd = (3 * model["hinges"]["nodes"][:, :, None]
              + np.arange(3)).reshape(-1, 12)
        np.add.at(R, hd, fh)
        if tangent:
            np.add.at(K, (hd[:, :, None], hd[:, None, :]), Kh)

    if np.any(pressure != 0.0):
        fp, Kp = pressure_forces(model, x, pressure, tangent)
        np.add.at(R, ed, -fp)
        if tangent:
            np.add.at(K, (ed[:, :, None], ed[:, None, :]), -Kp)

    if f_ext is not None:
        R -= np.asarray(f_ext, dtype=float).reshape(ndof)
    return R, K


# ----------------------------------------------------------------- solver

def potential(model, u, f_ext=None):
    """Total potential energy under dead loads: strain energy minus work.

    The strain energy density is S:E / 2 in every state: the elastic law in a
    taut element, E t e1^2 / 2 in a wrinkled one (S = E t e1 n1 n1), zero in a
    slack one, and N E L0 / 2 in a cable.
    """
    x = model["X"] + u
    F, E = membrane_kinematics(model, x)
    S, _, _ = membrane_stress(E, model["Et"], model["nu"], model["wrinkling"])
    w = 0.5 * np.sum(model["A0"] * np.einsum("eij,eij->e", S, E))
    for cab in model["cables"]:
        d = x[cab["pairs"][:, 1]] - x[cab["pairs"][:, 0]]
        Etot = (0.5 * (np.einsum("ek,ek->e", d, d) / cab["L0"] ** 2 - 1.0)
                + cab["prestrain"])
        N = cab["EA"] * Etot
        if cab["tension_only"]:
            N = np.where(Etot > 0.0, N, 0.0)
        w += 0.5 * np.sum(N * Etot * cab["L0"])
    if model["hinges"] is not None:
        w += hinge_forces(model, x, tangent=False)[2]
    if f_ext is not None:
        w -= np.dot(np.asarray(f_ext, dtype=float).reshape(-1), u.reshape(-1))
    return w


def solve_static(model, f_ext=None, pressure=0.0, u0=None, u_presc=None,
                 nsteps=1, tol=TOL, max_iter=MAX_ITER):
    """Static equilibrium under nodal forces and a follower pressure.

    The load is applied in `nsteps` equal increments from u0 (zero by
    default). Prescribed displacements `u_presc` (nn, 3) are imposed on the
    fixed degrees of freedom at the first increment.

    Under dead loads alone equilibrium is a minimum of the total potential,
    and each Newton step is a damped one: the tangent is shifted by mu times
    its mean diagonal until it is positive definite, and the step is accepted
    on an Armijo decrease of the potential. A follower pressure has no
    potential, so with one the step is accepted on a decrease of the residual
    norm instead. The residual is measured against the largest of the first
    residual of the call and the applied load, so a form-finding solve with
    no applied load is not judged against zero. A residual that has not
    halved in STALL_ITER iterations and is below TOL_STALL is accepted as
    converged and flagged `stalled`: an element sitting on the boundary
    between the taut and wrinkled states switches its tangent from one step
    to the next, and the residual floors there, near 1e-8, rather than at
    the round-off of the force integral. Returns a dict and mutates nothing.
    """
    nn = len(model["X"])
    free = ~model["fixed"].reshape(-1)
    u = np.zeros((nn, 3)) if u0 is None else np.array(u0, dtype=float)
    if u_presc is not None:
        fixed = model["fixed"]
        u[fixed] = np.asarray(u_presc, dtype=float)[fixed]
    f_ext = None if f_ext is None else np.asarray(f_ext, dtype=float)
    conservative = not np.any(pressure != 0.0)

    R, _ = assemble(model, u, f_ext, pressure, tangent=False)
    load_norm = 0.0 if f_ext is None else np.linalg.norm(f_ext.reshape(-1)[free])
    if not conservative:
        fp, _ = pressure_forces(model, model["X"] + u, pressure, tangent=False)
        load_norm = max(load_norm, np.linalg.norm(fp))
    scale = max(np.linalg.norm(R[free]), load_norm, 1e-300)

    history, total_iter = [], 0
    stalled = False
    mu = 0.0
    converged = True
    for step in range(1, nsteps + 1):
        lam = step / nsteps
        fl = None if f_ext is None else lam * f_ext
        pl = lam * np.asarray(pressure, dtype=float)
        R, K = assemble(model, u, fl, pl)
        r = np.linalg.norm(R[free])
        merit = potential(model, u, fl) if conservative else r
        ok = r <= tol * scale
        best, since = r, 0
        for _ in range(max_iter):
            if ok:
                break
            Kff = K[np.ix_(free, free)]
            if conservative:
                Kff = 0.5 * (Kff + Kff.T)
            dscale = np.mean(np.abs(np.diag(Kff))) + 1e-300
            eye = np.eye(len(Kff))
            accepted = False
            while not accepted and mu <= 1.0e8:
                du = _damped_step(Kff + mu * dscale * eye, -R[free],
                                  conservative)
                if du is not None:
                    slope = -np.dot(R[free], du)
                    alpha = 1.0
                    for _ in range(LINE_SEARCH):
                        trial = u.copy()
                        trial.reshape(-1)[free] += alpha * du
                        if conservative:
                            m_t = potential(model, trial, fl)
                            accepted = m_t <= merit - 1e-4 * alpha * slope
                        else:
                            Rt, _ = assemble(model, trial, fl, pl,
                                             tangent=False)
                            m_t = np.linalg.norm(Rt[free])
                            accepted = m_t < merit
                        if accepted:
                            break
                        alpha *= 0.5
                if accepted:
                    mu = 0.0 if mu <= MU_START else 0.1 * mu
                else:
                    mu = MU_START if mu == 0.0 else 10.0 * mu
            if not accepted:
                # no step lowers the merit: round-off, if already small
                if r <= TOL_STALL * scale:
                    ok = stalled = True
                break
            u = trial
            total_iter += 1
            R, K = assemble(model, u, fl, pl)
            r = np.linalg.norm(R[free])
            merit = m_t if conservative else r
            history.append(r / scale)
            ok = r <= tol * scale
            if r < 0.5 * best:
                best, since = r, 0
            else:
                since += 1
            if not ok and since >= STALL_ITER and r <= TOL_STALL * scale:
                ok = stalled = True
        if not ok:
            converged = False
            break

    x = model["X"] + u
    _, _, S, state = membrane_forces(model, x, tangent=False)
    return {"u": u, "converged": converged, "stalled": stalled,
            "iterations": total_iter,
            "residual": history, "stress": S, "state": state,
            "reaction": reactions(model, u, f_ext, pressure)}


def _damped_step(A, b, spd):
    """Solve A du = b; with spd, fail unless A is positive definite."""
    try:
        if spd:
            L = np.linalg.cholesky(A)
            return np.linalg.solve(L.T, np.linalg.solve(L, b))
        du = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return du if np.all(np.isfinite(du)) else None


def reactions(model, u, f_ext=None, pressure=0.0):
    """Support reactions (nn, 3): the force each support exerts on the sail.

    It is the residual f_int - f_ext on the fixed degrees of freedom, and zero
    elsewhere; the reactions and the applied loads sum to zero.
    """
    R, _ = assemble(model, u, f_ext, pressure, tangent=False)
    out = np.zeros_like(R)
    fixed = model["fixed"].reshape(-1)
    out[fixed] = R[fixed]
    return out.reshape(-1, 3)


def membrane_tension(model, x):
    """True (Cauchy) membrane tension of each triangle at positions x.

    The tension per unit current width is N = F S F^T / J, J the area
    ratio. Returns its principal values (ne, 2) [N/m], descending, the unit
    direction of the larger in space (ne, 3), and the element state. With
    the cloth thickness t, N / t is the stress.
    """
    F, E = membrane_kinematics(model, x)
    S, _, state = membrane_stress(E, model["Et"], model["nu"],
                                  model["wrinkling"])
    C = np.einsum("ekI,ekJ->eIJ", F, F)
    J = np.sqrt(np.linalg.det(C))
    # the in-plane principal values are those of S C / J, which is similar
    # to the symmetric C^1/2 S C^1/2 / J and so has real eigenvalues
    M = np.einsum("eIK,eKJ->eIJ", S, C) / J[:, None, None]
    half = 0.5 * (M[:, 0, 0] + M[:, 1, 1])
    disc = np.sqrt(np.maximum(half ** 2 - np.linalg.det(M), 0.0))
    values = np.stack([half + disc, half - disc], axis=1)
    N = np.einsum("ekI,eIJ,elJ->ekl", F, S, F) / J[:, None, None]
    return values, np.linalg.eigh(N)[1][:, :, 2], state


def principal_stresses(S):
    """Principal second Piola-Kirchhoff stresses (ne, 2), descending."""
    return np.linalg.eigvalsh(S)[:, ::-1]
