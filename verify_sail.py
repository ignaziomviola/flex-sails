"""Verification programme: prints the tables of docs/MEMBRANE.md and COUPLING.md.

    python3 verify_sail.py              all cases
    python3 verify_sail.py --case S4    one case
    python3 verify_sail.py --quick      the cheap ones (S1-S5, F1, C2-C4, C7)

S  structure:  S1 patch and frame invariance, S2 tangent consistency,
               S3 wrinkling law, S4 inflated strip against the exact arc,
               S5 battens against a beam in tension
F  fluid:      F1 panel loads, transfer conservation and the deck image
C  coupled:    C1 membrane wing against two-dimensional sail theory,
               C2 the stiff limit, C3 mainsail leech tension,
               C4 coupling accelerators, C5 mesh convergence,
               C6 free against frozen wake, C7 two surfaces that see
               each other
"""

import sys
import time

import numpy as np

import make_sample_inputs as msi
import panel_wing as pw
import sail_2d
import sail_fluid as sf
import sail_fsi as fsi
import sail_membrane as sm


def header(name, text):
    print(f"\n=== {name}: {text}")


def flat_grid(nc, ns, lx=1.0, ly=1.0, jitter=0.0, seed=0):
    X = np.zeros(((nc + 1) * (ns + 1), 3))
    rng = np.random.default_rng(seed)
    for i in range(nc + 1):
        for j in range(ns + 1):
            p = np.array([i * lx / nc, j * ly / ns, 0.0])
            if 0 < i < nc and 0 < j < ns:
                p[:2] += jitter * rng.uniform(-1, 1, 2) * np.array([lx / nc,
                                                                     ly / ns])
            X[sm.node_id(i, j, ns)] = p
    return X


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = sm.skew(axis[None])[0]
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


# ------------------------------------------------------------- structure

def case_S1():
    header("S1", "patch test and frame invariance")
    nc, ns = 6, 5
    X = flat_grid(nc, ns, 1.0, 0.8, jitter=0.3)
    tris = sm.grid_triangles(nc, ns)
    m = sm.build_model(X, tris, Et=1000.0, nu=0.3)
    interior = np.array([sm.node_id(i, j, ns) for i in range(1, nc)
                         for j in range(1, ns)])
    print("  homogeneous stretch on a distorted mesh (all elements taut)")
    print("     l1     l2   max |S - S_exact| / |S|   max interior |f| / |f|")
    for l1, l2 in ((1.01, 1.005), (1.2, 1.1), (1.5, 1.3)):
        x = X * np.array([l1, l2, 1.0])
        F, E = sm.membrane_kinematics(m, x)
        S, _, st = sm.membrane_stress(E, m["Et"], m["nu"])
        e1, e2 = 0.5 * (l1 ** 2 - 1), 0.5 * (l2 ** 2 - 1)
        Eb = 1000.0 / (1 - 0.09)
        # the stress is in each element's local frame, so compare invariants
        s_ex = np.sort([Eb * (e1 + 0.3 * e2), Eb * (e2 + 0.3 * e1)])
        s_num = np.sort(np.linalg.eigvalsh(S), axis=1)
        err = np.max(np.abs(s_num - s_ex)) / np.max(np.abs(s_ex))
        R, _ = sm.assemble(m, x - X, tangent=False)
        f_el, _, _, _ = sm.membrane_forces(m, x, tangent=False)
        rint = np.max(np.abs(R.reshape(-1, 3)[interior])) / np.max(np.abs(f_el))
        assert np.all(st == sm.TAUT)
        print(f"  {l1:5.2f} {l2:6.2f}   {err:22.2e}   {rint:24.2e}")

    print("  large rigid motion of a cambered, pre-strained mesh")
    print("    angle   max |S(Q x) - S(x)| / |S|   |f(Q x) - Q f(x)| / |f|")
    Xc = X.copy()
    Xc[:, 2] = 0.1 * np.sin(np.pi * Xc[:, 0]) * (1 + Xc[:, 1])
    mc = sm.build_model(Xc, tris, Et=1000.0, nu=0.3, prestrain=0.003)
    rng = np.random.default_rng(1)
    x0 = Xc + 0.02 * rng.standard_normal(Xc.shape)
    f0, _, S0, _ = sm.membrane_forces(mc, x0, tangent=False)
    for ang in (30.0, 90.0, 170.0):
        Q = rotation([1.0, 2.0, 0.5], np.radians(ang))
        x1 = x0 @ Q.T + np.array([3.0, -1.0, 2.0])
        f1, _, S1, _ = sm.membrane_forces(mc, x1, tangent=False)
        es = np.max(np.abs(S1 - S0)) / np.max(np.abs(S0))
        ef = (np.max(np.abs(f1.reshape(-1, 3, 3) - f0.reshape(-1, 3, 3) @ Q.T))
              / np.max(np.abs(f0)))
        print(f"  {ang:7.1f}   {es:26.2e}   {ef:24.2e}")


def case_S2():
    header("S2", "tangent against central differences of the residual")
    nc, ns = 3, 4
    X = flat_grid(nc, ns, 1.5, 1.6)
    X[:, 2] = 0.05 * np.sin(3 * X[:, 0])
    cab = [{"pairs": sm.edge_chain([sm.node_id(nc, j, ns)
                                    for j in range(ns + 1)]),
            "EA": 50.0, "prestrain": 0.01}]
    hinges = fsi.batten_hinges(X.reshape(nc + 1, ns + 1, 3), [(2, 5.0)])
    m = sm.build_model(X, sm.grid_triangles(nc, ns), 100.0, 0.3, 0.0,
                       cables=cab, hinges=hinges)
    rng = np.random.default_rng(0)
    print("  with a pre-strained cable, a batten and a follower pressure")
    print("  amplitude  taut wrinkled slack   max |K - K_fd| / max |K|")
    for amp in (0.02, 0.05, 0.1):
        u = amp * rng.standard_normal(X.shape)
        R, K = sm.assemble(m, u, pressure=3.0)
        Kfd = np.zeros_like(K)
        h = 1e-6
        for k in range(len(R)):
            up, um = u.copy().reshape(-1), u.copy().reshape(-1)
            up[k] += h
            um[k] -= h
            Kfd[:, k] = (sm.assemble(m, up.reshape(-1, 3), pressure=3.0,
                                     tangent=False)[0]
                         - sm.assemble(m, um.reshape(-1, 3), pressure=3.0,
                                       tangent=False)[0]) / (2 * h)
        _, E = sm.membrane_kinematics(m, X + u)
        st = sm.membrane_stress(E, m["Et"], m["nu"])[2]
        n = np.bincount(st, minlength=3)
        err = np.abs(K - Kfd).max() / np.abs(K).max()
        print(f"  {amp:9.2f} {n[0]:5d} {n[1]:8d} {n[2]:5d}   {err:22.2e}")


def case_S3():
    header("S3", "the wrinkling law")
    Et, nu = np.array([1000.0]), 0.3
    print("  strain (e1, e2, angle)           state      |S - S_exact| / |S|")
    for e1, e2, ang in ((0.01, -0.005, 0.0), (0.01, -0.005, 35.0),
                        (0.02, 0.001, 70.0), (-0.001, -0.002, 10.0)):
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        Rm = np.array([[c, -s], [s, c]])
        E = (Rm @ np.diag([e1, e2]) @ Rm.T)[None]
        S, _, st = sm.membrane_stress(E, Et, nu)
        if e1 <= 0.0:
            S_ex = np.zeros((2, 2))
        elif e2 + nu * e1 <= 0.0:
            S_ex = Et[0] * e1 * np.outer(Rm[:, 0], Rm[:, 0])
        else:
            Eb = Et[0] / (1 - nu * nu)
            S_ex = Rm @ np.diag([Eb * (e1 + nu * e2),
                                 Eb * (e2 + nu * e1)]) @ Rm.T
        err = (np.max(np.abs(S[0] - S_ex)) / max(np.max(np.abs(S_ex)), 1e-300)
               if np.any(S_ex) else np.max(np.abs(S[0])))
        label = ("taut", "wrinkled", "slack")[st[0]]
        print(f"  ({e1:+.3f}, {e2:+.4f}, {ang:4.0f})       {label:9s}  "
              f"{err:12.2e}")
    e1 = 0.01
    gaps = []
    for d in (1e-6, -1e-6):
        E = np.diag([e1, -nu * e1 + d])[None]
        gaps.append(sm.membrane_stress(E, Et, nu)[0][0])
    print(f"  continuity across the taut-wrinkled boundary: "
          f"|dS| / |S| = {np.max(np.abs(gaps[0] - gaps[1])) / (Et[0] * e1):.2e}"
          f" for a strain step of 2e-6")


def strip_arc(nc, p, L=1.0, W=0.2, Et=1000.0, nu=0.3, e0=0.01, ns=2):
    X = flat_grid(nc, ns, L, W)
    fixed = np.zeros_like(X, dtype=bool)
    fixed[:, 1] = True
    for j in range(ns + 1):
        fixed[sm.node_id(0, j, ns)] = True
        fixed[sm.node_id(nc, j, ns)] = True
    m = sm.build_model(X, sm.grid_triangles(nc, ns), Et, nu, e0, fixed=fixed)
    sol = sm.solve_static(m, pressure=p)
    mid = [sm.node_id(nc // 2, j, ns) for j in range(ns + 1)]
    sag = sol["u"][mid, 2].mean()
    luff = [sm.node_id(0, j, ns) for j in range(ns + 1)]
    Hx = -sol["reaction"][luff, 0].sum() / W
    return sol, sag, Hx


def exact_arc(p, L=1.0, Et=1000.0, nu=0.3, e0=0.01):
    """Pinned strip in plane strain under a follower pressure: a circular arc.

    Half-angle th: R = L / (2 sin th), stretch lam = th / sin th, tension per
    unit width N = lam S with S = Et/(1-nu^2) ((lam^2-1)/2 + (1+nu) e0), and
    N = p R. Returns the sag and the horizontal support force per width.
    """
    Eb = Et / (1 - nu * nu)

    def g(th):
        lam = th / np.sin(th)
        return lam * Eb * (0.5 * (lam * lam - 1) + (1 + nu) * e0) \
            - p * L / (2 * np.sin(th))
    lo, hi = 1e-8, 0.5 * np.pi * 0.999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if g(lo) * g(mid) <= 0 else (mid, hi)
    th = 0.5 * (lo + hi)
    R = L / (2 * np.sin(th))
    return R * (1 - np.cos(th)), p * R * np.cos(th), th


def case_S4():
    header("S4", "pinned strip inflated by a follower pressure: the exact arc")
    print("  E t = 1000 N/m, nu = 0.3, e0 = 0.01, L = 1 m, plane strain")
    print("      p   half-angle   nc   sag error   order   thrust error   "
          "Newton")
    for p in (5.0, 50.0, 500.0):
        sag_ex, H_ex, th = exact_arc(p)
        prev = None
        for nc in (8, 16, 32, 64):
            sol, sag, H = strip_arc(nc, p)
            e_s, e_h = (sag - sag_ex) / sag_ex, (H - H_ex) / H_ex
            order = (f"{np.log2(abs(prev / e_s)):5.2f}" if prev is not None
                     else "    -")
            prev = e_s
            print(f"  {p:5.0f} {np.degrees(th):10.2f}° {nc:4d}   {e_s:+.2e}   "
                  f"{order}   {e_h:+.2e}      {sol['iterations']:3d}")


def beam_string(q, N, EI, L):
    """Midspan deflection of a pinned beam under tension N and load q."""
    k = np.sqrt(N / EI)
    return q / N * (L * L / 8 - (1 - 1 / np.cosh(k * L / 2)) / k ** 2)


def batten_strip(nc, e0, alternate=True, L=1.0, W=0.2, EI=20.0, p=0.1,
                 Et=1000.0, nu=0.3):
    """One row of cloth between two battens, pinned at both ends, in plane
    strain, under a follower pressure. Returns (solution, w / w_exact)."""
    X = flat_grid(nc, 1, L, W)
    fixed = np.zeros_like(X, dtype=bool)
    fixed[:, 1] = True
    for j in (0, 1):
        fixed[sm.node_id(0, j, 1)] = fixed[sm.node_id(nc, j, 1)] = True
    hinges = fsi.batten_hinges(X.reshape(nc + 1, 2, 3), [(0, EI), (1, EI)])
    m = sm.build_model(X, sm.grid_triangles(nc, 1, alternate), Et, nu, e0,
                       fixed=fixed, hinges=hinges)
    sol = sm.solve_static(m, pressure=p)
    w = sol["u"][[sm.node_id(nc // 2, j, 1) for j in (0, 1)], 2].mean()
    N = Et / (1 - nu * nu) * (1 + nu) * e0 * W
    return sol, w / beam_string(p * W, N, 2 * EI, L)


def case_S5():
    header("S5", "battens against a pinned Euler-Bernoulli beam in tension")
    print("  two battens EI = 20 N m^2 bounding one row of cloth, L = 1 m, "
          "W = 0.2 m, p = 0.1 Pa")
    print("  cloth e0   kL     diagonal      nc = 8     16        32        64")
    for e0 in (1e-4, 1e-2):
        N = 1000.0 / 0.91 * 1.3 * e0 * 0.2
        kL = np.sqrt(N / 40.0)
        for alt in (True, False):
            errs = [batten_strip(nc, e0, alt)[1] - 1 for nc in (8, 16, 32, 64)]
            print(f"  {e0:8.0e}  {kL:5.3f}  {'alternate' if alt else 'single':10s}"
                  + "".join(f"  {e:+.2e}" for e in errs))


# ----------------------------------------------------------------- fluid

def case_F1():
    header("F1", "panel loads, transfer conservation and the deck image")
    pts = pw.pitch_mesh(msi.rectangular_mesh(), np.radians(5.0))
    on = sf.sail_onset(1.0, 1.0)
    print("  rectangular wing, AR 8, 4 x 12 panels, alpha 5 deg, rho = 1")
    print("  wake     CL (panels)  CL (vendored)   |dCL|    |dCDi|   time")
    for wake in ("frozen", "free"):
        t = time.time()
        s = sf.solve_fluid(pts, on, 1.0, rho=pw.RHO, wake=wake)
        print(f"  {wake:7s} {s['CL']:12.6f} {s['cl_vendored']:13.6f}  "
              f"{abs(s['CL'] - s['cl_vendored']):.1e}  "
              f"{abs(s['CD'] - s['cdi_vendored']):.1e}  {time.time() - t:5.1f} s")
    half = pts[:, 6:].copy()
    sd = sf.solve_fluid(half, on, 1.0, rho=pw.RHO, wake="frozen", deck=True)
    sfull = sf.solve_fluid(pts, on, 1.0, rho=pw.RHO, wake="frozen")
    print(f"  half wing on a deck: CL = {sd['CL']:.12f} against the full wing "
          f"{sfull['CL']:.12f}")
    pts0 = fsi.sail_planform(10, 3.5, 0.5, camber=0.08, alpha_deg=10)
    fl = sf.solve_fluid(pts0, sf.sail_onset(6.0, 10.0, shear=0.12,
                                            twist_deg=5.0), 6.0)
    print("  mainsail, 8 x 12, sheared and twisted onset: nodal against panel")
    for load in ("pressure", "kj"):
        r = fsi.transfer_report(fl, load)
        print(f"    load = {load:8s} force defect {r['force_defect']:.1e}, "
              f"moment defect {r['moment_defect']:.1e}")
    sx = sf.suction(fl)
    print(f"    in-plane (suction) resultant {np.linalg.norm(sx):.2f} N of "
          f"{np.linalg.norm(fl['F']):.2f} N")


# --------------------------------------------------------------- coupled

def membrane_wing(span, nc=8, Et=6000.0, e0=0.002, alpha=6.0, U=10.0,
                  width=0.5, deck=True, **kw):
    ns = int(round(span / width))
    pts0 = fsi.sail_planform(span, 1.0, 1.0, nchord=nc, nspan=ns,
                             alpha_deg=alpha)
    model = fsi.build_sail(pts0, Et, prestrain=e0, foot="free",
                           leech="pinned", deck=deck)
    on = sf.sail_onset(U, span)
    res = fsi.static_aeroelastic(model, pts0, on, U, deck=deck, tol=1e-6,
                                 **kw)
    return res


def case_C1():
    header("C1", "membrane wing on a deck against two-dimensional sail theory")
    U, Et, e0, alpha = 10.0, 6000.0, 0.002, 6.0
    q = 0.5 * sf.RHO_AIR * U ** 2
    print(f"  chord 1 m, pinned luff and leech, E t / q c = {Et / q:.2f}, "
          f"e0 = {e0}, alpha = {alpha} deg, 8 uniform chordwise panels")
    print("  AR_eff   coupling   CL      root cl   root depth   time")
    rows = []
    for span in (2.0, 4.0, 8.0, 16.0):
        t = time.time()
        res = membrane_wing(span)
        cl_root = fsi.sectional_cl(res["fluid"])[0]
        d_root = fsi.section_shape(res["points"], 0)["depth"]
        rows.append((2 * span, cl_root, d_root))
        print(f"  {2 * span:6.0f}   {res['iterations']:4d}     "
              f"{res['fluid']['CL']:.4f}   {cl_root:.4f}   {d_root:.5f}    "
              f"{time.time() - t:5.1f} s")
    (_, c1, d1), (_, c2, d2), (_, c3, d3) = rows[-3:]
    ext = {}
    for key, (a, b, c) in (("cl", (c1, c2, c3)), ("depth", (d1, d2, d3))):
        p = np.log2(abs((a - b) / (b - c)))
        ext[key] = (c + (c - b) / (2 ** p - 1), p)
    ref = sail_2d.solve(np.radians(alpha), Et / q, e0, n=8, spacing="uniform")
    lim = sail_2d.extrapolate(np.radians(alpha), Et / q, e0)
    print(f"  AR -> infinity (observed order {ext['cl'][1]:.2f}, "
          f"{ext['depth'][1]:.2f}): root cl {ext['cl'][0]:.4f}, "
          f"root depth {ext['depth'][0]:.5f}")
    print(f"  2D theory, the same 8 uniform panels:      cl {ref['CL']:.4f}, "
          f"depth {ref['depth']:.5f}, C_T {ref['C_T']:.3f}")
    print(f"  difference: cl {ext['cl'][0] / ref['CL'] - 1:+.2%}, "
          f"depth {ext['depth'][0] / ref['depth'] - 1:+.2%}")
    print(f"  2D theory, continuous limit:               cl {lim['CL']:.4f}, "
          f"depth {lim['depth']:.5f}, C_T {lim['C_T']:.3f} "
          f"(observed order {lim['CL_order']:.2f})")
    print("  2D chordwise convergence (uniform panels):")
    for n in (8, 16, 32, 64):
        r = sail_2d.solve(np.radians(alpha), Et / q, e0, n=n,
                          spacing="uniform")
        print(f"    n = {n:3d}: cl {r['CL']:.4f} ({r['CL'] / lim['CL'] - 1:+.2%})"
              f", depth {r['depth']:.5f}")


def case_C2():
    header("C2", "the stiff limit: the flying shape tends to the rigid one")
    U, q = 10.0, 0.5 * sf.RHO_AIR * 100.0
    pts0 = fsi.sail_planform(2.0, 1.0, 1.0, nchord=8, nspan=4, alpha_deg=6.0)
    rigid = sf.solve_fluid(pts0, sf.sail_onset(U, 2.0), U, deck=True)
    print(f"  membrane wing, AR_eff 4, rigid flat CL = {rigid['CL']:.5f}")
    print("  E t / q c     CL       CL - CL_rigid   root depth   "
          "depth x E t / q c")
    for Et in (6e3, 6e4, 6e5, 6e6):
        res = membrane_wing(2.0, Et=Et)
        d = fsi.section_shape(res["points"], 0)["depth"]
        print(f"  {Et / q:9.0f}   {res['fluid']['CL']:.5f}   "
              f"{res['fluid']['CL'] - rigid['CL']:+.3e}    {d:.3e}    "
              f"{d * Et / q:8.3f}")


BATTEN_EI = 50.0        # N m^2, three full-length battens at h/4, h/2, 3h/4


def mainsail(nc=8, ns=12, leech_e0=0.01, camber=0.08, battens=True, **kw):
    """The verification mainsail: luff 10 m, foot 3.5 m, headboard 0.5 m,
    moulded depth 8%, sheeted to 10 deg, luff, boom and headboard pinned."""
    pts0 = fsi.sail_planform(10.0, 3.5, 0.5, nchord=nc, nspan=ns,
                             camber=camber, alpha_deg=10.0)
    bat = [(ns * q // 4, BATTEN_EI) for q in (1, 2, 3)] if battens else []
    model = fsi.build_sail(pts0, 5e5, prestrain=0.002, foot="pinned",
                           head="pinned", leech_EA=5e4,
                           cable_prestrain=leech_e0, battens=bat)
    return pts0, model


def summary(res, pts0=None):
    rows = fsi.flying_shape_report(res)
    ns = len(rows) - 1
    if ns % 4:
        raise ValueError("the mainsail summary needs nspan divisible by 4")
    j3 = 3 * ns // 4
    fl = res["fluid"]
    return {"CL": fl["CL"], "CD": fl["CD"], "ce": fl["ce_height"],
            "twist": rows[0]["incidence_deg"] - rows[j3]["incidence_deg"],
            "depth_mid": rows[ns // 2]["depth"],
            "depth_max": max(r["depth"] for r in rows[1:-1])}


def case_C3():
    header("C3", "mainsail: the leech tension sets the twist")
    U = 6.0
    on = sf.sail_onset(U, 10.0)
    pts0, _ = mainsail()
    rigid = sf.solve_fluid(pts0, on, U)
    print("  luff 10 m, foot 3.5 m, headboard 0.5 m, moulded depth 8%, "
          "alpha 10 deg, U = 6 m/s, 8 x 12,")
    print(f"  E t = 5e5 N/m, e0 = 0.002, leech line EA = 5e4 N, battens EI = "
          f"{BATTEN_EI:g} N m^2 at h/4, h/2, 3h/4, frozen wake")
    print(f"  moulded shape held rigid: CL {rigid['CL']:.4f}, CD "
          f"{rigid['CD']:.4f}, CE {rigid['ce_height']:.3f} m")
    print("  battens  leech e0  tension  coupling  CL      CD      CE [m]  "
          "twist to 3h/4  depth at h/2  taut")
    for bat, le0 in ((True, 0.0), (True, 0.005), (True, 0.01), (True, 0.02),
                     (False, 0.01)):
        pts0, model = mainsail(leech_e0=le0, battens=bat)
        res = fsi.static_aeroelastic(model, pts0, on, U)
        s = summary(res)
        cab = model["cables"][0]
        N = sm.cable_forces(cab, res["points"].reshape(-1, 3), False)[2]
        print(f"  {'yes' if bat else 'no ':7s}  {le0:7.3f}  {N.max():5.0f} N  "
              f"{res['iterations']:5d}     {s['CL']:.4f}  {s['CD']:.4f}  "
              f"{s['ce']:.3f}   {s['twist']:6.2f} deg     {s['depth_mid']:.4f}"
              f"       {fsi.cloth_state(res)[0]:.0%}")


def case_C4():
    header("C4", "coupling accelerators on the battened mainsail")
    U = 6.0
    on = sf.sail_onset(U, 10.0)
    print("  accelerator   leech e0   iterations   final residual   CL")
    for le0 in (0.0, 0.01):
        for acc, om in (("iqn", 0.5), ("aitken", 0.5), ("constant", 0.5)):
            pts0, model = mainsail(leech_e0=le0)
            try:
                res = fsi.static_aeroelastic(model, pts0, on, U, accel=acc,
                                             omega=om, max_iter=40)
                txt = (f"{res['iterations']:6d}{'' if res['converged'] else '*'}"
                       f"        {res['history']['residual'][-1]:.1e}       "
                       f"{res['fluid']['CL']:.4f}")
            except RuntimeError as exc:
                txt = f"  failed: {exc}"
            print(f"  {acc:12s} {le0:8.3f}   {txt}")
    print("  (* not converged in 40 iterations)")


def case_C5():
    header("C5", "mainsail mesh convergence, leech e0 = 0.01, frozen wake")
    U = 6.0
    on = sf.sail_onset(U, 10.0)
    print("  battens  mesh     CL rigid  CL      CL / CL rigid  twist to 3h/4"
          "  depth at h/2  draft   time")
    for bat, meshes in ((True, ((8, 12), (16, 12), (24, 12), (8, 24),
                                (16, 24))),
                        (False, ((8, 12), (16, 12)))):
        for nc, ns in meshes:
            t = time.time()
            pts0, model = mainsail(nc, ns, battens=bat)
            rigid = sf.solve_fluid(pts0, on, U)["CL"]
            res = fsi.static_aeroelastic(model, pts0, on, U)
            s = summary(res)
            row = fsi.flying_shape_report(res)[ns // 2]
            print(f"  {'yes' if bat else 'no ':7s}  {nc:2d} x {ns:2d}  "
                  f"{rigid:.4f}    {s['CL']:.4f}  {s['CL'] / rigid:.4f}         "
                  f"{s['twist']:5.2f} deg      {row['depth']:.4f}       "
                  f"{row['draft']:.3f}  {time.time() - t:5.1f} s")


def case_C6():
    header("C6", "battened mainsail, free against frozen wake, leech e0 = 0.01")
    U = 6.0
    on = sf.sail_onset(U, 10.0)
    print("  wake     coupling   CL      CD      twist to 3h/4   time")
    for wake in ("frozen", "free"):
        t = time.time()
        pts0, model = mainsail()
        res = fsi.static_aeroelastic(model, pts0, on, U, wake=wake, tol=1e-3)
        s = summary(res)
        print(f"  {wake:7s} {res['iterations']:5d}      {s['CL']:.4f}  "
              f"{s['CD']:.4f}  {s['twist']:6.2f} deg      "
              f"{time.time() - t:6.1f} s")


def split_wing(nc, ns, sweeps=10):
    """A rectangular wing solved whole, and as two halves that see each
    other through sail_fluid.induced_field. Returns (full, (CL, CD) per
    Gauss-Seidel sweep)."""
    pts = pw.pitch_mesh(msi.rectangular_mesh(nchord=nc, nspan=ns),
                        np.radians(5.0))
    base = sf.sail_onset(1.0, 1.0)
    full = sf.solve_fluid(pts, base, 1.0, rho=1.0)
    h = ns // 2
    halves = (pts[:, :h + 1].copy(), pts[:, h:].copy())
    field, sols, hist = [None, None], [None, None], []
    for _ in range(sweeps):
        for k in (0, 1):
            sols[k] = sf.solve_fluid(halves[k], base, 1.0, rho=1.0,
                                     extra=field[1 - k])
            field[k] = sf.induced_field(sols[k])
        F = sols[0]["F"] + sols[1]["F"]
        q_area = 0.5 * (sols[0]["area"] + sols[1]["area"])
        hist.append((F[2] / q_area, F[0] / q_area))
    return full, hist


def case_C7():
    header("C7", "two surfaces that see each other: a wing split in two")
    print("  rectangular wing, AR 8, alpha 5 deg, frozen wake; the halves are"
          " solved in turn,")
    print("  each with the other's induced field in its boundary condition and"
          " loads")
    print("  mesh     CL whole   CL halves  CL error   CDi error  "
          "last sweep change")
    first = None
    for nc, ns in ((4, 12), (4, 24), (8, 24)):
        full, hist = split_wing(nc, ns)
        first = first or hist
        cl, cd = hist[-1]
        print(f"  {nc} x {ns:2d}  {full['CL']:.6f}   {cl:.6f}   "
              f"{cl / full['CL'] - 1:+.1e}   {cd / full['CD'] - 1:+.1e}    "
              f"{abs(hist[-1][0] - hist[-2][0]):.1e}")
    print("  Gauss-Seidel history of CL on 4 x 12: "
          + ", ".join(f"{c:.6f}" for c, _ in first[:5]))


CASES = {"S1": case_S1, "S2": case_S2, "S3": case_S3, "S4": case_S4,
         "S5": case_S5,
         "F1": case_F1, "C1": case_C1, "C2": case_C2, "C3": case_C3,
         "C4": case_C4, "C5": case_C5, "C6": case_C6, "C7": case_C7}
QUICK = ("S1", "S2", "S3", "S4", "S5", "F1", "C2", "C3", "C4", "C7")


def main():
    args = sys.argv[1:]
    if "--case" in args:
        names = [args[args.index("--case") + 1].upper()]
    elif "--quick" in args:
        names = list(QUICK)
    else:
        names = list(CASES)
    t0 = time.time()
    for name in names:
        CASES[name]()
    print(f"\n{len(names)} case(s) in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
