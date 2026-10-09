# The sail: aerodynamics, transfer, coupling and coupled verification

This document owns the sail frame and the onset flow, the deck image, the
panel loads and their transfer to the cloth, the coupling and its
accelerators, the supported sail configurations, and the measured results of
cases F1 and C1 to C6 of `verify_sail.py`.

## Frame and onset

The vendored lifting surface takes the span along y, the lift along +z and the
onset nominally along +x. A sail is set up in that frame: the mast along y, the
foot at y = 0, the apparent wind at the reference height along +x, and the
leeward side along +z. Lift is the force along +z and drag the force along +x.
The heeling moment is taken about the x axis through the foot, and the centre
of effort is its ratio to the lift.

`sail_fluid.sail_onset` gives the apparent wind a speed and a direction that
vary with height,

    u(y) = u_ref ((|y| + y0) / (y_ref + y0))^s,   beta(y) = beta_top |y| / y_twist,   (1)

where u_ref is the speed at the reference height y_ref; y0 is the height of the
foot above the water; s is the shear exponent; and beta is the rotation of the
wind about the mast towards +z, so that a positive twist raises the incidence
aloft. The onset is even in y, which the deck image requires. The vendored
`make_onset` varies the speed with z only, and `panel_wing.solve` accepts any
onset callable, so nothing upstream is changed.

## The deck image

A sail whose foot seals on the deck sees its image in the deck plane. The
image is realised by mesh doubling: the mesh is mirrored in y = 0 and joined at
the foot, and the loads are taken from the real half only. The doubling is
exact for an onset even in y. Case F1 checks it: a half wing on a deck gives
the CL of the whole wing to twelve digits. A gap between the boom and the deck
cannot be represented, because the vendored mesh is a single structured
surface.

## Panel loads and transfer

`panel_wing.compute_loads` returns totals only. `sail_fluid.panel_forces`
recomputes the vector Kutta–Joukowski force of each panel at the same bound
vortex midpoints, with the same total velocity and the same regularised
kernel, so that the panel forces sum to the vendored coefficients. Case F1
shows the two agree to the last digit printed, for a frozen and for a free
wake.

The structural mesh is the fluid mesh: node (i, j) of the cloth is
`points[i, j]` of the lifting surface. Each panel force acts at the midpoint of
its bound vortex, a quarter of the way down the panel and halfway across it. It
is lumped to the four corners by the bilinear weights of that point, 0.375 to
each forward corner and 0.125 to each aft corner. These weights are a partition
of unity that reproduces the point, so force and moment are conserved exactly.
F1 measures defects of 4.6e-16 in force and 3.8e-16 in moment.

**The cloth receives only the component of each panel force normal to the
panel.** That component is the pressure jump times the area. The in-plane
remainder is the leading-edge suction, which the luff support carries in a
sail. On the mainsail in a sheared and twisted onset, it is 47.6 N of a
493.4 N aerodynamic force (frozen wake). `sail_fluid.suction` reports it, and
`load="kj"` passes the full vector instead.

## Coupling

`sail_fsi.static_aeroelastic` iterates on the structural displacement u.
1. Solve the fluid on X + u.
2. Lump the forces f(u) to the nodes.
3. Solve the structure under f(u), warm-started from u, for u~.
4. Pass the residual r = u~ - u to the accelerator for the next iterate.

Convergence is ||r|| / ||u~|| below 1e-4 by default. If the structural solve
fails from an accelerated iterate, it is retried from the last equilibrium and
then from the moulded shape. The accelerator is that of bem-fem-fsi: IQN-ILS
by default, Aitken's Delta^2 process, or constant relaxation. The fluid and
the coupling both default to a frozen wake. A rigid reference computed with
one wake and a coupled result with the other would differ by more than many of
the effects measured below.

## Sail configurations

`sail_fsi.build_sail` always pins the luff to a rigid mast or forestay. The
rest is chosen per edge:

| argument | options |
| --- | --- |
| `foot` | `"pinned"` (boom), `"clew"` (loose foot, clew pinned), `"free"`; with `deck`, an unpinned foot slides on the deck |
| `leech` | `"free"`, or `"pinned"`, which is the membrane wing of two-dimensional sail theory |
| `head` | `"free"`, or `"pinned"`, a headboard fixed at the masthead |
| cables | `leech_EA`, `foot_EA`, `head_EA` on free edges, with `cable_prestrain`, which is how sheet tension enters |
| `battens` | a list of (j, EI) full-length battens along chordwise grid lines |

The verification mainsail has a luff of 10 m, a foot of 3.5 m and a headboard
of 0.5 m. Its moulded depth is 8% of the chord at mid-chord, and it is sheeted
to 10 degrees in a uniform apparent wind of 6 m s^-1. The cloth has
E t = 5e5 N m^-1 and a pre-strain of 0.002, with a leech line of EA = 5e4 N.
Three battens of EI = 50 N m^2 sit at a quarter, half and three quarters of
the height. The luff, the boom and the headboard are pinned. The headboard is
pinned because a free, unbattened head twists off by tens of degrees, and the
static problem then has no well-defined answer.

## Verification

### F1, panel loads, transfer and the deck image

Rectangular wing of aspect ratio eight, 4 × 12 panels, 5 degrees, density one:

| wake | CL from the panels | CL vendored | difference in CL and CDi | time |
| --- | --- | --- | --- | --- |
| frozen | 0.418125 | 0.418125 | 0 and 0 | 0.1 s |
| free | 0.419191 | 0.419191 | 0 and 0 | 11.5 s |

The half wing on a deck gives CL = 0.418124848357, equal to the full wing to
the twelve digits printed. The transfer defects on the mainsail are given
above.

### C1, membrane wing against two-dimensional sail theory

The independent reference is `sail_2d.py`: the linear sail theory of Thwaites
(1961) and Nielsen (1963) for a membrane pinned at its leading and trailing
edges. The cloth is extensible and pre-strained, and held in plane strain,
which is the mid-span state of a long wing pinned on its luff and leech. It
imports nothing from this repository. The three-dimensional case is a membrane
wing of chord 1 m, pinned on luff and leech, free at its head, with its foot
sliding on a deck. Its elasticity number is E t / q c = 97.96, with a
pre-strain of 0.002, an incidence of 6 degrees and 8 uniform chordwise panels
of 0.5 m span. The deck doubles the effective aspect ratio, and the root
section on the deck is the section furthest from a tip.

| effective aspect ratio | coupling iterations | CL | root cl | root depth | time |
| --- | --- | --- | --- | --- | --- |
| 4 | 9 | 0.9198 | 1.0504 | 0.07702 | 1.3 s |
| 8 | 8 | 1.1338 | 1.2819 | 0.08177 | 3.4 s |
| 16 | 8 | 1.2921 | 1.4156 | 0.08418 | 14.6 s |
| 32 | 8 | 1.3947 | 1.4802 | 0.08527 | 59.8 s |

The root values converge at the observed orders 1.05 (cl) and 1.16 (depth) in
the aspect ratio, consistent with the 1/AR of lifting-line theory. Their limits
are cl = 1.5406 and a depth of 0.08615. Two-dimensional theory on the same
8 uniform panels gives cl = 1.5333, a depth of 0.08547 and a tension
coefficient T / q c = 2.310. **The three-dimensional coupled model reaches
two-dimensional sail theory to within 0.47% in lift and 0.79% in depth.**

The chordwise discretisation is a separate error, common to both models. The
two-dimensional theory converges at first order in the panel count, because
the load is singular at the leading edge. Its continuous limit is cl = 1.6191,
a depth of 0.08631 and T / q c = 2.383 (observed order 1.01). With uniform
panels, the lift falls short of that limit by 5.30%, 2.56%, 1.26% and 0.63% for
8, 16, 32 and 64 panels.

### C2, the stiff limit

On the same membrane wing at an effective aspect ratio of four, the rigid flat
wing has CL = 0.40820.

| E t / q c | CL | CL - CL rigid | root depth | depth × E t / q c |
| --- | --- | --- | --- | --- |
| 98 | 0.91981 | +5.12e-1 | 7.70e-2 | 7.55 |
| 980 | 0.51933 | +1.11e-1 | 1.77e-2 | 17.33 |
| 9796 | 0.41773 | +9.53e-3 | 1.61e-3 | 15.79 |
| 97959 | 0.40912 | +9.22e-4 | 1.55e-4 | 15.22 |

The excess lift falls tenfold per decade of stiffness, and the depth times the
elasticity number tends to a constant: the linear aeroelastic limit.

### C3, the leech tension sets the twist

The verification mainsail on 8 × 12 panels has CL = 1.2319, CD = 0.0980 and its
centre of effort at 4.396 m when the moulded shape is held rigid. Twist is the
incidence at the foot minus that at three quarters of the height, which is the
top batten.

| battens | leech pre-strain | leech tension | iterations | CL | CD | CE [m] | twist | depth at h/2 | taut |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yes | 0 | 129 N | 5 | 1.0192 | 0.0711 | 4.231 | 3.60° | 0.0694 | 71% |
| yes | 0.005 | 376 N | 4 | 1.0838 | 0.0783 | 4.294 | 3.18° | 0.0744 | 70% |
| yes | 0.01 | 624 N | 4 | 1.1249 | 0.0834 | 4.323 | 2.73° | 0.0782 | 71% |
| yes | 0.02 | 1120 N | 4 | 1.1756 | 0.0900 | 4.355 | 2.11° | 0.0828 | 71% |
| no | 0.01 | 662 N | 6 | 1.2743 | 0.1029 | 4.434 | 3.08° | 0.0862 | 72% |

Tightening the leech from 129 to 1120 N reduces the twist from 3.60 to 2.11
degrees, raises the centre of effort by 0.12 m and the lift by 15%, and deepens
the mid-height section from 6.9% to 8.3%. Without battens, the same leech
tension gives a deeper, hooked leech and a CL above that of the moulded shape.
That configuration is not mesh-converged (C5).

### C4, accelerators

| accelerator | leech pre-strain 0 | leech pre-strain 0.01 |
| --- | --- | --- |
| IQN-ILS | 5 | 4 |
| Aitken | 6 | 5 |
| constant, 0.5 | 14 | 15 |

These are coupling iterations to a residual of 1e-4 on the battened mainsail.
All six runs converge to the same CL to four digits.

### C5, mesh convergence

Leech pre-strain 0.01, frozen wake. The rigid CL is that of the moulded shape
on the same mesh, so the ratio isolates the aeroelastic effect from the
discretisation error of the lifting surface.

| battens | mesh | CL rigid | CL | CL / CL rigid | twist to 3h/4 | depth at h/2 | draft at h/2 | time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yes | 8 × 12 | 1.2319 | 1.1249 | 0.9132 | 2.73° | 0.0782 | 0.523 | 5.0 s |
| yes | 16 × 12 | 1.2586 | 1.1563 | 0.9187 | 2.98° | 0.0772 | 0.526 | 18.4 s |
| yes | 24 × 12 | 1.2654 | 1.1678 | 0.9229 | 3.12° | 0.0765 | 0.527 | 129.3 s |
| yes | 8 × 24 | 1.2044 | 1.1025 | 0.9154 | 3.06° | 0.0783 | 0.524 | 7.2 s |
| yes | 16 × 24 | 1.2307 | 1.1359 | 0.9230 | 3.38° | 0.0772 | 0.527 | 57.4 s |
| no | 8 × 12 | 1.2319 | 1.2743 | 1.0344 | 3.08° | 0.0862 | 0.664 | 2.1 s |
| no | 16 × 12 | 1.2586 | 1.4234 | 1.1309 | 4.21° | 0.0837 | 0.754 | 15.5 s |

With battens, the aeroelastic loss of lift settles at ca. 8% of the rigid lift. The
ratio moves by 0.6% and 0.5% over the two chordwise refinements and by 0.2%
from 12 to 24 spanwise panels. The depth and draft at mid-height change in the
third decimal. The twist converges more slowly: it grows by 0.25 and 0.14
degrees over the chordwise refinements and by 0.33 degrees over the spanwise
one, so it is known to ca. 0.5 degrees on 8 × 12. The rigid CL itself spans
1.2044 to 1.2654 over these meshes, a range of 5%. That is the discretisation
error of the lifting surface, and it largely cancels in the ratio.

Without battens, the free leech hooks within one panel. The draft moves aft
from 0.66 to 0.75 of the chord, and the ratio grows from 1.03 to 1.13 when the
chordwise panels double. This is the absence of a length scale noted in
docs/MEMBRANE.md, not a defect of the solver, and it is why the mainsail
cases carry battens.

### C6, free against frozen wake

Battened mainsail on 8 × 12 panels, leech pre-strain 0.01, coupling residual
1e-3:

| wake | iterations | CL | CD | twist to 3h/4 | time |
| --- | --- | --- | --- | --- | --- |
| frozen | 4 | 1.1249 | 0.0834 | 2.73° | 4.6 s |
| free | 4 | 1.1786 | 0.0926 | 2.75° | 213.9 s |

The free wake raises CL by 4.8% and CD by 11%, and leaves the twist unchanged
to 0.02 degrees. It costs 46 times more.

## Open items

- **The flow is steady, inviscid and attached.** Separation behind the mast and
  at the leech is absent. So is the leading-edge suction's effect on the cloth,
  which the luff carries.
- **The free wake is too costly to couple routinely.** A fluid solve costs
  ca. 35 s for 96 panels, against 0.1 s for the frozen wake, and the vendored
  `solve` relaxes the wake from the onset streamlines on every call.
- **Battens follow grid lines, and the cloth has no bending stiffness of its
  own** (docs/MEMBRANE.md).
- **There is no independent reference for a three-dimensional flexible
  sail.** C1 checks the coupled model against two-dimensional theory, and C2
  against the rigid limit. The mainsail cases are self-consistency and
  convergence checks only.
