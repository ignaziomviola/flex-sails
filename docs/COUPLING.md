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

## Several sails

The vendored lifting surface takes one structured surface, so each sail is
meshed and solved on its own. Sails nonetheless see one another.
`sail_fluid.induced_field` turns a solved sail into a velocity field, which
`solve_fluid` and `static_aeroelastic` accept as `extra`. The field enters
the boundary condition and the loads, and it leaves the frozen wake on the
streamlines of the onset. `sail_plan.solve_sail_plan` solves every sail's
flying shape in turn, with the latest fields of the others, until no CL
changes by more than 1e-4. With frozen wakes this fixed point is the joint
solution of all the lifting surfaces.

Two details decide whether the fixed point is right:

- **The other sail enters the boundary condition with the cores of the
  vendored influence matrix, and the loads with those of its load
  evaluation.** The vendored code uses a near-zero core on every segment of
  the matrix, wake included, and the wake core only for loads. A first version
  used the wake core in both. The halves of a split wing then each saw their
  own tip vortices at full strength and the other half's softened, and the
  split wing came out 2.7% low in lift, independently of the mesh.
- **The other sail does not deflect the wake.** The field reaches the vendored
  `solve` only through its one onset call on the collocation array, recognised
  by identity.

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

### C7, two surfaces that see each other

A rectangular wing of aspect ratio eight at 5 degrees is solved whole, and as
two halves that see each other through `induced_field`:

| mesh | CL whole | CL halves | CL error | CDi error | last sweep change |
| --- | --- | --- | --- | --- | --- |
| 4 × 12 | 0.418125 | 0.418125 | -9.2e-11 | +1.3e-10 | 3.9e-10 |
| 4 × 24 | 0.409413 | 0.409413 | -3.4e-8 | +4.6e-8 | 6.0e-8 |
| 8 × 24 | 0.409498 | 0.409497 | -3.3e-8 | +4.4e-8 | 5.8e-8 |

The error equals the change over the last Gauss–Seidel sweep, so the two
halves reproduce the whole wing to the iteration tolerance. The first five
sweeps on 4 × 12 give CL = 0.372522, 0.413774, 0.417690, 0.418081 and
0.418120.

## Application: a J/80 sail plan upwind

`python3 sail_plan.py` solves a 135% genoa and a mainsail on the rig of the
J/80 one-design, together, upwind in a 10-knot breeze. It prints the summary
below and writes Figures 1 to 3 to `docs/figures/`. Each run takes ca. 100 s.

### Data

| quantity | value | source |
| --- | --- | --- |
| I, J, P, E | 31.5, 9.5, 30.0, 12.5 ft (9.60, 2.90, 9.14, 3.81 m) | J/80 class measurements (Wikipedia, KeelIndex) |
| cloth | woven polyester, E = 873 MPa, t = 0.25 mm, nu = 0.4, so E t = 218 kN m^-1 | Blicblau et al. (2008), finite element analysis of a woven polyester sail |
| wind gradient | power law, exponent 0.11, true wind 10 kn at 10 m above the sea | exponent recommended over the sea by Hsu et al. (1994) |
| true wind angle, boat speed | 42 degrees, 5.5 kn | assumed upwind operating point |
| genoa | LP 135% of J, luff on the forestay from 0.15 m to 0.3 m below the hounds, head 0.15 m; moulded depth 13% at 40% of the chord; sheeted at 10 degrees at the foot and 18 degrees at the head | assumed design and trim |
| mainsail | luff on the mast from a boom 0.9 m above the sheer, head 0.15 m; moulded depth 11% at 45%; sheeted at 2 and 10 degrees; three full battens of EI = 50 N m^2 | assumed design and trim |
| supports | genoa: luff, tack, clew and head pinned, foot and leech free with tapes of EA = 1e5 N at 1% pre-strain; mainsail: luff, boom and headboard pinned, leech free with the same tape | assumed |
| cloth pre-strain | 0.001 | assumed |
| mesh | 10 chordwise × 16 spanwise panels per sail, frozen wakes | |

The apparent wind is the true wind minus the boat velocity. At 5 m above the
sheer it is 7.18 m s^-1 at 26.7 degrees, at the boom 6.56 m s^-1 at
25.2 degrees, and at the masthead 7.52 m s^-1 at 27.4 degrees.

### Results

| | genoa | mainsail | plan |
| --- | --- | --- | --- |
| area [m^2] | 20.18 | 18.61 | 38.79 |
| CL | 2.139 | 0.704 | 1.450 |
| CD (inviscid) | 0.270 | 0.369 | 0.317 |
| aerodynamic force [N] | 1388 | 468 | 1833 |
| centre of effort above the sheer [m] | 4.41 | 5.03 | 4.56 |
| largest displacement from the moulded shape [mm] | 166 | 93 | |
| median sigma_1 [MPa] | 2.05 | 1.62 | |
| 95th percentile of sigma_1 [MPa] | 4.85 | 2.93 | |
| largest sigma_1 [MPa] | 15.1 (clew) | 10.3 | |
| cloth taut / wrinkled / slack | 47 / 52 / 1% | 42 / 58 / 0% | |
| leech tape tension [N] | 837 to 1649 | 938 to 1285 | |

The coefficients use each sail's own area and the apparent wind at 5 m. The
plan iteration converges in nine sweeps. Alone, the genoa has CL = 1.760 and
the mainsail CL = 1.956. Together, the mainsail's upwash raises the genoa to
2.139, and the genoa's downwash lowers the mainsail to 0.704. This is the
classical interaction of the two sails, which modelling them one at a time
would miss. Under load, the genoa
deepens from 13% to 16% at mid-height and 19% at three quarters, with the draft
moving aft from 40% to 62% and 66%. The mainsail deepens from 11% to 12%.

Figure 1 (`j80_displacement.png`): displacement magnitude from the moulded
shape (grey wireframe) on the flying shape, for (a) the genoa and (b) the
mainsail. Figure 2 (`j80_stress.png`): the larger principal stress sigma_1 on
the flying shape, with its stress trajectories, the black curves tangent
everywhere to sigma_1, for (a) the genoa and (b) the mainsail. The colour
saturates at the 99th percentile, 9.0 MPa.
Figure 3 (`j80_sections.png`): moulded (grey) and flying (black) sections at
the foot and at a quarter, half and three quarters of the height, in boat
axes, for (a) the genoa and (b) the mainsail.

In Figure 2 the trajectories fan out from the genoa clew and run up its
leech, and they cross the mainsail diagonally from the clew to the luff.

The trajectories (`sail_plan.stress_trajectories`) are integrated on the
sail's own grid coordinates. The Cauchy tension tensor is averaged to the
nodes in Cartesian components, which depend on no surface basis. At each
point it is interpolated bilinearly and projected on the tangent plane of the
bilinear surface through the nodes, where the principal direction solves
a^T N a p = lambda a^T a p. Working with the tensor rather than with
direction vectors removes the sign ambiguity of a principal direction. The
lines are integrated by the midpoint rule, each step cut at the cell edge
where the bilinear surface has a kink, and spaced evenly after Jobard and
Lefer (1997), with seeds taken in order of decreasing tension. Two tests
check them:
- a uniform stretch at 30 degrees on a grid with nodes displaced by up to
  25% of the spacing gives straight lines at 30 degrees, drifting sideways
  by 1.9 mm over 2.4 m. The drift falls from 5.0 to 1.9 to 0.31 mm as the
  step goes from 75 to 38 to 15 mm, i.e. at second order. Without the cut
  at cell edges it was 27 mm and fell at first order;
- on the inflated strip of S4, the lines follow the arc to its full sag of
  0.117 m and stay within 0.1 mm of constant span.

### Mesh dependence

| mesh per sail | plan CL | genoa CL | genoa max \|u\| [mm] | genoa median / 95th sigma_1 [MPa] | main CL | main max \|u\| [mm] | main median / 95th sigma_1 [MPa] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 × 16 | 1.431 | 2.104 | 153 | 1.98 / 4.75 | 0.702 | 86 | 1.58 / 2.80 |
| 10 × 16 | 1.450 | 2.139 | 166 | 2.05 / 4.85 | 0.704 | 93 | 1.62 / 2.93 |
| 16 × 16 | 1.481 | 2.195 | 180 | 2.07 / 5.06 | 0.706 | 98 | 1.66 / 2.88 |
| 10 × 24 | 1.487 | 2.184 | 168 | 2.12 / 4.63 | 0.732 | 90 | 1.64 / 2.87 |

From 8 to 16 chordwise panels, the median and the 95th percentile of the
stress change by up to 6%, and the mainsail's lift by 0.6%. The genoa's lift
grows by 4.3%, about half of which is the chordwise discretisation of the
lifting surface (C5). The largest displacement grows by 8 to 10% per
refinement. From 16 to 24 spanwise panels, the lift changes by 2.1% on the
genoa and 4.0% on the mainsail, the largest displacement by 1 to 3%, and the
stress percentiles by up to 5%. The lift of the plan is therefore known to
ca. 4%, its stresses to ca. 6%, and its largest displacements to ca. 10%. The
genoa has no battens, so its free leech hooks within one panel (C5 and
docs/MEMBRANE.md). The largest stresses sit at the pinned corners and grow
with refinement, as at any re-entrant corner of a membrane. They are reported
but are not converged numbers.

### What this case is not

- **The flow is inviscid and attached.** Over its lower half, the genoa's
  chord is at 14 to 17 degrees to the apparent wind, where a real genoa's leech flow
  separates. Its CL of 2.14 and the plan's CD of 0.32 are inviscid values, not
  predictions of measured coefficients.
- **There is no hull, deck or sea surface.** The feet see free air, which
  raises the induced drag.
- **The heel, the mast and its bend, the forestay sag and the cloth's
  orthotropy are absent.**
- **Nothing here is validated against a measured sail plan.** The input data
  are from the literature. The trim, the moulded shapes and the tape tensions
  are stated assumptions.

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
