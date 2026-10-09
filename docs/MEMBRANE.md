# The membrane: formulation, conventions and structural verification

This document owns the structural model of `sail_membrane.py`: its kinematics,
its constitutive law with wrinkling, its cables, battens and pressure, its
solver, and the measured results of the structural verification cases S1 to S5
of `verify_sail.py`.

## Formulation

Sailcloth carries in-plane tension and no bending. It is discretised with
three-node constant-strain triangles in a total Lagrangian formulation, so the
displacements and rotations may be large and the strains are taken as small
only in the constitutive law. Each triangle has a local orthonormal frame in its
reference configuration, so the reference surface may be curved (a moulded sail)
as well as flat. The deformation gradient is the 3 × 2 matrix
F = sum over nodes of x_a ⊗ grad N_a, and the Green–Lagrange strain is

    E = (F^T F - I) / 2 + e0 I,                                         (1)

where e0 is the pre-strain. A positive e0 is cloth cut smaller than the
reference geometry by the stretch (1 + 2 e0)^1/2, which is how tension is put
into the sail. The second Piola–Kirchhoff stress S is per unit reference width
[N m^-1], so the only material constant with units is the membrane stiffness
E t [N m^-1]. The internal force is the integral of B^T S over the reference
area, never K u, and the tangent is the material part B^T D B plus the
geometric part (grad N_a · S · grad N_b) I.

### Wrinkling

A membrane cannot carry compression. The stress is the relaxed-energy
tension-field law in its mixed stress-strain form:

| state | criterion | stress |
| --- | --- | --- |
| taut | smaller principal stress of the plane-stress law is positive | Saint Venant–Kirchhoff, plane stress |
| slack | larger principal strain e1 is not positive | zero |
| wrinkled | otherwise | E t e1 n1 ⊗ n1, along the major principal strain direction n1 |

The wrinkled stress is uniaxial: the cloth contracts freely across the wrinkles.
For an isotropic cloth it joins the taut law continuously at the boundary, where
e2 = -nu e1 and both give E t e1. Its tangent is exact,

    D = E t (a a^T + 2 e1 / (e1 - e2) b b^T),                           (2)

where a and b are the Voigt forms of n1 ⊗ n1 and of the symmetric part of
n1 ⊗ n2. The first term is the variation of the eigenvalue, and the second is the
rotation of the wrinkle direction. The strain energy density is S:E/2 in all
three states.

### Cables and pressure

Edge cables (leech, foot and head lines) are two-node bars with the same
Green–Lagrange kinematics and their own pre-strain, tension-only by default. The
pre-strain of a leech line is how sheet tension enters the model. A uniform
pressure may be applied as a follower load, each node taking a third of p times
the current area vector of every triangle, with its consistent non-symmetric
stiffness. The structural verification uses it. The aerodynamic load does not:
it arrives as nodal force vectors from the fluid (docs/COUPLING.md).

### Battens

A full-length batten is a line of rotation-free bending hinges along a
chordwise grid line j (`sail_fsi.batten_hinges`). At each interior station i,
the hinge measures the signed angle theta between two virtual triangles,
(a, b, c) and (b, a, d). Here a = (i, j); c and d are its chordwise neighbours
on the batten; and b is its spanwise neighbour, which fixes the bending axis.
The angle is the turn of the batten at a, theta = kappa h, where h is the mean
of the two adjacent panel chords. The energy

    W = k (theta - theta0)^2 / 2,   k = EI / h,                          (3)

is therefore the Euler–Bernoulli energy EI kappa^2 h / 2 per station, and theta0
makes the moulded shape stress free. The angle is written with complex-safe
operations, so its gradient is a complex step exact to round-off. The Hessian
is the exact Gauss–Newton term k grad theta grad theta^T, plus a central
difference of the gradient scaled by theta - theta0.

The hinges deliberately do not use the triangles of the mesh. With either
diagonal pattern, the two mesh triangles that share a spanwise edge have their
third vertices on the same row, so such a hinge senses the curvature of one row
and leaves the other free. A first version built the battens that way, and the
battened strip of S5 deflected 2300 times the beam value. A cloth bending
stiffness on the mesh triangles, with the discrete-shell weights of Grinspun et
al. (2003), was also tried and removed. Its stiffness depended on the cell
aspect ratio and tended to half the intended value under chordwise refinement.

### Solver

`solve_static` is a Newton iteration with optional load stepping. Under dead
loads, equilibrium is a minimum of the total potential. Each step solves
(K + mu d I) du = -R with d the mean diagonal of K, and mu raised from zero until
the shifted tangent is positive definite. The step is accepted on an Armijo
decrease of the potential. With a follower pressure there is no potential, and
the step is accepted on a decrease of the residual norm instead.

Three decisions in the solver look like defects and are not:

- **The damping is required.** A flat, slack or wrinkled membrane has a singular
  tangent, so the undamped Newton step does not exist at the configurations a
  sail starts from.
- **The residual is measured against the larger of the first residual of the call
  and the applied load**, not against the load alone. A form-finding solve with
  pre-strain and no load would otherwise be judged against zero.
- **A residual that stops falling below 1e-6 of that scale is accepted and flagged
  `stalled`.** An element on the boundary between the taut and wrinkled states
  switches its tangent from one step to the next. The residual then floors near
  1e-8, two decades above the 1e-9 target, and not at the round-off of the force
  integral. Without the flag, a converged mainsail is reported as a failure.

Newton converges quadratically once the element states settle. Reaching that
point takes 20 to 100 iterations from a moulded shape under its full
aerodynamic load, while elements change state. With a warm start inside the
coupling it takes one to 20 iterations. Load stepping does not shorten this
phase and is off by default.

Small residual stiffness in the wrinkled and slack states, the usual
regularisation, was tried and removed. With a fraction 1e-4, 1e-3 or 1e-2 of the
elastic law blended in, a mainsail with a free head and no battens did not
converge within 400 iterations per load step, against 251 without the blend. The small compressive
stress the blend admits makes the geometric stiffness indefinite.

## Conventions

- The structural and fluid meshes are the same grid, so a node is
  `node_id(i, j, nspan) = i (nspan + 1) + j`, with i chordwise from the luff and
  j spanwise from the foot, exactly as `points[i, j]` of the fluid mesh.
- Each grid quadrilateral is split into two triangles, with the diagonal
  alternating, and every triangle is ordered so that its normal is +z for a grid
  in the x–y plane.
- `reactions` returns the force each support exerts on the sail. Reactions and
  applied loads sum to zero.

## Verification

Every number below is printed by `python3 verify_sail.py --case S1` to `S4`.

**S1, patch test and frame invariance.** A homogeneous stretch of a mesh whose
interior nodes are displaced randomly by up to 30% of the spacing reproduces
the exact principal stresses, and leaves the interior nodes in equilibrium, to
round-off.

| stretches | max stress error | max interior force / max element force |
| --- | --- | --- |
| 1.01, 1.005 | 1.2e-13 | 2.1e-13 |
| 1.2, 1.1 | 9.5e-15 | 1.3e-14 |
| 1.5, 1.3 | 3.6e-15 | 8.5e-15 |

A rigid rotation of 30, 90 and 170 degrees of a cambered, pre-strained and
randomly deformed mesh leaves the stress unchanged to within 6.2e-15 and
rotates the nodal forces to within 1.0e-14.

**S2, tangent consistency.** The assembled tangent of a cambered mesh, with a
pre-strained leech cable, a batten and a follower pressure, agrees with
central differences of the residual. The relative error is 2.7e-10, 4.7e-11
and 4.8e-11 at the three random states, which mix taut, wrinkled and slack
elements (seven, 15 and two triangles in the first state).

**S3, the wrinkling law.** For a uniaxial strain with lateral contraction beyond
Poisson's ratio, the stress is E t e1 along the major strain direction to
1.3e-16, at 0 and 35 degrees. A taut and a slack state are reproduced exactly.
A strain step of 2e-6 across the taut–wrinkled boundary moves the stress by
1.1e-4 of its value, i.e. by the modulus times the step: the law is continuous
there.

**S4, an inflated strip against the exact arc.** A strip pinned at x = 0 and
x = L, held in plane strain and inflated by a follower pressure p, takes the
shape of a circular arc. Its half-angle th follows from

    th / sin th · E t / (1 - nu^2) · ((lam^2 - 1) / 2 + (1 + nu) e0) = p L / (2 sin th),   (3)

where lam = th / sin th is the stretch. This tests large rotation, large
stretch, pre-strain and the follower load together. The sag error is measured on
the midspan row of nodes, and the thrust error on the support reaction. Both
converge at second order.

| p [Pa] | half-angle | nc = 8 | 16 | 32 | 64 | order (last pair) |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 8.0° | +3.5e-3 | +1.1e-3 | +3.0e-4 | +7.6e-5 | 1.96 |
| 50 | 26.3° | +5.8e-3 | +1.6e-3 | +4.0e-4 | +1.0e-4 | 1.99 |
| 500 | 58.0° | +5.9e-3 | +1.6e-3 | +4.1e-4 | +1.0e-4 | 1.99 |

The thrust error falls from 2.0e-3 to 2.3e-5 at the lowest load and from
6.6e-3 to 8.2e-5 at the highest. Newton takes seven to nine iterations from
the flat strip in every case. The arc evaluated at a single node, rather than
averaged over the row, converges irregularly. The alternating diagonal gives
nodes of valence four and eight, which take consistent loads of different
size.

**S5, battens against a beam in tension.** Two battens of EI = 20 N m^2 bound
one row of pre-strained cloth, 1 m long and 0.2 m wide. The strip is pinned at
both ends, held in plane strain and loaded by a follower pressure of 0.1 Pa.
The exact midspan deflection is that of a pinned beam of stiffness 2 EI under
the cloth tension N and the load q = p W,

    w = q / N (L^2 / 8 - (1 - 1 / cosh(k L / 2)) / k^2),   k^2 = N / 2 EI.   (4)

| cloth e0 | kL | nc = 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| 1e-4 | 0.027 | +1.25e-2 | +3.12e-3 | +7.81e-4 | +1.95e-4 |
| 1e-2 | 0.267 | +1.24e-2 | +3.10e-3 | +7.75e-4 | +1.94e-4 |

The error is second order, and identical to three digits for the alternating
and the single diagonal pattern.

## Open items

- **The cloth has no bending stiffness.** For sailcloth, sqrt(D / T) is a few
  millimetres, so no practical mesh would resolve it. Its absence leaves a free
  leech without a length scale: without battens, the leech hooks within one
  panel, and the mainsail's CL grows by 12% from 8 to 16 chordwise panels
  (docs/COUPLING.md, case C5). Battens bound this. A bending element on the
  mesh that is isotropic for any cell aspect ratio is the open part.
- **Battens follow grid lines and bend about the spanwise axis only.** Batten
  torsion, partial-length battens and batten pockets are not modelled.
- **The cloth is isotropic.** Woven and laminated sailcloth is orthotropic, and
  its wrinkling criterion then differs. The mixed criterion above is exact for
  the isotropic law only.
- **The operators are dense.** A 16 × 24 sail has 1275 degrees of freedom, which
  is comfortable, but the cost grows with the cube of the node count.
