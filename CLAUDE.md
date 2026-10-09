# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this repository is

A static aeroelastic model of a flexible sail. The fluid is the free-wake thin
lifting surface of https://github.com/ignaziomviola/free-wake-lifting-surface,
carried in **verbatim and never modified here**. The structure is a
geometrically nonlinear membrane with wrinkling, written here. The coupling,
the sail adapter over the fluid and an independent two-dimensional sail theory
are also written here. The sibling repository
https://github.com/Vortex-Interaction-Laboratory/bem-fem-fsi couples the thick
panel method with a solid finite element model; its coupling accelerator is
reused here.

## The vendored fluid code is not maintained here

`panel_wing.py`, `make_sample_inputs.py`, `docs/fluid/README.md` and
`docs/fluid/HANDOVER.md` are byte-for-byte copies, checked by
`test_vendored.py` against `fluid_manifest.txt`. Never fix them here: fix
upstream, re-vendor, run `python3 test_vendored.py --write`, and update the
commit in `docs/FLUID.md` and `test_vendored.py` in the same commit.

## Authoritative documents

`docs/MEMBRANE.md` owns the structural formulation, the solver and the S cases.
`docs/COUPLING.md` owns the sail frame, the onset, the deck image, the
transfer, the coupling, the interaction of several sails, the sail
configurations, the F and C cases and the J/80 application.
`docs/FLUID.md` owns provenance. Their tables hold measured numbers printed by
`verify_sail.py`: update them when behaviour changes.

## Commands

```bash
python3 test_membrane.py          # 21 tests, ca. 2 s
python3 test_sail.py              # 22 tests, ca. 4 s
python3 test_vendored.py          # 4 tests: the fluid copies are unmodified
python3 -m unittest test_sail.TestCoupling -v

python3 verify_sail.py --quick    # S1-S5, F1, C2-C4, C7, ca. 6 min
python3 verify_sail.py --case C1  # one case
python3 verify_sail.py            # all thirteen, ca. 15 min (C6 is the free wake)

MPLBACKEND=Agg python3 sail_plan.py   # J/80 genoa and mainsail, ca. 2 min

python3 sail_2d.py                # 2D sail theory table
printf "\n\n\n\n\n\n\n\n\n\n\n\n\n\n" | MPLBACKEND=Agg python3 sail_fsi.py
```

The tests use the standard library `unittest`, and pytest is not used. numpy
and matplotlib are the only dependencies, and scipy is deliberately absent.
`pyflakes` is clean over every file written here.

## Architecture

```
panel_wing (vendored)  <-  sail_fluid  <-\
                                          sail_fsi  <-  verify_sail
sail_membrane  <-------------------------/
sail_plan  <-  sail_fsi, sail_fluid, sail_membrane   (several sails, J/80)
sail_2d  (imports nothing from here: the independent reference)
```

`sail_fluid` is the only module that imports the vendored code, and
`sail_fsi` the only one that imports both sides. Every block is a pure
function over explicit arguments: `build_model` returns a model dict, and
`solve_static` and `static_aeroelastic` return result dicts and mutate
nothing. Prompts, prints and figures live only under `main()` and in
`verify_sail.py`. `sail_plan.main()` is the one place that writes files, and
it writes only under `docs/figures/`.

The structural and fluid meshes are the same grid of nodes, `points[i, j]`
with i chordwise from the luff and j spanwise from the foot. The transfer is
therefore exact lumping, not interpolation.

## Deliberate decisions that look like defects

- **The cloth receives only the normal component of each panel force.** The
  in-plane rest is the leading-edge suction, which the luff support carries in a
  sail. It is about a tenth of the force on the mainsail, and `suction()` reports
  it. `load="kj"` passes the full vector.
- **The Newton step is damped and accepted on the potential.** A slack, flat or
  wrinkled membrane has a singular tangent.
- **A structural residual that stalls below 1e-6 is accepted and flagged.**
  Elements on the taut–wrinkled boundary floor it near 1e-8.
- **A failed structural solve is retried from the last equilibrium, then from
  the moulded shape.** An accelerated iterate need not be a configuration the
  cloth reaches smoothly.
- **The mainsail cases pin the headboard and carry three battens.** A free,
  unbattened head twists off by tens of degrees. A pure membrane's free leech
  has no length scale, so without battens it hooks within one panel, and CL
  grows by 12% from 8 to 16 chordwise panels. With battens, the ratio of
  flying to rigid CL changes by 0.6% (docs/COUPLING.md, case C5).
- **Battens are hinges built along the batten line, not on the mesh
  triangles.** The two mesh triangles that share a spanwise edge have their
  third vertices on the same row, so a hinge on them senses only one row. A
  cloth bending stiffness on the mesh triangles was removed for the same
  family of reasons: its stiffness depended on the cell aspect ratio.
- **`solve_fluid` and `static_aeroelastic` both default to a frozen wake.** A
  rigid reference computed with one wake and a coupled result with the other
  differ by more than the effect being measured: 1.3% in CL for the membrane
  wing of case C2.
- **Another sail enters through `extra`, not through the onset.** It reaches
  the boundary condition with the influence-matrix cores (CORE_WING on every
  segment) and the loads with the load cores (the wake core on the wake). It
  does not deflect the frozen wake. Using the wake core in the boundary
  condition left a split wing 2.7% low in lift on every mesh; with the
  matching cores it reproduces the whole wing to 1e-10 (case C7).
- **The other sail reaches the vendored `solve` by identity.** `solve`
  evaluates the onset once on `lat["colloc"]`, and the wrapper adds `extra`
  only when it receives that array object.
- **The deck image is mesh doubling, as in bem-fem-fsi.** It is exact for an
  onset even in height, which `sail_onset` guarantees by using |y|.

## Expected results that are not errors

- The 2D sail theory converges at first order in the panel count, not second.
  The load is singular at the leading edge.
- With 8 chordwise panels, the lift is 5.3% below the continuous 2D limit. The
  3D model is compared with 2D theory on the same chordwise panels, and the
  discretisation error is reported separately.
- Leech tension above the moulded state can deepen the sail as well as reduce
  its twist, so CL can exceed that of the moulded shape held rigid.
