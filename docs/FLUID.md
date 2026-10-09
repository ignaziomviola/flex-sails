# The vendored fluid solver: provenance and how to check it

The lifting-surface method in this repository is not written here. It is a
verbatim copy of

> https://github.com/ignaziomviola/free-wake-lifting-surface
> commit `07fc7003dc98dbc4142546b52cbcced24a55c84e` (9 August 2026)

and it is not modified here in any way. The upstream repository owns the fluid
physics and its verification. This repository owns the sail adapter
(`sail_fluid.py`), the membrane, the coupling and the two-dimensional sail
theory.

| file | role |
| --- | --- |
| `panel_wing.py` | the thin lifting surface: horseshoe lattice on a mesh, free-wake relaxation, vector Kutta–Joukowski loads |
| `make_sample_inputs.py` | upstream mesh and onset generators; the rectangular wing is reused by case F1 and the tests |
| `docs/fluid/README.md` | the upstream usage guide |
| `docs/fluid/HANDOVER.md` | the upstream method, pitfalls, constants and verification |

`docs/fluid/INDEX.md` is written here and is not vendored. It records how to
read the two upstream documents from this repository. `vortex_lattice_wing.py`
is not vendored because nothing here uses it.

## How the adapter reaches the vendored code

Only through its public functions, `build_lattice`, `solve`, `all_segments`,
`induced_velocity` and `pitch_mesh`, and through two module constants that
`sail_fluid.solve_fluid` sets for the duration of a call and restores:

- `MAX_ITER` is set to zero for a frozen-wake solve. `solve` then assembles once
  on the initial wake, which follows the onset streamlines from the trailing edge
  and the tips.
- stdout is redirected, because `solve` prints each wake iteration.

`panel_wing.py` imports `matplotlib.pyplot` at module level, so matplotlib is a
hard dependency of the solver and not only of the figures. `pyflakes` reports
one unused local in it (line 360), which stays because the file is not edited
here.

## Checking it

`python3 test_vendored.py` recomputes the SHA-256 of the four vendored files
against `fluid_manifest.txt`. To take an upstream change: fix it upstream,
copy the files, run `python3 test_vendored.py --write`, and update the commit
above and in `test_vendored.py` in the same commit.
