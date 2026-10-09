# Handover notes

This file records the state of the project, the decisions behind it, the numbers that verify it, and the pitfalls found along the way. Read it before extending the codes.

Last updated 9 August 2026.

This repository holds the two thin lifting-surface codes. A third code, extending the method to wings of finite thickness, lives in its own repository: https://github.com/ignaziomviola/free-wake-thick-panel-method. That code panels the closed wing surface with source and doublet panels under an internal Dirichlet condition, reuses the free-wake relaxation described below, and carries the shared Biot–Savart kernel as a copied module so that it clones and runs on its own; the copy must be kept in step with `panel_wing.py` if the kernel or its regularisation ever changes. The only trace of the split here is the guard in `panel_wing.load_mesh` rejecting the wrapped-surface mesh format and naming the other repository. The thin-surface numbers recorded below are the references that code is verified against, so keep them current.

## What exists

Two Python codes solve steady incompressible flow past a finite wing with a relaxed, force-free wake. Both impose flow tangency on a thin lifting surface and both relax the wake iteratively; they differ in the geometry and onset flow they admit.

| File | Geometry | Onset flow | Reports |
| --- | --- | --- | --- |
| `vortex_lattice_wing.py` | flat rectangular, set by aspect ratio | uniform | C_L, lifting-line estimate |
| `panel_wing.py` | generic mean camber surface from a mesh | sheared, from a profile | C_L, C_Di |
| `make_sample_inputs.py` | generates sample meshes and profiles | — | four input files |

Repository: https://github.com/ignaziomviola/free-wake-panel-method, public, branch `main`. Working copy at `~/Desktop/free-wake-panel-method/`.

## How the codes were built up

The work proceeded in four steps, each preserved in the current code:

1. A vortex-lattice method for a flat rectangular wing in uniform flow, prompting for aspect ratio, angle of attack α and panel counts, with horseshoe vortices at the panel quarter chord and flow tangency at the three-quarter chord.
2. Free-wake relaxation replacing the wake frozen along the free stream. The wake became filaments advected with the local velocity until aligned with the streamlines.
3. Tip filaments released along the side edge, one per chordwise station, anchored at the end of each bound segment. Before this change the tip legs were rigid on the wing surface and tip vorticity became free only at the trailing edge, so the flow at the tip stayed chordwise and no tip vortex developed over the chord.
4. Generalisation to an arbitrary mesh and a sheared onset flow, with loads from the vector Kutta–Joukowski theorem, which yields the induced drag along with the lift.

## Numerical method

Each panel carries a horseshoe vortex. The bound segment lies on the panel quarter-chord line; the legs follow the surface side edges to the trailing edge as polylines and become free filaments beyond it. Interior spanwise edges shed one shared filament each at the trailing edge. Both tip edges instead release one filament per chordwise station, directly from the bound-segment end.

Induced velocities follow from the Biot–Savart law with the regularisation of van Garrel, in which a finite core radius makes the velocity vanish smoothly on the filament axis. The core radius is a fifth of the spanwise panel spacing for wake filaments and `1e-6` for the influence coefficients, where the singularity is never approached.

One iteration assembles the influence matrix for the current wake, solves for the circulations Γ, evaluates the local velocity at every filament node, and marches each filament along the local streamline. Node positions are underrelaxed by a factor of 0.7, without which the tip filaments oscillate rather than converge.

Loads come from the vector Kutta–Joukowski theorem applied to the bound segments, using the total local velocity there. The self-induced contribution vanishes with the regularised kernel. Coefficients use the summed lattice area and, in `panel_wing.py`, the reference speed supplied by the user.

### Two pitfalls worth keeping

Convergence must be judged on the near field only, defined as within one span extent downstream of the wing. Far downstream the discrete filaments orbit the rolled-up vortex core indefinitely, so a global displacement criterion never converges even though the loads are settled.

Surface normals must come from the chordwise tangents of the camber surface interpolated to the collocation point, not from the flat panel corners. Flat-panel normals break the accuracy of the quarter-three-quarter rule on cambered surfaces: with them the quasi-two-dimensional cambered case reached only 0.2379 of the thin-aerofoil value with 32 chordwise panels, against 0.2393 with 16 panels once corrected.

### Key module constants

Set at the top of each file: `NWAKE` and `WAKE_LENGTH_SPANS` fix the wake discretisation; `TIP_FINE_STEP_CHORDS` and `TIP_FINE_LENGTH_CHORDS` refine the tip filaments near the wing; `MAX_ITER`, `OMEGA` and `TOL` control the relaxation; `CORE_WING` and `RC_WAKE_FRACTION` set the core radii. Tip-vortex height and chordwise onset depend on `RC_WAKE_FRACTION`, so state it whenever quoting tip results.

## Input conventions

The mesh is a NumPy archive with key `points`, of shape `(nchord+1, nspan+1, 3)`, holding the corner coordinates of the mean camber surface. The first index runs from leading to trailing edge, the second across the span. Camber, twist, taper, sweep and dihedral are all encoded in the coordinates. The onset flow is nominally along x and z points upwards. A pitch prompt rotates the mesh about y, so incidence changes need no remeshing.

The velocity profile is a comma-separated file of two columns, z and the streamwise speed U, with an optional header. The onset velocity is (U(z), 0, 0), interpolated linearly and held constant beyond the tabulated range.

## Verification

All cases below were run and the values are current for the committed code.

Rectangular flat wing, uniform flow:

| Case | Result |
| --- | --- |
| Aspect ratio four, eight, 20 at α = 5° | C_L = 0.332, 0.418, 0.490, falling 9.1%, 4.7% and 1.8% below the lifting-line estimate as the aspect ratio grows |
| α from 2° to 8°, aspect ratio eight | C_L proportional to α to within 0.2% |
| Lattice refinement, aspect ratio eight, α = 5° | C_L = 0.4425, 0.4256, 0.4182, 0.4114 for 4×2, 8×3, 12×4 and 20×6 panels, converging monotonically |
| Free wake against frozen wake, aspect ratio eight, α = 5° | 0.4193 against 0.4184, a difference in the fourth decimal |
| Tip filaments, aspect ratio six, α = 8° | filaments reach 0.09 to 0.14 chords above the wing plane at the trailing edge; C_L rises from 0.6100 to 0.6203 |

Mesh-based code:

| Case | Result |
| --- | --- |
| Rectangular mesh, aspect ratio eight, α = 5°, uniform flow | C_L = 0.4192, reproducing the analytical-geometry code to three decimals |
| Cambered wing, 2% camber, aspect ratio 20, α = 0° | C_L = 0.1989 against 0.2285 corrected for finite aspect ratio |
| Cambered quasi-two-dimensional wing, chordwise refinement | C_L = 0.1978, 0.2280, 0.2393 for four, eight and 16 panels, against the thin-aerofoil value 0.2513 |
| Swept, tapered, twisted sample wing, uniform flow | spanwise loading symmetric to 4×10⁻¹⁶; C_L = 0.4848, C_Di = 0.00877 |
| Same wing in the linear shear profile | C_L falls to 0.4630, matching the local dynamic pressure at the height of the wing |
| Fresh clone from GitHub | reproduces C_L = 0.4630, so the pushed copy is self-contained |

The chordwise convergence towards the thin-aerofoil limit is the weakest result: 16 chordwise panels still fall 4.8% short. Anyone tightening the camber prediction should start there, with a cosine chordwise distribution the obvious first attempt.

## Environment

Python 3.9.6 at `/Library/Developer/CommandLineTools/usr/bin/python3`, with numpy 2.0.2 and matplotlib 3.9.4 installed for the user with `python3 -m pip install --user`. No virtual environment. Homebrew is at `/opt/homebrew`; the GitHub command line tool version 2.97.0 is authenticated as `ignaziomviola` and wired as the git credential helper, so `git push` over HTTPS needs no token typing.

Both codes prompt on standard input and call `plt.show()`. For batch or cloud use, pipe the answers and force the non-interactive backend:

```bash
printf "wing_mesh.npz\nvelocity_profile.csv\n1.0\n5\n" | MPLBACKEND=Agg python3 panel_wing.py
```

In a notebook, `!python3 panel_wing.py` neither services the prompts nor displays the figure. Import the module and call `load_mesh`, `pitch_mesh`, `load_velocity_profile`, `make_onset`, `build_lattice`, `solve`, `spanwise_loading` and `plot_results` in sequence instead.

## Design decisions and their reasons

A thin lifting surface was chosen over a thick source–doublet formulation because it extends the existing vortex-lattice code directly and keeps the free wake and tip filaments intact; thickness effects are consequently absent. The mesh format is a NumPy archive because the solver needs structured chordwise and spanwise ordering to place the vortex rings and shed the wake at the trailing edge. The onset flow is a tabulated shear profile rather than a general function so that no user code is needed. The reference speed is prompted rather than derived, because under shear no single averaging convention is obviously right and an explicit value keeps coefficients comparable between runs.

## Open items

- Both repositories now carry the MIT licence, copyright Ignazio Maria Viola, with a `CITATION.cff` supplying the reference and the README asking for it. Two points remain open: the University of Edinburgh policy on research-code licensing was not checked, and the University may hold copyright in work produced under an employment contract, in which case the holder line needs amending; and no release is tagged and no digital object identifier minted, so the citation currently names the repository rather than an archived version. Depositing a release with Zenodo would supply one.
- No `requirements.txt`, which Binder would need, and no notebook for one-click cloud running.
- No command-line arguments; adding them would let `python3 panel_wing.py --mesh … --uref … --save fig.png` run non-interactively and unblock `!python3 …` in notebooks.
- No automated tests. The verification cases above are the natural candidates, the flat-plate value 0.4192 and the loading symmetry being the cheapest regression checks.
- Chordwise convergence for camber is slow, as noted above.
- The formulation is steady and inviscid: the wake carries no time history, separation is assumed only at the trailing edge and tips, and the tip-vortex core is not a viscous prediction.
