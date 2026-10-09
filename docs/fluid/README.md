# Free-wake lifting-surface methods for finite wings

Two Python codes compute the loads on a finite wing in steady incompressible flow, with a wake whose shape is computed rather than prescribed. The wake is discretised into vortex filaments that are advected iteratively with the local velocity until they align with the streamlines, so the trailing vortices roll up and the tip vortex develops along the chord instead of appearing only downstream of the trailing edge.

Both codes solve the same physics on a thin lifting surface; they differ in the geometry and onset flow they admit.

| Code | Geometry | Onset flow |
| --- | --- | --- |
| `vortex_lattice_wing.py` | flat rectangular wing, set by aspect ratio | uniform |
| `panel_wing.py` | generic mean camber surface, read from a mesh | sheared, read from a profile |

Both impose flow tangency on a surface of zero thickness, so thickness effects are absent by construction. The companion repository [free-wake-thick-panel-method](https://github.com/ignaziomviola/free-wake-thick-panel-method) extends the method to wings of finite thickness, panelling the closed surface with source and doublet panels under an internal Dirichlet condition while reusing the free wake described here.

## Requirements

Python 3 with numpy and matplotlib.

```bash
python3 -m pip install --user numpy matplotlib
```

## Rectangular wing in uniform flow

The code prompts for the aspect ratio, the angle of attack α, and the numbers of spanwise and chordwise panels.

```bash
python3 vortex_lattice_wing.py
```

Output is the lift coefficient C_L for a frozen wake and for the relaxed wake, the estimate of the Prandtl lifting-line theory for reference, and a figure with the spanwise loading, the relaxed wake in three dimensions, the wake rear view, and the tip-vortex development along the chord.

## Generic wing in a sheared onset flow

Generate the sample inputs, then run the code:

```bash
python3 make_sample_inputs.py
python3 panel_wing.py
```

The code prompts for the mesh file, the profile file, the reference speed U_ref, and an optional pitch angle that rotates the mesh about the y axis, so incidence changes need no remeshing. Pressing enter accepts the sample inputs. Output is C_L, the induced-drag coefficient C_Di, and a figure of six panels adding the onset profile and the planform top view to the four described above.

### Mesh file

A NumPy archive with key `points`, an array of shape `(nchord+1, nspan+1, 3)` holding the corner coordinates of the mean camber surface. The first index runs from the leading edge to the trailing edge, the second across the span, and the three components are x, y and z. Camber, twist, taper, sweep and dihedral are all encoded in these coordinates. The onset flow is nominally along x and z points upwards.

```python
import numpy as np
np.savez('mymesh.npz', points=points)   # points.shape == (nchord+1, nspan+1, 3)
```

### Velocity profile

A comma-separated file of two columns, z and the streamwise speed U, with an optional header. The onset velocity is (U(z), 0, 0), interpolated linearly between the tabulated heights and held constant beyond them.

```
# z, U
-4.0, 0.1
 0.0, 1.0
 4.0, 2.2
```

## Method

Each panel carries a horseshoe vortex whose bound segment lies on the panel quarter-chord line, with legs that follow the surface side edges to the trailing edge. Flow tangency is imposed at the three-quarter-chord collocation points, where the normals are built from the chordwise tangents of the camber surface. Beyond the trailing edge the legs become free filaments; at the tips each chordwise station releases its own filament from the end of its bound segment, which is what resolves the chordwise growth of the tip vortex.

Velocities follow from the Biot–Savart law with the regularisation of van Garrel, the core radius being a fifth of the spanwise panel spacing for the wake and negligible for the influence coefficients. One iteration solves the linear system for the circulations Γ on the current wake, evaluates the local velocity at every filament node, and marches each filament along the local streamline; node positions are underrelaxed by a factor of 0.7. Convergence is judged on the largest node displacement within one span extent of the wing, because the discrete filaments continue to orbit the rolled-up core far downstream without affecting the loads.

Loads come from the vector form of the Kutta–Joukowski theorem applied to the bound segments, using the local velocity there, which yields the induced drag along with the lift. Coefficients are nondimensionalised by the lattice area and, in `panel_wing.py`, by the reference speed supplied by the user.

## Verification

- A flat rectangular wing of aspect ratio eight at α = 5° gives C_L = 0.419 from both codes, the mesh-based code reproducing the analytical-geometry code to three decimal places.
- The same wing gives C_L within 1.8% of the lifting-line estimate at aspect ratio 20, and 9.1% below it at aspect ratio four, the difference narrowing as the aspect ratio grows.
- The lift coefficient is proportional to α to within 0.2% between 2° and 8°, and converges monotonically under refinement of the lattice.
- A cambered wing of aspect ratio 100 at α = 0° approaches the thin-aerofoil value 4πh under chordwise refinement, reaching 0.239 against 0.251 with 16 chordwise panels.
- The swept, tapered and twisted sample wing in uniform flow gives a spanwise loading symmetric to machine precision.
- Placing that wing in the sheared profile reduces C_L from 0.485 to 0.463, matching the local dynamic pressure at the height of the wing.

## Limitations

The formulation is inviscid and the surface is a thin lifting surface, so thickness effects are absent and no pressure distribution over a section of finite thickness is available; the companion repository linked above supplies both. The flow is assumed to separate only at the trailing edge and the tips. The chordwise onset and the height of the tip vortex therefore depend on the lattice resolution and on the core radius `RC_WAKE_FRACTION`, and the codes are not a substitute for a viscous simulation of the tip-vortex core. Loads are steady: the wake carries no time history.

## Licence and citation

Released under the MIT licence, in [LICENSE](LICENSE): the codes may be used, modified and redistributed, including in commercial work, provided the copyright notice and the permission notice are retained in any copy or substantial portion.

Retaining that notice is the licence condition. Citation is the academic one, and is asked for rather than compelled: if the codes contribute to published work, please cite them. The metadata in [CITATION.cff](CITATION.cff) drives the "Cite this repository" button on the repository page and yields, in the meantime,

> Viola, I. M. (2026). *free-wake-lifting-surface: free-wake lifting-surface methods for finite wings*. https://github.com/ignaziomviola/free-wake-lifting-surface
