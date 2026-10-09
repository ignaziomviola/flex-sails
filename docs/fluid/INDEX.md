# The vendored fluid documents

Written here, and not covered by `fluid_manifest.txt`. The two documents next to
this one are byte-for-byte copies from
[free-wake-lifting-surface](https://github.com/ignaziomviola/free-wake-lifting-surface)
at the commit recorded in [../FLUID.md](../FLUID.md), and they describe that
repository, not this one. Read them with three substitutions:

| Upstream says | Here |
| --- | --- |
| `vortex_lattice_wing.py` | not vendored: the sail uses `panel_wing.py` only |
| "the onset is (U(z), 0, 0) from a CSV profile" | `sail_fluid.sail_onset` builds the onset instead: speed varying with height y, direction rotating about the mast; `panel_wing.solve` takes any onset callable |
| "working copy at ~/Desktop/..." and "fresh clone" checks | refer to the upstream repository |

| File | What it owns |
| --- | --- |
| `README.md` | the upstream usage guide and input conventions |
| `HANDOVER.md` | the numerical method, its two pitfalls, its module constants and its verification table |

The upstream frame (span along y, lift along +z, onset along +x) is kept, and the
sail is set up in it: mast along y, apparent wind along +x, leeward along +z.
