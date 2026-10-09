"""Integrity of the vendored fluid solver (standard-library unittest).

Run:          python3 test_vendored.py
Regenerate:   python3 test_vendored.py --write

The lifting-surface method here is a verbatim copy of
ignaziomviola/free-wake-lifting-surface at the commit recorded in
docs/FLUID.md. This test recomputes the SHA-256 of every vendored file against
`fluid_manifest.txt` and fails if one has been edited. It checks local
integrity, not agreement with upstream, and touches no network: the failure
it prevents is a quiet fix to the fluid code made while chasing a coupled
result, which would leave every number in docs/COUPLING.md irreproducible
against the repository it claims to come from. A genuine fluid change belongs
upstream, followed by a re-vendor and a new manifest in the same commit.
"""

import hashlib
import pathlib
import sys
import unittest

MANIFEST = pathlib.Path(__file__).with_name("fluid_manifest.txt")
ROOT = pathlib.Path(__file__).parent

VENDORED_CODE = ("panel_wing.py", "make_sample_inputs.py")
# the fluid code's own documents; docs/fluid/INDEX.md is written here and is
# NOT in this list
VENDORED_DOCS = ("docs/fluid/README.md", "docs/fluid/HANDOVER.md")
VENDORED = VENDORED_CODE + VENDORED_DOCS

UPSTREAM = "https://github.com/ignaziomviola/free-wake-lifting-surface"
COMMIT = "07fc7003dc98dbc4142546b52cbcced24a55c84e"


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_manifest():
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, _, name = line.partition("  ")
        entries[name.strip()] = sha.strip()
    return entries


def write_manifest():
    lines = ["# SHA-256 of the files vendored verbatim from",
             f"# {UPSTREAM}",
             f"# commit {COMMIT}",
             "# Regenerate with: python3 test_vendored.py --write"]
    lines += [f"{digest(ROOT / name)}  {name}" for name in VENDORED]
    MANIFEST.write_text("\n".join(lines) + "\n")
    return MANIFEST


class TestVendoredFluid(unittest.TestCase):

    def test_manifest_covers_every_vendored_file(self):
        self.assertEqual(sorted(read_manifest()), sorted(VENDORED))

    def test_every_vendored_file_is_unmodified(self):
        for name, sha in read_manifest().items():
            path = ROOT / name
            self.assertTrue(path.exists(), f"{name} is missing")
            self.assertEqual(
                digest(path), sha,
                f"{name} has been modified since it was vendored from "
                f"{UPSTREAM} at {COMMIT}. The fluid solver is not maintained "
                f"here: change it upstream, re-vendor, and regenerate the "
                f"manifest with 'python3 test_vendored.py --write' in the "
                f"same commit.")

    def test_the_index_is_written_here(self):
        index = ROOT / "docs" / "fluid" / "INDEX.md"
        self.assertTrue(index.exists(), "docs/fluid/INDEX.md is missing")
        self.assertNotIn("docs/fluid/INDEX.md", read_manifest())

    def test_the_fluid_module_imports_without_output(self):
        """No prompt and no print at import. (It does import pyplot at
        module level, upstream, so matplotlib is a hard dependency.)"""
        import contextlib
        import importlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            importlib.import_module("panel_wing")
        self.assertEqual(buffer.getvalue(), "")


if __name__ == "__main__":
    if "--write" in sys.argv:
        print(f"wrote {write_manifest()}")
    else:
        unittest.main(verbosity=2)
