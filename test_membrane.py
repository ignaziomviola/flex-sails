"""Membrane finite elements (standard-library unittest).

    python3 test_membrane.py
"""

import unittest

import numpy as np

import sail_fsi as fsi
import sail_membrane as sm
import verify_sail as vs


class TestKinematics(unittest.TestCase):

    def test_homogeneous_stretch_is_exact_on_a_distorted_mesh(self):
        X = vs.flat_grid(5, 4, 1.0, 0.8, jitter=0.3)
        m = sm.build_model(X, sm.grid_triangles(5, 4), 1000.0, 0.3)
        x = X * np.array([1.2, 1.1, 1.0])
        _, E = sm.membrane_kinematics(m, x)
        ev = np.sort(np.linalg.eigvalsh(E), axis=1)
        np.testing.assert_allclose(ev, [[0.5 * (1.1 ** 2 - 1),
                                         0.5 * (1.2 ** 2 - 1)]] * len(E),
                                   atol=1e-13)

    def test_patch_interior_nodes_are_in_equilibrium(self):
        nc, ns = 5, 4
        X = vs.flat_grid(nc, ns, 1.0, 0.8, jitter=0.3)
        m = sm.build_model(X, sm.grid_triangles(nc, ns), 1000.0, 0.3)
        R, _ = sm.assemble(m, X * np.array([0.2, 0.1, 0.0]), tangent=False)
        interior = [sm.node_id(i, j, ns) for i in range(1, nc)
                    for j in range(1, ns)]
        self.assertLess(np.abs(R.reshape(-1, 3)[interior]).max(), 1e-10)

    def test_large_rigid_rotation_leaves_the_stress_unchanged(self):
        X = vs.flat_grid(4, 3)
        X[:, 2] = 0.1 * np.sin(np.pi * X[:, 0])
        m = sm.build_model(X, sm.grid_triangles(4, 3), 1000.0, 0.3, 0.003)
        f0, _, S0, _ = sm.membrane_forces(m, X, tangent=False)
        Q = vs.rotation([1.0, 2.0, 0.5], np.radians(120.0))
        f1, _, S1, _ = sm.membrane_forces(m, X @ Q.T + 1.0, tangent=False)
        np.testing.assert_allclose(S1, S0, atol=1e-12 * np.abs(S0).max())
        np.testing.assert_allclose(f1.reshape(-1, 3, 3),
                                   f0.reshape(-1, 3, 3) @ Q.T,
                                   atol=1e-12 * np.abs(f0).max())

    def test_cauchy_tension_of_a_rotated_stretch(self):
        X = vs.flat_grid(3, 3)
        m = sm.build_model(X, sm.grid_triangles(3, 3), 1000.0, 0.3)
        Q = vs.rotation([1.0, 2.0, 0.5], 1.0)
        values, d, _ = sm.membrane_tension(
            m, (X * np.array([1.02, 1.01, 1.0])) @ Q.T)
        Eb = 1000.0 / 0.91
        e1, e2 = 0.5 * (1.02 ** 2 - 1), 0.5 * (1.01 ** 2 - 1)
        np.testing.assert_allclose(
            values, [[Eb * (e1 + 0.3 * e2) * 1.02 / 1.01,
                      Eb * (e2 + 0.3 * e1) * 1.01 / 1.02]] * len(values),
            rtol=1e-12)
        np.testing.assert_allclose(np.abs(d @ Q), [[1.0, 0.0, 0.0]] * len(d),
                                   atol=1e-12)

    def test_stress_free_reference_has_no_internal_force(self):
        X = vs.flat_grid(3, 3)
        X[:, 2] = 0.2 * X[:, 0] * (1 - X[:, 0])
        m = sm.build_model(X, sm.grid_triangles(3, 3), 1000.0, 0.3)
        R, _ = sm.assemble(m, np.zeros_like(X), tangent=False)
        self.assertLess(np.abs(R).max(), 1e-10)


class TestWrinkling(unittest.TestCase):

    Et = np.array([1000.0])

    def stress(self, e1, e2, ang=0.0):
        c, s = np.cos(ang), np.sin(ang)
        Rm = np.array([[c, -s], [s, c]])
        S, D, st = sm.membrane_stress((Rm @ np.diag([e1, e2]) @ Rm.T)[None],
                                      self.Et, 0.3)
        return S[0], D[0], st[0], Rm

    def test_taut(self):
        S, _, st, _ = self.stress(0.01, 0.002)
        self.assertEqual(st, sm.TAUT)
        Eb = 1000.0 / 0.91
        np.testing.assert_allclose(np.diag(S), [Eb * 0.0106, Eb * 0.005])

    def test_wrinkled_is_uniaxial_along_the_major_strain(self):
        S, _, st, Rm = self.stress(0.01, -0.005, np.radians(35.0))
        self.assertEqual(st, sm.WRINKLED)
        np.testing.assert_allclose(S, 10.0 * np.outer(Rm[:, 0], Rm[:, 0]),
                                   atol=1e-12)

    def test_slack_carries_nothing(self):
        S, D, st, _ = self.stress(-0.001, -0.002)
        self.assertEqual(st, sm.SLACK)
        self.assertEqual(np.abs(S).max(), 0.0)
        self.assertEqual(np.abs(D).max(), 0.0)

    def test_stress_is_continuous_across_the_wrinkling_boundary(self):
        """A jump of the strain by 2e-9 moves the stress by at most the
        modulus times it, so no step: the law is continuous there."""
        a = self.stress(0.01, -0.003 + 1e-9)[0]
        b = self.stress(0.01, -0.003 - 1e-9)[0]
        self.assertLess(np.abs(a - b).max(), 2e-9 * 1000.0 / 0.91)

    def test_wrinkled_tangent_against_differences(self):
        E0 = np.array([[0.01, 0.002], [0.002, -0.006]])
        _, D, st = sm.membrane_stress(E0[None], self.Et, 0.3)
        self.assertEqual(st[0], sm.WRINKLED)
        h = 1e-8
        for k, dE in enumerate((np.array([[1, 0], [0, 0]]),
                                np.array([[0, 0], [0, 1]]),
                                np.array([[0, 0.5], [0.5, 0]]))):
            Sp = sm.membrane_stress((E0 + h * dE)[None], self.Et, 0.3)[0][0]
            Sm = sm.membrane_stress((E0 - h * dE)[None], self.Et, 0.3)[0][0]
            col = (Sp - Sm) / (2 * h)
            np.testing.assert_allclose(D[0][:, k],
                                       [col[0, 0], col[1, 1], col[0, 1]],
                                       rtol=1e-5, atol=1e-6)


class TestTangent(unittest.TestCase):

    def test_assembled_tangent_against_differences(self):
        """Membrane in all three states, a cable and a follower pressure."""
        nc, ns = 3, 3
        X = vs.flat_grid(nc, ns, 1.5, 1.2)
        X[:, 2] = 0.05 * np.sin(3 * X[:, 0])
        cab = [{"pairs": sm.edge_chain([sm.node_id(nc, j, ns)
                                        for j in range(ns + 1)]),
                "EA": 50.0, "prestrain": 0.01}]
        m = sm.build_model(X, sm.grid_triangles(nc, ns), 100.0, 0.3,
                           cables=cab)
        u = 0.05 * np.random.default_rng(0).standard_normal(X.shape)
        R, K = sm.assemble(m, u, pressure=3.0)
        _, E = sm.membrane_kinematics(m, X + u)
        states = set(sm.membrane_stress(E, m["Et"], m["nu"])[2])
        self.assertEqual(states, {sm.TAUT, sm.WRINKLED, sm.SLACK})
        h = 1e-6
        Kfd = np.empty_like(K)
        for k in range(len(R)):
            up, um = u.copy().reshape(-1), u.copy().reshape(-1)
            up[k] += h
            um[k] -= h
            Kfd[:, k] = (sm.assemble(m, up.reshape(-1, 3), pressure=3.0,
                                     tangent=False)[0]
                         - sm.assemble(m, um.reshape(-1, 3), pressure=3.0,
                                       tangent=False)[0]) / (2 * h)
        self.assertLess(np.abs(K - Kfd).max() / np.abs(K).max(), 1e-8)

    def test_potential_is_the_integral_of_the_residual(self):
        """dPi/du = R under dead loads, checked by differences."""
        X = vs.flat_grid(3, 3)
        m = sm.build_model(X, sm.grid_triangles(3, 3), 100.0, 0.3, 0.01)
        rng = np.random.default_rng(2)
        u = 0.05 * rng.standard_normal(X.shape)
        f = rng.standard_normal(X.shape)
        R, _ = sm.assemble(m, u, f, tangent=False)
        d = rng.standard_normal(X.shape)
        h = 1e-6
        dpi = (sm.potential(m, u + h * d, f)
               - sm.potential(m, u - h * d, f)) / (2 * h)
        self.assertAlmostEqual(dpi, float(R @ d.reshape(-1)), places=6)


class TestBattens(unittest.TestCase):

    def test_tangent_with_battens_against_differences(self):
        nc, ns = 3, 3
        X = vs.flat_grid(nc, ns, 1.5, 1.2)
        X[:, 2] = 0.05 * np.sin(3 * X[:, 0])
        hinges = fsi.batten_hinges(X.reshape(nc + 1, ns + 1, 3),
                                   [(1, 5.0), (ns, 3.0)])
        m = sm.build_model(X, sm.grid_triangles(nc, ns), 100.0, 0.3, 0.01,
                           hinges=hinges)
        u = 0.05 * np.random.default_rng(3).standard_normal(X.shape)
        R, K = sm.assemble(m, u)
        h = 1e-6
        Kfd = np.empty_like(K)
        for k in range(len(R)):
            up, um = u.copy().reshape(-1), u.copy().reshape(-1)
            up[k] += h
            um[k] -= h
            Kfd[:, k] = (sm.assemble(m, up.reshape(-1, 3), tangent=False)[0]
                         - sm.assemble(m, um.reshape(-1, 3),
                                       tangent=False)[0]) / (2 * h)
        self.assertLess(np.abs(K - Kfd).max() / np.abs(K).max(), 1e-7)

    def test_moulded_shape_is_stress_free_in_bending(self):
        X = vs.flat_grid(4, 2)
        X[:, 2] = 0.3 * X[:, 0] * (1 - X[:, 0])
        hinges = fsi.batten_hinges(X.reshape(5, 3, 3), [(1, 5.0)])
        m = sm.build_model(X, sm.grid_triangles(4, 2), 100.0, 0.3,
                           hinges=hinges)
        fh, _, W = sm.hinge_forces(m, X, tangent=False)
        self.assertLess(np.abs(fh).max(), 1e-12)
        self.assertEqual(W, 0.0)

    def test_batten_is_an_euler_bernoulli_beam(self):
        """Second order towards the pinned beam in tension."""
        e16 = vs.batten_strip(16, 1e-4)[1] - 1
        e32 = vs.batten_strip(32, 1e-4)[1] - 1
        self.assertLess(abs(e32), 1e-3)
        self.assertGreater(e16 / e32, 3.8)

    def test_batten_rigid_rotation_is_free(self):
        X = vs.flat_grid(4, 2)
        X[:, 2] = 0.3 * X[:, 0] * (1 - X[:, 0])
        hinges = fsi.batten_hinges(X.reshape(5, 3, 3), [(1, 5.0)])
        m = sm.build_model(X, sm.grid_triangles(4, 2), 100.0, 0.3,
                           hinges=hinges)
        Q = vs.rotation([0.3, 1.0, 0.2], np.radians(80.0))
        fh, _, W = sm.hinge_forces(m, X @ Q.T + 2.0, tangent=False)
        self.assertLess(np.abs(fh).max(), 1e-10)


class TestSolver(unittest.TestCase):

    def test_inflated_strip_against_the_exact_arc(self):
        sag_ex, H_ex, _ = vs.exact_arc(50.0)
        sol, sag, H = vs.strip_arc(16, 50.0)
        self.assertTrue(sol["converged"])
        self.assertLess(abs(sag / sag_ex - 1), 3e-3)
        self.assertLess(abs(H / H_ex - 1), 3e-3)

    def test_strip_converges_at_second_order(self):
        sag_ex, _, _ = vs.exact_arc(50.0)
        e8 = vs.strip_arc(8, 50.0)[1] / sag_ex - 1
        e16 = vs.strip_arc(16, 50.0)[1] / sag_ex - 1
        self.assertGreater(abs(e8 / e16), 3.5)

    def test_solve_is_pure(self):
        X = vs.flat_grid(3, 3)
        fixed = np.zeros_like(X, dtype=bool)
        fixed[[sm.node_id(0, j, 3) for j in range(4)]] = True
        fixed[[sm.node_id(3, j, 3) for j in range(4)]] = True
        m = sm.build_model(X, sm.grid_triangles(3, 3), 100.0, 0.3, 0.01,
                           fixed=fixed)
        X_before = m["X"].copy()
        f = np.zeros_like(X)
        f[:, 2] = 0.1
        u0 = np.zeros_like(X)
        sol = sm.solve_static(m, f_ext=f, u0=u0)
        self.assertTrue(sol["converged"])
        np.testing.assert_array_equal(m["X"], X_before)
        self.assertEqual(np.abs(u0).max(), 0.0)

    def test_reactions_balance_the_load(self):
        X = vs.flat_grid(4, 3)
        fixed = np.zeros_like(X, dtype=bool)
        for j in range(4):
            fixed[sm.node_id(0, j, 3)] = fixed[sm.node_id(4, j, 3)] = True
        m = sm.build_model(X, sm.grid_triangles(4, 3), 100.0, 0.3, 0.01,
                           fixed=fixed)
        f = np.zeros_like(X)
        f[:, 2] = 0.2
        sol = sm.solve_static(m, f_ext=f)
        np.testing.assert_allclose(sol["reaction"].sum(axis=0),
                                   -f.sum(axis=0), atol=1e-8)

    def test_tension_only_cable_carries_no_compression(self):
        X = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        cab = {"pairs": np.array([[0, 1]]), "L0": np.array([1.0]),
               "EA": 10.0, "prestrain": 0.0, "tension_only": True}
        fe, Ke, N = sm.cable_forces(cab, X * 0.9)
        self.assertEqual(N[0], 0.0)
        self.assertEqual(np.abs(fe).max(), 0.0)
        self.assertEqual(np.abs(Ke).max(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
