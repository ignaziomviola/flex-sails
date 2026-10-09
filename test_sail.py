"""Fluid adapter and coupling (standard-library unittest).

    python3 test_sail.py
"""

import unittest

import numpy as np

import make_sample_inputs as msi
import panel_wing as pw
import sail_2d
import sail_fluid as sf
import sail_fsi as fsi


class TestOnset(unittest.TestCase):

    def test_gradient_and_twist(self):
        on = sf.sail_onset(6.0, 10.0, shear=0.1, twist_deg=8.0, y_offset=1.0)
        v = on(np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0]]))
        self.assertAlmostEqual(np.linalg.norm(v[0]), 6.0)
        self.assertAlmostEqual(np.degrees(np.arctan2(v[0, 2], v[0, 0])), 8.0)
        self.assertAlmostEqual(np.linalg.norm(v[1]), 6.0 * (1.0 / 11.0) ** 0.1)
        self.assertEqual(v[1, 2], 0.0)

    def test_even_in_height_for_the_deck_image(self):
        on = sf.sail_onset(6.0, 10.0, shear=0.1, twist_deg=8.0)
        p = np.array([[1.0, 3.0, 0.2]])
        q = p * np.array([1.0, -1.0, 1.0])
        np.testing.assert_array_equal(on(p), on(q))


UNIFORM = sf.sail_onset(1.0, 1.0)


class TestFluid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pts = pw.pitch_mesh(msi.rectangular_mesh(), np.radians(5.0))
        cls.sol = sf.solve_fluid(cls.pts, UNIFORM, 1.0, rho=pw.RHO,
                                 wake="frozen")

    def test_panel_forces_sum_to_the_vendored_coefficients(self):
        self.assertAlmostEqual(self.sol["CL"], self.sol["cl_vendored"],
                               places=13)
        self.assertAlmostEqual(self.sol["CD"], self.sol["cdi_vendored"],
                               places=13)

    def test_frozen_wake_restores_the_vendored_constant(self):
        self.assertEqual(pw.MAX_ITER, 30)

    def test_deck_image_is_the_doubled_wing(self):
        half = self.pts[:, 6:].copy()
        sd = sf.solve_fluid(half, UNIFORM, 1.0, rho=pw.RHO, wake="frozen",
                            deck=True)
        self.assertAlmostEqual(sd["CL"], self.sol["CL"], places=12)

    def test_deck_needs_the_foot_on_the_deck(self):
        with self.assertRaises(ValueError):
            sf.mirror_in_deck(self.pts)

    def test_lumping_conserves_force_and_moment(self):
        for load in ("pressure", "kj"):
            r = fsi.transfer_report(self.sol, load)
            self.assertLess(r["force_defect"], 1e-13)
            self.assertLess(r["moment_defect"], 1e-13)

    def test_pressure_load_is_normal_and_suction_is_the_rest(self):
        f = sf.nodal_forces(self.sol, "pressure").reshape(-1, 3).sum(0)
        g = sf.nodal_forces(self.sol, "kj").reshape(-1, 3).sum(0)
        np.testing.assert_allclose(g - f, sf.suction(self.sol), atol=1e-14)

    def test_force_scales_with_density(self):
        s2 = sf.solve_fluid(self.pts, UNIFORM, 1.0, rho=2.0, wake="frozen")
        np.testing.assert_allclose(s2["F"], 2.0 * self.sol["F"], rtol=1e-13)
        self.assertAlmostEqual(s2["CL"], self.sol["CL"], places=13)


class TestInteraction(unittest.TestCase):

    def test_split_wing_is_the_whole_wing(self):
        import verify_sail as vs
        full, hist = vs.split_wing(4, 8, sweeps=8)
        self.assertAlmostEqual(hist[-1][0], full["CL"], places=8)
        self.assertAlmostEqual(hist[-1][1], full["CD"], places=8)

    def test_extra_field_leaves_the_wake_alone(self):
        pts = pw.pitch_mesh(msi.rectangular_mesh(), np.radians(5.0))
        a = sf.solve_fluid(pts, UNIFORM, 1.0, rho=pw.RHO)

        def side_wind(p, use):
            v = np.zeros((len(np.atleast_2d(p)), 3))
            v[:, 2] = 0.05
            return v
        b = sf.solve_fluid(pts, UNIFORM, 1.0, rho=pw.RHO, extra=side_wind)
        for fa, fb in zip(a["filaments"], b["filaments"]):
            np.testing.assert_array_equal(fa["nodes"], fb["nodes"])
        self.assertGreater(b["CL"], a["CL"])

    def test_extra_equal_to_the_onset_change(self):
        """A uniform extra upwash is a change of incidence for the boundary
        condition: same circulation as rotating the onset, wake aside."""
        pts = pw.pitch_mesh(msi.rectangular_mesh(), np.radians(5.0))

        def up(p, use):
            v = np.zeros((len(np.atleast_2d(p)), 3))
            v[:, 2] = 0.01
            return v

        def tilted(p):
            v = UNIFORM(p)
            v[:, 2] = 0.01
            return v
        b = sf.solve_fluid(pts, UNIFORM, 1.0, rho=pw.RHO, extra=up)
        c = sf.solve_fluid(pts, tilted, 1.0, rho=pw.RHO)
        np.testing.assert_allclose(b["gamma"], c["gamma"], rtol=2e-3)


class TestSailPlan(unittest.TestCase):

    def test_apparent_wind_triangle(self):
        import sail_plan as sp
        still = sp.apparent_wind(5.0, 40.0, 0.0)
        aws, awa = still(10.0)
        self.assertAlmostEqual(aws, 5.0)
        self.assertAlmostEqual(np.degrees(awa), 40.0)
        wind = sp.apparent_wind(5.0, 40.0, 2.5)
        low, high = wind(1.0), wind(10.0)
        self.assertLess(low[1], high[1])          # it veers aft aloft
        self.assertLess(high[1], np.radians(40.0))
        self.assertAlmostEqual(high[0] ** 2, 5.0 ** 2 + 2.5 ** 2
                               + 2 * 5.0 * 2.5 * np.cos(np.radians(40.0)))

    def test_j80_geometry(self):
        import sail_plan as sp
        case = dict(sp.J80_UPWIND)
        _, _, awa, _ = sp.apparent_onset(case)
        sails = sp.j80_sails(case, nchord=4, nspan=4)
        genoa = sp.to_boat(sails["genoa"][0], awa)
        main = sp.to_boat(sails["main"][0], awa)
        I, J, E = sp.J80["I"], sp.J80["J"], sp.J80["E"]
        np.testing.assert_allclose(genoa[0, :, 0] / genoa[0, :, 2], J / I)
        np.testing.assert_allclose(genoa[0, :, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(main[0, :, 0], J)
        foot = np.linalg.norm(main[-1, 0] - main[0, 0])
        self.assertAlmostEqual(foot, E, places=10)
        lp = np.linalg.norm(np.cross(genoa[-1, 0] - genoa[0, 0],
                                     [J, 0.0, I]) / np.hypot(J, I))
        self.assertAlmostEqual(lp, case["lp"] * J, delta=0.02 * J)


class TestStressTrajectories(unittest.TestCase):

    def test_uniform_stretch_on_a_distorted_grid_gives_straight_lines(self):
        import sail_membrane as sm
        import sail_plan as sp
        import verify_sail as vs
        nc, ns = 10, 8
        X = vs.flat_grid(nc, ns, 2.0, 1.6, jitter=0.25)
        m = sm.build_model(X, sm.grid_triangles(nc, ns), 1000.0, 0.3)
        th = np.radians(30.0)
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
        D = R @ np.diag([1.03, 1.005, 1.0]) @ R.T
        lines = sp.stress_trajectories(m, (X @ D.T).reshape(nc + 1, ns + 1,
                                                            3), 0.15)
        self.assertGreaterEqual(len(lines), 8)
        normal = np.array([-np.sin(th), np.cos(th), 0.0])
        drift = max(np.ptp(L @ normal) for L in lines)
        self.assertLess(drift, 3e-3)                # 1.9 mm over 2.4 m

    def test_inflated_strip_lines_follow_the_arc(self):
        import sail_membrane as sm
        import sail_plan as sp
        import verify_sail as vs
        nc, ns = 16, 4
        X = vs.flat_grid(nc, ns, 1.0, 0.4)
        fixed = np.zeros_like(X, dtype=bool)
        fixed[:, 1] = True
        for j in range(ns + 1):
            fixed[sm.node_id(0, j, ns)] = fixed[sm.node_id(nc, j, ns)] = True
        m = sm.build_model(X, sm.grid_triangles(nc, ns), 1000.0, 0.3, 0.01,
                           fixed=fixed)
        u = sm.solve_static(m, pressure=50.0)["u"]
        lines = sp.stress_trajectories(m, (X + u).reshape(nc + 1, ns + 1, 3),
                                       0.08)
        self.assertEqual(len(lines), 4)
        for L in lines:
            self.assertLess(np.ptp(L[:, 1]), 1e-3)
            self.assertGreater(np.ptp(L[:, 0]), 0.95)
            self.assertGreater(L[:, 2].max(), 0.1)


class TestTheory2D(unittest.TestCase):

    def test_rigid_limit_is_two_pi_alpha(self):
        r = sail_2d.solve(np.radians(4.0), 1e12, 0.002, n=16)
        self.assertAlmostEqual(r["CL"], 2 * np.pi * np.radians(4.0), places=8)

    def test_softer_cloth_cambers_more_and_lifts_more(self):
        a = sail_2d.solve(np.radians(6.0), 400.0, 0.002, n=32)
        b = sail_2d.solve(np.radians(6.0), 100.0, 0.002, n=32)
        self.assertGreater(b["depth"], a["depth"])
        self.assertGreater(b["CL"], a["CL"])

    def test_tension_closes_the_extensibility_law(self):
        r = sail_2d.solve(np.radians(6.0), 100.0, 0.002, n=32)
        self.assertAlmostEqual(
            r["C_T"], 100.0 / 0.91 * (r["excess"] + 1.3 * 0.002), places=8)


class TestCoupling(unittest.TestCase):

    def membrane_wing(self, accel="iqn", deck=True):
        pts0 = fsi.sail_planform(2.0, 1.0, 1.0, nchord=4, nspan=4,
                                 alpha_deg=6.0)
        model = fsi.build_sail(pts0, 6000.0, prestrain=0.002, foot="free",
                               leech="pinned", deck=deck)
        return fsi.static_aeroelastic(model, pts0, sf.sail_onset(10.0, 2.0),
                                      10.0, deck=deck, tol=1e-6, accel=accel)

    def test_membrane_wing_converges_and_cambers(self):
        res = self.membrane_wing()
        self.assertTrue(res["converged"])
        self.assertLess(res["iterations"], 15)
        d = fsi.section_shape(res["points"], 0)["depth"]
        self.assertGreater(d, 0.05)
        self.assertLess(d, 0.12)

    def test_converged_shape_is_a_fixed_point(self):
        """The structure under the loads of the converged shape returns it."""
        res = self.membrane_wing()
        self.assertLess(res["history"]["residual"][-1], 1e-6)
        rel = (np.linalg.norm(res["points"] - res["points_fluid"])
               / np.linalg.norm(res["u"]))
        self.assertLess(rel, 1e-6)

    def test_iqn_and_aitken_agree(self):
        a, b = self.membrane_wing("iqn"), self.membrane_wing("aitken")
        self.assertAlmostEqual(a["fluid"]["CL"], b["fluid"]["CL"], places=5)

    def test_mainsail_leech_tension_reduces_the_twist(self):
        on = sf.sail_onset(6.0, 10.0)
        twist = []
        for le0 in (0.0, 0.02):
            pts0 = fsi.sail_planform(10.0, 3.5, 0.5, nchord=4, nspan=6,
                                     camber=0.08, alpha_deg=10.0)
            model = fsi.build_sail(pts0, 5e5, prestrain=0.002, foot="pinned",
                                   head="pinned", leech_EA=5e4,
                                   cable_prestrain=le0)
            res = fsi.static_aeroelastic(model, pts0, on, 6.0)
            self.assertTrue(res["converged"])
            rows = fsi.flying_shape_report(res)
            twist.append(rows[0]["incidence_deg"] - rows[4]["incidence_deg"])
        self.assertGreater(twist[0], twist[1])

    def test_planform_and_section_shape(self):
        pts = fsi.sail_planform(10.0, 3.0, 1.0, nchord=16, nspan=4,
                                camber=0.1, draft=0.4, alpha_deg=7.0)
        s = fsi.section_shape(pts, 0)
        self.assertAlmostEqual(s["chord"], 3.0, places=12)
        self.assertAlmostEqual(s["incidence_deg"], 7.0, places=10)
        self.assertAlmostEqual(s["depth"], 0.1, places=3)
        self.assertAlmostEqual(s["draft"], 0.4, delta=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
