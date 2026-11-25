import unittest
import numpy as np

from bayesian.independent_evaluation import check_models
from sure.independent_recovery import (
    STATES, SPECS, features
)


class FrozenRecoveryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.estimator, cls.kernel = check_models()

    def assert_probability(self, p):
        p = np.asarray(p, dtype=float)

        self.assertEqual(p.shape, (3,))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all(p >= 0))
        self.assertAlmostEqual(
            float(p.sum()), 1.0, places=8
        )

    def test_state_order(self):
        self.assertEqual(
            tuple(STATES),
            ("unstarted", "processing", "completed")
        )

    def test_estimator_probabilities(self):
        for keyed in (False, True):
            for cap in (100, 150, 200):
                for obs in ("timeout", "error"):
                    f = features(obs, cap, keyed, cap)

                    p = self.estimator.predict(f)

                    self.assert_probability(p)

    def test_unkeyed_safety(self):
        for remaining in (50, 100, 350, 750, 1250):
            for budget in range(5):
                allowed = self.kernel.allowed(
                    False, remaining, budget
                )

                self.assertNotIn("retry", allowed)
                self.assertNotIn("failover", allowed)

    def test_zero_call_budget(self):
        for keyed in (False, True):
            allowed = self.kernel.allowed(
                keyed, 1250, 0
            )

            self.assertTrue(
                set(allowed).issubset({"wait"})
            )

    def test_expired_deadline(self):
        for keyed in (False, True):
            allowed = self.kernel.allowed(
                keyed, 0, 4
            )

            self.assertEqual(len(allowed), 0)

    def test_transition_probability_mass(self):
        beliefs = (
            np.array([1., 0., 0.]),
            np.array([0., 1., 0.]),
            np.array([0., 0., 1.]),
            np.array([0.3, 0.3, 0.4])
        )

        for keyed in (False, True):
            for level in (0, 1, 2):
                actions = self.kernel.allowed(
                    keyed, 1250, 4
                )

                for action in actions:
                    for belief in beliefs:
                        joint = self.kernel.joint(
                            belief, action, keyed,
                            150, "initial",
                            "timeout", level
                        )

                        total = sum(
                            float(v.sum())
                            for v in joint.values()
                        )

                        self.assertAlmostEqual(
                            total, 1.0, places=8
                        )

                        for (obs, duration), v in joint.items():
                            self.assertTrue(
                                np.all(np.isfinite(v))
                            )
                            self.assertTrue(
                                np.all(v >= 0)
                            )
                            self.assertGreater(duration, 0)

                            if obs == "confirmed":
                                self.assertAlmostEqual(
                                    float(v[0] + v[1]),
                                    0.0, places=8
                                )

    def test_unseen_observation_fallback(self):
        belief = np.array([0.3, 0.3, 0.4])

        posterior, unknown = self.kernel.update(
            belief, "wait", False,
            150, "initial", "timeout",
            "__unseen_test_observation__", 2
        )

        self.assertTrue(unknown)
        self.assert_probability(posterior)

        joint = self.kernel.joint(
            belief, "wait", False,
            150, "initial", "timeout", 2
        )

        expected = sum(
            joint.values(), np.zeros(3)
        )
        expected /= expected.sum()

        np.testing.assert_allclose(
            posterior, expected
        )

    def test_planner_action_eligibility(self):
        belief = np.array([0.3, 0.3, 0.4])

        for policy in ("v2", "v4",
                       "v4_no_time", "v4_time_only"):
            depth, cost, level = SPECS[policy]

            for keyed in (False, True):
                for deadline in (350, 750, 1250):
                    for budget in (0, 1, 3, 4):

                        utility, action = self.kernel.choose(
                            belief, keyed, 150,
                            "initial", "timeout",
                            deadline, budget,
                            depth, level, cost
                        )

                        allowed = self.kernel.allowed(
                            keyed, deadline - 150,
                            budget
                        )

                        self.assertIn(
                            action, (*allowed, "stop")
                        )

                        self.assertTrue(
                            np.isfinite(utility)
                        )

                        if not keyed:
                            self.assertNotIn(
                                action, ("retry", "failover")
                            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
