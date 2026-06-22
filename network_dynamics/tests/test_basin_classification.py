import unittest

import numpy as np

from network_dynamics.core.basin_common import classify_solution
from network_dynamics.core.config import BasinConfig


class BasinClassificationTests(unittest.TestCase):
    def test_first_crossing_allows_late_instability_after_sync(self):
        config = BasinConfig(
            tmax=5.0,
            dt=1.0,
            sync_tol=1e-3,
            success_definition="first_crossing",
        ).validate()

        sol = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        t = np.array([0.0, 1.0, 2.0])

        result = classify_solution(
            config=config,
            trial_seed=1,
            sol=sol,
            t=t,
        )

        self.assertTrue(result.success)
        self.assertFalse(result.integration_failed)
        self.assertEqual(result.sync_time, 0.0)

    def test_invalid_solution_without_first_crossing_still_fails(self):
        config = BasinConfig(
            tmax=5.0,
            dt=1.0,
            sync_tol=1e-3,
            success_definition="first_crossing",
        ).validate()

        sol = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        t = np.array([0.0, 1.0, 2.0])

        result = classify_solution(
            config=config,
            trial_seed=2,
            sol=sol,
            t=t,
        )

        self.assertFalse(result.success)
        self.assertTrue(result.integration_failed)


if __name__ == "__main__":
    unittest.main()
