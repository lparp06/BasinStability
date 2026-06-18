import unittest

import networkx as nx
import numpy as np

from network_dynamics.core.coupling_strengths import (
    CouplingStrengthInterval,
    coupling_strength_intervals_from_zeros,
    interval_coupling_strengths,
    laplacian_nonzero_eigenvalue_bounds,
)


class CouplingStrengthTests(unittest.TestCase):
    def test_laplacian_nonzero_eigenvalue_bounds_uses_graph_laplacian(self):
        G = nx.path_graph(5)

        first_nonzero, largest = laplacian_nonzero_eigenvalue_bounds(G)

        self.assertAlmostEqual(first_nonzero, 0.3819660112501051)
        self.assertAlmostEqual(largest, 3.618033988749895)

    def test_coupling_strength_interval_uses_msf_zero_and_laplacian_bounds(self):
        G = nx.path_graph(5)

        intervals = coupling_strength_intervals_from_zeros(G, [0.5, 5.0])

        self.assertEqual(len(intervals), 1)
        interval = intervals[0]
        self.assertAlmostEqual(interval.lower, 0.5 / 0.3819660112501051)
        self.assertAlmostEqual(interval.upper, 5.0 / 3.618033988749895)
        self.assertAlmostEqual(interval.msf_zero_low, 0.5)
        self.assertAlmostEqual(interval.msf_zero_high, 5.0)

    def test_empty_when_graph_spectrum_cannot_fit_msf_interval(self):
        G = nx.path_graph(5)

        intervals = coupling_strength_intervals_from_zeros(G, [0.5, 1.0])

        self.assertEqual(intervals, [])

    def test_interval_coupling_strengths_returns_evenly_spaced_values(self):
        interval = CouplingStrengthInterval(
            lower=1.0,
            upper=2.0,
            msf_zero_low=0.5,
            msf_zero_high=5.0,
            laplacian_first_nonzero=0.5,
            laplacian_largest=2.5,
        )

        strengths = interval_coupling_strengths(interval, n_strengths=3)

        self.assertTrue(np.allclose(strengths, [1.0, 1.5, 2.0]))


if __name__ == "__main__":
    unittest.main()
