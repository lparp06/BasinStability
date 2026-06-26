import unittest
import csv

import networkx as nx
import numpy as np

from network_dynamics.core.coupling import rank_one_inner_coupling_matrix
from network_dynamics.experiments.coupling_basin_scan import (
    ScanRequest,
    _auto_basin_csv_path,
    make_basin_config,
    write_basin_scan_csv,
)


class CouplingBasinScanTests(unittest.TestCase):
    def test_auto_basin_csv_path_includes_dynamics_stability_and_scheme(self):
        path = _auto_basin_csv_path(
            dynamics="lorenz",
            source=1,
            target=0,
        )

        self.assertEqual(
            str(path),
            "outputs/lorenz/stability/basin_lorenz_s1_t0.csv",
        )

    def test_rank_one_inner_coupling_matrix_uses_target_row_source_column(self):
        H = rank_one_inner_coupling_matrix(target=0, source=1, dimension=3)

        expected = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(np.array_equal(H, expected))

    def test_basin_config_uses_same_inner_coupling_as_msf_request(self):
        request = ScanRequest(
            graph_type="erdos-renyi",
            n_nodes=5,
            edge_probability=0.2,
            graph_seed=42,
            base_seed=42,
            n_trials=3,
            dynamics="lorenz",
            tmax=1.0,
            dt=0.01,
            n_strengths=2,
            coupling_low=None,
            coupling_high=None,
            interval_index=0,
            progress_interval=5,
            backend="cpu",
            n_workers=1,
            integrator="RK4",
            sync_tol=1e-3,
            tol_max=1e6,
            window_fraction=0.1,
            success_definition="first_crossing",
            sampling_low=-5.0,
            sampling_high=5.0,
            max_abs_threshold=1e9,
            a=10.0,
            b=2.0,
            c=28.0,
            msf_source=1,
            msf_target=0,
            K_min=0.0,
            K_max=50.0,
            n_K=1001,
            msf_cache="outputs/msf_zero_cache.csv",
            msf_transient_time=100.0,
            msf_measurement_time=3000.0,
            msf_dt=0.001,
            msf_chunk_size=None,
        )

        config = make_basin_config(
            request=request,
            graph=nx.path_graph(5),
            coupling_strength=0.7,
        )

        self.assertEqual(config.H[0, 1], 1.0)
        self.assertEqual(np.count_nonzero(config.H), 1)

    def test_basin_csv_excludes_msf_k_columns(self):
        path = self.create_temp_path()
        rows = [
            {
                "coupling_strength": 0.7,
                "basin_stability": 0.5,
                "successes": 1,
                "sync_failures": 1,
                "integration_failures": 0,
                "sync_time_mean": 2.5,
                "seconds": 0.1,
            }
        ]

        write_basin_scan_csv(path, rows, n_trials=2, dynamics="lorenz")

        with open(path, newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            self.assertIn("coupling_strength", reader.fieldnames)
            self.assertNotIn("K_msf_low", reader.fieldnames)
            self.assertNotIn("K_msf_high", reader.fieldnames)
            self.assertEqual(next(reader)["coupling_strength"], "0.7")

    def create_temp_path(self):
        import tempfile
        from pathlib import Path

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name) / "basin.csv"


if __name__ == "__main__":
    unittest.main()
