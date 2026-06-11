"""
results.py

Result containers for basin-stability experiments.
"""

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np


@dataclass
class TrialResult:
    """
    Result from one basin-stability trial.
    """

    trial_seed: int
    success: bool
    final_success: bool
    window_success: bool
    integration_failed: bool
    final_distance: Optional[float]
    window_max_distance: Optional[float]
    sync_time: Optional[float]
    error: Optional[str]

    def to_dict(self):
        return {
            "trial_seed": self.trial_seed,
            "success": self.success,
            "final_success": self.final_success,
            "window_success": self.window_success,
            "integration_failed": self.integration_failed,
            "final_distance": self.final_distance,
            "window_max_distance": self.window_max_distance,
            "sync_time": self.sync_time,
            "error": self.error,
        }


@dataclass
class BasinSummary:
    """
    Summary from a basin-stability experiment.
    """

    success_definition: str
    basin_stability: float
    n_trials: int
    successes: int
    sync_failures: int
    integration_failures: int
    sync_time_mean: float
    base_seed: int
    trial_seeds: list
    results: list
    config: Any

    @classmethod
    def from_results(cls, config, seeds, results):
        """
        Build a BasinSummary from a list of TrialResult objects.
        """

        successes = sum(result.success for result in results)
        integration_failures = sum(result.integration_failed for result in results)

        sync_failures = config.n_trials - successes - integration_failures

        if sync_failures < 0:
            raise RuntimeError(
                "Summary counts are inconsistent. "
                "Check success and integration_failed logic."
            )

        successful_sync_times = [
            result.sync_time
            for result in results
            if result.success and result.sync_time is not None
        ]

        if successful_sync_times:
            sync_time_mean = float(np.mean(successful_sync_times))
        else:
            sync_time_mean = np.inf

        basin_stability = successes / config.n_trials

        return cls(
            success_definition=config.success_definition,
            basin_stability=basin_stability,
            n_trials=config.n_trials,
            successes=successes,
            sync_failures=sync_failures,
            integration_failures=integration_failures,
            sync_time_mean=sync_time_mean,
            base_seed=config.base_seed,
            trial_seeds=list(seeds),
            results=results,
            config=config,
        )

    def to_dict(self):
        return {
            "success_definition": self.success_definition,
            "basin_stability": self.basin_stability,
            "n_trials": self.n_trials,
            "successes": self.successes,
            "sync_failures": self.sync_failures,
            "integration_failures": self.integration_failures,
            "sync_time_mean": self.sync_time_mean,
            "base_seed": self.base_seed,
            "trial_seeds": self.trial_seeds,
            "results": [result.to_dict() for result in self.results],
        }
