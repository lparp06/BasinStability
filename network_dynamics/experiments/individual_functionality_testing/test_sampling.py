import numpy as np

from network_dynamics.core.sampling import (
    sample_uniform_initial_condition,
    trial_seeds,
)


def main():
    print("=" * 70)
    print("TEST sampling.py")
    print("=" * 70)

    base_seed = 42
    n_trials = 10

    seeds = trial_seeds(base_seed=base_seed, n_trials=n_trials)

    print("Seeds:", seeds)

    assert len(seeds) == n_trials
    assert seeds[0] == base_seed

    rng1 = np.random.default_rng(43)
    rng2 = np.random.default_rng(43)

    ic1 = sample_uniform_initial_condition(
        rng1,
        n_nodes=5,
        dimension=3,
        low=-10,
        high=10,
    )

    ic2 = sample_uniform_initial_condition(
        rng2,
        n_nodes=5,
        dimension=3,
        low=-10,
        high=10,
    )

    print("Initial condition shape:", ic1.shape)
    print("Initial condition min/max:", ic1.min(), ic1.max())
    print("Initial condition seed 43:")
    print(ic1)

    assert ic1.shape == (15,)
    assert np.all(ic1 >= -10)
    assert np.all(ic1 <= 10)
    assert np.allclose(ic1, ic2)

    print("sampling.py passed.")


if __name__ == "__main__":
    main()
