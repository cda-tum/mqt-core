"""Test the DD module."""

from __future__ import annotations

from matplotlib import pyplot as plt
from quimb.calc import ent_cross_matrix

from mqt.core.dd import StateGenerator


def construct_configurations_rec(config: list[int], nqubits: int, configs: list[list[int]]) -> None:
    if len(config) == nqubits:
        configs.append(config)
        return

    nodes = config[0]
    for i in range(1, 2 * nodes + 1):
        construct_configurations_rec([i, *config.copy()], nqubits, configs)


def construct_configuration(nqubits: int) -> list[list[int]]:
    config = [1]
    configs: list[list[int]] = []
    construct_configurations_rec(config, nqubits, configs)
    return sorted(configs, key=lambda x: sum(x))


def test_statistics() -> None:
    n_trials = 1000
    nqubits = 4
    configs = construct_configuration(nqubits)

    gen = StateGenerator(seed=12345)
    node_data: list[str] = []
    entropy_data: list[list[float]] = [[] for _ in range(nqubits)]
    for config in configs:
        id = str(config)
        for _ in range(n_trials):
            node_data.append(id)
            state = gen.get_random_state_from_structured_dd(nqubits, config)
            ent_matrix = ent_cross_matrix(state)
            for i in range(nqubits):
                # computes the entanglement of one qubit with the remaining system
                entropy_data[i].append(ent_matrix[i][i])

    for i in range(nqubits):
        plt.clf()
        plt.scatter(node_data, entropy_data[i])
        plt.savefig(f"entropy_{nqubits}_{n_trials}_q{nqubits-i-1}.png")


def test_state_generation_product_state() -> None:
    """Test the the generation of a single-qubit state."""
    n_trials = 10
    gen = StateGenerator(seed=12345)
    for _ in range(n_trials):
        state = gen.get_random_state_from_structured_dd(5, [1, 1, 1, 1, 1])
        ent_matrix = ent_cross_matrix(state)
        print(ent_matrix)


def test_state_generation_reduced_state() -> None:
    """Test the the generation of a single-qubit state."""
    n_trials = 10
    gen = StateGenerator(seed=12345)
    for _ in range(n_trials):
        state = gen.get_random_state_from_structured_dd(5, [2, 2, 2, 2, 1])
        ent_matrix = ent_cross_matrix(state)
        print(ent_matrix)


def test_state_generation_full_state() -> None:
    """Test the the generation of a single-qubit state."""
    n_trials = 10
    gen = StateGenerator(seed=12345)
    for _ in range(n_trials):
        state = gen.get_random_state_from_structured_dd(5, [16, 8, 4, 2, 1])
        ent_matrix = ent_cross_matrix(state)
        print(ent_matrix)
