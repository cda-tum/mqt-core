# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pennylane as qml
from catalyst import for_loop, qjit


@for_loop(0, 3, 1)
def loop_body(i, *args):
    qml.Hadamard(wires=i)
    qml.Hadamard(wires=i + 1)
    qml.CZ(wires=[i, i + 1])
    qml.Hadamard(wires=i)
    qml.Hadamard(wires=i + 1)
    return args


@qjit
@qml.qnode(qml.device("lightning.qubit", wires=4))
def circuit():
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    loop_body()
    return qml.probs(wires=3)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
