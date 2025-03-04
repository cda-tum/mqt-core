# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pennylane as qml
from catalyst import qjit


@qjit
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.MultiControlledX(control_wires=[1, 2], wires=0)
    return qml.probs(wires=1)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
