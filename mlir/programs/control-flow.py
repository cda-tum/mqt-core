from catalyst import qjit, measure, cond, for_loop, while_loop, grad
import pennylane as qml
from jax import numpy as jnp

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
    print(circuit())
    print(circuit.mlir)

if __name__ == "__main__":
    main()
