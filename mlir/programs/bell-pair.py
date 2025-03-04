from catalyst import qjit, measure, cond, for_loop, while_loop, grad
import pennylane as qml
from jax import numpy as jnp

@qjit
@qml.qnode(qml.device("lightning.qubit", wires=3))
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    qml.MultiControlledX(control_wires=[1,2], wires=0)
    return qml.probs(wires=1)

def main() -> None:
    print(circuit())
    print(circuit.mlir)

if __name__ == "__main__":
    main()
