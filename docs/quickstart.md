---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Quickstart

The central interface for working with quantum circuits in the Munich Quantum Toolkit is the {py:class}`~mqt.core.ir.QuantumComputation` class.
It represents quantum circuits as a sequential list of operations.
Operations can be directly applied to the {py:class}`~mqt.core.ir.QuantumComputation`:

```{code-cell} ipython3
from mqt.core import QuantumComputation

# Build a `QuantumComputation` representing a Bell-state preparation circuit.
nqubits = 2
qc = QuantumComputation(nqubits)

qc.h(0)  # Apply Hadamard gate on qubit 0
qc.cx(0, 1)  # Apply a CNOT (controlled X-Gate) with control on qubit 0 and target on qubit 1

# Get Circuit in OpenQASM 3.0 format
print(qc.qasm3_str())
```

The circuit class provides a lot of flexibility as every unitary gate can be declared as a controlled gate:

```{code-cell} ipython3
from mqt.core.ir.operations import Control

nqubits = 2
qc = QuantumComputation(nqubits)

# Controlled Hadamard Gate
qc.ch(0, 1)

# Negatively controlled S-gate: S-Gate on target is performed if control is in |0> state.
qc.cs(Control(0, Control.Type.Neg), 1)

print(qc.qasm3_str())
```

Providing a set of `Control` objects allows declaring any (unitary) gate as a multi-controlled gate:

```{code-cell} ipython3
nqubits = 3
qc = QuantumComputation(nqubits)

# Toffoli gate in mqt-core:
qc.mcx({0, 1}, 2)

# Control type can be individually declared
qc.mcs({Control(0, Control.Type.Neg), Control(1, Control.Type.Pos)}, 2)

print(qc.qasm3_str())
```

## Layout Information

A {py:class}`~mqt.core.ir.QuantumComputation` also contains information about the mapping of algorithmic (or logical/virtual/circuit) qubits to and from device (or physical) qubits.
These are contained in the {py:attr}`~mqt.core.ir.QuantumComputation.initial_layout` and {py:attr}`~mqt.core.ir.QuantumComputation.output_permutation` members which are instances of the {py:class}`~mqt.core.ir.Permutation` class. If no layout is given the trivial layout is assumed.

When printing the OpenQASM representation of the {py:class}`~mqt.core.ir.QuantumComputation` the input and output permutations are given as comments in the first two lines of the QASM string. The format is:

`// i Q_0, Q_1, ..., Q_n` ... algorithmic qubit $i$ is mapped to device qubit $Q_i$.

`// o Q_0, Q_1, ..., Q_n` ... the value of algorithmic qubit $i$ (assumed to be stored in classical bit $c[i]$) is measured at device qubit $Q_i$.

```{code-cell} ipython3
nqubits = 3
qc = QuantumComputation(nqubits)
qc.initial_layout[2] = 0
qc.initial_layout[0] = 1
qc.initial_layout[1] = 2

qc.output_permutation[2] = 0
qc.output_permutation[0] = 1
qc.output_permutation[1] = 2


print(qc.qasm3_str())
```

The layout information can also be automatically determined from measurements
using the {py:meth}`~mqt.core.ir.QuantumComputation.initialize_io_mapping` method:

```{code-cell} ipython3
nqubits = 3
qc = QuantumComputation(nqubits, nqubits)  # 3 qubits, 3 classical bits

qc.h(0)
qc.x(1)
qc.s(2)
qc.measure(1, 0)  # c[0] is measured at qubit 1
qc.measure(2, 1)  # c[1] is measured at qubit 2
qc.measure(0, 2)  # c[2] is measured at qubit 0
qc.initialize_io_mapping()  # determine permutation from measurement

print(qc.qasm3_str())
```

## Visualizing Circuits

Circuits can be printed in a human-readable format:

```{code-cell} ipython3
from mqt.core import QuantumComputation

nqubits = 2
qc = QuantumComputation(nqubits, 1)

qc.h(0)
qc.cx(0, 1)
qc.measure(1, 0)

print(qc)
```

## Operations

The operations in a {py:class}`~mqt.core.ir.QuantumComputation` object are of type {py:class}`~mqt.core.ir.operations.Operation`.
Every type of operation in `mqt-core` is derived from this class.
Operations can also be explicitly constructed.
Each {py:class}`~mqt.core.ir.operations.Operation` has a type in the form of an {py:class}`~mqt.core.ir.operations.OpType`.

### `StandardOperation`

A {py:class}`~mqt.core.ir.operations.StandardOperation` is used to represent basic unitary gates. These can also be declared with arbitrary targets and controls.

```{code-cell} ipython3
from math import pi

from mqt.core.ir.operations import OpType, StandardOperation

nqubits = 3

# u3 gate on qubit 0 in a 3-qubit circuit
u_gate = StandardOperation(target=0, params=[pi / 4, pi, -pi / 2], op_type=OpType.u)

# controlled x-rotation
crx = StandardOperation(control=Control(0), target=1, params=[pi], op_type=OpType.rx)

# multi-controlled x-gate
mcx = StandardOperation(controls={Control(0), Control(1)}, target=2, op_type=OpType.x)

# add operations to a quantum computation
qc = QuantumComputation(nqubits)
qc.append(u_gate)
qc.append(crx)
qc.append(mcx)

print(qc)
```

### `NonUnitaryOperation`

A {py:class}`~mqt.core.ir.operations.NonUnitaryOperation` is used to represent operations involving measurements or resets.

```{code-cell} ipython3
from mqt.core.ir.operations import NonUnitaryOperation

nqubits = 2
qc = QuantumComputation(nqubits, nqubits)
qc.h(0)

# measure qubit 0 on classical bit 0
meas_0 = NonUnitaryOperation(target=0, classic=0)

# reset all qubits
reset = NonUnitaryOperation(targets=[0, 1], op_type=OpType.reset)

qc.append(meas_0)
qc.append(reset)

print(qc.qasm3_str())
```

### `SymbolicOperation`

A {py:class}`~mqt.core.ir.operations.SymbolicOperation` can represent all gates of a {py:class}`~mqt.core.ir.operations.StandardOperation` but the gate parameters can be symbolic.
Symbolic expressions are represented in MQT using the {py:class}`~mqt.core.ir.symbolic.Expression` type, which represent linear combinations of symbolic {py:class}`~mqt.core.ir.symbolic.Term` objects over some set of {py:class}`~mqt.core.ir.symbolic.Variable` objects.

```{code-cell} ipython3
from mqt.core.ir.operations import SymbolicOperation
from mqt.core.ir.symbolic import Expression, Term, Variable

nqubits = 1

x = Variable("x")
y = Variable("y")
sym = Expression([Term(x, 2), Term(y, 3)])
print(sym)

sym += 1
print(sym)

# Create symbolic gate
u1_symb = SymbolicOperation(target=0, params=[sym], op_type=OpType.p)

# Mixed symbolic and instantiated parameters
u2_symb = SymbolicOperation(target=0, params=[sym, 2.0], op_type=OpType.u2)
```

### `CompoundOperation`

A {py:class}`~mqt.core.ir.operations.CompoundOperation` bundles multiple {py:class}`~mqt.core.ir.operations.Operation` objects together.

```{code-cell} ipython3
from mqt.core.ir.operations import CompoundOperation

nqubits = 2
comp_op = CompoundOperation()

# create bell pair circuit
comp_op.append(StandardOperation(0, op_type=OpType.h))
comp_op.append(StandardOperation(target=0, control=Control(1), op_type=OpType.x))

qc = QuantumComputation(nqubits)
qc.append(comp_op)

print(qc)
```

Circuits can be conveniently turned into operations which allows to create nested circuits:

```{code-cell} ipython3
from mqt.core import QuantumComputation

nqubits = 2
comp = QuantumComputation(nqubits)
comp.h(0)
comp.cx(0, 1)

qc = QuantumComputation(nqubits)
qc.append(comp.to_operation())

print(qc)
```

## Interfacing with other SDKs

Since a {py:class}`~mqt.core.ir.QuantumComputation` can be imported from and exported to an OpenQASM 3.0 (or OpenQASM 2.0) string, any library that can work with OpenQASM is easy to use in conjunction with the {py:class}`~mqt.core.ir.QuantumComputation` class.

In addition, `mqt-core` can import [Qiskit](https://qiskit.org/) {py:class}`~qiskit.circuit.QuantumCircuit` objects directly.

```{code-cell} ipython3
from qiskit import QuantumCircuit

from mqt.core.plugins.qiskit import qiskit_to_mqt

# GHZ circuit in qiskit
qiskit_qc = QuantumCircuit(3)
qiskit_qc.h(0)
qiskit_qc.cx(0, 1)
qiskit_qc.cx(0, 2)

qiskit_qc.draw(output="mpl", style="iqp")
```

```{code-cell} ipython3
mqt_qc = qiskit_to_mqt(qiskit_qc)
print(mqt_qc)
```
