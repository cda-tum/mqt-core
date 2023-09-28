from typing import ClassVar, overload

class CompoundOperation(Operation):
    """
    Quantum operation comprised of multiple sub-operations.
    """

    def __getitem__(self, arg0: int) -> Operation: ...
    def __init__(self, nq: int, ops: list[Operation]) -> None:
        """
        Create a compound operation from a list of operations.
        """
    def __len__(self) -> int:
        """
        Return number of sub-operations.
        """
    def acts_on(self, arg0: int) -> bool: ...
    def clone(self) -> Operation:
        """
        Return deep clone of the operation.
        """
    def empty(self) -> bool: ...
    def equals(self, other: Operation, p1: Permutation, p2: Permutation) -> bool: ...
    def get_used_qubits(self) -> set[int]:
        """
        Return set of qubits used by the operation.
        """
    def is_compound_operation(self) -> bool: ...
    def is_non_unitary_operation(self) -> bool: ...
    def set_n_qubits(self, arg0: int) -> None: ...
    def size(self) -> int:
        """
        Return number of sub-operations.
        """
    def to_open_qasm(self, arg0: list[tuple[str, str]], arg1: list[tuple[str, str]]) -> str: ...

class Control:
    @overload
    def __init__(self, qubit: int) -> None:
        """
        Create a positive control qubit.
        """
    @overload
    def __init__(self, qubit: int, type: ControlType) -> None:
        """
        Create a control qubit of the specified control type.
        """
    @property
    def control_type(self) -> ...:
        """
        The type of the control qubit. Can be positive or negative.
        """
    @control_type.setter
    def control_type(self, arg0: ControlType) -> None: ...
    @property
    def qubit(self) -> int:
        """
        The qubit index of the control qubit.
        """
    @qubit.setter
    def qubit(self, arg0: int) -> None: ...

class ControlType:
    __members__: ClassVar[dict[ControlType, str]]  # readonly
    Neg: ClassVar[ControlType]  # value = <ControlType.Neg: 0>
    Pos: ClassVar[ControlType]  # value = <ControlType.Pos: 1>
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Expression:
    """
    Class representing a symbolic sum of terms. The expression is of the form `constant + term_1 + term_2 + ... + term_n`.
    """

    __hash__: ClassVar[None] = None
    constant: float
    @overload
    def __add__(self, arg0: Expression) -> Expression: ...
    @overload
    def __add__(self, arg0: Term) -> Expression: ...
    @overload
    def __add__(self, arg0: float) -> Expression: ...
    def __eq__(self, arg0: Expression) -> bool: ...
    def __getitem__(self, arg0: int) -> Term: ...
    @overload
    def __init__(self) -> None:
        """
        Create an empty expression.
        """
    @overload
    def __init__(self, terms: list[Term], constant: float = 0.0) -> None:
        """
        Create an expression with a given list of terms and a constant (0 by default).
        """
    @overload
    def __init__(self, term: Term, constant: float = 0.0) -> None:
        """
        Create an expression with a given term and a constant (0 by default).
        """
    @overload
    def __init__(self, constant: float) -> None:
        """
        Create a constant expression involving no symbolic terms.
        """
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    def __mul__(self, arg0: float) -> Expression: ...
    @overload
    def __radd__(self, arg0: Term) -> Expression: ...
    @overload
    def __radd__(self, arg0: float) -> Expression: ...
    def __rmul__(self, arg0: float) -> Expression: ...
    @overload
    def __rsub__(self, arg0: Term) -> Expression: ...
    @overload
    def __rsub__(self, arg0: float) -> Expression: ...
    def __rtruediv__(self, arg0: float) -> Expression: ...
    @overload
    def __sub__(self, arg0: Expression) -> Expression: ...
    @overload
    def __sub__(self, arg0: Term) -> Expression: ...
    @overload
    def __sub__(self, arg0: float) -> Expression: ...
    def __truediv__(self, arg0: float) -> Expression: ...
    def evaluate(self, assignment: dict[Variable, float]) -> float:
        """
        Return the value of this expression given by summing the values of all instantiated terms and the constant given by the assignment.
        """
    def is_constant(self) -> bool:
        """
        Return true if this expression is constant, i.e., all terms have coefficient 0 or no terms are involved.
        """
    def is_zero(self) -> bool:
        """
        Return true if this expression is zero, i.e., all terms have coefficient 0 and the constant is 0 as well.
        """
    def num_terms(self) -> int:
        """
        Return the number of terms in this expression.
        """
    @property
    def terms(self) -> list[Term]: ...

class NonUnitaryOperation(Operation):
    """
    Non-unitary operations such as classically controlled quantum gates.
    """

    @overload
    def __init__(self, nq: int, targets: list[int], classics: list[int]) -> None:
        """
        Create an nq qubit multi-qubit non-unitary operation controlled by a classical bit.
        """
    @overload
    def __init__(self, nq: int, target: int, classic: int) -> None:
        """
        Create an nq qubit non-unitary operation on qubit target controlled by a classical bit.
        """
    @overload
    def __init__(self, nq: int, targets: list[int], op_type: OpType) -> None:
        """
        Create an nq qubit multi-qubit non-unitary operation of specified type.
        """
    def acts_on(self, arg0: int) -> bool:
        """
        Return set of qubits acted on by the operation.
        """
    def clone(self) -> Operation:
        """
        Return deep clone of the operation.
        """
    @overload
    def equals(self, arg0: Operation, p1: Permutation, p2: Permutation) -> bool: ...
    @overload
    def equals(self, arg0: Operation) -> bool: ...
    def get_used_qubits(self) -> set[int]:
        """
        Return set of qubits used by the operation.
        """
    def is_non_unitary_operation(self) -> bool: ...
    def is_unitary(self) -> bool: ...
    def to_open_qasm(self, arg0: list[tuple[str, str]], arg1: list[tuple[str, str]]) -> str: ...
    @property
    def classics(self) -> list[int]:
        """
        Return the classical bits.
        """
    @property
    def n_targets(self) -> int: ...
    @property
    def targets(self) -> list[int]:
        """
        Return the target qubits.
        """
    @targets.setter
    def targets(self, arg1: list[int]) -> None: ...

class OpType:
    """
    Enum class for representing quantum operations.
    """

    __members__: ClassVar[dict[OpType, str]]  # readonly
    barrier: ClassVar[OpType]  # value = <OpType.barrier: 3>
    classiccontrolled: ClassVar[OpType]  # value = <OpType.classiccontrolled: 38>
    compound: ClassVar[OpType]  # value = <OpType.compound: 34>
    dcx: ClassVar[OpType]  # value = <OpType.dcx: 26>
    ecr: ClassVar[OpType]  # value = <OpType.ecr: 27>
    gphase: ClassVar[OpType]  # value = <OpType.gphase: 1>
    h: ClassVar[OpType]  # value = <OpType.h: 4>
    i: ClassVar[OpType]  # value = <OpType.i: 2>
    iswap: ClassVar[OpType]  # value = <OpType.iswap: 23>
    measure: ClassVar[OpType]  # value = <OpType.measure: 35>
    none: ClassVar[OpType]  # value = <OpType.none: 0>
    peres: ClassVar[OpType]  # value = <OpType.peres: 24>
    peresdag: ClassVar[OpType]  # value = <OpType.peresdag: 25>
    phase: ClassVar[OpType]  # value = <OpType.phase: 16>
    reset: ClassVar[OpType]  # value = <OpType.reset: 36>
    rx: ClassVar[OpType]  # value = <OpType.rx: 19>
    rxx: ClassVar[OpType]  # value = <OpType.rxx: 28>
    ry: ClassVar[OpType]  # value = <OpType.ry: 20>
    ryy: ClassVar[OpType]  # value = <OpType.ryy: 29>
    rz: ClassVar[OpType]  # value = <OpType.rz: 21>
    rzx: ClassVar[OpType]  # value = <OpType.rzx: 31>
    rzz: ClassVar[OpType]  # value = <OpType.rzz: 30>
    s: ClassVar[OpType]  # value = <OpType.s: 8>
    sdag: ClassVar[OpType]  # value = <OpType.sdag: 9>
    swap: ClassVar[OpType]  # value = <OpType.swap: 22>
    sx: ClassVar[OpType]  # value = <OpType.sx: 17>
    sxdag: ClassVar[OpType]  # value = <OpType.sxdag: 18>
    t: ClassVar[OpType]  # value = <OpType.t: 10>
    tdag: ClassVar[OpType]  # value = <OpType.tdag: 11>
    teleportation: ClassVar[OpType]  # value = <OpType.teleportation: 37>
    u2: ClassVar[OpType]  # value = <OpType.u2: 15>
    u3: ClassVar[OpType]  # value = <OpType.u3: 14>
    v: ClassVar[OpType]  # value = <OpType.v: 12>
    vdag: ClassVar[OpType]  # value = <OpType.vdag: 13>
    x: ClassVar[OpType]  # value = <OpType.x: 5>
    xx_minus_yy: ClassVar[OpType]  # value = <OpType.xx_minus_yy: 32>
    xx_plus_yy: ClassVar[OpType]  # value = <OpType.xx_plus_yy: 33>
    y: ClassVar[OpType]  # value = <OpType.y: 6>
    z: ClassVar[OpType]  # value = <OpType.z: 7>
    @staticmethod
    def name(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    @overload
    def __init__(self, value: int) -> None: ...
    @overload
    def __init__(self, arg0: str) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def value(self) -> int: ...

class Operation:
    """
    Generic quantum operation.
    """

    controls: set[Control]
    gate: ...
    n_qubits: int
    targets: list[int]
    def acts_on(self, qubit: int) -> bool:
        """
        Check if the operation acts on the specified qubit.
        """
    def get_starting_qubit(self) -> int:
        """
        Get the starting qubit index of the operation.
        """
    def get_used_qubits(self) -> set[int]:
        """
        Get the qubits used by the operation (both control and targets).
        """
    def is_classic_controlled_operation(self) -> bool: ...
    def is_compound_operation(self) -> bool: ...
    def is_controlled(self) -> bool: ...
    def is_non_unitary_operation(self) -> bool: ...
    def is_standard_operation(self) -> bool: ...
    def is_symbolic_operation(self) -> bool: ...
    def is_unitary(self) -> bool: ...
    @property
    def n_controls(self) -> int: ...
    @property
    def n_targets(self) -> int: ...
    @property
    def name(self) -> str: ...
    @name.setter
    def name(self) -> None: ...

class Permutation:
    """
    Class representing a permutation of qubits.
    """

    def __getitem__(self, arg0: int) -> int: ...
    def __iter__(self) -> Iterator: ...
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
    @overload
    def apply(self, arg0: set[Control]) -> set[Control]:
        """
        Apply the permutation to a set of controls and return the permuted controls.
        """
    @overload
    def apply(self, arg0: list[int]) -> list[int]:
        """
        Apply the permutation to a set of targets and return the permuted targets.
        """

class QuantumComputation:
    """
    Representation of quantum circuits within MQT Core
    """

    gphase: float
    name: str
    def __getitem__(self, arg0: int) -> ...: ...
    @overload
    def __init__(self) -> None:
        """
        Constructs an empty QuantumComputation.
        """
    @overload
    def __init__(self, nq: int) -> None:
        """
        Constructs an empty QuantumComputation with the given number of qubits.
        """
    @overload
    def __init__(self, filename: str) -> None:
        """
        Read QuantumComputation from given file. Supported formats are [OpenQASM, Real, GRCS, TFC, QC]
        """
    def __len__(self) -> int:
        """
        Get the number of operations in the quantum computation.
        """
    def add_ancillary_register(self, n: int, name: str = "") -> None:
        """
        Add a register of n ancillary qubits with name name.
        """
    def add_classical_bit_register(self, n: int, name: str = "") -> None:
        """
        Add a register of n classical bits with name name.
        """
    def add_qubit_register(self, n: int, name: str = "") -> None:
        """
        Add a register of n qubits with name name.
        """
    def add_variable(self, var: ... | double | float) -> None:
        """
        Add variable var to the quantum computation.
        """
    def add_variables(self, vars: list[... | double | float]) -> None:
        """
        Add variables vars to the quantum computation.
        """
    def append_operation(self, op: OpType) -> None:
        """
        Append operation op to the quantum computation.
        """
    @overload
    def barrier(self, q: int) -> None:
        """
        Apply a barrier on qubit q.
        """
    @overload
    def barrier(self, qs: list[int]) -> None:
        """
        Apply a barrier on qubits qs.
        """
    @overload
    def classic_controlled(self, op: OpType, q: int, c: tuple[int, int], t: int, params: list[float]) -> None:
        """
        Apply a classically controlled gate op on qubit q with classical control bit c and target qubit t.
        """
    @overload
    def classic_controlled(
        self, op: OpType, q: int, ctrl: Control, c: tuple[int, int], t: int, params: list[float]
    ) -> None:
        """
        Apply a classically controlled, parameterized gate op on qubit q with classical control bit c, target qubit t and parameters params.
        """
    def clone(self) -> QuantumComputation:
        """
        Clone this QuantumComputation object.
        """
    @overload
    def dcx(self, q1: int, q2: int) -> None:
        """
        Apply a double CNOT gate on qubits q1 and q2.
        """
    @overload
    def dcx(self, q1: int, q2: int, ctrl: Control) -> None:
        """
        Apply a controlled double CNOT gate on qubits q1 and q2 with control ctrl.
        """
    @overload
    def dcx(self, q1: int, q2: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled double CNOT gate on qubits q1 and q2 with controls controls.
        """
    @overload
    def dump(self, filename: str) -> None:
        """
        Dump the quantum computation to file.Supported formats are: - OpenQASM 2.0 (.qasm) - Real (.real) - GRCS (.grcs) - TFC (.tfc) - qc (.qc) - Tensor (.tensor)
        """
    @overload
    def dump(self, filename: str, format: str) -> None:
        """
        Dump the quantum computation to file with specified format.Supported formats are: - OpenQASM 2.0 (.qasm) - Real (.real) - GRCS (.grcs) - TFC (.tfc) - qc (.qc) - Tensor (.tensor)
        """
    @overload
    def ecr(self, q1: int, q2: int) -> None:
        """
        Apply an echoed cross-resonance gate on qubits q1 and q2.
        """
    @overload
    def ecr(self, q1: int, q2: int, ctrl: Control) -> None:
        """
        Apply a controlled echoed cross-resonance gate on qubits q1 and q2 with control ctrl.
        """
    @overload
    def ecr(self, q1: int, q2: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled echoed cross-resonance gate on qubits q1 and q2 with controls controls.
        """
    @overload
    def from_file(self, filename: str) -> None:
        """
        Import the quantum computation from file.Supported formats are: - OpenQASM 2.0 (.qasm) - Real (.real) - GRCS (.grcs) - TFC (.tfc) - QC (.qc)
        """
    @overload
    def from_file(self, filename: str, format: str) -> None:
        """
        Import the quantum computation from file with specified format.Supported formats are: - OpenQASM 2.0 (.qasm) - Real (.real) - GRCS (.grcs) - TFC (.tfc) - qc (.qc)
        """
    def get_variables(self) -> set[...]:
        """
        Get all variables used in the quantum computation.
        """
    @overload
    def h(self, q: int) -> None:
        """
        Apply the Hadamard gate on qubit q.
        """
    @overload
    def h(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled Hadamard gate on qubit q with control ctrl.
        """
    @overload
    def h(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled Hadamard gate on qubit q with controls controls.
        """
    @overload
    def i(self, q: int) -> None:
        """
        Apply the identity on qubit q.
        """
    @overload
    def i(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled identity gate on qubit q with control ctrl.
        """
    @overload
    def i(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled identity gate on qubit q with controls controls.
        """
    def initialize_io_mapping(self) -> None:
        """
        Initialize the I/O mapping of the quantum computation.If no initial mapping is given, the identity mapping will be assumed.If no output permutation is given, it is derived from the measurements
        """
    def instantiate(self, assignment: dict[..., float]) -> None:
        """
        Instantiate the quantum computation by replacing all variables with their values dictated by the dict assignment which maps Variable objects to float.
        """
    def is_variable_free(self) -> bool:
        """
        Check if the quantum computation is free of variables.
        """
    @overload
    def iswap(self, q1: int, q2: int) -> None:
        """
        Apply an iSWAP gate on qubits q1 and q2.
        """
    @overload
    def iswap(self, q1: int, q2: int, ctrl: Control) -> None:
        """
        Apply a controlled iSWAP gate on qubits q1 and q2 with control ctrl.
        """
    @overload
    def iswap(self, q1: int, q2: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled iSWAP gate on qubits q1 and q2 with controls controls.
        """
    @overload
    def measure(self, q: int, c: int) -> None:
        """
        Measure qubit q and store the result in classical register c.
        """
    @overload
    def measure(self, q: int, c: tuple[str, int]) -> None:
        """
        Measure qubit q and store the result in a named classical register c.
        """
    @overload
    def measure(self, qs: list[int], cs: list[int]) -> None:
        """
        Measure qubits qs and store the result in classical register cs.
        """
    @overload
    def peres(self, q1: int, q2: int) -> None:
        """
        Apply a Peres gate on qubits q1 and q2.
        """
    @overload
    def peres(self, q1: int, q2: int, ctrl: Control) -> None:
        """
        Apply a controlled Peres gate on qubits q1 and q2 with control ctrl.
        """
    @overload
    def peres(self, q1: int, q2: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled Peres gate on qubits q1 and q2 with controls controls.
        """
    @overload
    def peresdag(self, q1: int, q2: int) -> None:
        """
        Apply an inverse Peres gate on qubits q1 and q2.
        """
    @overload
    def peresdag(self, q1: int, q2: int, ctrl: Control) -> None:
        """
        Apply a controlled inverse Peres gate on qubits q1 and q2 with control ctrl.
        """
    @overload
    def peresdag(self, q1: int, q2: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled inverse Peres gate on qubits q1 and q2 with controls controls.
        """
    @overload
    def phase(self, q: int, lambda_: float) -> None:
        """
        Apply a phase gate on qubit q with parameter lambda_.
        """
    @overload
    def phase(self, q: int, ctrl: Control, lambda_: float) -> None:
        """
        Apply a controlled phase gate on qubit q with control ctrl and parameter lambda_.
        """
    @overload
    def phase(self, q: int, controls: set[Control], lambda_: float) -> None:
        """
        Apply a multi-controlled phase gate on qubit q with controls controls and parameter lambda_.
        """
    @overload
    def reset(self, q: int) -> None:
        """
        Reset qubit q.
        """
    @overload
    def reset(self, qs: list[int]) -> None:
        """
        Reset qubits qs.
        """
    @overload
    def rx(self, q: int, theta: float) -> None:
        """
        Apply an X-rotation gate on qubit q with angle theta.
        """
    @overload
    def rx(self, q: int, ctrl: Control, theta: float) -> None:
        """
        Apply a controlled X-rotation gate on qubit q with control ctrl and angle theta.
        """
    @overload
    def rx(self, q: int, controls: set[Control], theta: float) -> None:
        """
        Apply a multi-controlled X-rotation gate on qubit q with controls controls and angle theta.
        """
    @overload
    def rxx(self, q1: int, q2: int, phi: float) -> None:
        """
        Apply an XX-rotation gate on qubits q1 and q2 with angle phi.
        """
    @overload
    def rxx(self, q1: int, q2: int, ctrl: Control, phi: float) -> None:
        """
        Apply a controlled XX-rotation gate on qubits q1 and q2 with control ctrl and angle phi.
        """
    @overload
    def rxx(self, q1: int, q2: int, controls: set[Control], phi: float) -> None:
        """
        Apply a multi-controlled XX-rotation gate on qubits q1 and q2 with controls controls and angle phi.
        """
    @overload
    def ry(self, q: int, theta: float) -> None:
        """
        Apply a Y-rotation gate on qubit q with angle theta.
        """
    @overload
    def ry(self, q: int, ctrl: Control, theta: float) -> None:
        """
        Apply a controlled Y-rotation gate on qubit q with control ctrl and angle theta.
        """
    @overload
    def ry(self, q: int, controls: set[Control], theta: float) -> None:
        """
        Apply a multi-controlled Y-rotation gate on qubit q with controls controls and angle theta.
        """
    @overload
    def ryy(self, q1: int, q2: int, phi: float) -> None:
        """
        Apply a YY-rotation gate on qubits q1 and q2 with angle phi.
        """
    @overload
    def ryy(self, q1: int, q2: int, ctrl: Control, phi: float) -> None:
        """
        Apply a controlled YY-rotation gate on qubits q1 and q2 with control ctrl and angle phi.
        """
    @overload
    def ryy(self, q1: int, q2: int, controls: set[Control], phi: float) -> None:
        """
        Apply a multi-controlled YY-rotation gate on qubits q1 and q2 with controls controls and angle phi.
        """
    @overload
    def rz(self, q: int, phi: float) -> None:
        """
        Apply a Z-rotation gate on qubit q with angle phi.
        """
    @overload
    def rz(self, q: int, ctrl: Control, phi: float) -> None:
        """
        Apply a controlled Z-rotation gate on qubit q with control ctrl and angle phi.
        """
    @overload
    def rz(self, q: int, controls: set[Control], phi: float) -> None:
        """
        Apply a multi-controlled Z-rotation gate on qubit q with controls controls and angle phi.
        """
    @overload
    def rzx(self, q1: int, q2: int, phi: float) -> None:
        """
        Apply a ZX-rotation gate on qubits q1 and q2 with angle phi.
        """
    @overload
    def rzx(self, q1: int, q2: int, ctrl: Control, phi: float) -> None:
        """
        Apply a controlled ZX-rotation gate on qubits q1 and q2 with control ctrl and angle phi.
        """
    @overload
    def rzx(self, q1: int, q2: int, controls: set[Control], phi: float) -> None:
        """
        Apply a multi-controlled ZX-rotation gate on qubits q1 and q2 with controls controls and angle phi.
        """
    @overload
    def rzz(self, q1: int, q2: int, phi: float) -> None:
        """
        Apply a ZZ-rotation gate on qubits q1 and q2 with angle phi.
        """
    @overload
    def rzz(self, q1: int, q2: int, ctrl: Control, phi: float) -> None:
        """
        Apply a controlled ZZ-rotation gate on qubits q1 and q2 with control ctrl and angle phi.
        """
    @overload
    def rzz(self, q1: int, q2: int, controls: set[Control], phi: float) -> None:
        """
        Apply a multi-controlled ZZ-rotation gate on qubits q1 and q2 with controls controls and angle phi.
        """
    @overload
    def s(self, q: int) -> None:
        """
        Apply an S gate on qubit q.
        """
    @overload
    def s(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled S gate on qubit q with control ctrl.
        """
    @overload
    def s(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled S gate on qubit q with controls controls.
        """
    @overload
    def sdag(self, q: int) -> None:
        """
        Apply an Sdag gate on qubit q.
        """
    @overload
    def sdag(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled Sdag gate on qubit q with control ctrl.
        """
    @overload
    def sdag(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled Sdag gate on qubit q with controls controls.
        """
    def set_logical_qubit_ancillary(self, q: int) -> None:
        """
        Set logical qubit q to be an ancillary qubit.
        """
    @overload
    def swap(self, q1: int, q2: int) -> None:
        """
        Apply a SWAP gate on qubits q1 and q2.
        """
    @overload
    def swap(self, q1: int, q2: int, ctrl: Control) -> None:
        """
        Apply a controlled SWAP (Fredkin) gate on qubits q1 and q2 with control ctrl.
        """
    @overload
    def swap(self, q1: int, q2: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled SWAP gate on qubits q1 and q2 with controls controls.
        """
    @overload
    def sx(self, q: int) -> None:
        """
        Apply a square root of X gate on qubit q.
        """
    @overload
    def sx(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled square root of X gate on qubit q with control ctrl.
        """
    @overload
    def sx(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled square root of X gate on qubit q with controls controls.
        """
    @overload
    def sxdag(self, q: int) -> None:
        """
        Apply the inverse of the square root of X gate on qubit q.
        """
    @overload
    def sxdag(self, q: int, ctrl: Control) -> None:
        """
        Apply the controlled inverse of the square root of X gate on qubit q with control ctrl.
        """
    @overload
    def sxdag(self, q: int, controls: set[Control]) -> None:
        """
        Apply the multi-controlled inverse of the square root of X gate on qubit q with controls controls.
        """
    @overload
    def t(self, q: int) -> None:
        """
        Apply a T gate on qubit q.
        """
    @overload
    def t(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled T gate on qubit q with control ctrl.
        """
    @overload
    def t(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled T gate on qubit q with controls controls.
        """
    @overload
    def tdag(self, q: int) -> None:
        """
        Apply a Tdag gate on qubit q.
        """
    @overload
    def tdag(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled Tdag gate on qubit q with control ctrl.
        """
    @overload
    def tdag(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled Tdag gate on qubit q with controls controls.
        """
    def to_open_qasm(self) -> str:
        """
        Dump the quantum computation to a string in OpenQASM 2.0 format.
        """
    @overload
    def u2(self, q: int, phi: float, lambda_: float) -> None:
        """
        Apply a U2 gate on qubit q with parameters phi, lambda_.
        """
    @overload
    def u2(self, q: int, ctrl: Control, phi: float, lambda_: float) -> None:
        """
        Apply a controlled U2 gate on qubit q with control ctrl and parameters phi, lambda_.
        """
    @overload
    def u2(self, q: int, controls: set[Control], phi: float, lambda_: float) -> None:
        """
        Apply a multi-controlled U2 gate on qubit q with controls controls and parameters phi, lambda_.
        """
    @overload
    def u3(self, q: int, theta: float, phi: float, lambda_: float) -> None:
        """
        Apply a U3 gate on qubit q with parameters theta, phi, lambda_.
        """
    @overload
    def u3(self, q: int, ctrl: Control, theta: float, phi: float, lambda_: float) -> None:
        """
        Apply a controlled U3 gate on qubit q with control ctrl and parameters theta, phi, lambda_.
        """
    @overload
    def u3(self, q: int, controls: set[Control], theta: float, phi: float, lambda_: float) -> None:
        """
        Apply a multi-controlled U3 gate on qubit q with controls controls and parameters theta, phi, lambda_.
        """
    @overload
    def v(self, q: int) -> None:
        """
        Apply a V gate on qubit q.
        """
    @overload
    def v(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled V gate on qubit q with control ctrl.
        """
    @overload
    def v(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled V gate on qubit q with controls controls.
        """
    @overload
    def vdag(self, q: int) -> None:
        """
        Apply a Vdag gate on qubit q.
        """
    @overload
    def vdag(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled Vdag gate on qubit q with control ctrl.
        """
    @overload
    def vdag(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled Vdag gate on qubit q with controls controls.
        """
    @overload
    def x(self, q: int) -> None:
        """
        Apply an X gate on qubit q.
        """
    @overload
    def x(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled X gate on qubit q with control ctrl.
        """
    @overload
    def x(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled X gate on qubit q with controls controls.
        """
    @overload
    def xx_minus_yy(self, q1: int, q2: int, phi: float, lambda_: float) -> None:
        """
        Apply an XX-YY-rotation gate on qubits q1 and q2 with angles phi and lambda_.
        """
    @overload
    def xx_minus_yy(self, q1: int, q2: int, ctrl: Control, phi: float, lambda_: float) -> None:
        """
        Apply a controlled XX-YY-rotation gate on qubits q1 and q2 with control ctrl and angles phi and lambda_.
        """
    @overload
    def xx_minus_yy(self, q1: int, q2: int, controls: set[Control], phi: float, lambda_: float) -> None:
        """
        Apply a multi-controlled XX-YY-rotation gate on qubits q1 and q2 with controls controls and angles phi and lambda_.
        """
    @overload
    def xx_plus_yy(self, q1: int, q2: int, phi: float, lambda_: float) -> None:
        """
        Apply an XX+YY-rotation gate on qubits q1 and q2 with angles phi and lambda_.
        """
    @overload
    def xx_plus_yy(self, q1: int, q2: int, ctrl: Control, phi: float, lambda_: float) -> None:
        """
        Apply a controlled XX+YY-rotation gate on qubits q1 and q2 with control ctrl and angles phi and lambda_.
        """
    @overload
    def xx_plus_yy(self, q1: int, q2: int, controls: set[Control], phi: float, lambda_: float) -> None:
        """
        Apply a multi-controlled XX+YY-rotation gate on qubits q1 and q2 with controls controls and angles phi and lambda_.
        """
    @overload
    def y(self, q: int) -> None:
        """
        Apply a Y gate on qubit q.
        """
    @overload
    def y(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled Y gate on qubit q with control ctrl.
        """
    @overload
    def y(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled Y gate on qubit q with controls controls.
        """
    @overload
    def z(self, q: int) -> None:
        """
        Apply a Z gate on qubit q.
        """
    @overload
    def z(self, q: int, ctrl: Control) -> None:
        """
        Apply a controlled Z gate on qubit q with control ctrl.
        """
    @overload
    def z(self, q: int, controls: set[Control]) -> None:
        """
        Apply a multi-controlled Z gate on qubit q with controls controls.
        """
    @property
    def depth(self) -> int: ...
    @property
    def initial_layout(self) -> ...: ...
    @property
    def n_ancillae(self) -> int: ...
    @property
    def n_cbits(self) -> int: ...
    @property
    def n_individual_ops(self) -> int: ...
    @property
    def n_ops(self) -> int: ...
    @property
    def n_qubits(self) -> int: ...
    @property
    def n_qubits_without_ancillae(self) -> int: ...
    @property
    def n_single_qubit_ops(self) -> int: ...
    @property
    def output_permutation(self) -> ...: ...

class StandardOperation(Operation):
    """
    Standard quantum operation.This class is used to represent all standard operations, i.e. operations that can be represented by a single gate.This includes all single qubit gates, as well as multi-qubit gates like CNOT, SWAP, etc. as well primitives like barriers and measurements.
    """

    @overload
    def __init__(self) -> None:
        """
        Create an empty standard operation. This is equivalent to the identity gate.
        """
    @overload
    def __init__(
        self, nq: int, target: int, op_type: OpType, params: list[float] = [], starting_qubit: int = 0
    ) -> None:
        """
        Create a single-qubit standard operation of specified type.
        """
    @overload
    def __init__(
        self, nq: int, targets: list[int], op_type: OpType, params: list[float] = [], starting_qubit: int = 0
    ) -> None:
        """
        Create a multi-qubit standard operation of specified type.
        """
    @overload
    def __init__(
        self,
        nq: int,
        control: Control,
        target: int,
        op_type: OpType,
        params: list[float] = [],
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a controlled standard operation of specified type.
        """
    @overload
    def __init__(
        self,
        nq: int,
        control: Control,
        targets: list[int],
        op_type: OpType,
        params: list[float] = [],
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a controlled multi-target standard operation of specified type.
        """
    @overload
    def __init__(
        self,
        nq: int,
        controls: set[Control],
        target: int,
        op_type: OpType,
        params: list[float] = [],
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a multi-controlled standard operation of specified type.
        """
    @overload
    def __init__(
        self,
        nq: int,
        controls: set[Control],
        targets: list[int],
        op_type: OpType,
        params: list[float] = [],
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a multi-controlled multi-target standard operation of specified type.
        """
    @overload
    def __init__(self, nq: int, controls: set[Control], target: int, starting_qubit: int = 0) -> None:
        """
        Create a multi-controlled single-target operation of specified type involving nq consecutive control qubits starting_qubit.
        """
    @overload
    def __init__(
        self,
        nq: int,
        controls: set[Control],
        target0: int,
        target1: int,
        op_type: OpType,
        params: list[float] = [],
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a multi-controlled two-target operation of specified type involving nq consecutive control qubits starting_qubit.
        """
    def clone(self) -> Operation:
        """
        Return deep clone of the operation.
        """
    @overload
    def equals(self, arg0: Operation) -> bool: ...
    @overload
    def equals(self, arg0: Operation, arg1: ..., arg2: ...) -> bool: ...
    def is_standard_operation(self) -> bool: ...
    def to_open_qasm(self, arg0: list[tuple[str, str]], arg1: list[tuple[str, str]]) -> str: ...

class SymbolicOperation(Operation):
    """
    Class representing a symbolic operation.This encompasses all symbolic versions of `StandardOperation` that involve (float) angle parameters.
    """

    @staticmethod
    def get_instantiated_operation(*args, **kwargs) -> StandardOperation: ...
    @staticmethod
    def instantiate(*args, **kwargs) -> None: ...
    @overload
    def __init__(self) -> None:
        """
        Create an empty symbolic operation.
        """
    @overload
    def __init__(
        self,
        nq: int,
        target: int,
        op_type: OpType,
        params: list[... | double | float] | None = None,
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a symbolic operation acting on a single qubit.Params is a list of parameters that can be either `Expression` or `float`.
        """
        if params is None:
            params = []
    @overload
    def __init__(
        self,
        nq: int,
        targets: list[int],
        op_type: OpType,
        params: list[... | double | float] | None = None,
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a symbolic operation acting on multiple qubits.Params is a list of parameters that can be either `Expression` or `float`.
        """
        if params is None:
            params = []
    @overload
    def __init__(
        self,
        nq: int,
        control: Control,
        target: int,
        op_type: OpType,
        params: list[... | double | float] | None = None,
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a controlled symbolic operation.Params is a list of parameters that can be either `Expression` or `float`.
        """
        if params is None:
            params = []
    @overload
    def __init__(
        self,
        nq: int,
        control: Control,
        targets: list[int],
        op_type: OpType,
        params: list[... | double | float] | None = None,
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a controlled multi-target symbolic operation.Params is a list of parameters that can be either `Expression` or `float`.
        """
        if params is None:
            params = []
    @overload
    def __init__(
        self,
        nq: int,
        controls: set[Control],
        target: int,
        op_type: OpType,
        params: list[... | double | float] | None = None,
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a multi-controlled symbolic operation.Params is a list of parameters that can be either `Expression` or `float`.
        """
        if params is None:
            params = []
    @overload
    def __init__(
        self,
        nq: int,
        controls: set[Control],
        targets: list[int],
        op_type: OpType,
        params: list[... | double | float] | None = None,
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a multi-controlled multi-target symbolic operation.Params is a list of parameters that can be either `Expression` or `float`.
        """
        if params is None:
            params = []
    @overload
    def __init__(
        self,
        nq: int,
        controls: set[Control],
        target0: int,
        target1: int,
        op_type: OpType,
        params: list[Expression | double | float] | None = None,
        starting_qubit: int = 0,
    ) -> None:
        """
        Create a multi-controlled two-target symbolic operation.Params is a list of parameters that can be either `Expression` or `float`.
        """
        if params is None:
            params = []
    def clone(self) -> Operation:
        """
        Create a deep copy of this operation.
        """
    @overload
    def equals(self, arg0: Operation, arg1: Permutation, arg2: Permutation) -> bool: ...
    @overload
    def equals(self, arg0: Operation) -> bool: ...
    def get_parameter(self, arg0: int) -> Expression | double | float: ...
    def get_parameters(self) -> list[Expression | double | float]: ...
    def is_standard_operation(self) -> bool:
        """
        Return true if this operation is not parameterized by a symbolic parameter.
        """
    def is_symbolic_operation(self) -> bool:
        """
        Return true if this operation is actually parameterized by a symbolic parameter.
        """

class Term:
    """
    A symbolic term which consists of a variable with a given coefficient.
    """

    @overload
    def __init__(self, coefficient: float, variable: Variable) -> None:
        """
        Create a term with a given coefficient and variable.
        """
    @overload
    def __init__(self, variable: Variable) -> None:
        """
        Create a term with a given variable and coefficient 1.
        """
    def __mul__(self, arg0: float) -> Term: ...
    def __rmul__(self, arg0: float) -> Term: ...
    def __rtruediv__(self, arg0: float) -> Term: ...
    def __truediv__(self, arg0: float) -> Term: ...
    def add_coefficient(self, coeff: float) -> None:
        """
        Add coeff a to the coefficient of this term.
        """
    def evaluate(self, assignment: dict[Variable, float]) -> float:
        """
        Return the value of this term given by multiplying the coefficient of this term to the variable value dictated by the assignment.
        """
    def has_zero_coefficient(self) -> bool:
        """
        Return true if the coefficient of this term is zero.
        """
    @property
    def coefficient(self) -> float:
        """
        Return the coefficient of this term.
        """
    @property
    def variable(self) -> Variable:
        """
        Return the variable of this term.
        """

class Variable:
    """
    A symbolic variable.
    """

    __hash__: ClassVar[None] = None
    def __eq__(self, arg0: Variable) -> bool: ...
    def __gt__(self, arg0: Variable) -> bool: ...
    def __init__(self, name: str = "") -> None:
        """
        Create a variable with a given variable name. Variables are uniquely identified by their name, so if a variable with the same name already exists, the existing variable will be returned.
        """
    def __lt__(self, arg0: Variable) -> bool: ...
    def __ne__(self, arg0: Variable) -> bool: ...
    @property
    def name(self) -> str: ...
