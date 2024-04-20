from collections.abc import Iterable, Iterator, Mapping, MutableMapping, MutableSequence, Sequence
from os import PathLike
from typing import overload

from .._compat.typing import Self
from .operations import Control, Operation, OpType
from .symbolic import Expression, Variable

class Permutation(MutableMapping[int, int]):
    """A class to represent a permutation of the qubits in a quantum circuit.

    Args:
        permutation: The permutation to initialize the object with.

    """

    def __init__(self: Self, permutation: dict[int, int] | None = None) -> None:
        """Initialize the permutation."""

    def __getitem__(self: Self, idx: int) -> int:
        """Get the value of the permutation at the given index.

        Args:
            idx: The index to get the value of the permutation at.

        Returns:
            The value of the permutation at the given index.
        """

    def __setitem__(self: Self, idx: int, val: int) -> None:
        """Set the value of the permutation at the given index.

        Args:
            idx: The index to set the value of the permutation at.
            val: The value to set the permutation at the given index to.
        """

    def __delitem__(self: Self, key: int) -> None:
        """Delete the value of the permutation at the given index.

        Args:
            key: The index to delete the value of the permutation at.
        """

    def __iter__(self: Self) -> Iterator[int]:
        """Return an iterator over the indices of the permutation."""

    def __len__(self: Self) -> int:
        """Return the number of indices in the permutation."""

    def __eq__(self: Self, other: object) -> bool:
        """Check if the permutation is equal to another permutation."""

    def __ne__(self: Self, other: object) -> bool:
        """Check if the permutation is not equal to another permutation."""

    def __hash__(self: Self) -> int:
        """Return the hash of the permutation."""

    @overload
    def apply(self: Self, controls: set[Control]) -> set[Control]:
        """Apply the permutation to a set of controls.

        Args:
            controls: The set of controls to apply the permutation to.

        Returns:
            The set of controls with the permutation applied.
        """

    @overload
    def apply(self: Self, targets: list[int]) -> list[int]:
        """Apply the permutation to a list of targets.

        Args:
            targets: The list of targets to apply the permutation to.

        Returns:
            The list of targets with the permutation applied.
        """

class QuantumComputation(MutableSequence[Operation]):
    """The main class for representing quantum computations within the MQT.

    Acts as mutable sequence of :class:`~mqt.core.operations.Operation` objects, which represent the individual operations in the quantum computation.

    Args:
        nq: The number of qubits in the quantum computation.
        nc: The number of classical bits in the quantum computation.
        filename: The filename of the file to load the quantum computation from. Supported formats are OpenQASM2, OpenQASM3, Real, GRCS, TFC, QC.
    """

    # --------------------------------------------------------------------------
    #                               Constructors
    # --------------------------------------------------------------------------
    @overload
    def __init__(self: Self) -> None: ...
    @overload
    def __init__(self: Self, nq: int, nc: int = 0) -> None: ...
    @overload
    def __init__(self: Self, filename: str | PathLike[str]) -> None: ...
    @staticmethod
    def from_qasm(qasm: str) -> QuantumComputation:
        """Create a QuantumComputation object from an OpenQASM string.

        Args:
            qasm: The OpenQASM string to create the QuantumComputation object from.

        Returns:
            The QuantumComputation object created from the OpenQASM string.
        """

    # --------------------------------------------------------------------------
    #                          General Properties
    # --------------------------------------------------------------------------

    name: str
    """
    The name of the quantum computation.
    """
    global_phase: float
    """
    The global phase of the quantum computation.
    """

    @property
    def num_qubits(self: Self) -> int:
        """The total number of qubits in the quantum computation."""

    @property
    def num_ancilla_qubits(self: Self) -> int:
        r"""The number of ancilla qubits in the quantum computation.

        Note:
            Ancilla qubits are qubits that always start in a fixed state (usually :math:`|0\\rangle`).
        """

    @property
    def num_garbage_qubits(self: Self) -> int:
        """The number of garbage qubits in the quantum computation.

        Note:
            Garbage qubits are qubits whose final state is not relevant for the computation.
        """

    @property
    def num_measured_qubits(self: Self) -> int:
        """The number of qubits that are measured in the quantum computation.

        Computed as :math:`|qubits| - |garbage|`.
        """

    @property
    def num_data_qubits(self: Self) -> int:
        """The number of data qubits in the quantum computation.

        Computed as :math:`|qubits| - |ancilla|`.
        """

    @property
    def num_classical_bits(self: Self) -> int:
        """The number of classical bits in the quantum computation."""

    @property
    def num_ops(self: Self) -> int:
        """The number of operations in the quantum computation."""

    def num_single_qubit_ops(self: Self) -> int:
        """Return the number of single-qubit operations in the quantum computation."""

    def num_total_ops(self: Self) -> int:
        """Return the total number of operations in the quantum computation.

        Recursively counts sub-operations (e.g., from :class:`~mqt.core.operations.CompoundOperation` objects).
        """

    def depth(self: Self) -> int:
        """Return the depth of the quantum computation."""

    def invert(self: Self) -> None:
        """Invert the quantum computation (in-place) by inverting each operation and reversing the order of the operations."""

    def to_operation(self: Self) -> Operation:
        """Convert the quantum computation to a single operation.

        This gives ownership of the operations to the resulting operation, so the quantum computation will be empty after this operation.

        When the quantum computation contains more than one operation, the resulting operation is a :class:`~mqt.core.operations.CompoundOperation`.

        Returns:
            The operation representing the quantum computation.
        """

    # --------------------------------------------------------------------------
    #                 Mutable Sequence Interface
    # --------------------------------------------------------------------------

    def __len__(self: Self) -> int:
        """Return the number of operations in the quantum computation."""

    @overload
    def __getitem__(self: Self, idx: int) -> Operation:
        """Get the operation at the given index.

        Note:
            This gives write access to the operation at the given index.

        Args:
            idx: The index of the operation to get.

        Returns:
            The operation at the given index.
        """

    @overload
    def __getitem__(self: Self, idx: slice) -> list[Operation]:
        """Get a slice of operations from the quantum computation.

        Note:
            This gives write access to the operations in the given slice.

        Args:
            idx: The slice of operations to get.

        Returns:
            The operations in the given slice.
        """

    @overload
    def __setitem__(self: Self, idx: int, op: Operation) -> None:
        """Set the operation at the given index.

        Args:
            idx: The index of the operation to set.
            op: The operation to set at the given index.
        """

    @overload
    def __setitem__(self: Self, idx: slice, ops: Iterable[Operation]) -> None:
        """Set the operations in the given slice.

        Args:
            idx: The slice of operations to set.
            ops: The operations to set in the given slice.
        """

    @overload
    def __delitem__(self: Self, idx: int) -> None:
        """Delete the operation at the given index.

        Args:
            idx: The index of the operation to delete.
        """

    @overload
    def __delitem__(self: Self, idx: slice) -> None:
        """Delete the operations in the given slice.

        Args:
            idx: The slice of operations to delete.
        """

    def insert(self: Self, idx: int, op: Operation) -> None:
        """Insert an operation at the given index.

        Args:
            idx: The index to insert the operation at.
            op: The operation to insert.
        """

    def append(self: Self, op: Operation) -> None:
        """Append an operation to the end of the quantum computation.

        Args:
            op: The operation to append.
        """

    def reverse(self: Self) -> None:
        """Reverse the order of the operations in the quantum computation (in-place)."""

    def clear(self: Self) -> None:
        """Clear the quantum computation of all operations."""

    # --------------------------------------------------------------------------
    #                          (Qu)Bit Registers
    # --------------------------------------------------------------------------

    def add_ancillary_register(self: Self, n: int, name: str = "q") -> None:
        """Add an ancillary register to the quantum computation.

        Args:
            n: The number of qubits in the ancillary register.
            name: The name of the ancillary register.
        """

    def add_classical_register(self: Self, n: int, name: str = "c") -> None:
        """Add a classical register to the quantum computation.

        Args:
            n: The number of bits in the classical register.
            name: The name of the classical register.
        """

    def add_qubit_register(self: Self, n: int, name: str = "anc") -> None:
        """Add a qubit register to the quantum computation.

        Args:
            n: The number of qubits in the qubit register.
            name: The name of the qubit register.
        """

    def unify_quantum_registers(self: Self, name: str = "q") -> None:
        """Unify all quantum registers in the quantum computation.

        Args:
            name: The name of the unified quantum register.
        """

    # --------------------------------------------------------------------------
    #                  Initial Layout and Output Permutation
    # --------------------------------------------------------------------------

    initial_layout: Permutation
    """
    The initial layout of the qubits in the quantum computation.

    This is a permutation of the qubits in the quantum computation. It is mainly
    used to track the mapping of circuit qubits to device qubits during quantum
    circuit compilation. The keys are the device qubits (in which a compiled circuit
    is expressed in), and the values are the circuit qubits (in which the original
    quantum circuit is expressed in).

    Any operations in the quantum circuit are expected to be expressed in terms
    of the keys of the initial layout.

    Examples:
        - If no initial layout is explicitly specified (which is the default),
          the initial layout is assumed to be the identity permutation.
        - Assume a three-qubit circuit has been compiled to a four qubit device
          and circuit qubit 0 is mapped to device qubit 1, circuit qubit 1 is
          mapped to device qubit 2, and circuit qubit 2 is mapped to device qubit 3.
          Then the initial layout is {1: 0, 2: 1, 3: 2}.

    """
    output_permutation: Permutation
    """
    The output permutation of the qubits in the quantum computation.

    This is a permutation of the qubits in the quantum computation. It is mainly
    used to track where individual qubits end up at the end of the quantum computation,
    for example after a circuit has been compiled to a specific device and SWAP
    gates have been inserted, which permute the qubits. The keys are the qubits
    in the circuit and the values are the actual qubits being measured.

    Similar to the initial layout, the keys in the output permutation are the
    qubits actually present in the circuit and the values are the qubits in the
    "original" circuit.

    Examples:
        - If no output permutation is explicitly specified and the circuit does
          not contain measurements at the end, the output permutation is assumed
          to be the identity permutation.
        - If the circuit contains measurements at the end, these measurements
          are used to infer the output permutation. Assume a three-qubit circuit
          has been compiled to a four qubit device and, at the end of the circuit,
          circuit qubit 0 is measured into classical bit 2, circuit qubit 1 is
          measured into classical bit 1, and circuit qubit 3 is measured into
          classical bit 0. Then the output permutation is {0: 2, 1: 1, 3: 0}.
    """

    def initialize_io_mapping(self: Self) -> None:
        """Initialize the I/O mapping of the quantum computation.

        If no initial layout is explicitly specified, the initial layout is assumed
        to be the identity permutation. If the circuit contains measurements at the
        end, these measurements are used to infer the output permutation.
        """

    # --------------------------------------------------------------------------
    #                       Ancilla and Garbage Handling
    # --------------------------------------------------------------------------

    @property
    def ancillary(self: Self) -> list[bool]:
        """A list of booleans indicating whether each qubit is ancillary."""

    def set_circuit_qubit_ancillary(self: Self, q: int) -> None:
        """Set a circuit (i.e., logical) qubit to be ancillary.

        Args:
            q: The index of the circuit qubit to set as ancillary.
        """

    def set_circuit_qubits_ancillary(self: Self, q_min: int, q_max: int) -> None:
        """Set a range of circuit (i.e., logical) qubits to be ancillary.

        Args:
            q_min: The minimum index of the circuit qubits to set as ancillary.
            q_max: The maximum index of the circuit qubits to set as ancillary.
        """

    def is_circuit_qubit_ancillary(self: Self, q: int) -> bool:
        """Check if a circuit (i.e., logical) qubit is ancillary.

        Args:
            q: The index of the circuit qubit to check.

        Returns:
            True if the circuit qubit is ancillary, False otherwise.
        """

    @property
    def garbage(self: Self) -> list[bool]:
        """A list of booleans indicating whether each qubit is garbage."""

    def set_circuit_qubit_garbage(self: Self, q: int) -> None:
        """Set a circuit (i.e., logical) qubit to be garbage.

        Args:
            q: The index of the circuit qubit to set as garbage.
        """

    def set_circuit_qubits_garbage(self: Self, q_min: int, q_max: int) -> None:
        """Set a range of circuit (i.e., logical) qubits to be garbage.

        Args:
            q_min: The minimum index of the circuit qubits to set as garbage.
            q_max: The maximum index of the circuit qubits to set as garbage.
        """

    def is_circuit_qubit_garbage(self: Self, q: int) -> bool:
        """Check if a circuit (i.e., logical) qubit is garbage.

        Args:
            q: The index of the circuit qubit to check.

        Returns:
            True if the circuit qubit is garbage, False otherwise.
        """

    # --------------------------------------------------------------------------
    #                        Symbolic Circuit Handling
    # --------------------------------------------------------------------------

    @property
    def variables(self: Self) -> set[Variable]:
        """The set of variables in the quantum computation."""

    def add_variable(self: Self, var: Expression | float) -> None:
        """Add a variable to the quantum computation.

        Args:
            var: The variable to add.
        """

    def add_variables(self: Self, vars_: Sequence[Expression | float]) -> None:
        """Add multiple variables to the quantum computation.

        Args:
            vars_: The variables to add.
        """

    def is_variable_free(self: Self) -> bool:
        """Check if the quantum computation is free of variables.

        Returns:
            True if the quantum computation is free of variables, False otherwise.
        """

    def instantiate(self: Self, assignment: Mapping[Variable, float]) -> QuantumComputation:
        """Instantiate the quantum computation with the given variable assignment.

        Args:
            assignment: The variable assignment to instantiate the quantum computation with.

        Returns:
            The instantiated quantum computation.
        """

    def instantiate_inplace(self: Self, assignment: Mapping[Variable, float]) -> None:
        """Instantiate the quantum computation with the given variable assignment in-place.

        Args:
            assignment: The variable assignment to instantiate the quantum computation with.
        """

    # --------------------------------------------------------------------------
    #                             Output Handling
    # --------------------------------------------------------------------------

    def qasm2_str(self: Self) -> str:
        """Return the OpenQASM2 representation of the quantum computation as a string.

        Note:
            This uses some custom extensions to OpenQASM 2.0 that allow for easier
            definition of multi-controlled gates. These extensions might not be
            supported by all OpenQASM 2.0 parsers. Consider using the :meth:`qasm3_str`
            method instead, which uses OpenQASM 3.0 that natively supports
            multi-controlled gates. The export also assumes the bigger, non-standard
            `qelib1.inc` from Qiskit is available.

        Returns:
            The OpenQASM2 representation of the quantum computation as a string.
        """

    def qasm2(self: Self, filename: PathLike[str] | str) -> None:
        """Write the OpenQASM2 representation of the quantum computation to a file.

        See Also:
            :meth:`qasm2_str`

        Args:
            filename: The filename of the file to write the OpenQASM2 representation to.
        """

    def qasm3_str(self: Self) -> str:
        """Return the OpenQASM3 representation of the quantum computation as a string.

        Returns:
            The OpenQASM3 representation of the quantum computation as a string.
        """

    def qasm3(self: Self, filename: PathLike[str] | str) -> None:
        """Write the OpenQASM3 representation of the quantum computation to a file.

        See Also:
            :meth:`qasm3_str`

        Args:
            filename: The filename of the file to write the OpenQASM3 representation to.
        """

    # --------------------------------------------------------------------------
    #                               Operations
    # --------------------------------------------------------------------------

    def i(self: Self, q: int) -> None:
        r"""Apply an identity operation.

        .. math::
            I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def ci(self: Self, control: Control | int, target: int) -> None:
        """Apply a controlled identity operation.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`i`
        """

    def mci(self: Self, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled identity operation.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`i`
        """

    def x(self: Self, q: int) -> None:
        r"""Apply a Pauli-X gate.

        .. math::
            X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cx(self: Self, control: Control | int, target: int) -> None:
        """Apply a controlled Pauli-X (CNOT / CX) gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`x`
        """

    def mcx(self: Self, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled Pauli-X (Toffoli / MCX) gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`x`
        """

    def y(self: Self, q: int) -> None:
        r"""Apply a Pauli-Y gate.

        .. math::
            Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cy(self: Self, control: Control | int, target: int) -> None:
        """Apply a controlled Pauli-Y gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`y`
        """

    def mcy(self: Self, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled Pauli-Y gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`y`
        """

    def z(self: Self, q: int) -> None:
        r"""Apply a Pauli-Z gate.

        .. math::
            Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cz(self: Self, control: Control | int, target: int) -> None:
        """Apply a controlled Pauli-Z (CZ) gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`z`
        """

    def mcz(self: Self, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled Pauli-Z (MCZ) gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`z`
        """

    def h(self: Self, q: int) -> None:
        r"""Apply a Hadamard gate.

        .. math::
            H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def ch(self: Self, control: Control | int, target: int) -> None:
        """Apply a controlled Hadamard gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`h`
        """

    def mch(self: Self, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled Hadamard gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`h`
        """

    def s(self: Self, q: int) -> None:
        r"""Apply an S gate (phase gate).

        .. math::
            S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cs(self: Self, control: Control | int, target: int) -> None:
        """Apply a controlled S gate (CS gate).

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`s`
        """

    def mcs(self: Self, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled S gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`s`
        """

    def sdg(self: Self, q: int) -> None:
        r"""Apply an :math:`S^{\dagger}` gate.

        .. math::
            S^{\dagger} = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}

        Args:
            q: The target qubit
        """

    def csdg(self: Self, control: Control | int, target: int) -> None:
        r"""Apply a controlled :math:`S^{\dagger}` gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`sdg`
        """

    def mcsdg(self: Self, controls: set[Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`S^{\dagger}` gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`sdg`
        """

    def t(self: Self, q: int) -> None:
        r"""Apply a T gate.

        .. math::
            T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}

        Args:
            q: The target qubit
        """

    def ct(self: Self, control: Control | int, target: int) -> None:
        """Apply a controlled T gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`t`
        """

    def mct(self: Self, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled T gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`t`
        """

    def tdg(self: Self, q: int) -> None:
        r"""Apply a :math:`T^{\dagger}` gate.

        .. math::
            T^{\dagger} = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}

        Args:
            q: The target qubit
        """

    def ctdg(self: Self, control: Control | int, target: int) -> None:
        r"""Apply a controlled :math:`T^{\dagger}` gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`tdg`
        """

    def mctdg(self: Self, controls: set[Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`T^{\dagger}` gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`tdg`
        """

    def v(self: Self, q: int) -> None:
        r"""Apply a V gate.

        .. math::
            V = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -i \\ -i & 1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cv(self: Self, control: Control | int, target: int) -> None:
        """Apply a controlled V gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`v`
        """

    def mcv(self: Self, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled V gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`v`
        """

    def vdg(self: Self, q: int) -> None:
        r"""Apply a :math:`V^{\dagger}` gate.

        .. math::
            V^{\dagger} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix}

        Args:
            q: The target qubit
        """

    def cvdg(self: Self, control: Control | int, target: int) -> None:
        r"""Apply a controlled :math:`V^{\dagger}` gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`vdg`
        """

    def mcvdg(self: Self, controls: set[Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`V^{\dagger}` gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`vdg`
        """

    def sx(self: Self, q: int) -> None:
        r"""Apply a :math:`\sqrt{X}` gate.

        .. math::
            \sqrt{X} = \frac{1}{2} \begin{pmatrix} 1 + i & 1 - i \\ 1 - i & 1 + i \end{pmatrix}

        Args:
            q: The target qubit
        """

    def csx(self: Self, control: Control | int, target: int) -> None:
        r"""Apply a controlled :math:`\sqrt{X}` gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`sx`
        """

    def mcsx(self: Self, controls: set[Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`\sqrt{X}` gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`sx`
        """

    def sxdg(self: Self, q: int) -> None:
        r"""Apply a :math:`\sqrt{X}^{\dagger}` gate.

        .. math::
            \sqrt{X}^{\dagger} = \frac{1}{2} \begin{pmatrix} 1 - i & 1 + i \\ 1 + i & 1 - i \end{pmatrix}

        Args:
            q: The target qubit
        """

    def csxdg(self: Self, control: Control | int, target: int) -> None:
        r"""Apply a controlled :math:`\sqrt{X}^{\dagger}` gate.

        Args:
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`sxdg`
        """

    def mcsxdg(self: Self, controls: set[Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`\sqrt{X}^{\dagger}` gate.

        Args:
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`sxdg`
        """

    def rx(self: Self, theta: float | Expression, q: int) -> None:
        r"""Apply an :math:`R_x(\theta)` gate.

        .. math::
            R_x(\theta) = e^{-i\theta X/2} = \cos(\theta/2) I - i \sin(\theta/2) X = \begin{pmatrix} \cos(\theta/2) & -i \sin(\theta/2) \\ -i \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}

        Args:
            theta: The rotation angle
            q: The target qubit
        """

    def crx(self: Self, theta: float | Expression, control: Control | int, target: int) -> None:
        r"""Apply a controlled :math:`R_x(\theta)` gate.

        Args:
            theta: The rotation angle
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`rx`
        """

    def mcrx(self: Self, theta: float | Expression, controls: set[Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`R_x(\theta)` gate.

        Args:
            theta: The rotation angle
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`rx`
        """

    def ry(self: Self, theta: float | Expression, q: int) -> None:
        r"""Apply an :math:`R_y(\theta)` gate.

        .. math::
            R_y(\theta) = e^{-i\theta Y/2} = \cos(\theta/2) I - i \sin(\theta/2) Y = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}

        Args:
            theta: The rotation angle
            q: The target qubit
        """

    def cry(self: Self, theta: float | Expression, control: Control | int, target: int) -> None:
        r"""Apply a controlled :math:`R_y(\theta)` gate.

        Args:
            theta: The rotation angle
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`ry`
        """

    def mcry(self: Self, theta: float | Expression, controls: set[Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`R_y(\theta)` gate.

        Args:
            theta: The rotation angle
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`ry`
        """

    def rz(self: Self, theta: float | Expression, q: int) -> None:
        r"""Apply an :math:`R_z(\theta)` gate.

        .. math::
            R_z(\theta) = e^{-i\theta Z/2} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}

        Args:
            theta: The rotation angle
            q: The target qubit
        """

    def crz(self: Self, theta: float | Expression, control: Control | int, target: int) -> None:
        r"""Apply a controlled :math:`R_z(\theta)` gate.

        Args:
            theta: The rotation angle
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`rz`
        """

    def mcrz(self: Self, theta: float | Expression, controls: set[Control | int], target: int) -> None:
        r"""Apply a multi-controlled :math:`R_z(\theta)` gate.

        Args:
            theta: The rotation angle
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`rz`
        """

    def p(self: Self, theta: float | Expression, q: int) -> None:
        r"""Apply a phase gate.

        .. math::
            P(\theta) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}

        Args:
            theta: The rotation angle
            q: The target qubit
        """

    def cp(self: Self, theta: float | Expression, control: Control | int, target: int) -> None:
        """Apply a controlled phase gate.

        Args:
            theta: The rotation angle
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`p`
        """

    def mcp(self: Self, theta: float | Expression, controls: set[Control | int], target: int) -> None:
        """Apply a multi-controlled phase gate.

        Args:
            theta: The rotation angle
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`p`
        """

    def u2(self: Self, phi: float | Expression, lambda_: float | Expression, q: int) -> None:
        r"""Apply a :math:`U_2(\phi, \lambda)` gate.

        .. math::
            U_2(\phi, \lambda) =
            \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -e^{i\lambda} \\ e^{i\phi} & e^{i(\phi + \lambda)} \end{pmatrix}

        Args:
            phi: The rotation angle
            lambda_: The rotation angle
            q: The target qubit
        """

    def cu2(
        self: Self, phi: float | Expression, lambda_: float | Expression, control: Control | int, target: int
    ) -> None:
        r"""Apply a controlled :math:`U_2(\phi, \lambda)` gate.

        Args:
            phi: The rotation angle
            lambda_: The rotation angle
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`u2`
        """

    def mcu2(
        self: Self, phi: float | Expression, lambda_: float | Expression, controls: set[Control | int], target: int
    ) -> None:
        r"""Apply a multi-controlled :math:`U_2(\phi, \lambda)` gate.

        Args:
            phi: The rotation angle
            lambda_: The rotation angle
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`u2`
        """

    def u(self: Self, theta: float | Expression, phi: float | Expression, lambda_: float | Expression, q: int) -> None:
        r"""Apply a :math:`U(\theta, \phi, \lambda)` gate.

        .. math::
            U(\theta, \phi, \lambda) =
            \begin{pmatrix} \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\ e^{i\phi}\sin(\theta/2) & e^{i(\phi + \lambda)}\cos(\theta/2) \end{pmatrix}

        Args:
            theta: The rotation angle
            phi: The rotation angle
            lambda_: The rotation angle
            q: The target qubit
        """

    def cu(
        self: Self,
        theta: float | Expression,
        phi: float | Expression,
        lambda_: float | Expression,
        control: Control | int,
        target: int,
    ) -> None:
        r"""Apply a controlled :math:`U(\theta, \phi, \lambda)` gate.

        Args:
            theta: The rotation angle
            phi: The rotation angle
            lambda_: The rotation angle
            control: The control qubit
            target: The target qubit

        See Also:
            :meth:`u`
        """

    def mcu(
        self: Self,
        theta: float | Expression,
        phi: float | Expression,
        lambda_: float | Expression,
        controls: set[Control | int],
        target: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`U(\theta, \phi, \lambda)` gate.

        Args:
            theta: The rotation angle
            phi: The rotation angle
            lambda_: The rotation angle
            controls: The control qubits
            target: The target qubit

        See Also:
            :meth:`u`
        """

    def swap(self: Self, target1: int, target2: int) -> None:
        r"""Apply a SWAP gate.

        .. math::
            SWAP = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cswap(self: Self, control: Control | int, target1: int, target2: int) -> None:
        """Apply a controlled SWAP gate.

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`swap`
        """

    def mcswap(self: Self, controls: set[Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled SWAP gate.

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`swap`
        """

    def dcx(self: Self, target1: int, target2: int) -> None:
        r"""Apply a DCX (double CNOT) gate.

        .. math::
            DCX = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cdcx(self: Self, control: Control | int, target1: int, target2: int) -> None:
        """Apply a controlled DCX (double CNOT) gate.

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`dcx`
        """

    def mcdcx(self: Self, controls: set[Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled DCX (double CNOT) gate.

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`dcx`
        """

    def ecr(self: Self, target1: int, target2: int) -> None:
        r"""Apply an ECR (echoed cross-resonance) gate.

        .. math::
            ECR = \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 0 & 1 & i \\ 0 & 0 & i & 1 \\ 1 & -i & 0 & 0 \\ -i & 1 & 0 & 0 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cecr(self: Self, control: Control | int, target1: int, target2: int) -> None:
        """Apply a controlled ECR (echoed cross-resonance) gate.

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`ecr`
        """

    def mcecr(self: Self, controls: set[Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled ECR (echoed cross-resonance) gate.

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`ecr`
        """

    def iswap(self: Self, target1: int, target2: int) -> None:
        r"""Apply an iSWAP gate.

        .. math::
            iSWAP = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def ciswap(self: Self, control: Control | int, target1: int, target2: int) -> None:
        """Apply a controlled iSWAP gate.

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`iswap`
        """

    def mciswap(self: Self, controls: set[Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled iSWAP gate.

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`iswap`
        """

    def iswapdg(self: Self, target1: int, target2: int) -> None:
        r"""Apply an :math:`iSWAP^{\dagger}` gate.

        .. math::
            iSWAP^{\dagger} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & -i & 0 \\ 0 & -i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def ciswapdg(self: Self, control: Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`iSWAP^{\dagger}` gate.

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`iswapdg`
        """

    def mciswapdg(self: Self, controls: set[Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`iSWAP^{\dagger}` gate.

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`iswapdg`
        """

    def peres(self: Self, target1: int, target2: int) -> None:
        r"""Apply a Peres gate.

        .. math::
            Peres = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cperes(self: Self, control: Control | int, target1: int, target2: int) -> None:
        """Apply a controlled Peres gate.

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`peres`
        """

    def mcperes(self: Self, controls: set[Control | int], target1: int, target2: int) -> None:
        """Apply a multi-controlled Peres gate.

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`peres`
        """

    def peresdg(self: Self, target1: int, target2: int) -> None:
        r"""Apply a :math:`Peres^{\dagger}` gate.

        .. math::
            Peres^{\dagger} = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix}

        Args:
            target1: The first target qubit
            target2: The second target qubit
        """

    def cperesdg(self: Self, control: Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`Peres^{\dagger}` gate.

        Args:
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`peresdg`
        """

    def mcperesdg(self: Self, controls: set[Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`Peres^{\dagger}` gate.

        Args:
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`peresdg`
        """

    def rxx(self: Self, theta: float | Expression, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{xx}(\theta)` gate.

        .. math::
            R_{xx}(\theta) = e^{-i\theta XX/2} = \cos(\theta/2) I\otimes I - i \sin(\theta/2) X \otimes X = \begin{pmatrix} \cos(\theta/2) & 0 & 0 & -i \sin(\theta/2) \\ 0 & \cos(\theta/2) & -i \sin(\theta/2) & 0 \\ 0 & -i \sin(\theta/2) & \cos(\theta/2) & 0 \\ -i \sin(\theta/2) & 0 & 0 & \cos(\theta/2) \end{pmatrix}

        Args:
            theta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def crxx(self: Self, theta: float | Expression, control: Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`R_{xx}(\theta)` gate.

        Args:
            theta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`rxx`
        """

    def mcrxx(self: Self, theta: float | Expression, controls: set[Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`R_{xx}(\theta)` gate.

        Args:
            theta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`rxx`
        """

    def ryy(self: Self, theta: float | Expression, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{yy}(\theta)` gate.

        .. math::
            R_{yy}(\theta) = e^{-i\theta YY/2} = \cos(\theta/2) I\otimes I - i \sin(\theta/2) Y \otimes Y = \begin{pmatrix} \cos(\theta/2) & 0 & 0 & i \sin(\theta/2) \\ 0 & \cos(\theta/2) & -i \sin(\theta/2) & 0 \\ 0 & -i \sin(\theta/2) & \cos(\theta/2) & 0 \\ i \sin(\theta/2) & 0 & 0 & \cos(\theta/2) \end{pmatrix}

        Args:
            theta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def cryy(self: Self, theta: float | Expression, control: Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`R_{yy}(\theta)` gate.

        Args:
            theta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`ryy`
        """

    def mcryy(self: Self, theta: float | Expression, controls: set[Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`R_{yy}(\theta)` gate.

        Args:
            theta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`ryy`
        """

    def rzz(self: Self, theta: float | Expression, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{zz}(\theta)` gate.

        .. math::
            R_{zz}(\theta) = e^{-i\theta ZZ/2} = \begin{pmatrix} e^{-i\theta/2} & 0 & 0 & 0 \\ 0 & e^{i\theta/2} & 0 & 0 \\ 0 & 0 & e^{i\theta/2} & 0 \\ 0 & 0 & 0 & e^{-i\theta/2} \end{pmatrix}

        Args:
            theta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def crzz(self: Self, theta: float | Expression, control: Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`R_{zz}(\theta)` gate.

        Args:
            theta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`rzz`
        """

    def mcrzz(self: Self, theta: float | Expression, controls: set[Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`R_{zz}(\theta)` gate.

        Args:
            theta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`rzz`
        """

    def rzx(self: Self, theta: float | Expression, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{zx}(\theta)` gate.

        .. math::
            R_{zx}(\theta) = e^{-i\theta ZX/2} = \cos(\theta/2) I\otimes I - i \sin(\theta/2) Z \otimes X = \begin{pmatrix} \cos(\theta/2) & -i \sin(\theta/2) & 0 & 0 \\ -i \sin(\theta/2) & \cos(\theta/2) & 0 & 0 \\ 0 & 0 & \cos(\theta/2) & i \sin(\theta/2) \\ 0 & 0 & i \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}

        Args:
            theta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def crzx(self: Self, theta: float | Expression, control: Control | int, target1: int, target2: int) -> None:
        r"""Apply a controlled :math:`R_{zx}(\theta)` gate.

        Args:
            theta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`rzx`
        """

    def mcrzx(self: Self, theta: float | Expression, controls: set[Control | int], target1: int, target2: int) -> None:
        r"""Apply a multi-controlled :math:`R_{zx}(\theta)` gate.

        Args:
            theta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`rzx`
        """

    def xx_minus_yy(
        self: Self, theta: float | Expression, beta: float | Expression, target1: int, target2: int
    ) -> None:
        r"""Apply an :math:`R_{XX - YY}(\theta, \beta)` gate.

        .. math::
            R_{XX - YY}(\theta, \beta) = R_{z_2}(\beta) \cdot e^{-i\frac{\theta}{2} \frac{XX-YY}{2}} \cdot R_{z_2}(-\beta) =
            \begin{pmatrix} \cos(\theta/2) & 0 & 0 & -i \sin(\theta/2) e^{-i\beta} \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ -i \sin(\theta/2) e^{i\beta} & 0 & 0 & \cos(\theta/2) \end{pmatrix}

        Args:
            theta: The rotation angle
            beta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def cxx_minus_yy(
        self: Self,
        theta: float | Expression,
        beta: float | Expression,
        control: Control | int,
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a controlled :math:`R_{XX - YY}(\theta, \beta)` gate.

        Args:
            theta: The rotation angle
            beta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`xx_minus_yy`
        """

    def mcxx_minus_yy(
        self: Self,
        theta: float | Expression,
        beta: float | Expression,
        controls: set[Control | int],
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R_{XX - YY}(\theta, \beta)` gate.

        Args:
            theta: The rotation angle
            beta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`xx_minus_yy`
        """

    def xx_plus_yy(self: Self, theta: float | Expression, beta: float | Expression, target1: int, target2: int) -> None:
        r"""Apply an :math:`R_{XX + YY}(\theta, \beta)` gate.

        .. math::
            R_{XX + YY}(\theta, \beta) = R_{z_1}(\beta) \cdot e^{-i\frac{\theta}{2} \frac{XX+YY}{2}} \cdot R_{z_1}(-\beta) =
            \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos(\theta/2) & -i \sin(\theta/2) e^{-i\beta} & 0 \\ 0 & -i \sin(\theta/2) e^{i\beta} & \cos(\theta/2) & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

        Args:
            theta: The rotation angle
            beta: The rotation angle
            target1: The first target qubit
            target2: The second target qubit
        """

    def cxx_plus_yy(
        self: Self,
        theta: float | Expression,
        beta: float | Expression,
        control: Control | int,
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a controlled :math:`R_{XX + YY}(\theta, \beta)` gate.

        Args:
            theta: The rotation angle
            beta: The rotation angle
            control: The control qubit
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`xx_plus_yy`
        """

    def mcxx_plus_yy(
        self: Self,
        theta: float | Expression,
        beta: float | Expression,
        controls: set[Control | int],
        target1: int,
        target2: int,
    ) -> None:
        r"""Apply a multi-controlled :math:`R_{XX + YY}(\theta, \beta)` gate.

        Args:
            theta: The rotation angle
            beta: The rotation angle
            controls: The control qubits
            target1: The first target qubit
            target2: The second target qubit

        See Also:
            :meth:`xx_plus_yy`
        """

    def gphase(self: Self, theta: float) -> None:
        r"""Apply a global phase gate.

        .. math::
            GPhase(\theta) = (e^{i\theta})

        Args:
            theta: The rotation angle
        """

    @overload
    def measure(self: Self, qubit: int, cbit: int) -> None:
        """Measure a qubit and store the result in a classical bit.

        Args:
            qubit: The qubit to measure
            cbit: The classical bit to store the result
        """

    @overload
    def measure(self: Self, qubit: int, creg_bit: tuple[str, int]) -> None:
        """Measure a qubit and store the result in a bit of a classical register.

        Args:
            qubit: The qubit to measure
            creg_bit: The classical register and index to store the result
        """

    @overload
    def measure(self: Self, qubits: Sequence[int], cbits: Sequence[int]) -> None:
        """Measure multiple qubits and store the results in classical bits.

        This method is equivalent to calling :meth:`measure` multiple times.

        Args:
            qubits: The qubits to measure
            cbits: The classical bits to store the results
        """

    def measure_all(self: Self, add_bits: bool = True) -> None:
        """Measure all qubits and store the results in classical bits.

        Details:
            If `add_bits` is `True`, a new classical register (named "`meas`") with
            the same size as the number of qubits will be added to the circuit
            and the results will be stored in it. If `add_bits` is `False`, the
            classical register must already exist and have a sufficient number
            of bits to store the results.

        Args:
            add_bits: Whether to explicitly add a classical register
        """

    @overload
    def reset(self: Self, q: int) -> None:
        """Add a reset operation to the circuit.

        Args:
            q: The qubit to reset
        """

    @overload
    def reset(self: Self, qubits: Sequence[int]) -> None:
        """Add a reset operation to the circuit.

        Args:
            qubits: The qubits to reset
        """

    @overload
    def barrier(self: Self) -> None:
        """Add a barrier to the circuit."""

    @overload
    def barrier(self: Self, q: int) -> None:
        """Add a barrier to the circuit.

        Args:
            q: The qubit to add the barrier to
        """

    @overload
    def barrier(self: Self, qubits: Sequence[int]) -> None:
        """Add a barrier to the circuit.

        Args:
            qubits: The qubits to add the barrier to
        """

    @overload
    def classic_controlled(
        self: Self,
        op: OpType,
        target: int,
        creg: tuple[int, int],
        expected_value: int = 1,
        params: Sequence[float] = (),
    ) -> None:
        """Add a classic-controlled operation to the circuit.

        Args:
            op: The operation to apply
            target: The target qubit
            creg: The classical register (index and number of bits)
            expected_value: The expected value of the classical register
            params: The parameters of the operation
        """

    @overload
    def classic_controlled(
        self: Self,
        op: OpType,
        target: int,
        control: Control | int,
        creg: tuple[int, int],
        expected_value: int = 1,
        params: Sequence[float] = (),
    ) -> None:
        """Add a classic-controlled operation to the circuit.

        Args:
            op: The operation to apply
            target: The target qubit
            control: The control qubit
            creg: The classical register (index and number of bits)
            expected_value: The expected value of the classical register
            params: The parameters of the operation
        """

    @overload
    def classic_controlled(
        self: Self,
        op: OpType,
        target: int,
        controls: set[Control | int],
        creg: tuple[int, int],
        expected_value: int = 1,
        params: Sequence[float] = (),
    ) -> None:
        """Add a classic-controlled operation to the circuit.

        Args:
            op: The operation to apply
            target: The target qubit
            controls: The control qubits
            creg: The classical register (index and number of bits)
            expected_value: The expected value of the classical register
            params: The parameters of the operation
        """

__all__ = [
    "Permutation",
    "QuantumComputation",
]
