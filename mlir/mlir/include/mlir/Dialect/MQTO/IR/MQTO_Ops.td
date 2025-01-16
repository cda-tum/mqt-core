#ifndef MQTO_OPS
#define MQTO_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def MQTO_Dialect : Dialect {
    // The dialect name used in the MLIR file to prefix operations.
    let name = "mqto";

    let summary = "The MQT optimization (mqto) dialect.";

    let description = [{
        This dialect is using value semantics for qubits. This means that
        variables of type qubit and the elements of qubit registers do not
        identify a physical, they rather refer to the state of a qubit. A qubit
        state can be assigned only once and can also be used only once.
        Otherwise, the no-cloning theorem would be violated.

        The first requirement is enforced by the SSA form of MLIR. The second
        requirement is not explicitly enforced. However, when converting the
        input dialect mqt that is using reference semantics to the mqto dialect,
        qubit state values will only be used once by construction. This is also
        the reason why the mqto dialect is not meant to be used by hand, but
        rather as an intermediate representation for optimization passes.

        Due to the value semantics, the mqto dialect allows dataflow analysis
        and optimizations. Those are very common in classical compiler
        optimizations. Hence, the existing classical compiler optimizations can
        easier be reused for quantum programs.

        For more information, see the paper "QIRO:A Static Single Assignment
        based Quantum Program Representation for Optimization"
        (https://doi.org/10.1145/3491247)
    }];

    // The C++ namespace that the dialect, and all sub-components, get placed
    // in. This must start with "::mlir::" and cannot start with "::mqt::", for
    // example.
    let cppNamespace = "::mlir::mqto";

    // This dialect defines its own types for qubits and qubit registers. To
    // parse and print these, respective hooks must be defined. With this flag
    // set to 1, the default hooks are created and used.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class MQTO_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<MQTO_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : MQTO_Type<"Qubit", "Qubit"> {
    let summary = "A value-semantic qubit (state).";
}
def QuregType : MQTO_Type<"QubitRegister", "QubitRegister"> {
    let summary = "An array of value-semantic qubits (i.e. quantum register).";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// This is the base class for all operations in the MQTO dialect. It is a
// template that takes the operation mnemonic and a list of traits. The MQTO_Op
// class is a subclass of the Op class defined in the MLIR core. Every operation
// in the MQTO dialect has no side effects due to the value semantics of qubits.
// This is accomplished by the [NoMemoryEffect] trait.
class MQTO_Op<string mnemonic, list<Trait> traits = []> :
        Op<MQTO_Dialect, mnemonic, traits # [NoMemoryEffect]>;

def OperationOp : MQTO_Op<"operation"> {
    let summary = "A standard operation";

    let description = [{
        This class represents a standard quantum operation. It has a gate name
        attribute and a variadic number of input qubits. It returns the same
        number of qubits as it takes as input.

        Example:
        ```mlir
        %2:2 = "mqto.operation"(%0, %1) <{gate_name = "CX"}> : (!mqto.Qubit, !mqto.Qubit) -> (!mqto.Qubit, !mqto.Qubit)
        %3 = "mqto.operation"(%2#0) <{gate_name = "H"}> : (!mqto.Qubit) -> !mqto.Qubit
        ```
    }];

    let arguments = (ins
        StrAttr:$gate_name,
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    // This operation has a verifier that checks if the number of input qubits
    // is equal to the number of output qubits. It is implemented in the
    // corresponding .cpp file.
    let hasVerifier = 1;
}

def MeasureOp : MQTO_Op<"measure"> {
    let summary = "A measure operation";

    let description = [{
        This class represents a measure operation. It takes a single qubit as
        input and returns a qubit and a bit. After the measurement, the returned
        qubit is either in the state |0> or |1> and is indicated by the returned
        bit.

        Example:
        ```mlir
        %1, %2 = "mqto.measure"(%0) : (!mqto.Qubit) -> (!mqto.Qubit, i1)
        ```
    }];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        QubitType:$out_qubit,
        I1:$out_bit
    );
}

def AllocOp : MQTO_Op<"allocQubitRegister"> {
    let summary = "Allocates a qubit register";

    let description = [{
        This class represents an operation that allocates a qubit register. The
        size can either be specified during runtime as an operand or as an
        attribute when the size is known at compile time. Before the qubits in
        the register can be used, they need to be extracted from the register
        with the extract operation.
    }];

    let arguments = (ins
        Optional<I64>:$size,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$size_attr
    );

    let results = (outs
        QuregType:$qureg
    );

    // This operation has a verifier that checks if the size is given either as
    // operand or attribute and not both. It is implemented in the corresponding
    // .cpp file.
    let hasVerifier = 1;
}

def DeallocOp : MQTO_Op<"deallocQubitRegister"> {
    let summary = "Deallocates a qubit register";

    let description = [{
        This class represents an operation that deallocates a qubit register.
        The qubit register is no longer valid after this operation. To be valid
        all extracted qubits had to be inserted back into the register. However,
        this property is not enforced by the dialect.
    }];

    let arguments = (ins
        QuregType:$qureg
    );
}

def ExtractOp : MQTO_Op<"extractQubit"> {
    let summary = "Extract a specific qubit from a given register";

    let description = [{
        This class represents an operation that extracts a specific qubit from
        a given register. Before, the register must be allocated with the
        allocateQubitRegister operation. Upon invocation of this operation, the
        qubit is removed from the register and can be used in operations. A gap
        is left in the register that can be filled with the insert operation.
    }];

    let arguments = (ins
        QuregType:$in_qureg,
        Optional<I64>:$index,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$index_attr
    );

    let results = (outs
        QuregType:$out_qureg,
        QubitType:$out_qubit
    );

    // This operation has a verifier that checks if the index is given either as
    // operand or attribute and not both. It is implemented in the corresponding
    // .cpp file.
    let hasVerifier = 1;
}

def InsertOp : MQTO_Op<"insertQubit"> {
    let summary = "Insert a qubit to an empty register index";

    let description = [{
        This class represents an operation that inserts a qubit back into a qubit
        register. Before, the same underlying qubit must have been extracted
        from the same register at the same index. However, this property is not
        enforced by the dialect. It is naturally satisfied when a program in the
        mqt input dialect is converted to the mqto dialect.
    }];

    let arguments = (ins
        QuregType:$in_qureg,
        QubitType:$in_qubit,
        Optional<I64>:$index,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$index_attr
    );

    let results = (outs
        QuregType:$out_qureg
    );

    // This operation has a verifier that checks if the index is given either as
    // operand or attribute and not both. It is implemented in the corresponding
    // .cpp file.
    let hasVerifier = 1;
}

#endif // MQTO_OPS
