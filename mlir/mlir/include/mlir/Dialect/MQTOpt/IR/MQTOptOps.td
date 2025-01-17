#ifndef MQTOPT_OPS
#define MQTOPT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def MQTOptDialect : Dialect {
    // The dialect name used in the MLIR file to prefix operations.
    let name = "mqtopt";

    let summary = "The MQT optimization (mqtopt) dialect.";

    let description = [{
        This dialect is using value semantics for qubits. This means that
        variables of type qubit and the elements of qubit registers do not
        identify a physical, they rather refer to the state of a qubit. A qubit
        state can be assigned only once and can also be used only once.
        Otherwise, the no-cloning theorem would be violated.

        The first requirement is enforced by the SSA form of MLIR. The second
        requirement is not explicitly enforced. However, when converting the
        input dialect mqt that is using reference semantics to the mqtopt dialect,
        qubit state values will only be used once by construction. This is also
        the reason why the mqtopt dialect is not meant to be used by hand, but
        rather as an intermediate representation for optimization passes.

        Due to the value semantics, the mqtopt dialect allows dataflow analysis
        and optimizations. Those are very common in classical compiler
        optimizations. Hence, the existing classical compiler optimizations can
        easier be reused for quantum programs.

        For more information, see the paper "QIRO:A Static Single Assignment
        based Quantum Program Representation for Optimization"
        (https://doi.org/10.1145/3491247)
    }];

    // The C++ namespace that the dialect, and all sub-components, get placed
    // in. Here, `mlir::` must either appear at the front or not at all.
    let cppNamespace = "::mqt::ir::opt";

    // This dialect defines its own types for qubits and qubit registers. To
    // parse and print these, respective hooks must be defined. With this flag
    // set to 1, the default hooks are created and used.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class MQTOptType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<MQTOptDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : MQTOptType<"Qubit", "Qubit"> {
    let summary = "A value-semantic qubit (state).";
}

def QuregType : MQTOptType<"QubitRegister", "QubitRegister"> {
    let summary = "An array of value-semantic qubits (i.e. quantum register).";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// This is the base class for all operations in the MQTOpt dialect. It is a
// template that takes the operation mnemonic and a list of traits. The MQTOptOp
// class is a subclass of the Op class defined in the MLIR core. Every operation
// in the MQTOpt dialect has no side effects due to the value semantics of qubits.
// This is accomplished by the [NoMemoryEffect] trait.
class MQTOptOp<string mnemonic, list<Trait> traits = []> :
        Op<MQTOptDialect, mnemonic, traits # [NoMemoryEffect]>;

class GateOp<string mnemonic, list<Trait> traits = []> :
    MQTOptOp<mnemonic, traits> {
}

class UnitaryOp<string mnemonic, list<Trait> traits = []> :
    GateOp<mnemonic, traits>;

class UnaryUnitaryOp<string mnemonic, list<Trait> traits = []> :
    UnitaryOp<mnemonic, traits> {
    let arguments = (ins
        QubitType:$in_qubit,
        Variadic<QubitType>:$ctrl_in_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
        `(` `)` attr-dict $in_qubit ( `ctrl` `(` $ctrl_in_qubits^ `)` )?
        `:` `(` type($in_qubit) ( `,` type($ctrl_in_qubits)^ )? `)`
        `->` `(` type($out_qubits) `)`
    }];
}

def XOp : UnaryUnitaryOp<"x"> {
    let summary = "X operation";

    let description = [{
        This class represents a standard quantum operation. It has a gate name
        attribute and a variadic number of input qubits. It returns the same
        number of qubits as it takes as input.

        Example:
        ```mlir
        %0_1 = "mqtopt.x"(%0) : (!mqtopt.Qubit) -> !mqtopt.Qubit
        %0_1, %1_1 = "mqtopt.x"(%0_1, [%1]) : (!mqtopt.Qubit, !mqtopt.Qubit) -> !mqtopt.Qubit
        %0_2, %1_2, %2_1,  = "mqtopt.x"(%0_1, [%1_1], [%2]) : (!mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit) -> !mqtopt.Qubit
        ```
    }];
}

class NonUnitaryOp<string mnemonic, list<Trait> traits = []> :
    GateOp<mnemonic, traits>;

def MeasureOp : NonUnitaryOp<"measure", [SameVariadicResultSize]> {
    let summary = "A measure operation";

    let description = [{
        This class represents a measure operation. It takes a single qubit as
        input and returns a qubit and a bit. After the measurement, the returned
        qubit is either in the state |0> or |1> and is indicated by the returned
        bit.

        Example:
        ```mlir
        %0_1, %1 = "mqtopt.measure"(%0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        ```
    }];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits,
        Variadic<I1>:$out_bits
    );
}

class ResourceOp<string mnemonic, list<Trait> traits = []> : MQTOptOp<mnemonic, traits>;

def AllocOp : ResourceOp<"allocQubitRegister"> {
    let summary = "Allocates a qubit register";

    let description = [{
        This class represents an operation that allocates a qubit register. The
        size can either be specified during runtime as an operand or as an
        attribute when the size is known at compile time. Before the qubits in
        the register can be used, they need to be extracted from the register
        with the extract operation.

        Example:
        ```mlir
        %0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        %c0_i64 = arith.constant 2 : i64
        %1 = "mqtopt.allocQubitRegister"(%c0_i64) : (i64) -> !mqtopt.QubitRegister
        ```
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

def DeallocOp : ResourceOp<"deallocQubitRegister"> {
    let summary = "Deallocates a qubit register";

    let description = [{
        This class represents an operation that deallocates a qubit register.
        The qubit register is no longer valid after this operation. To be valid
        all extracted qubits had to be inserted back into the register. However,
        this property is not enforced by the dialect.

        Example:
        ```
        "mqtopt.deallocQubitRegister"(%0) : (!mqtopt.QubitRegister) -> ()
        ```
    }];

    let arguments = (ins
        QuregType:$qureg
    );
}

def ExtractOp : ResourceOp<"extractQubit"> {
    let summary = "Extract a specific qubit from a given register";

    let description = [{
        This class represents an operation that extracts a specific qubit from
        a given register. Before, the register must be allocated with the
        allocateQubitRegister operation. Upon invocation of this operation, the
        qubit is removed from the register and can be used in operations. A gap
        is left in the register that can be filled with the insert operation.

        Extract:

        %out_qureg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

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

def InsertOp : ResourceOp<"insertQubit"> {

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

#endif // MQTOPT_OPS
