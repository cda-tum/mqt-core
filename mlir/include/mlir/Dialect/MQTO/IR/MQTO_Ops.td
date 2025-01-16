#ifndef MQTO_OPS
#define MQTO_OPS

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def MQTO_Dialect : Dialect {
    let name = "mqto";
    let description = [{
        The MQT optimization (MQTO) dialect.
    }];
    let cppNamespace = "::mlir::mqto";
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

class MQTO_Op<string mnemonic, list<Trait> traits = []> : Op<MQTO_Dialect, mnemonic, traits # [NoMemoryEffect]>;

def OperationOp : MQTO_Op<"operation"> {
    let summary = "A standard operation";
    let description = [{
        A very nice description goes here.

        Example:
        ```mlir
        %q1 = "mqto.operation" (%q0) {"gate_name" = "Hadamard" : string} : (!mqto.Qubit) -> !mqto.Qubit
        %q4, %q5 = "mqto.operation" (%q2, %q3) {"gate_name" = "CX" : string} : (!mqto.Qubit, !mqto.Qubit) -> (!mqto.Qubit, !mqto.Qubit)
        ```
    }];
    let arguments = (ins
        StrAttr:$gate_name,
        Variadic<QubitType>:$in_qubits
    );
    let results = (outs
        Variadic<QubitType>:$out_qubits
    );
    let hasVerifier = 1;
}

def MeasureOp : MQTO_Op<"measure"> {
    let summary = "A measure operation";
    let description = [{
        A very nice description goes here.
    }];
    let arguments = (ins
        QubitType:$in_qubit
    );
    let results = (outs
        QubitType:$out_qubit,
        UI1:$out_bit
    );
}

def AllocOp : MQTO_Op<"allocQubitRegister"> {
    let summary = "Allocates a qubit register";
    let description = [{
        .
    }];
    let arguments = (ins
        Optional<UI64>:$nqubits,
        OptionalAttr<ConfinedAttr<UI64Attr, [IntNonNegative]>>:$nqubits_attr
    );
    let results = (outs
        QuregType:$qureg
    );
    let hasVerifier = 1;
}

def DeallocOp : MQTO_Op<"deallocQubitRegister"> {
    let summary = "Deallocates a qubit register";
    let description = [{
        .
    }];
    let arguments = (ins
        QuregType:$qureg
    );
}

def ExtractOp : MQTO_Op<"extractQubit"> {
    let summary = "Extract a specific qubit from a given register";
    let description = [{
        .
    }];
    let arguments = (ins
        QuregType:$in_qureg,
        Optional<UI64>:$index,
        OptionalAttr<ConfinedAttr<UI64Attr, [IntNonNegative]>>:$index_attr
    );
    let results = (outs
        QuregType:$out_qureg,
        QubitType:$out_qubit
    );
    let hasVerifier = 1;
}

def InsertOp : MQTO_Op<"insertQubit"> {
    let summary = "Insert a qubit to an empty register index";
    let description = [{
        .
    }];
    let arguments = (ins
        QuregType:$in_qureg,
        QubitType:$in_qubit,
        Optional<UI64>:$index,
        OptionalAttr<ConfinedAttr<UI64Attr, [IntNonNegative]>>:$index_attr
    );
    let results = (outs
        QuregType:$out_qureg
    );
    let hasVerifier = 1;
}

#endif // MQTO_OPS
