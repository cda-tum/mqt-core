#ifndef MQTOPT_INTERFACES
#define MQTOPT_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def UnitaryInterface : OpInterface<"UnitaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unitary
        operations in the MQTOpt dialect. Unitary operations are quantum
        operations that are reversible and can be represented by a unitary
        matrix. The interface provides methods to access the operands and
        results of the operation.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);
        if (const auto in_qubits = gate.getInQubits().size() + gate.getPosCtrlQubits().size() + gate.getNegCtrlQubits().size();
            in_qubits != gate.getOutQubits().size()) {
            return $_op->emitError() <<
                "number of input qubits (" << in_qubits << ") " <<
                "and output qubits (" << gate.getOutQubits().size() << ") must be the same";
        }
        return mlir::success();
    }];
}

def NoParametersInterface : OpInterface<"NoParametersInterface"> {
    let description = [{
        This interface provides a generic way to interact with operations that
        do not have any parameters.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);
        if (!gate.getParams().empty()) {
            return $_op->emitError() <<
                "operation does not accept any parameters but got " << gate.getParams().size();
        }
        return mlir::success();
    }];
}

def OneParameterInterface : OpInterface<"OneParameterInterface"> {
    let description = [{
        This interface provides a generic way to interact with operations that
        have exactly one parameter.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);
        if (gate.getParams().size() != 1) {
            return $_op->emitError() <<
                "operation expects exactly one parameter but got " << gate.getParams().size();
        }
        return mlir::success();
    }];
}

def TwoParametersInterface : OpInterface<"TwoParametersInterface"> {
    let description = [{
        This interface provides a generic way to interact with operations that
        have exactly two parameters.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);
        if (gate.getParams().size() != 2) {
            return $_op->emitError() <<
                "operation expects exactly two parameters but got " << gate.getParams().size();
        }
        return mlir::success();
    }];
}

def ThreeParametersInterface : OpInterface<"ThreeParametersInterface"> {
    let description = [{
        This interface provides a generic way to interact with operations that
        have exactly three parameters.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let verify = [{
        auto gate = mlir::cast<ConcreteOp>($_op);
        if (gate.getParams().size() != 3) {
            return $_op->emitError() <<
                "operation expects exactly three parameters but got " << gate.getParams().size();
        }
        return mlir::success();
    }];
}

#endif // MQTOPT_INTERFACES
