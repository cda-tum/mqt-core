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

    let methods = [
        InterfaceMethod<
            /*desc=*/        "Returns all input qubits of the operation excl. controlling qubits.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getInQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all positively controlling qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getPosCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getPosCtrlQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all negatively controlling qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getNegCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getNegCtrlQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all output qubits of the operation incl. all controlling qubits.",
            /*returnType=*/  "mlir::ResultRange",
            /*methodName=*/  "getOutQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getOutQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all controlling qubits of the operation incl. positive and negative controls.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                std::vector<mlir::Value> controls{};
                const auto& posCtrlQubits = $_op.getPosCtrlQubits();
                controls.insert(controls.end(), posCtrlQubits.begin(), posCtrlQubits.end());
                const auto& negCtrlQubits = $_op.getNegCtrlQubits();
                controls.insert(controls.end(), negCtrlQubits.begin(), negCtrlQubits.end());
                return controls;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all input qubits of the operation incl. all controlling qubits.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getAllInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                std::vector<mlir::Value> operands{};
                const auto& inQubits = $_op.getInQubits();
                operands.insert(operands.end(), inQubits.begin(), inQubits.end());
                const auto& controls = $_op.getCtrlQubits();
                operands.insert(operands.end(), controls.begin(), controls.end());
                return operands;
            }]>
    ];

    let verify = [{
        auto unitaryOp = mlir::cast<ConcreteOp>($_op);
        if (const auto in_qubits = unitaryOp.getInQubits().size() + unitaryOp.getPosCtrlQubits().size() + unitaryOp.getNegCtrlQubits().size();
            in_qubits != unitaryOp.getOutQubits().size()) {
            return $_op->emitError() <<
                "number of input qubits (" << in_qubits << ") " <<
                "and output qubits (" << unitaryOp.getOutQubits().size() << ") must be the same";
        }
        return mlir::success();
    }];
}

def UnaryInterface : OpInterface<"UnaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unary
        operations in the MQTOpt dialect. Unary operations are quantum
        operations that take exactly one input qubit plus optional controlling
        qubits.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let verify = [{
        auto unaryOp = mlir::cast<ConcreteOp>($_op);
        if (unaryOp.getInQubits().size() != 1) {
            return $_op->emitError() <<
                "number of input qubits (" << unaryOp.getInQubits().size() << ") must be the one";
        }
        return mlir::success();
    }];
}

def BinaryInterface : OpInterface<"BinaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unary
        operations in the MQTOpt dialect. Unary operations are quantum
        operations that take exactly two input qubits plus optional controlling
        qubits.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let verify = [{
        auto binaryOp = mlir::cast<ConcreteOp>($_op);
        if (binaryOp.getInQubits().size() != 2) {
            return $_op->emitError() <<
                "number of input qubits (" << binaryOp.getInQubits().size() << ") must be the two";
        }
        return mlir::success();
    }];
}

def TernaryInterface : OpInterface<"TernaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unary
        operations in the MQTOpt dialect. Unary operations are quantum
        operations that take exactly three input qubits plus optional controlling
        qubits.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let verify = [{
        auto ternaryOp = mlir::cast<ConcreteOp>($_op);
        if (ternaryOp.getInQubits().size() != 3) {
            return $_op->emitError() <<
                "number of input qubits (" << ternaryOp.getInQubits().size() << ") must be the three";
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
        auto noParamOp = mlir::cast<ConcreteOp>($_op);
        if (!noParamOp.getParams().empty()) {
            return $_op->emitError() <<
                "operation does not accept any parameters but got " << noParamOp.getParams().size();
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
        auto oneParamOp = mlir::cast<ConcreteOp>($_op);
        if (oneParamOp.getParams().size() != 1) {
            return $_op->emitError() <<
                "operation expects exactly one parameter but got " << oneParamOp.getParams().size();
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
        auto twoParamOp = mlir::cast<ConcreteOp>($_op);
        if (twoParamOp.getParams().size() != 2) {
            return $_op->emitError() <<
                "operation expects exactly two parameters but got " << twoParamOp.getParams().size();
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
        auto threeParamOp = mlir::cast<ConcreteOp>($_op);
        if (threeParamOp.getParams().size() != 3) {
            return $_op->emitError() <<
                "operation expects exactly three parameters but got " << threeParamOp.getParams().size();
        }
        return mlir::success();
    }];
}

#endif // MQTOPT_INTERFACES
