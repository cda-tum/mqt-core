#ifndef MQTDYN_INTERFACES
#define MQTDYN_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def UnitaryInterface : OpInterface<"UnitaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unitary
        operations in the MQTDyn dialect. Unitary operations are quantum
        operations that are reversible and can be represented by a unitary
        matrix. The interface provides methods to access the operands and
        results of the operation.
    }];

    let cppNamespace = "::mqt::ir::dyn";

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
}

def UnaryInterface : OpInterface<"UnaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unary
        operations in the MQTDyn dialect. Unary operations are quantum
        operations that take exactly one input qubit plus optional controlling
        qubits.
    }];

    let cppNamespace = "::mqt::ir::dyn";

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
        operations in the MQTDyn dialect. Unary operations are quantum
        operations that take exactly two input qubits plus optional controlling
        qubits.
    }];

    let cppNamespace = "::mqt::ir::dyn";

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
        operations in the MQTDyn dialect. Unary operations are quantum
        operations that take exactly three input qubits plus optional controlling
        qubits.
    }];

    let cppNamespace = "::mqt::ir::dyn";

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

    let cppNamespace = "::mqt::ir::dyn";

    let verify = [{
        auto noParamOp = mlir::cast<ConcreteOp>($_op);
        const auto numParams = noParamOp.getParams().size() + (noParamOp.getStaticParams().has_value() ? noParamOp.getStaticParams().value().size() : 0);
        if (numParams != 0) {
            return $_op->emitError() <<
                "operation does not accept any parameters but got " << numParams;
        }
        if (noParamOp.getParamsMask().has_value()) {
            return $_op->emitError() <<
                "operation does not accept any parameters but has a parameter mask";
        }
        return mlir::success();
    }];
}

def OneParameterInterface : OpInterface<"OneParameterInterface"> {
    let description = [{
        This interface provides a generic way to interact with operations that
        have exactly one parameter.
    }];

    let cppNamespace = "::mqt::ir::dyn";

    let verify = [{
        auto oneParamOp = mlir::cast<ConcreteOp>($_op);
        const auto numParams = oneParamOp.getParams().size() + (oneParamOp.getStaticParams().has_value() ? oneParamOp.getStaticParams().value().size() : 0);
        if (numParams != 1) {
            return $_op->emitError() <<
                "operation expects exactly one parameter but got " << numParams;
        }
        if (oneParamOp.getParamsMask().has_value()) {
            return $_op->emitError() <<
                "operation expects exactly one parameter but has a parameter mask";
        }
        return mlir::success();
    }];
}

def TwoParametersInterface : OpInterface<"TwoParametersInterface"> {
    let description = [{
        This interface provides a generic way to interact with operations that
        have exactly two parameters.
    }];

    let cppNamespace = "::mqt::ir::dyn";

    let verify = [{
        auto twoParamOp = mlir::cast<ConcreteOp>($_op);
        const auto numParams = twoParamOp.getParams().size() +
                               (twoParamOp.getStaticParams().has_value()
                                   ? twoParamOp.getStaticParams().value().size()
                                   : 0);
        if (numParams != 2) {
            return $_op->emitError()
                << "operation expects exactly two parameters but got "
                << numParams;
        }
        if (!twoParamOp.getParams().empty() &&
            twoParamOp.getStaticParams().has_value() &&
            !twoParamOp.getParamsMask().has_value()) {
            return $_op->emitError()
                << "operation has mixed dynamic and static parameters but no parameter mask";
        }
        if (twoParamOp.getParamsMask().has_value() &&
            twoParamOp.getParamsMask().value().size() != 2) {
            return $_op->emitError()
                << "operation expects exactly two parameters but has a parameter mask with "
                << twoParamOp.getParamsMask().value().size()
                << " entries";
        }
        if (twoParamOp.getParamsMask().has_value()) {
            auto array = twoParamOp.getParamsMask().value();
            const auto true_entries = static_cast<std::size_t>(
                    std::count_if(array.begin(), array.end(), [](bool b) { return b; }));
            if ((!twoParamOp.getStaticParams().has_value() ||
                 twoParamOp.getStaticParams().value().empty()) &&
                true_entries != 0) {
                return $_op->emitError()
                    << "operation has no static parameter but has a parameter mask with "
                    << true_entries
                    << " true entries";
            }
            if (twoParamOp.getStaticParams().value().size() != true_entries) {
                return $_op->emitError() <<
                    "operation has "
                    << (twoParamOp.getStaticParams().value().size() == 1
                        ? "one static parameter"
                        : "two static parameters" )
                    << " but has a parameter mask with "
                    << true_entries
                    << " true entries";
            }
        }
        return mlir::success();
    }];
}

def ThreeParametersInterface : OpInterface<"ThreeParametersInterface"> {
    let description = [{
        This interface provides a generic way to interact with operations that
        have exactly three parameters.
    }];

    let cppNamespace = "::mqt::ir::dyn";

    let verify = [{
        auto threeParamOp = mlir::cast<ConcreteOp>(op);
        const auto numParams = threeParamOp.getParams().size() +
                               (threeParamOp.getStaticParams().has_value()
                                    ? threeParamOp.getStaticParams().value().size()
                                    : 0);
        if (numParams != 3) {
            return op->emitError()
                   << "operation expects exactly three parameters but got "
                   << numParams;
        }
        if (!threeParamOp.getParams().empty() &&
            threeParamOp.getStaticParams().has_value() &&
            !threeParamOp.getParamsMask().has_value()) {
            return op->emitError() << "operation has mixed dynamic and static "
                                      "parameters but no parameter mask";
        }
        if (threeParamOp.getParamsMask().has_value() &&
            threeParamOp.getParamsMask().value().size() != 3) {
            return op->emitError()
                   << "operation expects exactly three parameters but has a "
                      "parameter mask with "
                   << threeParamOp.getParamsMask().value().size() << " entries";
        }
        if (threeParamOp.getParamsMask().has_value()) {
            auto array = threeParamOp.getParamsMask().value();
            const auto true_entries = static_cast<std::size_t>(
                    std::count_if(array.begin(), array.end(), [](bool b) { return b; }));
            if ((!threeParamOp.getStaticParams().has_value() ||
                 threeParamOp.getStaticParams().value().empty()) &&
                true_entries != 0) {
                return op->emitError() << "operation has no static parameter but has a "
                                          "parameter mask with "
                                       << true_entries << " true entries";
            }
            if (threeParamOp.getStaticParams().value().size() != true_entries) {
                return op->emitError()
                       << "operation has "
                       << (threeParamOp.getStaticParams().value().size() == 1
                               ? "one static parameter"
                               : (threeParamOp.getStaticParams().value().size() == 2
                                      ? "two static parameters"
                                      : "three static parameters"))
                       << " but has a parameter mask with " << true_entries
                       << " true entries";
            }
        }
        return mlir::success();
    }];
}

#endif // MQTDYN_INTERFACES