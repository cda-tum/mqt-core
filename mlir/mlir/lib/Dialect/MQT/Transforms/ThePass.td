#ifndef THE_PASS
#define THE_PASS

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/MQT/MQTOps.td"

def performThePass : NativeCodeCall<
    "performThePass($_builder, $0.getDefiningOp(), $1, $2, $3, $4, $5, $6)">;

def : Pat<(CustomOp:$old_value $params, $in_qubits, StrAttr:$gate_name, OptionalAttr<UnitAttr>:$adjoint, $in_ctrl_qubits, $in_ctrl_values),
          (CustomOp $params, $in_qubits, $gate_name, $adjoint, $in_ctrl_qubits, $in_ctrl_values)>;

#endif // THE_PASS
