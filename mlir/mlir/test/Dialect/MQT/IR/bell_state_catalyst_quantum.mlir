module @circuit {
  func.func public @jit_circuit() -> tensor<2xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
  module @module_circuit {
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
        transform.yield 
      }
    }
    func.func public @circuit() -> tensor<2xf64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
      %c0_i64 = arith.constant 0 : i64
      quantum.device shots(%c0_i64) ["/Users/patrickhopf/Code/mqt/mqt-mlir/.venv/lib/python3.12/site-packages/pennylane_lightning/liblightning_qubit_catalyst.dylib", "LightningSimulator", "{'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
      %0 = quantum.alloc( 2) : !quantum.reg
      %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
      %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
      %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits, %2 : !quantum.bit, !quantum.bit
      %3 = quantum.compbasis %out_qubits_0#1 : !quantum.obs
      %4 = quantum.probs %3 : tensor<2xf64>
      %5 = quantum.insert %0[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit
      %6 = quantum.insert %5[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit
      quantum.dealloc %6 : !quantum.reg
      quantum.device_release
      return %4 : tensor<2xf64>
    }
  }
  func.func @setup() {
    quantum.init
    return
  }
  func.func @teardown() {
    quantum.finalize
    return
  }
}