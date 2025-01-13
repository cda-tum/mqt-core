// RUN: quantum-opt %s | quantum-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        // CHECK: %{{.*}} = quantum.alloc( 1) : !quantum.reg
        %0 = quantum.alloc( 1) : !quantum.reg
        return
    }
}
