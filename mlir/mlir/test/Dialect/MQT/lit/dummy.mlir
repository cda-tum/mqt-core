// RUN: quantum-opt %s | quantum-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        // CHECK: %{{.*}} = mqt.alloc( 1) : !mqt.reg
        %0 = mqt.alloc( 1) : !mqt.reg
        return
    }
}
