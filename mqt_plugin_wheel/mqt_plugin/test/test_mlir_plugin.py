# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file performs the frontend lit tests that the peephole transformations are correctly lowered.

We check the transform jax primitives for each pass is correctly injected
during tracing, and these transform primitives are correctly lowered to the mlir before
running -apply-transform-sequence.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long
from __future__ import annotations

import pennylane as qml
from catalyst import CompileError, pipeline
from utils import qjit_for_tests as qjit


def flush_peephole_opted_mlir_to_iostream(QJIT) -> None:
    """The QJIT compiler does not offer a direct interface to access an intermediate mlir in the pipeline.
    The `QJIT.mlir` is the mlir before any passes are run, i.e. the "0_<qnode_name>.mlir".
    Since the QUANTUM_COMPILATION_PASS is located in the middle of the pipeline, we need
    to retrieve it with keep_intermediate=True and manually access the "2_QuantumCompilationPass.mlir".
    Then we delete the kept intermediates to avoid pollution of the workspace.
    """


def test_MQT_plugin() -> bool | None:
    """Generate MLIR for the MQT plugin via entry-point.

    HACK required to make this test work!

    1. in catalyst/pipelines.py, add the following line:

        def get_enforce_runtime_invariants_stage(_options: CompileOptions) -> List[str]:
            ...
            enforce_runtime_invariants = [
                ...
                "builtin.module(mqt-core-round-trip)", # HACK
                "builtin.module(apply-transform-sequence)",
                ...
            ]

    2. in catalyst/compiler.py, add the following line:

        def run_from_ir(self, ir: str, module_name: str, workspace: Directory):
            ...
            cmd = self.get_cli_command(tmp_infile_name, output_ir_name, module_name, workspace)
            cmd += ["--debug"] # HACK
            try:
                ...

    """
    my_pipeline = {
        # "mqt.mqt-core-round-trip": {"cmap": [[0, 1], [1, 0]]},
    }

    try:

        @qjit(keep_intermediate=True, verbose=True)
        @pipeline(my_pipeline)
        @qml.qnode(qml.device(name="lightning.qubit", wires=3))
        def test_pipeline_mqtplugin_workflow() -> None:
            qml.CNOT(wires=[1, 0])
            qml.CNOT(wires=[2, 1])
            qml.CNOT(wires=[1, 0])
            qml.CNOT(wires=[2, 0])

        test_pipeline_mqtplugin_workflow()
        flush_peephole_opted_mlir_to_iostream(test_pipeline_mqtplugin_workflow)

    except CompileError as error:  # Expecting failure, because MQT plugin does not cover full roundtrip (yet)
        error_msg = str(error)  # Recover the output after application of the MQT conversion pass
        try:
            mlir_str = error_msg.split("module @module_test_pipeline_mqtplugin_workflow_transformed {")[1]
            mlir_str = error_msg.split("}\n}")[0]
        except Exception:
            return False

        transformed = [
            r"%c = stablehlo.constant dense<0> : tensor<i64>",
            r"%extracted = tensor.extract %c[] : tensor<i64>",
            r"quantum.device shots(%extracted) ",  # ... there is more in this line
            r"%c_0 = stablehlo.constant dense<1> : tensor<i64>",
            r'%0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister',
            r"%c_1 = stablehlo.constant dense<0> : tensor<i64>",
            r"%extracted_2 = tensor.extract %c_1[] : tensor<i64>",
            r'%out_qureg, %out_qubit = "mqtopt.extractQubit"(%0, %extracted_2) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)',
            r"%1 = mqtopt.H() %out_qubit : !mqtopt.Qubit",
            r"%c_3 = stablehlo.constant dense<0> : tensor<i64>",
            r"%extracted_4 = tensor.extract %c_3[] : tensor<i64>",
            r'%2 = "mqtopt.insertQubit"(%out_qureg, %1, %extracted_4) : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister',
            r'"mqtopt.deallocQubitRegister"(%2) : (!mqtopt.QubitRegister) -> ()',
            r"quantum.device_release",
            r"return",
        ]

        return all(t in mlir_str for t in transformed)

    return False


test_MQT_plugin()
