# Compile output selection

Status: complete.

## Contract

`compile_program` selects the returned representation independently of the
hardware target. Explicit `output` returns its typed program. The default
preserves the input family without a target or with a hardware-only
`CompilerTarget`; Qiskit targets select `QuantumCircuit`, and QDMI targets
select a submission-ready `CompiledProgram`. `program_format` selects the
existing QDMI envelope and is mutually exclusive with `output`.

Source and path inputs return program objects. Raw QC import and QCO are
pre-compilation checkpoints and cannot be combined with a target. jeff does not
retain physical placement. Supported exporter subsets remain unchanged.

## Implementation and validation

The bindings retain frontend format identity, resolve target and output
constraints, and call the shared native pipeline. QC output retains layout by
deferring resource-changing export cleanup. Structural control flow is limited
by both the output representation and the target's known payload contract.
Qiskit export receives the complete environment when compiling, and retains a
target-only overload for direct conversions.

`bindings/patterns.txt` uses literal output overloads and a constrained type
variable for same-family defaults. Existing Python tests in
`test/python/test_mlir*.py` and `test/python/qdmi/test_compilation.py` cover
compilation, simulation, submission, ownership, layout, parameters, phase, and
capability limits. The native payload test checks the capability intersection
directly. These cover representative combinations without an input/target/output
Cartesian product.

Validation passed: 4,036 native tests; 1,120 MLIR/QDMI Python tests with Qiskit
2.6.0; and 722 compatibility tests with Qiskit 2.5.2. Return-type assertions
passed under Python 3.11 typing rules. Generated stubs, repository and
whole-file C++ lint, executable documentation, and link checks passed. The
output-selection and target-selection guides were also inspected in a browser.
Hosted CI is a separate validation step.
