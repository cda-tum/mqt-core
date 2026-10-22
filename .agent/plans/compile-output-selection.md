# Compile output selection

Status: complete; local validation passed, hosted CI remains separate.

## Contract

`compile_program` selects the returned representation independently of the
hardware target. Explicit `output` returns its typed program. The default
preserves the input family without a target or with a hardware-only
`CompilerTarget`; Qiskit targets select `QuantumCircuit`, and QDMI targets
select a submission-ready `CompiledProgram`. `output` is the sole compiler
format selector. Exact QDMI transport selection belongs to
`submit_program(..., program_format=...)`; automatic precompilation retains the
selected native payload for repeated submissions.

Python does not offer exact-wire-format precompilation into `CompiledProgram`.
Use automatic reusable precompilation, a typed export, or exact-format
compile-and-submit. The native environment-based API retains offline payload
compilation.

Source and path inputs return program objects. Raw QC import and QCO are
pre-compilation checkpoints and cannot be combined with a target. jeff does not
retain physical placement. Supported exporter subsets remain unchanged.

## Implementation and validation

The bindings retain frontend format identity, resolve target and output
constraints, and call the shared native pipeline. QC output retains layout by
deferring resource-changing export cleanup. Capabilities are restricted by both
the output representation and the target's known contract. Different formats
share structural control-flow constraints; matching language versions and
profiles also share format-specific capabilities, independently of text or
binary encoding. The shared pipeline validates the final QIR feature
requirements, including loops introduced during lowering. Qiskit export receives
the complete environment when compiling, and retains a target-only overload for
direct conversions. All 11 supported capability IDs and three structural limits
are covered. Required QIR loop capabilities must be unconstrained because final
lowering does not retain a proof of their bounds. QDMI 1.4 has no detailed
per-format capability descriptors; its device snapshots use format baselines.

`bindings/patterns.txt` uses literal output overloads and a constrained type
variable for same-family defaults. Existing Python tests in
`test/python/test_mlir*.py` and `test/python/qdmi/test_compilation.py` cover
compilation, simulation, submission, ownership, layout, parameters, phase, and
capability limits. The native payload test checks the capability intersection
directly. These cover representative combinations without an input/target/output
Cartesian product. Tests use direct simulation, export, submission, and module
inspection; serialization, ownership, phase, and layout checks retain their
distinct oracles.

Validation on main revision `422a69c1a3`: 4,040 native tests; 1,471 MLIR, QDMI,
and magic-state-distillation Python tests with Qiskit 2.6.0; and 722
compatibility tests with Qiskit 2.5.2. Generated stubs, Python 3.11 return-type
checks, whole-file C++ lint, and executable documentation with internal links
passed. Twenty-six compiler tests passed after the final style-only native
edits.
