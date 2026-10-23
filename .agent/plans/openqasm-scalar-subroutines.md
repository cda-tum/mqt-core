# Classical subroutines and borrowed arrays

Status: complete and locally validated for PR #2816.

## Goal and scope

Implement #2593 with scalar subroutines and borrowed array parameters. The
[OpenQASM guide](../../docs/mlir/OpenQASM.md) defines supported syntax,
initialization, aliasing, and backend limits. Array-valued outputs (#2592)
remain separate.

## Decisions

- Use `func.func`/`func.call` and ranked memrefs. Bit vectors remain by-value;
  borrowed arrays retain caller storage without argument copies.
- Infer readonly use and export canonical scalar types instead of storing source
  metadata. Preserve signed arithmetic and bit patterns.
- Inline borrowed helpers before jeff's value-array lowering; retain scalar
  helpers. CBit permits mapped inlining for the QIR path.
- Run SCCP before dead-value cleanup: unreachable quantum branches otherwise
  acquire poison values that violate positional linearity.
- Static empty rank-reduced views use shape-only casts because LLVM 23 subviews
  reject zero extents. Remove this workaround when upstream accepts them.

## Validation

501 focused tests passed: OpenQASM target (214), translation (239), compiler
pipeline (11), jeff round-trip (28), and DD utilities (9). They cover retained
calls, one-time compound-index evaluation, initialization, empty/strided views,
QIR JIT, and serialized jeff. Run the component GoogleTest binaries under
`build/release/mlir/unittests/` after `cmake --build --preset release`. C++ and
repository lint passed. Header checks found only the existing redundant
`indices{}` initializer in `Frontend.h`. No full suite, benchmarks, or
automations were run.
