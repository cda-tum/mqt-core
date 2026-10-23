# Classical scalar subroutines

Status: implemented and locally validated on #2815 (`4c1296c41`).

## Goal and scope

Implement the first stage of #2593: defined, nonrecursive OpenQASM `def`
subroutines with by-value `bool`, `int`, `uint`, `float[64]`, and `bit[N]`
parameters and at most one result. Support local classical variables, calls to
earlier definitions, void calls, and a final top-level `return`.

Preserve function boundaries through QC/QCO and OpenQASM export. Array
references, array outputs (#2592), quantum parameters, recursion, externs,
runtime angle signatures, and early or nested returns remain unsupported and
receive diagnostics. Mutable globals cannot be captured; global constants can.

## Decisions

- Reuse ordinary `func.func` and `func.call`, not unitary quantum calls.
- Pass bit vectors as integer values, with fresh local CBit storage when the
  callee mutates them. Caller storage must not alias that local storage.
- Reuse assignment conversions: unsized `bit` and `bool` use truthiness;
  explicitly sized `bit[N]` values retain exact-width conversion rules.
- Keep final-return-only control flow in this first PR. General early returns
  would require return state through nested structured loops or whole-function
  control-flow reconstruction and are independent follow-up work.
- Export canonical scalar types where MLIR loses source spelling, preserving bit
  patterns and signed arithmetic rather than adding general type metadata.
- Existing unsupported backend signatures must fail explicitly; this change does
  not introduce a new host-result ABI.
- Stack on #2815 and preserve its consuming OpenQASM export API.

## Validation

The OpenQASM target and translation GoogleTest binaries pass 212 and 238 tests,
respectively. New coverage checks QC/QCO and export/reimport semantics, nested
calls, local mutation isolation, narrow and wide values, Boolean conversions,
loops, rejected input, builtin-name collisions, and snapshot resource limits. DD
sampling checks returned values before and after the round trip.

Whole-changed-file C++ lint against the parent branch and repository lint pass.
A header-inclusive parser check also reports no project warnings. No benchmarks,
full test suite, new QIR/jeff signature tests, or automations are part of this
stage.
