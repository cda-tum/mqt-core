# Classical subroutines and borrowed arrays

Status: implemented and locally validated for PR #2816, targeting `main`.

## Goal and scope

Implement #2593: defined, nonrecursive OpenQASM `def` subroutines with by-value
`bool`, `int`, `uint`, `float[64]`, and `bit[N]` parameters and at most one
result. Support local classical variables, calls to earlier definitions, void
calls, and a final top-level `return`.

Add readonly and mutable array parameters with fixed dimensions or a fixed rank
and runtime dimensions (`#dim`). Borrow caller storage, including strided and
rank-reduced slices, through ordinary ranked memrefs without argument copies.

Preserve function boundaries through QC/QCO and OpenQASM export. Array outputs
(#2592), quantum parameters, recursion, externs, runtime scalar angle
signatures, and early or nested returns remain unsupported and receive
diagnostics. Mutable globals cannot be captured; global constants can.

## Decisions

- Reuse ordinary `func.func` and `func.call`, not unitary quantum calls.
- Pass bit vectors as integer values, with fresh local CBit storage when the
  callee mutates them. Caller storage must not alias that local storage.
- Reuse assignment conversions: unsized `bit` and `bool` use truthiness;
  explicitly sized `bit[N]` values retain exact-width conversion rules.
- Keep final-return-only control flow. General early returns would require
  return state through nested structured loops or whole-function control-flow
  reconstruction and are independent follow-up work.
- Export canonical scalar types where MLIR loses source spelling, preserving bit
  patterns and signed arithmetic rather than adding general type metadata.
- Require all selected array elements initialized on entry, also for mutable
  references. Calls preserve these facts; they do not initialize unknown caller
  storage. Reject overlap involving mutable arguments when disjointness cannot
  be proved. Dynamic actual dimensions cannot satisfy a fixed-shape parameter
  unless their lengths are known; use a rank-only parameter instead.
- Infer readonly export parameters from their uses, including nested callees,
  rather than adding metadata. Backend inlining may remove helper boundaries,
  but must retain caller-visible writes and the existing host-result ABI.
- Preserve the consuming OpenQASM export API merged in #2815.

## Work items

- [x] Complete and validate syntax, borrowing checks, dynamic dimensions, and QC
      emission.
- [x] Preserve references and calls on OpenQASM export/reimport and DD
      execution.
- [x] Validate supported QIR/jeff lowering and unsupported signature
      diagnostics.
- [x] Review the complete diff and run focused tests.

## Validation

The final functional code passes 501 focused tests: 214 OpenQASM target tests,
239 translation tests, 11 compiler pipeline tests, 28 jeff round-trip
regressions, and 9 DD/memref/call tests. Coverage includes static and dynamic
shapes, nested calls, negative and strided indices, rank reduction, empty views,
angle tables, overlapping assignments, initialization and alias rejection, and
caller-visible mutation. Round trips check both retained calls and DD execution;
backend tests exercise QIR JIT and serialized jeff execution, including scalar
bit-vector isolation and default cleanup of constant scalar conditions.

Borrowed helpers inline before jeff value-array lowering; scalar helpers remain
native. CBit supports mapped inlining, and SCCP precedes dead-value cleanup to
avoid poisoning unreachable quantum passthrough values. Empty rank-reduced views
use shape-only casts because upstream subviews reject zero extents; all retained
extents must be static in this case.

Whole-file C++ lint covered all 15 changed sources against the then-parent
branch (#2815), with a separate check of the final test-helper placement fix.
Repository lint passed. An additional header-inclusive parser check found only
the existing redundant `indices{}` initializer warning from #2574, which is
unchanged. No benchmarks, full test suite, or automations are part of this
extension.
