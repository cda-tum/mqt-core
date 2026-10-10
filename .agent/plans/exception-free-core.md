# Exception-free Core APIs

Status: in progress. The migration is based on main `62125acad`. Platform checks
and whole-PR performance acceptance remain open. See the
[audit](../audits/exception-free-rebase.md) for integration fixes and retained
coverage.

## Work remaining

- [ ] Compare matched successful workloads against upstream, including cold and
  reused DDSIM workers and state transfer.
- [ ] Validate the published revision's Windows/macOS packages and hosted
  checks.

## Contract

All Core libraries require LLVM/MLIR 23.1 or newer. Native fallible APIs use
`llvm::FailureOr<T>` and `llvm::LogicalResult`; MLIR-facing APIs use the same
types through MLIR aliases. Infallible APIs return ordinary values or `void`.
Successful absence remains `optional<T>`, and borrowed results use pointers.
Scoped thread-local diagnostics preserve severity, category, message, and QDMI
status. Binding invocation and test capture each share one implementation. See
[native error handling](../../docs/cpp_api.md#handle-native-errors).

Standalone driver and device builds embed diagnostic support. QDMI C interfaces
carry status codes and use local logging. Private driver APIs and their static
tests use scoped diagnostics without separate error-output parameters. Keep
C-status conversion at each runtime boundary. Do not pass `llvm::Error` between
independently embedded, hidden LLVM support archives: their error class
identities differ, and consuming such an error can abort. The driver keeps
exception unwinding enabled through its C ABI allocation-recovery boundary.

Preserve C++20, Python APIs, QDMI statuses, and result semantics. Private JSON
translation units catch dependency exceptions and parse valid input once,
retaining schema, duplicate-key, encoding, and numerical checks. DD kernels
remain exception-free; MSVC retains normal STL configuration and exception ABI.

Balanced DD reference ownership and valid measurement targets are native
preconditions. Python validates them at its binding boundary. QCO qubit mappings
and QIR address resolution establish valid targets internally. Keep fallibility
for public input, provider, numerical, compiler, and I/O errors. Typed gate
construction and operations on already-validated owned state need no repeated
checks. Benchmark evaluation validates outcomes once through its probability
callback; hashing uses LLVM SHA-256.

The JIT validates C calling conventions, native ABI attributes, and runtime call
signatures before binding host functions. ORC errors use scoped diagnostics. QIR
resource-allocation failures with explicit error outputs remain recoverable and
release partial allocations. Host allocation failures and other unhandled
runtime failures diagnose and terminate. The thin `noexcept` ABI facade retains
exception support; algorithms do not. Setup and compilation errors remain
recoverable. Session-owned classical allocations survive quantum resets until
reference-count release or session destruction. JIT constructors and destructors
bind the session runtime, as entry-point execution does.

DDSIM worker isolation and concurrency are supplied by the parent multi-program
change. This layer adapts worker compilation and compact DD reconstruction to
explicit results and carries structured diagnostics over the existing framed
transport. Crashes fail only their assigned programs without replay; completed
siblings retain their results. Cancellation, worker reuse, indexed results,
stable IDs, and standard-interface driver replacement keep their parent
contracts.

## Validation

Validation on Linux aarch64 with GCC 13.3 and LLVM/MLIR 23.1.0:

- `cmake --preset release -DENABLE_IPO=OFF`, the release build, and
  `ctest --preset release -j 8` pass: 4,046 runnable tests and the existing
  unsupported `ScQDMIJobSpecificationTest.QueryJobId` skip.
- The rebuilt package passes all 1,976 Python tests with
  `uv run --no-sync pytest`.
- `uvx nox -s stubs` passes with no generated API changes.
- `uvx nox --non-interactive -s docs` passes Doxygen, Sphinx, executable
  examples, and internal links.
- Repository lint passes. Whole-file C++ lint checked all 179 selected files;
  its trailing-comma finding is fixed. The final affected-file check also
  includes the newly changed allocator source. All 129 DD package tests pass
  after the formatting fix.

Earlier installed GCC Development/default consumers built and ran, missing-SDK
lookups behaved as specified, and wheel libraries exported no LLVM-owned
symbols. Those checks predate this rebase. Windows/macOS packaging and hosted
checks remain unverified.

Performance acceptance requires matched upstream and revised builds using the
external harness. Report successful workloads separately, including cold and
reused DDSIM workers and state transfer. Earlier measurements do not establish
performance of the current implementation.
