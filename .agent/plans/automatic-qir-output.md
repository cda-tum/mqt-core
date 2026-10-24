# Automatic QIR output

Sampled DDSIM QIR jobs provide their full textual output through CUSTOM1 while
shots and histograms remain the recorded Result/Boolean projection. OpenQASM and
zero-shot state extraction do not provide QIR output. No result enums or
supported program shapes are added.

The QIR JIT captures one terminal shot with the runtime formatter and its Result
digit offsets. DDSIM transports this template with packed shots and expands it
directly into the caller's buffer on request. Size queries do not expand it. The
decoder checks each shot's width before materialization uses fixed strides.
Programs with global destructors use ordinary execution so teardown observes the
last shot's collapsed state and Result values. Adaptive execution retains its
full output. Constant numeric records keep their terminal eligibility.

LLVM string streams write into the owned output strings, removing extraction and
capture cleanup. Borrowed streams must outlive their JIT session, including
destructor execution. Double formatting keeps `std::format` for round-trip
precision; LLVM's default double formatting loses significant digits. Binary DD
serialization retains its existing stream interface. The JIT/runtime stream API
is internal to the build; source embedders must supply `llvm::raw_ostream`
sinks.

The worker protocol remains version 4 while its format is unreleased. DDSIM has
no capture parameter. Generic QDMI custom slots remain part of the shared API.

Fresh paired public-API measurements against ddbdcf9e16 used terminal Bell/GHZ,
adaptive GHZ/repeat-until-success, and the existing typed-output fixture. The
fixture's median submit/wait time fell from 28.4 to 27.0 ms; other timing ranges
overlap. All five cases produced identical seeded output and optimized IR. DD
execution dominates adaptive GHZ; heap volume, live heap and RSS show no
material stream-related improvement. The stream change is retained for simpler
ownership and the measured formatting benefit, without a general speedup claim.

Tests retain distinct contracts for direct JIT text output, disabled output,
empty outcomes, scalar values, bit projection, state retention, labels,
lifecycle ordering and malformed protocol messages. The lifecycle test counts
constructor and destructor records without a synthetic IR counter. Precision
loss and missing destructor output were deliberately introduced; both retained
tests failed. Earlier probes also detected frozen snapshots, reversed output
bits and missing packed-shot width checks. One specialist Ponytail review found
no additional defect. Final validation covers native QIR/DDSIM suites, Python
QDMI/QIR-runner suites, whole-file C++ lint, repository lint and executable
docs.
