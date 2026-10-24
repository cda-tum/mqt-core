# Automatic QIR output

Status: complete on the upstream streaming and packed-shot implementation; local
validation passed, hosted CI pending.

Sampled DDSIM QIR jobs provide their full textual output through CUSTOM1 while
shots and histograms remain the recorded Result/Boolean projection. CUSTOM2 is a
validated compatibility no-op. OpenQASM and zero-shot state extraction do not
provide QIR output. No result enums or supported program shapes are added.

The QIR JIT uses the runtime formatter to capture one terminal shot and its
Result digit offsets. DDSIM transports this template with comma-terminated shots
and materializes it directly into the caller's buffer on request. Size queries
do not expand it. The decoder checks each shot's width before the caller uses
fixed strides to read it. This avoids repeated quantum execution and retaining
or sending expanded terminal output. Eager expansion was rejected because it
imposed measurable allocation and transfer costs on counts-only callers.

Programs with global destructors use ordinary execution: destructors may read
the last shot's collapsed state and Result values. This also keeps terminal
output as a header/constructor prefix followed by repeated shot records, with no
suffix-insertion machinery. Adaptive execution retains its full output. Constant
numeric records retain their existing terminal eligibility.

Validation uses the QIR JIT, runtime, and DDSIM native suites, Python QDMI and
QIR-runner tests, whole-file C++ lint, repository lint, and executable docs. The
existing repeated-output regression detects frozen result digits, reversed bit
substitution, and incorrect destructor-observed results when those defects are
deliberately introduced. The protocol regression also detects a missing packed
shot width check. Public submit/wait/retrieval measurements and worker
allocation profiles justify the bounded template transport; no claim of hardware
acceleration follows from this change.
