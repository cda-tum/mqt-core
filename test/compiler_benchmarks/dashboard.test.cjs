// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { test } = require("node:test");
const context = vm.createContext({
  URL,
  document: { currentScript: { src: "https://example.invalid/dashboard.js" } },
  fetch: () => new Promise(() => {}),
});
vm.runInContext(
  fs.readFileSync("benchmarks/dashboard/dashboard.js", "utf8"),
  context,
);
const { environmentKey, runKey, change, historyRuns } = vm.runInContext(
  "({environmentKey, runKey, change, historyRuns})",
  context,
);
const cell = { value: 10, version: "suite", input_version: "input" };
const observation = (machine, commit, date, position) => ({
  machine,
  commit,
  environment: "python",
  rows: [{ time: cell }],
  measurement: {
    series: "series",
    baseline: "release",
    created_at: date,
    position,
    reference_position: 5,
  },
});

test("configuration series group pairs without losing repeated observations", () => {
  const first = observation("first", "release", "2026-09-12", 5);
  const second = observation("second", "release", "2026-09-19", 5);
  assert.equal(environmentKey(first), environmentKey(second));
  assert.notEqual(runKey(first), runKey(second));
  second.measurement.series = "new-toolchain";
  assert.notEqual(environmentKey(first), environmentKey(second));
});

test("incompatible inputs, suites and absent baselines do not produce deltas", () => {
  assert.equal(change({ ...cell, value: 15 }, cell).text, "+50.0%");
  for (const update of [
    { input_version: "changed" },
    { version: "changed" },
    { value: null },
  ])
    assert.equal(change({ ...cell, ...update }, cell).text, "—");
  assert.equal(change(cell, undefined).text, "—");
  assert.equal(change(cell, { ...cell, value: 0 }).text, "—");
});

test("main history starts at release, retains gaps and selects latest candidate", () => {
  const runs = [
    observation("manual", "unrelated", "2026-09-20", undefined),
    observation("repeat", "candidate", "2026-09-19", 7),
    observation("repeat", "release", "2026-09-19", 7),
    observation("pair", "candidate", "2026-09-12", 7),
    observation("pair", "release", "2026-09-12", 7),
    observation("initial", "release", "2026-09-11", 5),
  ];
  const history = historyRuns(runs, [
    { commits: [{ commit: "missing", position: 6 }] },
  ]);
  assert.equal(
    history.commits.map((run) => run.commit).join(),
    "release,missing,candidate",
  );
  assert.equal(history.commits[0].machine, "initial");
  assert.equal(history.commits[1].rows.length, 0);
  assert.equal(history.commits[2].machine, "repeat");
  assert.equal(
    history.references.map((run) => run.machine).join(),
    "initial,pair,repeat",
  );
});
