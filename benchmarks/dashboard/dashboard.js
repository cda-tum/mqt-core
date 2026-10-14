// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

"use strict";

const $ = (id) => document.getElementById(id);
const assetBase = new URL(".", document.currentScript.src);
const topologyLabel = { all_to_all: "All-to-all", line: "Line" };
const colors = { all_to_all: "var(--mqt-blue)", line: "var(--mqt-orange)" };
const environmentKey = (run) =>
  JSON.stringify([run.measurement?.series || run.machine, run.environment]);
const caseKey = (row) =>
  JSON.stringify([row.circuit, row.qubits, row.topology]);
const runKey = (run) => JSON.stringify([run.machine, run.commit]);
const compatible = (a, b) =>
  a?.version && a.version === b?.version && a.input_version === b.input_version;
let database;

function formatValue(value, metric) {
  if (value == null) return "Unavailable";
  return metric.id === "time"
    ? `${(value * 1000).toLocaleString("en", { maximumFractionDigits: 3 })} ms`
    : value.toLocaleString("en", { maximumFractionDigits: 0 });
}

function change(current, baseline) {
  if (!baseline || current?.value == null || baseline.value == null)
    return { text: "—", title: "No comparable baseline measurement" };
  if (!compatible(current, baseline))
    return {
      text: "—",
      title: "Different benchmark suites or inputs; no comparison",
    };
  if (baseline.value === 0)
    return {
      text: current.value === 0 ? "0.0%" : "—",
      title: "Baseline is zero",
    };
  const delta = (current.value / baseline.value - 1) * 100;
  return {
    text: `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%`,
    title: "Change relative to the selected baseline",
    className: delta < -0.05 ? "better" : delta > 0.05 ? "worse" : "",
  };
}

function options(select, entries, value) {
  select.replaceChildren(
    ...entries.map(([key, label]) => new Option(label, key)),
  );
  if (entries.some(([key]) => key === value)) select.value = value;
}

function availableRuns() {
  return database.runs.filter(
    (run) => environmentKey(run) === $("environment").value,
  );
}

function setBaseline() {
  const runs = availableRuns();
  const current = runs.find((run) => runKey(run) === $("revision").value);
  const paired = runs.find(
    (run) =>
      run.machine === current.machine &&
      run.commit === current.measurement?.baseline,
  );
  $("baseline").value = paired && paired !== current ? runKey(paired) : "";
}

function setRevisions() {
  const runs = availableRuns();
  const label = (run) =>
    `${run.commit.slice(0, 8)} · ${new Date(run.measurement?.created_at || run.date).toLocaleString("en-GB")}${run.commit === run.measurement?.baseline ? " · reference" : ""}`;
  const candidate =
    runs.find((run) => run.commit !== run.measurement?.baseline) || runs[0];
  options(
    $("revision"),
    runs.map((run) => [runKey(run), label(run)]),
    runKey(candidate),
  );
  options(
    $("baseline"),
    [["", "No comparison"], ...runs.map((run) => [runKey(run), label(run)])],
    runs[1] ? runKey(runs[1]) : "",
  );
  if (candidate.measurement) setBaseline();
  const qubits = [
    ...new Set(runs.flatMap((run) => run.rows.map((row) => row.qubits))),
  ].sort((a, b) => a - b);
  options(
    $("history-qubits"),
    qubits.map((q) => [String(q), String(q)]),
    String(qubits.at(-1)),
  );
}

// Keep repeated references as observations; select one candidate per main commit.
function historyRuns(runs, batches) {
  runs = runs.filter((run) => Number.isInteger(run.measurement?.position));
  const reference = runs[0]?.measurement?.baseline;
  const references = runs
    .filter((run) => run.commit === reference)
    .sort((a, b) =>
      a.measurement.created_at.localeCompare(b.measurement.created_at),
    );
  const candidates = new Map();
  for (const run of [...runs].reverse()) {
    if (run.commit !== reference) candidates.set(run.commit, run);
  }
  if (references.length) candidates.set(reference, references[0]);
  for (const batch of batches) {
    for (const entry of batch.commits || []) {
      if (!candidates.has(entry.commit))
        candidates.set(entry.commit, {
          commit: entry.commit,
          rows: [],
          measurement: { position: entry.position },
        });
    }
  }
  return {
    references,
    commits: [...candidates.values()].sort(
      (a, b) =>
        (a.commit === reference
          ? a.measurement.reference_position
          : a.measurement.position) -
        (b.commit === reference
          ? b.measurement.reference_position
          : b.measurement.position),
    ),
  };
}

function svgElement(name, attributes, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes))
    element.setAttribute(key, String(value));
  if (text != null) element.textContent = text;
  return element;
}

function drawChart(metric, rows, baselineRows, run, axis) {
  const coordinate = (row) => (axis ? row.position : row.qubits);
  const article = document.createElement("article");
  article.className = "chart";
  const heading = document.createElement("h3");
  heading.textContent = metric.label;
  const unit = document.createElement("p");
  unit.className = "chart-unit";
  unit.textContent =
    metric.id === "time" ? "Milliseconds · median" : metric.unit;
  const svg = svgElement("svg", {
    viewBox: "0 0 400 245",
    role: "img",
    "aria-label": `${metric.label} by ${axis?.label || "qubit count"}; exact values in the measurements table`,
  });
  const topologies = [...new Set(rows.map((row) => row.topology))];
  const qubits = [...new Set(rows.map(coordinate))].sort((a, b) => a - b);
  const baseline = baselineRows.filter((row) =>
    rows.some(
      (current) =>
        caseKey(current) === caseKey(row) &&
        compatible(current[metric.id], row[metric.id]),
    ),
  );
  const scale = metric.id === "time" ? 1000 : 1;
  const allValues = [...rows, ...baseline]
    .map((row) => row[metric.id]?.value)
    .filter((v) => v != null);
  const max = Math.max(1, ...allValues.map((v) => v * scale)) * 1.14;
  const x = (q) =>
    48 +
    (qubits.length < 2
      ? 164
      : ((q - qubits[0]) / (qubits.at(-1) - qubits[0])) * 328);
  const y = (v) => 204 - ((v * scale) / max) * 178;
  for (let tick = 0; tick <= 4; tick++) {
    const value = (max * tick) / 4;
    const height = 204 - (tick * 178) / 4;
    svg.append(
      svgElement("line", {
        x1: 48,
        x2: 376,
        y1: height,
        y2: height,
        stroke: "var(--mqt-border)",
      }),
    );
    svg.append(
      svgElement(
        "text",
        { x: 39, y: height + 4, "text-anchor": "end" },
        value.toLocaleString("en", {
          maximumFractionDigits: value < 10 ? 1 : 0,
        }),
      ),
    );
  }
  for (const q of qubits.filter(
    (_, index) =>
      !axis ||
      index % Math.max(1, Math.ceil(qubits.length / 4)) === 0 ||
      index === qubits.length - 1,
  ))
    svg.append(
      svgElement(
        "text",
        { x: x(q), y: 225, "text-anchor": "middle" },
        axis ? axis.ticks.get(q) : q,
      ),
    );
  svg.append(
    svgElement(
      "text",
      { x: 212, y: 243, "text-anchor": "middle" },
      axis?.label || "Qubits",
    ),
  );
  for (const [source, isBaseline] of [
    [baseline, true],
    [rows, false],
  ]) {
    for (const topology of topologies) {
      const points = source
        .filter((row) => row.topology === topology)
        .sort((a, b) => coordinate(a) - coordinate(b));
      let segment = [];
      const flush = () => {
        if (segment.length)
          svg.append(
            svgElement("polyline", {
              points: segment.join(" "),
              fill: "none",
              stroke: colors[topology],
              "stroke-width": isBaseline ? 1.6 : 2.5,
              "stroke-dasharray": isBaseline ? "5 5" : "none",
              opacity: isBaseline ? 0.6 : 1,
            }),
          );
        segment = [];
      };
      let previous;
      for (const row of points) {
        if (
          axis &&
          previous &&
          !compatible(previous[metric.id], row[metric.id])
        )
          flush();
        previous = row;
        const value = row[metric.id]?.value;
        if (value == null) {
          flush();
          continue;
        }
        segment.push(`${x(coordinate(row))},${y(value)}`);
      }
      flush();
      for (const row of points.filter(
        (entry) => entry[metric.id]?.value != null,
      )) {
        const cell = row[metric.id];
        const label = `${isBaseline ? "Baseline" : "Revision"}, ${topologyLabel[topology]}, ${axis ? row.label : `${row.qubits} qubits`}: ${formatValue(cell.value, metric)}`;
        const point = svgElement("circle", {
          cx: x(coordinate(row)),
          cy: y(cell.value),
          r: isBaseline ? 3 : 4,
          fill: isBaseline ? "var(--mqt-background)" : colors[topology],
          stroke: colors[topology],
          "stroke-width": 1.5,
          tabindex: 0,
          "aria-label": label,
        });
        point.append(svgElement("title", {}, label));
        svg.append(point);
      }
    }
  }
  const values = document.createElement("p");
  values.className = "chart-value";
  if (axis) {
    values.textContent = `${qubits.length} ${axis.label.toLowerCase()} · gaps mark missing or incompatible measurements`;
    article.append(heading, unit, svg, values);
    return article;
  }
  const largest = qubits.at(-1);
  values.textContent = allValues.length
    ? `${largest} qubits · ${rows
        .filter((row) => row.qubits === largest)
        .map(
          (row) =>
            `${topologyLabel[row.topology]} ${formatValue(row[metric.id]?.value, metric)}`,
        )
        .join(" · ")}`
    : "No measurements for this selection.";
  const link = document.createElement("a");
  const query = new URLSearchParams({
    "x-axis": "commit",
    "p-circuit": `'${$("circuit").value}'`,
    commits: run.commit.slice(0, 8),
    machine: run.machine,
    python: run.python,
  });
  for (const [name, value] of Object.entries(run.requirements))
    query.set(name, value);
  for (const qubitCount of qubits) query.append("p-qubits", qubitCount);
  for (const topology of topologies)
    query.append("p-topology", `'${topology}'`);
  link.href = new URL(`asv/#${metric.benchmark}?${query}`, assetBase).href;
  link.textContent = "Explore in asv ↗";
  article.append(heading, unit, svg, values, link);
  return article;
}

function render() {
  const runs = availableRuns();
  const current = runs.find((run) => runKey(run) === $("revision").value);
  const baseline = runs.find((run) => runKey(run) === $("baseline").value);
  const selected = (row) =>
    row.circuit === $("circuit").value &&
    ($("topology").value === "all" || row.topology === $("topology").value);
  const rows = current.rows.filter(selected);
  const baselineRows = baseline?.rows.filter(selected) || [];
  const baselineMap = new Map(baselineRows.map((row) => [caseKey(row), row]));
  const successful = current.rows
    .flatMap((row) => database.metrics.map((metric) => row[metric.id]))
    .filter((cell) => cell?.value != null).length;
  const expected = current.rows.length * database.metrics.length;
  const measured = current.measured_at
    ? new Date(current.measured_at).toLocaleString("en-GB")
    : "unknown";
  $("run-info").textContent =
    `${successful}/${expected} measurements · ${current.commit.slice(0, 8)} · Measured ${measured}`;
  $("commit-link").href =
    `https://github.com/munich-quantum-toolkit/core/commit/${current.commit}`;
  $("selection-title").textContent =
    `${$("circuit").selectedOptions[0].textContent} · ${rows.length} cases`;
  const notices = [];
  if (current.measurement) {
    notices.push(
      baseline?.machine === current.machine && baseline !== current
        ? "Baseline and candidate were measured together. Each observation retains its raw samples and configuration."
        : "Each observation retains its raw samples and configuration. Selected observations were not measured as a pair.",
    );
    if (current.measurement.generator_sources_changed)
      notices.push(
        "Core generator sources differ; changes can include different compiler inputs.",
      );
  }
  if (!baseline)
    notices.push(
      runs.length === 1
        ? "One measured revision. Add another revision to compare compiler changes."
        : "Select a baseline revision to compare measurements.",
    );
  if (successful < expected)
    notices.push(
      "This run is incomplete. Unavailable measurements are shown explicitly.",
    );
  if (
    rows.some((row) =>
      database.metrics.some(
        (metric) => row[metric.id]?.current_suite === false,
      ),
    )
  )
    notices.push("This run uses an older benchmark suite.");
  if (
    rows.some((row) =>
      database.metrics.some((metric) => {
        const previous = baselineMap.get(caseKey(row))?.[metric.id];
        return previous?.version && !compatible(previous, row[metric.id]);
      }),
    )
  )
    notices.push(
      "Different benchmark suites or generator inputs are not compared.",
    );
  $("notice").hidden = !notices.length;
  $("notice").textContent = notices.join(" ");
  document.querySelector(".baseline-key").hidden = !baseline;
  $("charts").replaceChildren(
    ...database.metrics.map((metric) =>
      drawChart(metric, rows, baselineRows, current),
    ),
  );
  $("measurements").replaceChildren(
    ...rows.map((row) => {
      const tr = document.createElement("tr");
      const add = (value, className = "", title = "") => {
        const td = document.createElement("td");
        td.textContent = value;
        td.className = className;
        td.title = title;
        tr.append(td);
      };
      add(row.qubits);
      add(topologyLabel[row.topology]);
      for (const metric of database.metrics) {
        const cell = row[metric.id];
        const spread =
          cell?.q25 != null && cell?.q75 != null
            ? `; middle 50%: ${formatValue(cell.q25, metric)}–${formatValue(cell.q75, metric)}`
            : "";
        add(
          formatValue(cell?.value, metric),
          cell?.value == null ? "missing" : "",
          cell?.repeat ? `${cell.repeat} samples${spread}` : "",
        );
        const delta = change(cell, baselineMap.get(caseKey(row))?.[metric.id]);
        add(delta.text, delta.className, delta.title);
      }
      return tr;
    }),
  );
  renderHistory(runs);
  const machine = database.machines[current.machine];
  $("provenance").textContent =
    `${current.machine} · ${machine.cpu || machine.arch} · Python ${current.python} · ${Object.entries(
      current.requirements,
    )
      .map(([name, version]) => `${name} ${version}`)
      .join(
        " · ",
      )}${current.measurement ? ` · ${current.measurement.os} · ${current.measurement.compiler} · LLVM ${current.measurement.llvm} · CPU affinity ${current.measurement.cpu_affinity.join(",")}` : ""} · Dashboard generated ${new Date(database.generated_at).toLocaleString("en-GB")}`;
}

function renderHistory(runs) {
  const series = runs[0]?.measurement?.series;
  const batches = (database.batches || [])
    .filter((batch) => batch.series === series)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  $("history").hidden = !series || !batches.length;
  if ($("history").hidden) return;
  const history = historyRuns(runs, batches);
  const qubits = Number($("history-qubits").value);
  const topologies =
    $("topology").value === "all"
      ? Object.keys(topologyLabel)
      : [$("topology").value];
  const rowsFor = (observations, reference = false) =>
    observations.flatMap((run, index) =>
      topologies.map((topology) => {
        const row = run.rows.find(
          (item) =>
            item.circuit === $("circuit").value &&
            item.qubits === qubits &&
            item.topology === topology,
        );
        return {
          ...row,
          topology,
          qubits,
          position: index,
          run,
          label: reference
            ? new Date(run.measurement.created_at).toLocaleString("en-GB")
            : run.commit.slice(0, 8),
        };
      }),
    );
  const rows = rowsFor(history.commits);
  const referenceRows = rowsFor(history.references, true);
  const ticks = new Map(rows.map((row) => [row.position, row.label]));
  $("history-charts").replaceChildren(
    ...database.metrics.map((metric) =>
      drawChart(metric, rows, [], null, { label: "Main commits", ticks }),
    ),
  );
  const table = (target, entries, reference = false) => {
    $(target).replaceChildren(
      ...entries.map((row) => {
        const tr = document.createElement("tr");
        const first = referenceRows.find(
          (item) => item.topology === row.topology,
        );
        const delta = reference ? change(row.time, first?.time) : null;
        const values = [
          row.label,
          topologyLabel[row.topology],
          ...database.metrics.map((metric) =>
            formatValue(row[metric.id]?.value, metric),
          ),
          reference
            ? delta.text
            : row.time?.value == null
              ? "Missing / retry pending"
              : "Measured",
        ];
        for (const value of values) {
          const td = document.createElement("td");
          td.textContent = value;
          tr.append(td);
        }
        if (delta) tr.lastChild.title = delta.title;
        tr.title = `${row.run.commit} · ${row.run.measurement.created_at || "Not measured"}`;
        return tr;
      }),
    );
  };
  table("history-measurements", rows);
  table("reference-measurements", referenceRows, true);
  const latest = batches[0];
  $("history-status").textContent = latest
    ? `Last batch: ${latest.status} · ${new Date(latest.finished_at).toLocaleString("en-GB")} · main ${latest.head.slice(0, 8)}. ${history.commits.filter((run) => run.rows.length).length}/${history.commits.length} commits measured.`
    : "No batch records for this series yet.";
  $("batch-attempts").replaceChildren(
    ...batches.flatMap((batch) =>
      batch.attempts
        .filter((attempt) => attempt.status !== "ok")
        .map((attempt) => {
          const li = document.createElement("li");
          li.textContent = `${new Date(batch.created_at).toLocaleString("en-GB")} · ${attempt.commit.slice(0, 8)} · ${attempt.role}: ${attempt.status}. ${attempt.error || "Run interrupted"}`;
          return li;
        }),
    ),
  );
  $("batch-failures").hidden = !$("batch-attempts").children.length;
}

async function load() {
  try {
    const response = await fetch(new URL("measurements.json", assetBase), {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    database = await response.json();
    if (database.schema !== 1 || !Array.isArray(database.runs))
      throw new Error("Unsupported measurement format");
    if (!database.runs.length) {
      $("status").textContent =
        "No measurements yet. Run the compiler benchmark suite, then rebuild the dashboard.";
      return;
    }
    const environments = new Map(
      database.runs.map((run) => [
        environmentKey(run),
        run.measurement
          ? `${run.measurement.host} · ${(run.measurement.series || run.machine).slice(0, 8)} · ${run.environment}`
          : `${run.machine} · ${run.environment}`,
      ]),
    );
    options($("environment"), [...environments]);
    setRevisions();
    $("environment").addEventListener("change", () => {
      setRevisions();
      render();
    });
    $("revision").addEventListener("change", () => {
      setBaseline();
      render();
    });
    for (const id of ["baseline", "circuit", "topology", "history-qubits"])
      $(id).addEventListener("change", render);
    $("filters").addEventListener("submit", (event) => event.preventDefault());
    render();
    $("status").hidden = true;
    $("dashboard").hidden = false;
    registerFilterTool();
  } catch (error) {
    $("status").textContent =
      `Measurements could not be loaded (${error.message}). Check that the dashboard server is running, then reload this page.`;
  }
}

function registerFilterTool() {
  const context = document.modelContext;
  if (!context?.registerTool) return;
  const lifecycle = new AbortController();
  window.addEventListener("pagehide", () => lifecycle.abort(), { once: true });
  try {
    Promise.resolve(
      context.registerTool(
        {
          name: "select_compiler_benchmark",
          title: "Select compiler benchmark",
          description:
            "Select a Core circuit and connectivity in the visible benchmark dashboard. Keeps the selected machine and revisions unchanged.",
          inputSchema: {
            type: "object",
            properties: {
              circuit: { type: "string", enum: ["ghz", "qft", "bv"] },
              topology: { type: "string", enum: ["all", "all_to_all", "line"] },
            },
            required: ["circuit", "topology"],
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: false },
          execute(input) {
            if (
              !input ||
              Object.keys(input).some(
                (key) => !["circuit", "topology"].includes(key),
              ) ||
              !["ghz", "qft", "bv"].includes(input.circuit) ||
              !["all", "all_to_all", "line"].includes(input.topology)
            ) {
              throw new Error("Select a supported circuit and topology.");
            }
            $("circuit").value = input.circuit;
            $("topology").value = input.topology;
            render();
            return {
              circuit: input.circuit,
              topology: input.topology,
              revision: availableRuns().find(
                (run) => runKey(run) === $("revision").value,
              ).commit,
              baseline:
                availableRuns().find(
                  (run) => runKey(run) === $("baseline").value,
                )?.commit || null,
              cases: $("measurements").rows.length,
            };
          },
        },
        { signal: lifecycle.signal },
      ),
    ).catch((error) => console.warn("Benchmark tool unavailable:", error));
  } catch (error) {
    console.warn("Benchmark tool unavailable:", error);
  }
}

load();
