/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

"use strict";
(() => {
  const data = window.MQSF_DATA;
  const assets = window.MQSF_ASSETS;
  const viz = window.MQSF_VIZ;
  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  const scenario = (id) => data.scenarios.find((s) => s.id === id);
  const variant = (id = "parity", unrolled = false) =>
    scenario(id)?.variants.find(
      (v) => v.id === (unrolled ? "unrolled" : "structured"),
    ) || {};
  const app = data.application || {};
  const targets = data.targets || [];
  const references = data.references || {};
  const run =
    variant("qpe").executions?.["qir-adaptive"] ||
    variant("qpe").execution ||
    {};
  const state = {
    slide: 0,
    step: 0,
    animation: null,
    token: 0,
    jump: "",
    overview: 0,
  };
  const svg = (body, box = "0 0 1600 650", name = "Illustration") =>
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${box}" role="img" aria-label="${esc(name)}">${body}</svg>`;
  const text = (x, y, s, size = 32, color = "#142b45", anchor = "middle") =>
    `<text x="${x}" y="${y}" font-family="Inter,Arial,sans-serif" font-size="${size}" fill="${color}" text-anchor="${anchor}">${esc(s)}</text>`;
  const line = (x1, y1, x2, y2, color = "#2f70b8", width = 4) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>`;
  const logo = (name, cls = "logo") =>
    `<img class="${cls}" src="${assets[name]}" alt="${esc(name === "tum-cda" ? "Chair for Design Automation, TUM" : name.toUpperCase())}" />`;
  const heading = (title, sub = "") =>
    `<h2 data-morph="heading">${title}</h2>${sub ? `<p class="subtitle">${sub}</p>` : ""}`;
  const qr = (name, url, caption) =>
    `<div class="qr-block"><img src="${assets[`qr-${name}`]}" alt="QR code: ${esc(url)}"/><div><div class="url">${esc(url)}</div><div class="caption">${esc(caption)}</div></div></div>`;
  const fmt = (n, digits = 2) =>
    Number(n).toLocaleString("en-US", { maximumFractionDigits: digits });
  const sourceNote = (s) => `<p class="source-line">${s}</p>`;
  function code(
    a,
    { start = 0, count = 11, hot = [], title, compact = false, key } = {},
  ) {
    if (!a?.code)
      return `<div class="code-frame"><p class="small">Capture unavailable</p></div>`;
    const lines = a.code.split("\n");
    start = Math.max(0, Math.min(start, Math.max(0, lines.length - 1)));
    const shown = lines.slice(start, start + count);
    const indent = Math.min(
      ...shown.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length),
      100,
    );
    const body = shown
      .map((raw, i) => {
        const l = raw.slice(indent),
          highlighted = a.lines_html?.[start + i]?.replace(
            new RegExp(`^ {0,${indent}}`),
            "",
          );
        const content = highlighted ?? esc(l);
        return `<span class="code-line ${hot.includes(start + i) ? "hot" : ""}"><span class="line-no">${start + i + (a.excerpt_start_line || 1)}</span><span class="code-text">${content}</span></span>`;
      })
      .join("");
    return `<div data-morph="code-${esc(key || a.id || title || a.label || a.language || "source")}" class="code-frame ${compact ? "compact" : ""}"><div class="code-title">${esc(title || a.label || a.language || "Actual source")}</div><pre class="highlight">${body}</pre></div>`;
  }
  const focus = (a, needle, before = 1) =>
    Math.max(
      0,
      (a?.code?.split("\n").findIndex((s) => s.includes(needle)) ?? 0) - before,
    );
  const flatten = window.MQSF_CIRCUIT.flatten;
  const circuit = window.MQSF_CIRCUIT.render;
  function histogram(capture, elapsed = Infinity, box = "0 0 900 480") {
    const bins = new Map();
    const timed = capture.shot_events || [];
    const shots = Number.isFinite(elapsed)
      ? timed.filter((s) => s.time_ms <= elapsed).map((s) => s.outcome)
      : capture.shots || [];
    shots.forEach((s) =>
      bins.set(parseInt(s, 2), (bins.get(parseInt(s, 2)) || 0) + 1),
    );
    const totals = capture.counts || {},
      max = Math.max(1, ...Object.values(totals));
    const all = Object.keys(totals).map((s) => parseInt(s, 2));
    // Keep the visible QPE window stable; aggregate real tails instead of hiding them.
    const lo = 78,
      hi = 93,
      values = [
        bins.get(-1) || 0,
        ...Array.from({ length: hi - lo + 1 }, (_, i) => bins.get(lo + i) || 0),
        0,
      ];
    values[0] = [...bins]
      .filter(([i]) => i < lo)
      .reduce((s, [, n]) => s + n, 0);
    values[values.length - 1] = [...bins]
      .filter(([i]) => i > hi)
      .reduce((s, [, n]) => s + n, 0);
    const x0 = 65,
      w = 790,
      plotH = 285,
      y0 = 360,
      bw = w / values.length;
    let body = line(x0, y0, x0 + w, y0, "#9cb3c9", 2);
    [0, 0.5, 1].forEach((f) => {
      const y = y0 - f * plotH;
      body +=
        line(x0, y, x0 + w, y, "#e0e9f2", 1) +
        text(x0 - 12, y + 8, String(Math.round(max * f)), 23, "#5b7088", "end");
    });
    const latest = shots.length ? parseInt(shots.at(-1), 2) : -100;
    values.forEach((n, i) => {
      const x = x0 + i * bw + 5,
        h = (n / max) * plotH,
        key =
          i === 0
            ? `<${lo}`
            : i === values.length - 1
              ? `>${hi}`
              : String(lo + i - 1);
      body += `<rect class="bar ${Number(key) === latest ? "hot" : ""}" x="${x}" y="${y0 - h}" width="${bw - 9}" height="${h}" data-count="${n}"/>`;
      if (n > max * 0.07) body += text(x + (bw - 9) / 2, y0 - h - 10, n, 22);
      if (i === 0 || i === values.length - 1 || i % 2 === 0 || key === "85")
        body += text(x + (bw - 9) / 2, y0 + 35, key, 22, "#5b7088");
    });
    body +=
      text(450, 428, "8-bit phase estimate · integer / 256", 27, "#5b7088") +
      text(
        450,
        475,
        `${fmt(shots.length, 0)} / ${fmt(capture.num_shots || 0, 0)} actual shots`,
        31,
        "#2f70b8",
      );
    return `<div class="histogram" data-shots="${shots.length}" data-elapsed-ms="${elapsed}" data-observed-bins="${all.length}">${svg(body, box, "Actual QPE measurement distribution")}</div>`;
  }
  const appStage = (id) => {
    const v = targets[0]?.compilation || {};
    return [...(v.stages || []), ...(v.exports || [])].find((a) => a.id === id);
  };
  const stagesFor = (target) => [
    ...(target?.compilation?.stages || []),
    ...(target?.compilation?.exports || []),
  ];
  const targetStage = (target, id) =>
    stagesFor(target).find((a) => a.id === id);
  function walkerScene(progress = 1) {
    const p = app.propagation,
      frames = p?.curves?.[0]?.frames || [];
    if (!frames.length) return "";
    const index = Math.max(
        0,
        Math.min(frames.length - 1, progress * (frames.length - 1)),
      ),
      low = Math.floor(index),
      high = Math.min(low + 1, frames.length - 1),
      t = index - low;
    const f = frames[low],
      next = frames[high],
      tau = f.tau + (next.tau - f.tau) * t;
    const weights = f.walkers.map(
        (w, i) => w.weight + (next.walkers[i].weight - w.weight) * t,
      ),
      max = Math.max(...weights);
    const columns = 32,
      dx = 34,
      dy = 34;
    let body = "";
    f.walkers.forEach((w, i) => {
      const xx = 48 + (i % columns) * dx,
        yy = 30 + Math.floor(i / columns) * dy,
        r = 14 * Math.sqrt(weights[i] / max);
      body += `<g data-morph="walker-${w.id}"><circle cx="${xx}" cy="${yy}" r="${r}" fill="#2f70b8" opacity="${0.25 + (0.75 * weights[i]) / max}"/></g>`;
    });
    body += text(
      580,
      195,
      `τ = ${fmt(tau, 2)} Ha⁻¹ · dot area ∝ walker weight`,
      34,
      "#2f70b8",
    );
    return svg(
      body,
      "0 0 1160 220",
      "Measured AFQMC walker weights replayed over imaginary time",
    );
  }

  function deviceView(
    target = targets[0],
    activeSites = [],
    layout = null,
    blend = null,
    activeEdges = null,
  ) {
    const meta = target?.metadata || data.device,
      sites = meta.sites || [],
      edges = meta.edges || [];
    const full = edges.length > sites.length * 5,
      points = new Map();
    if (full)
      sites.forEach((s, i) =>
        points.set(s.id, {
          x: 410 + 245 * Math.cos((i * 2 * Math.PI) / sites.length),
          y: 310 + 245 * Math.sin((i * 2 * Math.PI) / sites.length),
        }),
      );
    else {
      // The full Emerald lattice fixes drawing coordinates; only advertised edges are drawn.
      const geometry = target?.id === "iqm.emerald" ? data.device.edges : edges;
      const graph = new Set(
          geometry.flatMap(([a, b]) => [`${a}:${b}`, `${b}:${a}`]),
        ),
        rows = [];
      [...sites]
        .sort((a, b) => a.id - b.id)
        .forEach((s) => {
          const r = rows.at(-1);
          if (r && graph.has(`${r.at(-1).id}:${s.id}`)) r.push(s);
          else rows.push([s]);
        });
      rows.forEach((r, y) => {
        let offset = 0;
        const connection = r.flatMap((s, x) =>
          (rows[y - 1] || [])
            .filter((a) => graph.has(`${a.id}:${s.id}`))
            .map((a) => points.get(a.id).x - x),
        )[0];
        if (connection !== undefined) offset = connection;
        r.forEach((s, x) => points.set(s.id, { x: x + offset, y }));
      });
      const xs = [...points.values()].map((p) => p.x),
        minX = Math.min(...xs),
        maxX = Math.max(...xs),
        maxY = Math.max(1, rows.length - 1),
        scale = Math.min(640 / Math.max(1, maxX - minX), 470 / maxY);
      points.forEach((p) => {
        p.x = 410 + (p.x - (minX + maxX) / 2) * scale;
        p.y = 300 + (p.y - maxY / 2) * scale;
      });
    }
    const active = new Set(activeSites),
      colors = [
        "#2f70b8",
        "#087d92",
        "#8864bd",
        "#dd8d23",
        "#d46172",
        "#5b83a1",
      ];
    const fidelities = new Map(
      (target?.compiler_model?.operations || [])
        .filter((o) => o.numQubits === 2)
        .flatMap((o) =>
          (o.siteOverrides || [])
            .filter((v) => v.sites?.length === 2 && Number.isFinite(v.fidelity))
            .map((v) => [
              v.sites
                .slice()
                .sort((a, b) => a - b)
                .join(":"),
              v.fidelity,
            ]),
        ),
    );
    const fidelityColor = (f) =>
      f === undefined
        ? "#b8cee2"
        : f < 0.98
          ? "#ce9152"
          : f < 0.995
            ? "#54a0b2"
            : "#2f70b8";
    let body = edges
      .map(([a, b]) => {
        const p = points.get(a),
          q = points.get(b);
        if (!p || !q) return "";
        const f = fidelities.get([a, b].sort((x, y) => x - y).join(":")),
          hot =
            activeEdges === null
              ? active.has(a) && active.has(b)
              : activeEdges.has([a, b].sort((x, y) => x - y).join(":"));
        return `<line data-morph="device-edge-${a}-${b}" x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}" stroke="${hot ? "#087d92" : fidelityColor(f)}" stroke-width="${hot ? 6 : full ? 1 : 3.5}" opacity="${full ? 0.22 : 1}"><title>${a}–${b}${f === undefined ? "" : `: reported fidelity ${fmt(f * 100, 3)}%`}</title></line>`;
      })
      .join("");
    body += text(
      410,
      620,
      full ? "All-to-all connectivity" : "",
      24,
      "#526d88",
    );
    const r = sites.length > 80 ? 12 : sites.length > 40 ? 17 : 20;
    const tokens = (layout || [])
      .slice(0, 6)
      .map((site, i) => {
        const destination = points.get(site),
          origin = points.get(blend?.before?.[i]) || destination;
        if (!destination) return null;
        const t = blend?.fraction ?? 1,
          dx = destination.x - origin.x,
          dy = destination.y - origin.y,
          distance = Math.hypot(dx, dy),
          // Opposing exchanges take opposite arcs; recorded endpoints stay exact.
          arc = distance
            ? (Math.min(45, distance / 4) * Math.sin(Math.PI * t)) / distance
            : 0;
        return {
          x: origin.x + dx * t - dy * arc,
          y: origin.y + dy * t + dx * arc,
          i,
        };
      })
      .filter(Boolean);
    // Keep labels apart during schematic movement, never change recorded placements.
    if (blend?.fraction > 0 && blend.fraction < 1)
      for (let pass = 0; pass < 12; pass++)
        for (let i = 0; i < tokens.length; i++)
          for (let j = i + 1; j < tokens.length; j++) {
            const a = tokens[i],
              b = tokens[j],
              dx = b.x - a.x || 0.001,
              dy = b.y - a.y,
              distance = Math.hypot(dx, dy),
              shift = Math.max(
                0,
                (2 * (r + 5) + 4 - distance) / (2 * distance),
              );
            a.x -= dx * shift;
            a.y -= dy * shift;
            b.x += dx * shift;
            b.y += dy * shift;
          }
    body += sites
      .map((s) => {
        const p = points.get(s.id),
          hot = active.has(s.id);
        const covered = tokens.some(
          (t) => Math.hypot(t.x - p.x, t.y - p.y) < r + 18,
        );
        return `<g data-morph="device-node-${s.id}" transform="translate(${p.x} ${p.y})"><circle r="${r}" fill="${hot ? "#d4e9fa" : "#ecf3fa"}" stroke="${hot ? "#087d92" : "#92b2d0"}" stroke-width="${hot ? 4 : 1.5}"/>${covered ? "" : text(0, 6, s.id, r > 15 ? 17 : 12, "#58718b")}</g>`;
      })
      .join("");
    tokens.forEach(({ x, y, i }) => {
      body += `<g data-morph="logical-qubit-${i}" transform="translate(${x} ${y})"><circle r="${r + 5}" fill="${colors[i]}" stroke="white" stroke-width="3"/>${text(0, 7, `q${i}`, 17, "white")}</g>`;
    });
    return `<div class="target-map" data-morph="target-map">${svg(body, "0 0 820 630", `${target?.label || "Emerald"} captured coupling graph`)}</div>`;
  }
  function programMetrics(target, id) {
    const m = target?.compilation?.metrics?.[id] || {},
      c = targetStage(target, id)?.circuit,
      ops = c ? flatten(c) : [];
    const two =
      m.two_qubit_operations ??
      ops.filter((o) => o.qubits?.length === 2).length;
    return `<div class="program-metrics" data-stage="${id}">${[
      [
        ops.filter(
          (o) =>
            o.qubits?.length === 1 && !["measure", "reset"].includes(o.name),
        ).length,
        "one-qubit",
      ],
      [two, "two-qubit"],
      [m.depth ?? 0, "depth"],
      [m.active_qubits ?? c?.qubits?.length ?? 0, "active qubits"],
    ]
      .map(
        ([v, l]) =>
          `<div data-morph="metric-${id}-${l.replaceAll(" ", "-")}"><b>${fmt(v, 0)}</b><span>${l}</span></div>`,
      )
      .join("")}</div>`;
  }
  // Display folds retain the original line numbers and every control boundary.
  // This changes the view only; full unmodified programs remain in evidence.json.gz.
  function structureCode(a, title, qir = false) {
    const lines = (a?.code || "").split("\n"),
      rows = [];
    const start = qir ? lines.findIndex((l) => l.startsWith("define ")) : 0;
    let end = qir
      ? lines.findIndex((l, i) => i > start && l === "}") + 1
      : lines.length;
    if (end <= start) end = lines.length;
    const foldable = (l) =>
      qir
        ? l.trim() &&
          !/^define|^\}|^\w+:/u.test(l) &&
          !/(?:\b(?:br |phi |icmp |ret |add )|__(?:reset|mz)__|__read_result)/u.test(
            l,
          )
        : /^\s*(?:r\(|ctrl @ z |cz |rx\(|rz\()/u.test(l);
    for (let i = Math.max(0, start); i < end;) {
      if (!lines[i].trim()) {
        i++;
        continue;
      }
      let last = i + 1;
      if (foldable(lines[i]))
        while (last < end && foldable(lines[last])) last++;
      if (last - i > 1)
        rows.push(
          `<span class="code-line folded"><span class="line-no">${i + 1}–${last}</span><span class="code-text">${" ".repeat(lines[i].match(/^ */)[0].length)}⋯ ${last - i} ${qir ? "straight-line instructions" : "gate operations"}</span></span>`,
        );
      else
        rows.push(
          `<span data-effect="${/reset/.test(lines[i]) ? "reset" : /phi |add |icmp.*slt|loop/.test(lines[i]) ? "loop" : /br |read_result|__mz__/.test(lines[i]) ? "branch" : "gate"}" class="code-line ${/for |if |while |br |phi |icmp |measure|reset|read_result/.test(lines[i]) ? "hot" : ""}"><span class="line-no">${i + 1}</span><span class="code-text">${qir && /^\w+:/.test(lines[i]) ? esc(lines[i].replace(/ {2,};/, "  ;")) : (a.lines_html?.[i] ?? esc(lines[i]))}</span></span>`,
        );
      i = last;
    }
    const middle = Math.ceil(rows.length / 2);
    return `<div class="code-frame structure-code ${qir ? "qir-structure" : ""}" data-lines="${rows.length}"><div class="code-title">${title}</div><div class="${qir ? "code-columns" : ""}"><pre class="highlight">${(qir ? rows.slice(0, middle) : rows).join("")}</pre>${qir ? `<pre class="highlight">${rows.slice(middle).join("")}</pre>` : ""}</div><p class="code-foot">${qir ? "Complete entry-point control flow · label spacing condensed" : "Complete program structure"} · explicit folds · original line numbers</p></div>`;
  }
  const evidence = data.compiler_evidence || {};
  const totalTime = (capture = run) => {
    const recordedEnd = Math.max(
      0,
      ...(capture.events || []).map((e) => e.time_ms + (e.duration_ms || 0)),
      ...(capture.lifecycle || []).map((e) => e.time_ms),
    );
    return capture.trace_kind === "python-observed-adapter-calls"
      ? recordedEnd
      : Math.max(
          recordedEnd,
          capture.completed_ms || 0,
          capture.duration_ms || 0,
        );
  };
  const qdmiStart = (capture) =>
    capture.events?.find((e) => /try_submit_job|job_submit/.test(e.operation))
      ?.time_ms || 0;
  const interpolateTime = (capture, progress) =>
    qdmiStart(capture) + progress * (totalTime(capture) - qdmiStart(capture));
  function runtimeSequence(capture, elapsed) {
    const adapter = capture.trace_kind === "python-observed-adapter-calls";
    const names = adapter
      ? ["PennyLane", "Core adapter", "Core driver", "DDSIM"]
      : ["QDMI client", "DDSIM device", "JIT runtime", "Shots"];
    const events = [
      ...(capture.events || []),
      ...(capture.runtime_events || []),
      ...(capture.lifecycle || []),
    ].sort((a, b) => a.time_ms - b.time_ms);
    const latest = events.filter((e) => e.time_ms <= elapsed).at(-1);
    // The adapter capture observes Python boundaries; C-interface links show architecture only.
    const selected = adapter
      ? [
          [
            1,
            2,
            "try_submit_job(programs)",
            events.find(
              (e) => e.operation === "Device.try_submit_job(programs)",
            ),
          ],
          [
            1,
            2,
            "check() → RUNNING",
            events.find((e) => e.state === "running"),
          ],
          [
            0,
            1,
            "result()",
            events.find((e) => e.operation.includes("PennyLaneJob.result():")),
          ],
          [
            1,
            2,
            "get_shots(program_index)",
            events.find((e) => Number.isInteger(e.program_index)),
          ],
          [1, 0, "Decoded results", events.find((e) => e.state === "done")],
        ]
      : [
          [
            0,
            1,
            "Submit adaptive QIR",
            events.find((e) => e.operation.includes("job_submit")),
          ],
          [
            1,
            2,
            "Compile & initialize JIT",
            events.find((e) => e.operation === "JIT compilation begins"),
          ],
          [
            2,
            3,
            "Execute adaptive shots",
            events.find(
              (e) => e.operation === "Serial adaptive sampling begins",
            ),
          ],
          [
            1,
            0,
            "Return results",
            events.find((e) => e.operation.includes("get_results")),
          ],
        ];
    const xs = [90, 310, 530, 750];
    let body = names
      .map(
        (name, i) =>
          text(xs[i], 32, name, 22) + line(xs[i], 52, xs[i], 345, "#cbdced", 2),
      )
      .join("");
    if (adapter)
      body +=
        `<path d="M530 53H750" stroke="#9bb7d1" stroke-width="3" stroke-dasharray="5 5"/>` +
        text(640, 76, "Device C interface", 16, "#698198");
    selected.forEach(([a, b, label, e], i) => {
      const on = e && elapsed >= e.time_ms,
        y = 105 + i * 54;
      body += `<g data-morph="sequence-${i}" opacity="${on ? 1 : 0.13}">${line(xs[a], y, xs[b], y)}<path d="M${xs[b] + (b > a ? -10 : 10)} ${y - 7}L${xs[b]} ${y}L${xs[b] + (b > a ? -10 : 10)} ${y + 7}" fill="none" stroke="#2f70b8" stroke-width="3"/>${text((xs[a] + xs[b]) / 2, y - 13, label, 21)}</g>`;
    });
    return `<div class="sequence-svg">${svg(body, "0 0 840 370", "Recorded calls and runtime observations")}</div><p class="trace-current">${esc(latest?.operation?.replace(/^MQT_DDSIM_QDMI_/, "") || "Ready to submit")}</p>`;
  }

  function energyChart(progress = 1, compare = false) {
    const p = app.propagation;
    if (!p) return "";
    const curves = p.curves,
      ref = app.chemistry.fci_energy;
    const values = curves
      .flatMap((c) =>
        c.energy.flatMap((e, i) => [
          e - c.stderr[i] - ref,
          e + c.stderr[i] - ref,
        ]),
      )
      .map((e) => e * 1000);
    const padding = Math.max(
      1,
      (Math.max(...values) - Math.min(...values)) * 0.08,
    );
    const low = Math.min(0, ...values) - padding,
      high = Math.max(0, ...values) + padding;
    const x = (t) => 90 + (t / p.tau.at(-1)) * 640,
      y = (e) => 305 - ((e - low) / (high - low)) * 240;
    let body = text(
      90,
      30,
      "Energy − exact reference (mHa)",
      25,
      "#142b45",
      "start",
    );
    for (let i = 0; i <= 4; i++) {
      const value = low + ((high - low) * i) / 4,
        yy = y(value);
      body +=
        line(90, yy, 730, yy, "#e1eaf2", 1) +
        text(77, yy + 7, fmt(value, 0), 21, "#5b7088", "end");
      body += text(
        90 + i * 160,
        337,
        fmt((p.tau.at(-1) * i) / 4, 1),
        21,
        "#5b7088",
      );
    }
    body +=
      line(90, y(0), 730, y(0), "#078b9c", 2) +
      text(405, 378, "Imaginary time τ (Ha⁻¹)", 25);
    const count = Math.max(
      1,
      Math.min(p.tau.length, Math.ceil(progress * p.tau.length)),
    );
    curves.forEach((c, j) => {
      if (j && !compare) return;
      const coords = (sign = 0) =>
        c.energy
          .slice(0, count)
          .map(
            (e, i) =>
              `${x(p.tau[i])},${y((e - ref + sign * c.stderr[i]) * 1000)}`,
          );
      if (!j)
        body += `<polygon points="${[...coords(1), ...coords(-1).reverse()].join(" ")}" fill="#2f70b8" opacity=".12"/>`;
      body += `<polyline data-morph="energy-${j}" points="${coords().join(" ")}" stroke="${j ? "#8998a8" : "#2f70b8"}" stroke-width="${j ? 3 : 4}" fill="none" ${j ? 'stroke-dasharray="8 5"' : ""}/>`;
    });
    body +=
      text(280, 423, "Quantum-shadow trial", 24, "#2f70b8") +
      (compare ? text(615, 423, "Hartree–Fock trial", 24, "#778899") : "");
    return `<div class="energy-chart">${svg(body, "0 0 820 440", "Measured energy convergence, both full time ranges retained")}</div>`;
  }
  function jobWindow(progress) {
    const capture = app.execution || {},
      elapsed = interpolateTime(capture, progress);
    const count = (capture.events || []).filter(
      (e) =>
        Number.isInteger(e.program_index) &&
        e.time_ms + e.duration_ms <= elapsed,
    ).length;
    const number = app.workload.snapshots;
    const status =
      (capture.lifecycle || []).filter((e) => e.time_ms <= elapsed).at(-1)
        ?.state ||
      (count === number ? "done" : count ? "extracting-results" : "submitted");
    let body = "";
    for (let i = 0; i < 96; i++)
      body += `<rect x="${18 + (i % 16) * 44}" y="${18 + Math.floor(i / 16) * 28}" width="33" height="19" rx="4" fill="${i / 96 < count / number ? "#2f70b8" : status === "running" ? "#a7d6e1" : "#e2ebf3"}"/>`;
    body += text(
      360,
      218,
      `${fmt(count, 0)} / ${fmt(number, 0)} program results`,
      26,
    );
    return `<div class="batch-grid" data-programs="${count}" data-elapsed-ms="${elapsed}"><div class="runtime-strip"><span>${esc(status.replaceAll("-", " "))}</span><span>${fmt(elapsed - qdmiStart(capture), 0)} ms</span></div><div class="clock-line"><div style="width:${progress * 100}%"></div></div>${runtimeSequence(capture, elapsed)}${svg(body, "0 0 750 240", "Indexed results from the actual QDMI batch")}</div>`;
  }
  function applicationScene(phase) {
    const source = appStage("source")?.circuit;
    const cards = [
      [
        "Classical preparation",
        "Molecular Hamiltonian → trial state",
        logo("pennylane") +
          `<div class="molecule"><i>H</i><span></span><i>H</i></div>`,
        "Workstation / CPU node",
      ],
      [
        "Quantum shadow collection",
        `${fmt(app.workload?.snapshots, 0)} circuits × ${fmt(app.workload?.shots_per_snapshot, 0)} shots`,
        circuit(source, { limit: 18, key: "shadow-input" }) +
          `<div class="shadow-cards">${(app.shadow_preview || []).map((x) => `<span>${esc(x.outcome)}<small>program ${x.program_index}</small></span>`).join("")}</div>`,
        "QDMI device · DDSIM",
      ],
      [
        "Classical Monte Carlo",
        "Propagate → overlap → reweight → reduce",
        walkerScene(0.5) +
          `<img class="weight-equation" src="${assets["eq-afqmc-energy"]}" alt="Weighted mean of local walker energies"/>`,
        "Slurm CPU nodes",
      ],
    ];
    return `<div class="application-flow">${cards.map(([title, description, visual, where], i) => `<section data-morph="app-pillar-${i}" class="app-card ${phase < i ? "future" : ""}"><span class="location-label">${where}</span><h3>${title}</h3><p>${description}</p><div class="app-visual">${visual}</div><div class="stack-binding">${i === 0 ? "Frontend · program preparation" : i === 1 ? "Compiler Collection → QDMI" : "Orchestrator → classical resources"}</div></section>`).join("")}</div><div class="app-stack-strip">${logo("mqt")} Compiler Collection <span>→</span> ${logo("qdmi")} Shared device interface <span>→</span> ${logo("mqsc")} Orchestrator</div>`;
  }
  function hybridExecution(step, progress = 1) {
    const shown = Math.min(step, 3),
      active = Math.max(0, shown - 1);
    const generation =
      step <= 1
        ? code(
            { code: app.source, lines_html: app.source_lines_html },
            {
              count: 30,
              compact: true,
              title: "Actual PennyLane quantum function",
            },
          ) +
          code(
            { code: app.batch_source, lines_html: app.batch_source_lines_html },
            { count: 8, compact: true, title: "Actual batch submission" },
          )
        : code(
            { code: app.batch_source, lines_html: app.batch_source_lines_html },
            { count: 8, compact: true, title: "Actual batch submission" },
          ) +
          circuit(appStage("source")?.circuit, {
            limit: 18,
            key: "batch-payload",
          });
    const qprogress = step === 2 ? progress : step > 2 ? 1 : 0,
      cprogress = step === 3 ? progress : step > 3 ? 1 : 0;
    const nodes = new Set(app.propagation.workers.map((w) => w.node)).size;
    const allocation = `<div class="slurm-allocation">${logo("slurm")}<span>${nodes} compute nodes · ${app.propagation.workers.length} ranks<br>Propagate → reweight → reduce</span></div>`;
    const columns = [
      generation,
      jobWindow(qprogress),
      allocation +
        `${walkerScene(cprogress)}<img class="weight-equation" src="${assets["eq-afqmc-mean"]}" alt="Importance-weighted local energies"/>${energyChart(cprogress, step >= 4)}`,
    ];
    return `<div class="execution-pillars stage-${step >= 4 ? 4 : shown}">${columns.map((body, i) => `<section data-morph="execution-pillar-${i}" class="execution-pillar ${i < shown ? "shown" : "future"} ${i === active ? "focused" : ""}"><div class="location-label">${["Classical CPU", "Quantum device / simulator", "Parallel Slurm CPUs"][i]}</div><h3>${["Payload generation", "Quantum batch execution", "Walker propagation"][i]}</h3><div class="pillar-body">${body}</div></section>`).join("")}</div><div class="execution-foot">${step ? `<span>${fmt(app.workload?.snapshots, 0)} programs · ${fmt(app.workload?.total_shots, 0)} shots · ${app.propagation?.walkers} walkers</span>` : ""}<span>Captured Slurm playground · ideal DDSIM · replay slowed for presentation · band: walker standard error</span></div>`;
  }
  const compilerLabels = [
    "Input program",
    "Device specification",
    "Optimization",
    "Placement & routing",
    "Native synthesis",
    "Executable",
  ];
  function compilerFlow(
    active,
    content,
    { target = targets[0], complete = false } = {},
  ) {
    const miniature = (i) => {
      if (i === 0 || i === 2 || i === 4)
        return circuit(
          targetStage(
            target,
            ["source", "", "optimized", "", "target-native-synthesis"][i],
          )?.circuit,
          { limit: 8, key: `tile-${i}` },
        );
      if (i === 1 || i === 3)
        return deviceView(
          target,
          [],
          i === 3 ? target.compilation?.layout?.initial : null,
        ).replaceAll('data-morph="', 'data-morph="mini-' + i + "-");
      return logo("qir");
    };
    return `<div class="compiler-tiles">${compilerLabels.map((label, i) => `<div data-morph="compile-tile-${i}" class="compile-tile ${i === active ? "selected" : complete || i < active ? "complete" : ""}"><span>${i + 1}</span><b>${label}</b><div class="tile-visual ${complete || i <= active ? "" : "future"}">${miniature(i)}</div></div>`).join("")}</div><div class="compiler-focus" data-morph="compiler-focus">${content}</div>`;
  }
  function deviceCard(target, next = false) {
    if (next)
      return `<div class="spec-card next-device"><h3>Your device could be next</h3><div class="device-port">⇄</div>${svg(viz.qdmiBadge(0, 0, 460, "Powered by"), "-2 -2 464 99", "Powered by QDMI")}<p>One shared interface.<br>Your capabilities and constraints.</p></div>`;
    const i = targets.indexOf(target),
      name = ["iqm", "ibm", "aws"][i],
      formats =
        i === 0
          ? `${logo("qir")}<span>QIR</span>${logo("iqm")}<span>JSON</span>`
          : i === 1
            ? "<span>QPY</span>"
            : "<span>OpenQASM 3</span>";
    const names = ["R · CZ", "RZ · SX · X · CZ", "GPI · GPI2 · RZZ"];
    return `<div class="spec-card" data-morph="device-card"><div class="spec-title">${logo(name)}<h3>${esc(target.label)}</h3></div><div class="spec-layout">${deviceView(target)}<div class="spec-data"><b>${target.metadata.qubits}</b><span>qubits</span><strong>${names[i]}</strong><span>Native gates</span><div class="spec-formats">${formats}</div>${svg(viz.qdmiBadge(0, 0, 350, "Powered by"), "0 0 350 90", "QDMI device")}</div></div><div class="device-links">${qr(["iqm", "ibm", "braket"][i], ["iqm-finland.github.io/QDMI-on-IQM", "ibm-qdmi-device.readthedocs.io", "amazon-braket-qdmi-device.readthedocs.io"][i], target.id)}</div></div>`;
  }
  function compilationScene(step, progress) {
    const target = targets[step >= 11 ? 2 : step >= 9 ? 1 : 0];
    let active = 0,
      body = "";
    if (step === 0)
      body = `<div class="circuit-hero"><h3>${app.chemistry.molecule} · one actual shadow program</h3>${circuit(targetStage(target, "source")?.circuit, { limit: 32, key: "payload" })}${programMetrics(target, "source")}</div>`;
    if ([1, 9, 11].includes(step)) {
      active = 1;
      body = deviceCard(target);
    }
    if (step === 2) {
      active = 2;
      body = `<div class="before-after"><section><h3>Prepared program</h3>${circuit(targetStage(target, "source")?.circuit, { limit: 24, key: "payload" })}${programMetrics(target, "source")}</section><section><h3>After optimization</h3>${circuit(targetStage(target, "optimized")?.circuit, { limit: 24, key: "optimized" })}${programMetrics(target, "optimized")}</section></div>`;
    }
    if (step >= 3 && step <= 6) {
      active = 3;
      body = routingReplay(target, step - 3, progress);
    }
    if (step === 7 || step === 10 || step === 12) {
      active = 4;
      body = nativeSynthesisScene(target, progress);
    }
    if (step === 8) {
      active = 5;
      body = `<div class="emitted-program">${logo("qir")}<div><h3>QIR 2.1 executable</h3><p>Native operations · physical qubits</p><div class="binary-card">LLVM IR / bitcode</div></div><span class="big-arrow">→</span>${logo("iqm")}<div><h3>IQM Emerald</h3><p>Payload checked on DDSIM</p></div></div>`;
    }
    if (step === 13) {
      active = 5;
      body = `<div class="abstract-compiler"><div>${logo("pennylane")}<h3>Application</h3></div><span>+</span><div>${logo("qdmi")}<h3>Device capabilities</h3></div><span>→</span><div>${logo("mqt")}<h3>Compiler Collection</h3></div><span>→</span><div>${logo("qir")}<h3>Executable program</h3></div></div>`;
    }
    if (step === 14) {
      active = 1;
      body = deviceCard(target, true);
    }
    return (
      compilerFlow(active, body, { target, complete: step >= 8 }) +
      sourceNote(
        "Provider metadata → local QDMI target model · execution on DDSIM · no hardware jobs",
      )
    );
  }
  function nativeSynthesisScene(target, progress) {
    const native = targetStage(target, "target-native-synthesis")?.circuit;
    const operations = window.MQSF_CIRCUIT.schedule(native);
    const position = progress * Math.max(0, ...operations.map((o) => o.column));
    const selected = operations.filter(
      (o) => o.column === Math.floor(position),
    );
    const active = selected.map((o) => o.index);
    const site = (q) => native.qubits.find((w) => w.id === q).site;
    const sites = selected.flatMap((o) => o.qubits.map(site));
    const edges = new Set(
      selected
        .filter((o) => o.qubits.length === 2)
        .map((o) =>
          o.qubits
            .map(site)
            .sort((a, b) => a - b)
            .join(":"),
        ),
    );
    const routed = targetStage(target, "place-and-route")?.circuit;
    return `<div class="architecture-grid synthesis-replay"><div><h3>Routed operations → native gates</h3><div class="synthesis-input">${circuit(routed, { limit: 18, key: "synthesis-input" })}</div><div class="synthesis-output">${circuit(native, { limit: 18, active, camera: position, key: "synthesis-output" })}</div>${programMetrics(target, "target-native-synthesis")}</div><div>${deviceView(target, sites, null, null, edges)}</div></div>`;
  }
  function routingReplay(target, phase, progress) {
    const trace = target.compilation.routing_trace || [];
    const trial = trace.find((e) => e.phase === "selected")?.trial ?? 0;
    const sweep =
      phase === 3
        ? trace.find((e) => e.phase === "final-routing")
        : trace.find(
            (e) =>
              e.trial === trial &&
              e.phase === (phase === 2 ? "backward" : "forward"),
          );
    if (!phase) {
      const initial =
        trace
          .find((e) => e.trial === trial && e.phase === "forward")
          ?.before.slice(0, target.compilation.layout.initial.length) ||
        target.compilation.layout.initial;
      return `<div class="architecture-grid"><div><h3>Selected trial · initial placement</h3>${circuit(targetStage(target, "optimized")?.circuit, { limit: 24, key: "initial-placement" })}<div class="mapping-legend">${initial.map((p, i) => `<span>q${i} → ${p}</span>`).join("")}</div><div class="telemetry"><strong>0<small> SWAPs</small></strong><span>${target.compilation.routing_trace.filter((e) => e.phase === "score").length} candidate scores → selected trial</span></div></div><div>${deviceView(target, initial, initial)}</div></div>`;
    }
    if (!sweep?.steps?.length)
      throw new Error("Routing replay requires captured sweep steps");
    const steps = sweep.steps,
      cursor = Math.min(steps.length - 1, progress * steps.length),
      index = Math.floor(cursor),
      event = steps[index];
    let layout = sweep.before.slice(
        0,
        target.compilation.layout.initial.length,
      ),
      before = [...layout],
      swaps = 0;
    steps.slice(0, index + 1).forEach((e, i) => {
      if (e.operation === "qco.swap") {
        if (i === index) before = [...layout];
        const [a, b] = e.sites;
        layout = layout.map((site) => (site === a ? b : site === b ? a : site));
        swaps++;
      } else if (i === index) before = [...layout];
    });
    const physical = [...new Set(steps.flatMap((e) => e.sites))].sort(
      (a, b) => a - b,
    );
    const captured = {
      qubits: physical.map((id) => ({ id, site: id, label: `p${id}` })),
      operations: steps.map((e, i) => ({
        id: `sweep-${i}`,
        name: e.operation.replace(/^qco\./, ""),
        qubits: e.sites,
      })),
    };
    const labels = [
      "",
      "Forward traversal",
      "Backward traversal",
      "Final forward routing",
    ];
    return `<div class="architecture-grid"><div><h3>${labels[phase]}</h3>${circuit(captured, { limit: 20, active: index, key: "sweep" })}<div class="telemetry"><b>Trial ${trial + 1} · operation ${index + 1} / ${steps.length}</b><strong>${swaps}<small> SWAPs</small></strong></div><div class="mapping-legend">${layout.map((p, i) => `<span>q${i} → ${p}</span>`).join("")}</div></div><div>${deviceView(target, event.sites, layout, { before, fraction: progress === 1 ? 1 : cursor - index })}</div></div>`;
  }
  function measuredGrowth() {
    const points = evidence.grover_size?.points || [];
    if (!points.length)
      return '<p class="note">Grover measurements are being regenerated.</p>';
    const x0 = 160,
      y0 = 530,
      w = 1300,
      h = 420,
      minQ = points[0].qubits,
      maxQ = points.at(-1).qubits;
    const max = Math.ceil(
      Math.log10(Math.max(...points.map((p) => p.unrolled_bytes))),
    );
    const min = Math.floor(
      Math.log10(Math.min(...points.map((p) => p.preserved_bytes))),
    );
    const x = (q) => x0 + ((q - minQ) / (maxQ - minQ)) * w,
      y = (v) => y0 - ((Math.log10(v) - min) / (max - min)) * h;
    let body = text(
      160,
      45,
      "Emitted program size · bytes · logarithmic scale",
      30,
      "#142b45",
      "start",
    );
    for (let power = min; power <= max; power++)
      body +=
        line(x0, y(10 ** power), x0 + w, y(10 ** power), "#dbe5ef", 1) +
        text(
          x0 - 24,
          y(10 ** power) + 8,
          `10${["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸"][power] || "^" + power}`,
          26,
          "#526d88",
          "end",
        );
    for (const [key, color, label] of [
      ["unrolled_bytes", "#cc704b", "Fully unrolled"],
      ["preserved_bytes", "#2f70b8", "Structure preserved"],
    ]) {
      body += `<polyline points="${points.map((p) => `${x(p.qubits)},${y(p[key])}`).join(" ")}" fill="none" stroke="${color}" stroke-width="5"/>`;
      body += points
        .map(
          (p) =>
            `<circle cx="${x(p.qubits)}" cy="${y(p[key])}" r="5" fill="${color}"/>`,
        )
        .join("");
      body += text(1220, key === "unrolled_bytes" ? 50 : 90, label, 30, color);
    }
    points
      .filter((_, i) => i % 2 === 0)
      .forEach((p) => {
        body += text(x(p.qubits), 574, p.qubits, 27, "#526d88");
      });
    body += text(810, 626, "Number of qubits", 31);
    return `<div class="growth-chart">${svg(body, "0 0 1640 660", "Measured MQT Core Grover program sizes, preserved and fully unrolled")}</div>`;
  }
  function underHood(step, progress) {
    if (step === 0 || step === 6)
      return compilerFlow(
        0,
        `<div class="scene compact-world">${viz.architecture("detail")}</div>`,
        { complete: true },
      );
    let body = "";
    if (step === 1)
      body = `<div class="ir-columns">${["source", "qc", "qco"]
        .map((id, i) => {
          const a = appStage(id);
          return `<section><h3>${["OpenQASM 3", "QC · references", "QCO · values"][i]}</h3>${code(a, { start: i ? focus(a, i === 1 ? "qc.h" : "qco.h", 1) : 0, count: i ? 9 : 14, title: i ? undefined : "AFQMC state preparation · payload lines 1–14", compact: true })}</section>`;
        })
        .join("")}</div>`;
    if (step === 2) {
      const o = evidence.optimization || {};
      body = `<div class="before-after">${["before", "after"].map((id, i) => `<section><h3>${i ? "After canonicalization" : "Before canonicalization"}</h3>${code(o[id], { start: focus(o[id], i ? "qco.rx" : "qco.h", 0), count: i ? 6 : 4, compact: true })}${circuit(o[id]?.circuit, { limit: 12, key: "canonical-" + id })}</section>`).join("")}</div>`;
    }
    if (step === 3) body = searchScene(progress);
    if (step === 4) body = weylScene(progress);
    if (step === 5)
      body = `<div class="qir-output">${structureCode(appStage("qir-adaptive"), "Actual device-native QIR", true)}</div>`;
    return compilerFlow([0, 0, 2, 3, 4, 5, 0][step], body, { complete: true });
  }
  function searchScene(progress) {
    const trace = targets[0]?.compilation.routing_trace || [];
    const search = trace.find((e) => e.search?.nodes?.length)?.search;
    if (!search) return routingReplay(targets[0], 1, progress);
    const nodes = search.nodes,
      count = Math.max(1, Math.ceil(search.visited.length * progress));
    const visited = new Set(search.visited.slice(0, count));
    const current =
      nodes.find(
        (n) =>
          n.id === (progress >= 1 ? search.goal : search.visited[count - 1]),
      ) || nodes[0];
    const rows = new Map(),
      points = new Map();
    nodes.forEach((n) => {
      const depth = n.depth ?? (points.get(n.parent)?.depth + 1 || 0);
      const row = rows.get(depth) || [];
      row.push(n);
      rows.set(depth, row);
      points.set(n.id, { depth });
    });
    rows.forEach((row, depth) =>
      row.forEach((n, i) =>
        points.set(n.id, {
          x: 80 + ((i + 1) * 1050) / (row.length + 1),
          y: 60 + depth * 65,
          depth,
        }),
      ),
    );
    let body = "";
    nodes.forEach((n) => {
      const p = points.get(n.id),
        a = points.get(n.parent);
      body += `<g data-morph="search-node-${n.id}">${a ? line(a.x, a.y, p.x, p.y, "#bdd0e2", 2) : ""}<circle cx="${p.x}" cy="${p.y}" r="${n.id === current.id ? 14 : 7}" fill="${n.id === current.id ? "#087d92" : visited.has(n.id) ? "#2f70b8" : "#cbdced"}"/></g>`;
    });
    const layout = current.layout.slice(
      0,
      targets[0].compilation.layout.initial.length,
    );
    return `<div class="search-grid" data-search-current="${current.id}"><div><h3>A* searches candidate SWAP sequences</h3>${svg(body, `0 0 1200 ${Math.max(430, rows.size * 65 + 50)}`, "Recorded A* candidates and visited sequence")}<p class="note">${count} visited · ${nodes.length} recorded candidates · ${search.truncated ? "bounded trace" : "actual search"}<br>score = path cost + estimated remaining cost</p></div><div>${deviceView(targets[0], [], layout)}</div></div>`;
  }
  function weylScene(progress) {
    const w = evidence.weyl || {},
      points = w.points || [];
    const angle = 0.35 * progress;
    const project = ([a, b, c]) => [
      100 +
        (a / (Math.PI / 4)) * 570 -
        (b / (Math.PI / 4)) * (165 + 80 * Math.sin(angle)),
      360 -
        (b / (Math.PI / 4)) * 180 -
        (c / (Math.PI / 4)) * (110 + 65 * Math.sin(angle)),
    ];
    const vertices = (w.vertices || []).map((v) => project(v.coordinates));
    let body = "";
    [
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 2],
      [1, 3],
      [2, 3],
      [0, 4],
      [1, 4],
      [2, 4],
    ]
      .filter(([a, b]) => vertices[a] && vertices[b])
      .forEach(([a, b]) => {
        body += line(...vertices[a], ...vertices[b], "#9bb7d1", 3);
      });
    body += vertices.length
      ? `<polygon points="${vertices.map((v) => v.join(",")).join(" ")}" fill="#dcebf8" opacity=".2"/>`
      : "";
    (w.vertices || []).forEach((v, i) => {
      body += text(
        vertices[i][0],
        vertices[i][1] + (i < 2 ? 35 : i === 4 ? 36 : -20),
        v.label,
        25,
      );
    });
    const count = Math.ceil(points.length * progress);
    points.slice(0, count).forEach((p, i) => {
      const [xx, yy] = project(p.coordinates);
      body += `<circle cx="${xx}" cy="${yy}" r="${9 + 5 * progress}" fill="#2f70b8" opacity="${0.25 + 0.65 * progress}"/>`;
    });
    return `<div class="weyl-grid"><div>${svg(body, "0 0 820 530", "Weyl chamber of measured two-qubit fused matrices")}</div><div><h3>Fuse. Decompose. Synthesize.</h3><p>Two-qubit operations share a compact geometric representation.</p><div class="synthesis-equation">U = (k₁ ⊗ k₂)<br>exp[i(a XX + b YY + c ZZ)]<br>(k₃ ⊗ k₄)</div><p class="fusion-example">CX + SWAP → two CZ gates</p><p class="note">One accepted H₂ fusion · iSWAP local-equivalence class</p><p class="note">Local one-qubit rotations surround the nonlocal interaction.</p></div></div>`;
  }
  function structureScene(step) {
    if (step === 0) return measuredGrowth();
    if (step === 9)
      return `<div class="exchange-scene">${logo("jeff")}<h3>Structured programs across the ecosystem</h3><p>Loops · branches · calls · quantum operations</p><div class="exchange-flow">${logo("mqt")}<span>⇄</span>${logo("jeff")}<span>⇄</span><div>Tools &amp; frameworks</div></div><p class="note">jeff exchange-format collaboration · with the authors of “Why Are We Unrolling?”</p></div>`;
    const rus = step >= 7,
      v = variant(rus ? "rus" : "parity"),
      a = v.stages.find((a) => a.id === "source"),
      qir = v.exports.find((a) => a.id === "qir-adaptive");
    if (step === 1 || step === 7)
      return `<div class="structured-grid"><div>${code(a, { count: 30, compact: true, title: rus ? "Repeat-until-success · complete source" : "Syndrome extraction · complete source" })}</div><div>${circuit(a.circuit, { limit: rus ? 18 : 16, key: rus ? "rus" : "syndrome" })}<div class="structure-promise">OpenQASM 3.1<br><span>Loops, reset and measurement feedback</span></div></div></div>`;
    if (step === 2)
      return `<div class="structured-flow"><div>${code(a, { count: 30, compact: true })}</div><div>${logo("mqt")}<h3>Compiler Collection</h3><p>Optimize → map → synthesize</p><span class="big-arrow">→</span>${logo("qir")}</div></div>`;
    return `<div class="adaptive-source focus-${step}">${structureCode(qir, "QIR 2.1 Adaptive Profile · native operations, preserved control flow", true)}</div>${rus ? "" : `<div class="dynamic-features">${["reset", "branches", "loops"].map((label, i) => `<span class="${step === i + 4 ? "selected" : ""}">${label}</span>`).join("")}</div>`}`;
  }
  function adaptiveRuntime(step, progress) {
    if (step === 0)
      return `<div class="scene compact-world">${viz.architecture("detail")}</div><p class="runtime-transition">From compilation to execution</p>`;
    const elapsed =
      step === 1 ? qdmiStart(run) : interpolateTime(run, progress);
    const source = variant("qpe").stages?.find((a) => a.id === "source");
    return `<div class="adaptive-runtime"><div class="qpe-overview"><h3>Iterative quantum phase estimation</h3>${circuit(source?.circuit, { limit: 18, key: "qpe" })}<div class="runtime-products">${logo("qdmi")}<span>→</span>${logo("mqt")}<strong>DDSIM</strong><span>→</span>${logo("llvm")}<strong>JIT</strong></div><p>Adaptive QIR 2.1 · quantum operations<br>and classical computation in one runtime</p></div><div class="execution-grid"><div>${runtimeSequence(run, elapsed)}</div><div><div class="clock-line"><div style="width:${(elapsed / totalTime()) * 100}%"></div></div>${histogram(run, elapsed)}</div></div></div>${sourceNote("Recorded serial execution · four logical qubits · eight output bits · phase 1/3 · timings slowed for inspection")}`;
  }
  function projectResources() {
    return `<div class="project-resources"><div class="project-resource"><img class="project-qr" src="${assets["qr-core-repo"]}" alt="MQT Core repository"/><div><div class="project-title">MQT Core <span class="project-credit-logos">${logo("mqsc")}${logo("tum-cda")}</span></div><p class="project-stats">★ ${fmt(references.core_stats?.github?.stargazers_count ?? 0, 0)} · ↓ ${esc(references.core_stats?.dashboard_snapshot?.total_downloads ?? "—")} PyPI</p><p class="project-credit">MQSC · TUM CDA · MQT community</p></div></div><div class="project-resource"><img class="project-qr" src="${assets["qr-qdmi-repo"]}" alt="QDMI repository"/><div><div class="project-title">QDMI <span class="project-credit-logos">${logo("tum")}${logo("lrz")}${logo("mqv")}${logo("mqsc")}</span></div><p class="project-stats">★ ${fmt(references.qdmi_stats?.github?.stargazers_count ?? 0, 0)} · Open, vendor-neutral standard</p><p class="project-credit">Created by TUM CDA, TUM CAPS &amp; LRZ<br>Maintained by MQV &amp; MQSC, with the community</p></div></div></div>`;
  }
  const slides = [];
  function add(id, section, title, builds, render, notes = "", playbacks = {}) {
    slides.push({ id, section, title, builds, render, notes, playbacks });
  }
  add(
    "title",
    "Munich Quantum Software Forum · October 2026",
    "System Software for Quantum Computing",
    0,
    () =>
      `<div class="title-layout"><div class="title-main"><p class="talk-date">MQSF 2026 · Munich · 14–15 October</p><h1>System Software<br>for Quantum Computing</h1><p class="title-slogan">From the Metal to the User</p><div class="speaker"><b>Lukas Burgholzer</b><div class="affiliations"><p>CTO &amp; Co-founder<br><span>Munich Quantum Software Company</span></p><p>Senior Researcher<br><span>Chair for Design Automation · TUM</span></p></div><p class="speaker-social">@burgholzer <span>·</span> LinkedIn / Lukas Burgholzer</p></div></div><aside class="title-event">${logo("mqsc", "title-brand")}${logo("tum-cda", "title-cda")}${qr("company", "mq.sc", "System Software for Quantum Computing")}</aside></div>`,
    "Introduce both roles. The system-software subtitle appears once; the ecosystem story begins on the next click.",
  );
  const phases = viz.architecturePhases;
  const architectureSteps = ["problem", "stack", "detail"].flatMap((mode) =>
    phases[mode].map((label, phase) => ({ mode, phase, label })),
  );
  add(
    "integration",
    "01 · System software",
    "Connecting the quantum ecosystem",
    architectureSteps.length - 1,
    (s) => {
      const { mode, phase } = architectureSteps[s];
      return (
        heading(
          {
            problem: "Software that connects the quantum ecosystem",
            stack: "A shared software stack",
            detail: "From the Metal to the User",
          }[mode],
          {
            problem: "",
            stack: "Modular · Efficient · Extensible",
            detail: "Common interfaces. Compilation. Orchestration.",
          }[mode],
        ) +
        `<div class="scene world-scene" data-morph="world">${viz.architecture(mode, phase)}</div>` +
        (mode === "stack"
          ? `<div class="architecture-reference">${logo("mqss")}${qr("mqss", "doi.org/10.1145/3773656.3773669", "Burgholzer, Echavarria, et al., 2026")}</div>`
          : mode === "detail"
            ? projectResources()
            : "")
      );
    },
    "Each click reveals one path or component. A workstation reaches cloud-hosted quantum systems; on-premise paths go through HPC. QDMI is a community standard, not an MQSC-owned interface. Compiler Collection and Orchestrator use the MQT Core repository.",
  );
  add(
    "hybrid-algorithm",
    "02 · A practical hybrid application",
    "Quantum-assisted auxiliary-field Quantum Monte Carlo",
    2,
    (s) =>
      heading(
        "Quantum-assisted auxiliary-field Quantum Monte Carlo",
        "A molecular energy calculation across classical and quantum resources.",
      ) +
      applicationScene(s) +
      `<div class="reference-strip">${logo("aws")}${qr("afqmc", "github.com/amazon-braket/amazon-braket-examples", "Adapted from the public Amazon Braket AFQMC example")}</div>`,
    "Preparation constructs a trial state and random measurement bases. Quantum execution collects classical shadows once. Classical walkers reuse the data to estimate a weighted energy. This is a bounded software demonstration, not a quantum advantage claim.",
  );
  add(
    "afqmc-execution",
    "02 · A practical hybrid application",
    "Executing a hybrid workflow",
    4,
    (s, t) =>
      heading("Executing a hybrid workflow") +
      `<aside class="execution-repos">${["core", "qdmi"].map((name) => `<div><img src="${assets["qr-" + name + "-repo"]}" alt="${name === "core" ? "MQT Core" : "QDMI"} repository"/><span>${name === "core" ? "MQT Core" : "QDMI"}</span></div>`).join("")}</aside>` +
      hybridExecution(s, t ?? 1),
    "Three click-controlled pillars: payload generation, one multi-program QDMI job, then parallel walker propagation in Slurm. Only the final click overlays the Hartree–Fock trial. The error band covers walker uncertainty, not finite-shadow error or systematic biases.",
    {
      2: { duration: 7000, transition: 700 },
      3: { duration: 8000, transition: 700 },
    },
  );
  add(
    "routing",
    "03 · Compiling for a device",
    "From quantum program to executable",
    14,
    (s, t) =>
      heading("From quantum program to executable") +
      compilationScene(s, t ?? 1),
    "Follow the actual AFQMC payload. Select a target, optimize, place, replay real routing sweeps, synthesize and generate executable QIR. Changing device cards and recompiling are separate clicks. Hardware metadata is modeled locally; verification execution uses DDSIM.",
    Object.fromEntries(
      [4, 5, 6, 7, 10, 12].map((i) => [i, { duration: 5500, transition: 700 }]),
    ),
  );
  add(
    "compiler-stages",
    "03 · Inside the MQT Compiler Collection",
    "Inside the compiler",
    6,
    (s, t) => heading("Inside the compiler") + underHood(s, t ?? 1),
    "The machine-room view exposes actual IR, canonicalization, a recorded A* search and measured two-qubit matrix coordinates. The Weyl chamber is a mathematical view of fused interactions, not a replay of an uninstrumented synthesis search.",
    {
      3: { duration: 5000, transition: 700 },
      4: { duration: 4000, transition: 700 },
    },
  );
  add(
    "structured",
    "04 · Structured quantum programs",
    "Why are we unrolling?",
    9,
    (s) =>
      heading("Why are we unrolling?") +
      structureScene(s) +
      `<div class="reference-strip paper-reference"><div class="paper-logos">${["tum-cda", "oxford", "quantinuum", "xanadu", "mqsc", "unitary", "ubc"].map((name) => logo(name)).join("")}</div>${qr("unrolling", "arxiv.org/abs/2609.16171", "Rovara, Haag, et al., 2026 · Why Are We Unrolling?")}</div>`,
    "Measured Grover output size motivates preserving structure. The complete compact syndrome source compiles to adaptive QIR with reset, branches and loops intact. RUS uses a literature circuit with its required failure correction. jeff retains structure for exchange between tools.",
  );
  add(
    "adaptive-execution",
    "04 · Adaptive QIR execution",
    "A runtime for structured quantum programs",
    2,
    (s, t) =>
      heading("A runtime for structured quantum programs") +
      adaptiveRuntime(s, t ?? 1),
    "Return to the overall architecture and focus on runtime. DDSIM executes adaptive QIR using a JIT. Calls, JIT boundaries and each shot completion share a monotonic clock. This intentionally instrumented serial capture does not demonstrate optimized shot branching.",
    { 2: { duration: 9000, transition: 700 } },
  );
  add(
    "ecosystem",
    "From the metal to the user",
    "System Software for Quantum Computing",
    0,
    () =>
      heading(
        "System Software for Quantum Computing",
        "Common Interfaces. Compilation. Orchestration.",
      ) +
      `<div class="scene world-scene closing-world" data-morph="world">${viz.architecture("ecosystem")}</div><div class="closing-resources"><div><div class="closing-logos">${logo("mqt")}${logo("qdmi")}${logo("mqss")}${logo("mqv")}${logo("tum-cda")}${logo("lrz")}</div><p class="closing-credit">Built with the MQT and QDMI communities — driven into practice by MQSC.</p></div>${qr("company", "mq.sc", "System Software for Quantum Computing")}</div>`,
    "Zoom back out from runtime to the common architecture. Credit the MQT and QDMI communities and their originating institutions; close with MQSC bringing the stack into practice.",
  );

  function resize() {
    document.documentElement.style.setProperty(
      "--scale",
      Math.min(innerWidth / 1920, innerHeight / 1080),
    );
  }
  function render(progress = null, transition = 0) {
    state.token++;
    state.animation?.cancel();
    state.animation = null;
    if (progress !== null && !transition) window.MQSF_MOTION.finish();
    draw(progress, progress === null ? 950 : transition);
    const slide = slides[state.slide];
    $("section-label").textContent = slide.section;
    $("deck").classList.toggle("title-slide", state.slide === 0);
    $("slide-number").innerHTML =
      `${String(state.slide + 1).padStart(2, "0")} / ${slides.length}<span class="build-dots" aria-label="Build ${state.step + 1} of ${slide.builds + 1}">${Array.from({ length: slide.builds + 1 }, (_, i) => (i === state.step ? "<b>•</b>" : "•")).join("")}</span>`;
    $("build-progress").style.width =
      `${((state.slide + (state.step + 1) / (slide.builds + 1)) / slides.length) * 100}%`;
    $("speaker-notes").textContent =
      `${state.slide + 1}. ${slide.title} — ${slide.notes}`;
    history.replaceState(null, "", `#${state.slide + 1}.${state.step}`);
  }
  function draw(progress = null, morph = 0) {
    const html = slides[state.slide].render(state.step, progress);
    if (morph) window.MQSF_MOTION.replace($("slide"), html, morph);
    else window.MQSF_MOTION.update($("slide"), html);
    $("slide").dataset.slide = String(state.slide + 1);
    $("slide").dataset.step = String(state.step);
    $("slide").dataset.progress = progress === null ? "" : String(progress);
  }
  function animateTimeline(playback) {
    const token = state.token,
      duration = playback.duration;
    let begin, pending;
    let transitioning = !!playback.transition;
    const cancel = () => cancelAnimationFrame(pending);
    const end = () => {
      if (token !== state.token) return;
      cancel();
      window.MQSF_MOTION.finish();
      state.animation = null;
      draw();
    };
    state.animation = { end, cancel, loop: !!playback.loop };
    function frame(now) {
      if (token !== state.token || !state.animation) return;
      // RAF timestamps share a frame clock; performance.now() may be ahead of it.
      begin ??= now;
      if (transitioning) {
        // Let shared geometry reach the initial frame before replay updates it.
        if (now - begin < playback.transition) {
          pending = requestAnimationFrame(frame);
          return;
        }
        window.MQSF_MOTION.finish();
        transitioning = false;
        begin = now;
      }
      const elapsed = Math.max(0, now - begin),
        cycle = playback.loop ? elapsed % (duration + 2400) : elapsed;
      draw(Math.min(1, cycle / duration));
      if (playback.loop || elapsed < duration)
        pending = requestAnimationFrame(frame);
      else end();
    }
    pending = requestAnimationFrame(frame);
  }
  function play(playback) {
    if (!playback || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      render();
      return;
    }
    render(0, playback.transition || 0);
    animateTimeline(playback);
  }
  function go(slide, step = 0, forward = false) {
    state.slide = Math.max(0, Math.min(slide, slides.length - 1));
    state.step = Math.max(0, Math.min(step, slides[state.slide].builds));
    play(forward ? slides[state.slide].playbacks[state.step] : null);
  }
  function next() {
    if (state.animation) {
      state.animation.end();
    }
    const current = slides[state.slide];
    if (state.step < current.builds) {
      state.step++;
      play(current.playbacks[state.step]);
    } else if (state.slide < slides.length - 1) go(state.slide + 1, 0, true);
  }
  function previous() {
    if (state.step > 0) go(state.slide, state.step - 1);
    else if (state.slide > 0)
      go(state.slide - 1, slides[state.slide - 1].builds);
  }
  function overview(open) {
    const d = $("overview");
    if (!open) {
      d.close();
      $("deck").focus();
      return;
    }
    state.animation?.end();
    state.overview = state.slide;
    $("slide-grid").innerHTML = slides
      .map(
        (s, i) =>
          `<button data-slide="${i}" class="${i === state.overview ? "selected" : ""}"><b>${i + 1}</b>${esc(s.title)}</button>`,
      )
      .join("");
    $("slide-grid")
      .querySelectorAll("button")
      .forEach((b) =>
        b.addEventListener("click", () => {
          go(Number(b.dataset.slide));
          overview(false);
        }),
      );
    d.showModal();
    $("slide-grid").children[state.overview].focus();
  }
  document.addEventListener("keydown", (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if ($("overview").open) {
      if (e.key === "Escape") {
        e.preventDefault();
        overview(false);
        return;
      }
      const offset = {
        ArrowLeft: -1,
        ArrowRight: 1,
        ArrowUp: -5,
        ArrowDown: 5,
      }[e.key];
      if (offset) {
        e.preventDefault();
        state.overview = Math.max(
          0,
          Math.min(slides.length - 1, state.overview + offset),
        );
        $("slide-grid")
          .querySelectorAll("button")
          .forEach((b, i) =>
            b.classList.toggle("selected", i === state.overview),
          );
        $("slide-grid").children[state.overview].focus();
      }
      if (e.key === "Enter") {
        e.preventDefault();
        go(state.overview);
        overview(false);
      }
      return;
    }
    if (/^[0-9]$/.test(e.key)) {
      state.jump = (state.jump + e.key).slice(-2);
      $("jump-indicator").textContent = `Go to ${state.jump} ↵`;
      $("jump-indicator").hidden = false;
      return;
    }
    if (e.key === "Enter" && state.jump) {
      e.preventDefault();
      go(Number(state.jump) - 1);
      state.jump = "";
      $("jump-indicator").hidden = true;
      return;
    }
    if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) {
      e.preventDefault();
      next();
    }
    if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) {
      e.preventDefault();
      previous();
    }
    if (e.key === "Home") {
      e.preventDefault();
      go(0);
    }
    if (e.key === "End") {
      e.preventDefault();
      go(slides.length - 1, slides.at(-1).builds);
    }
    if (e.key === "Escape" || e.key.toLowerCase() === "g") {
      e.preventDefault();
      state.jump = "";
      $("jump-indicator").hidden = true;
      overview(true);
    }
    if (e.key.toLowerCase() === "b")
      $("blackout").hidden = !$("blackout").hidden;
    if (e.key.toLowerCase() === "p")
      $("speaker-notes").hidden = !$("speaker-notes").hidden;
    if (e.key.toLowerCase() === "f")
      document.fullscreenElement
        ? document.exitFullscreen()
        : document.documentElement.requestFullscreen();
  });
  let touchX = 0;
  document.addEventListener(
    "touchstart",
    (e) => {
      touchX = e.changedTouches[0].screenX;
    },
    { passive: true },
  );
  document.addEventListener(
    "touchend",
    (e) => {
      const dx = e.changedTouches[0].screenX - touchX;
      if (Math.abs(dx) > 80) dx < 0 ? next() : previous();
    },
    { passive: true },
  );
  addEventListener("resize", resize);
  addEventListener("beforeprint", () => {
    state.animation?.end();
    $("print-deck").innerHTML = slides
      .map(
        (s, i) =>
          `<section class="print-slide"><header class="slide-header"><span>${esc(s.section)}</span>${logo("mqsc")}</header><article data-slide="${i + 1}" data-step="${s.builds}">${s.render(s.builds)}</article><footer class="slide-footer"><span>Lukas Burgholzer · MQSC</span><span>mq.sc</span><span>${i + 1} / ${slides.length}</span></footer></section>`,
      )
      .join("");
  });
  addEventListener("afterprint", () => {
    $("print-deck").replaceChildren();
  });
  window.MQSF_DECK = {
    slides: slides.map((s) => ({
      id: s.id,
      title: s.title,
      playbacks: s.playbacks,
      builds: s.builds,
      notes: s.notes,
    })),
    getState: () => ({
      slide: state.slide,
      step: state.step,
      animating: !!state.animation,
    }),
    go,
    next,
    previous,
    finishAnimation: () => state.animation?.end(),
    finishMotion: () => window.MQSF_MOTION.finish(),
  };
  $("mqsc-logo").src = assets.mqsc;
  resize();
  const hash = location.hash.match(/^#(\d+)(?:\.(\d+))?$/);
  go(hash ? Number(hash[1]) - 1 : 0, hash ? Number(hash[2] || 0) : 0);
  $("deck").focus();
})();
