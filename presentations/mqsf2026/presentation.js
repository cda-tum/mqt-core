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
  // Preserve every source line. Long compiler listings get a complete overview and a readable lens.
  function fullCode(
    a,
    title,
    { columns = 1, height = 510, width = 800, hot = null, lens = false } = {},
  ) {
    if (!a?.code) return code(a, { title });
    const lines = a.code
      .split("\n")
      .filter((l, i, all) => i < all.length - 1 || l.length);
    const rows = Math.ceil(lines.length / columns);
    const longest = Math.max(...lines.map((l) => l.length), 1);
    const font = Math.min(
      23,
      height / rows / 1.18,
      (width / columns - 40) / longest / 0.61,
    );
    const effect = (l) =>
      /reset/.test(l)
        ? "reset"
        : /phi |add |icmp.*slt|for |while |loop/.test(l)
          ? "loop"
          : /if |br |read_result|measure|__mz__/.test(l)
            ? "branch"
            : "gate";
    const parts = Array.from(
      { length: columns },
      (_, c) =>
        `<pre class="highlight">${lines
          .slice(c * rows, (c + 1) * rows)
          .map((l, j) => {
            const i = c * rows + j;
            return `<span class="code-line ${hot && effect(l) === hot ? "hot" : ""}" data-effect="${effect(l)}"><span class="line-no">${i + 1}</span><span class="code-text">${a.lines_html?.[i] ?? esc(l)}</span></span>`;
          })
          .join("")}</pre>`,
    ).join("");
    return `<div class="code-frame full-code" data-morph="full-${esc(title)}" data-complete-lines="${lines.length}"><div class="code-title">${esc(title)}</div><div class="complete-code" style="--code-columns:${columns};--code-font:${font}px">${parts}</div>${lens && font < 14 ? `<div class="code-lens">${code(a, { start: 0, count: Math.min(18, lines.length), title: "Entry Point · Enlarged View", compact: true })}</div>` : ""}</div>`;
  }
  const evidence = data.compiler_evidence || {};
  const totalTime = (capture = run) => {
    const end = Math.max(
      0,
      ...(capture.events || []).map((e) => e.time_ms + (e.duration_ms || 0)),
      ...(capture.lifecycle || []).map((e) => e.time_ms),
    );
    return capture.program_completions
      ? end
      : Math.max(end, capture.completed_ms || 0, capture.duration_ms || 0);
  };
  const qdmiStart = (capture) =>
    capture.submission_started_ms ??
    capture.events?.find((e) => /try_submit_job|job_submit/.test(e.operation))
      ?.time_ms ??
    0;
  const interpolateTime = (capture, progress) =>
    qdmiStart(capture) + progress * (totalTime(capture) - qdmiStart(capture));
  const icon = (name, cls = "pictogram") =>
    `<span class="${cls}">${svg(viz.icon(name, 0, 0, 100), "0 0 100 100", name)}</span>`;
  const badge = (brand, label) =>
    `<div class="product-badge">${logo(brand)}<span>${label}</span></div>`;
  function runtimeSequence(capture, elapsed) {
    const adapter =
      capture.trace_kind === "python-adapter-and-native-program-completions";
    const events = [
      ...(capture.events || []),
      ...(capture.runtime_events || []),
      ...(capture.lifecycle || []),
    ].sort((a, b) => a.time_ms - b.time_ms);
    const first = (predicate) => events.find(predicate);
    const links = adapter
      ? [
          [
            0,
            1,
            "Compile OpenQASM → QIR",
            first((e) => e.state === "compiling"),
          ],
          [
            1,
            2,
            "Submit One Program Batch",
            first((e) => /try_submit_job/.test(e.operation)),
          ],
          [2, 2, "Execute Programs", first((e) => e.state === "running")],
          [
            2,
            1,
            "Retrieve Indexed Results",
            first((e) => Number.isInteger(e.program_index)),
          ],
          [1, 0, "Return Samples", first((e) => e.state === "done")],
        ]
      : [
          [
            0,
            1,
            "Submit Adaptive QIR",
            first((e) => /job_submit/.test(e.operation)),
          ],
          [
            1,
            2,
            "Compile and Initialize JIT",
            first((e) => e.operation === "JIT compilation begins"),
          ],
          [
            2,
            2,
            "Execute Adaptive Shots",
            first((e) => e.operation === "Serial adaptive sampling begins"),
          ],
          [
            1,
            0,
            "Retrieve Results",
            first((e) => /get_results/.test(e.operation)),
          ],
        ];
    const names = adapter
        ? ["Application", "Core / QDMI", "DDSIM"]
        : ["QDMI", "DDSIM", "LLVM JIT"],
      xs = [82, 320, 558];
    let body = names
      .map(
        (n, i) =>
          text(xs[i], 30, n, 24) + line(xs[i], 53, xs[i], 430, "#cbdced", 2),
      )
      .join("");
    links.forEach(([a, b, label, e], i) => {
      const on = !!e && elapsed >= e.time_ms,
        y = 105 + i * 72;
      body += `<g data-morph="sequence-${i}" opacity="${on ? 1 : 0.12}">${a === b ? `<path d="M${xs[a]} ${y - 10}h34v26h-34" fill="none" stroke="#2f70b8" stroke-width="3"/>` : line(xs[a], y, xs[b], y)}${a !== b ? `<path d="M${xs[b] + (b > a ? -7 : 7)} ${y - 5}L${xs[b]} ${y}L${xs[b] + (b > a ? -7 : 7)} ${y + 5}" fill="none" stroke="#2f70b8" stroke-width="2"/>` : ""}${text(a === b ? xs[a] - 20 : (xs[a] + xs[b]) / 2, y - 17, label, 18, "#142b45", a === b ? "end" : "middle")}</g>`;
    });
    return `<div class="sequence-svg">${svg(body, "0 0 640 450", "Timestamped QDMI workflow")}</div>`;
  }
  function energyChart(progress = 1, compare = true) {
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
      high = Math.max(Math.abs(low) * 1.8, ...values) + padding;
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
        (Math.abs(yy - y(0)) > 28
          ? text(77, yy + 7, fmt(value, 0), 21, "#5b7088", "end")
          : "");
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
      text(77, y(0) + 7, "0", 21, "#087d92", "end") +
      text(405, 378, "Imaginary Time τ (Ha⁻¹)", 25);
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
  function batchState(elapsed) {
    const capture = app.execution || {};
    const complete = (rows) =>
      new Set(
        (rows || [])
          .filter((e) => e.time_ms + (e.duration_ms || 0) <= elapsed)
          .map((e) => e.program_index),
      );
    return {
      compiled: complete(capture.compilation),
      done: complete(capture.program_completions),
      retrieved: complete(
        capture.retrievals ||
          capture.events?.filter((e) => Number.isInteger(e.program_index)),
      ),
    };
  }
  function batchMatrix(elapsed) {
    const { compiled, done, retrieved } = batchState(elapsed),
      n = app.workload?.snapshots || 0;
    // One cell per actual program; group no unrelated events into synthetic jobs.
    const cols = 64,
      cell = 9,
      gap = 2,
      rows = Math.ceil(n / cols);
    const paths = ["", "", "", ""];
    for (let i = 0; i < n; i++) {
      const color = retrieved.has(i)
        ? 3
        : done.has(i)
          ? 2
          : compiled.has(i)
            ? 1
            : 0;
      paths[color] +=
        `M${(i % cols) * (cell + gap)} ${Math.floor(i / cols) * (cell + gap)}h${cell}v${cell}h-${cell}z`;
    }
    // Each program retains its own cell; four paths avoid thousands of DOM updates per frame.
    const body = paths
      .map(
        (path, i) =>
          `<path d="${path}" fill="${["#edf2f7", "#bbd1e8", "#16a5a5", "#2f70b8"][i]}"/>`,
      )
      .join("");
    return `<div class="batch-grid" data-programs="${retrieved.size}" data-completed="${done.size}" data-compiled="${compiled.size}" data-elapsed-ms="${elapsed}">${svg(body, `0 0 ${cols * (cell + gap)} ${rows * (cell + gap)}`, "One cell per program: compiled, completed, retrieved")}<div class="batch-counts"><span>${fmt(compiled.size, 0)} Compiled</span><span>${fmt(done.size, 0)} Completed</span><span>${fmt(retrieved.size, 0)} Retrieved</span></div></div>`;
  }
  function preparationSketch(progress = 1) {
    const p = Math.max(0, Math.min(1, progress));
    let body = `<defs><radialGradient id="molecule-orbital"><stop stop-color="#2f70b8" stop-opacity=".35"/><stop offset="1" stop-color="#2f70b8" stop-opacity="0"/></radialGradient></defs><ellipse cx="110" cy="92" rx="97" ry="63" fill="url(#molecule-orbital)"/><ellipse cx="174" cy="92" rx="97" ry="63" fill="url(#molecule-orbital)"/>${line(106, 92, 178, 92, "#6fa8d8", 8)}<circle cx="106" cy="92" r="21" fill="#2f70b8"/><circle cx="178" cy="92" r="21" fill="#62aaca"/>${text(141, 165, "Molecular Hamiltonian", 26)}`;
    body += `<g opacity="${Math.min(1, p * 3)}"><path d="M145 190V229" stroke="#86aaca" stroke-width="3"/><rect x="38" y="241" width="211" height="96" rx="15" fill="#eaf2fa" stroke="#a9c6e1"/>${text(143, 284, "|Ψₜ⟩ = U(θ)|HF⟩", 27)}${text(143, 317, "Trial State", 22, "#5b7088")}</g>`;
    body += `<g opacity="${Math.max(0, Math.min(1, p * 3 - 1))}">${line(265, 288, 313, 288)}<rect x="329" y="241" width="171" height="96" rx="15" fill="#e8f6f7" stroke="#8ecbd0"/>${text(414, 279, "U(Q)", 30, "#087d92")}${text(414, 316, "Random Bases", 22)}</g>`;
    body += `<g opacity="${Math.max(0, Math.min(1, p * 3 - 2))}">${line(416, 355, 416, 399)}${[2, 1, 0].map((i) => `<rect x="${82 + i * 13}" y="${421 - i * 9}" width="352" height="91" rx="9" fill="white" stroke="#7fa8cb"/>`).join("")}${[0, 1, 2].map((i) => line(111, 441 + i * 24, 411, 441 + i * 24, "#9bb7d1", 2)).join("")}<rect x="147" y="430" width="42" height="61" rx="5" fill="#eaf2fa" stroke="#2f70b8"/><rect x="251" y="430" width="42" height="61" rx="5" fill="#e8f6f7" stroke="#087d92"/>${text(168, 470, "Uₜ", 22)}${text(272, 470, "UQ", 22)}${text(265, 553, "Batch of Quantum Programs", 26)}</g>`;
    return svg(
      body,
      "0 0 540 590",
      "Molecule, trial-state ansatz, random Gaussian rotations, quantum circuit batch",
    );
  }
  function sampleStream(progress = 1) {
    const preview = app.shadow_preview || [],
      shown = Math.ceil(progress * preview.length);
    return `<div class="sample-stream">${preview
      .slice(0, shown)
      .map(
        (sample, i) =>
          `<div data-morph="shadow-sample-${i}" style="opacity:${Math.min(1, progress * preview.length - i)}"><small>Program ${sample.program_index}</small><strong>${esc(sample.outcome)}</strong></div>`,
      )
      .join("")}</div>`;
  }
  const appHeading = "Quantum-Assisted Auxiliary-Field Monte Carlo";
  const appReference = () =>
    `<div class="application-reference">${logo("aws")}${qr("afqmc", "github.com/amazon-braket/amazon-braket-examples", "Amazon Braket · Public AFQMC Example")}</div>`;
  function appPanel(i, body) {
    return `<section data-morph="app-pillar-${i}" class="app-panel"><h3>${icon(["workstation", "cryostat", "server"][i])}<span>${["Classical Preparation", "Quantum Shadow Collection", "Classical Monte Carlo"][i]}</span></h3><div class="app-panel-body">${body}</div><div class="app-badges">${i === 0 ? badge("pennylane", "") : i === 1 ? badge("mqt", "Compiler Collection") + badge("qdmi", "") : badge("mqsc", "Orchestrator")}</div></section>`;
  }
  function appBody(i, progress = 1) {
    if (i === 0) return preparationSketch(progress);
    if (i === 1)
      return `<p class="panel-kicker">${fmt(app.workload?.snapshots, 0)} Circuits × ${fmt(app.workload?.shots_per_snapshot, 0)} Shots</p>${circuit(appStage("source")?.circuit, { full: true, key: "shadow-input", barrier: app.circuit_sections?.trial_primitive_operations })}${sampleStream(progress)}`;
    return `<p class="panel-kicker">Propagate → Overlap → Reweight → Reduce</p>${walkerScene(progress)}<div class="equation-caption"><img src="${assets["eq-afqmc-mean"]}" alt="Weighted local energy"/><span>Weighted Mean</span></div>${energyChart(progress)}`;
  }
  function applicationScene(step, progress = 1) {
    return `<div class="app-panels intro">${[0, 1, 2].map((i) => `<div class="${i > step ? "future" : ""}">${appPanel(i, appBody(i, i === step ? progress : 1))}</div>`).join("")}</div>${appReference()}`;
  }
  function preparationCode() {
    const definitions = app.source_definitions || {};
    // The build supplies verbatim executable lines with docstrings omitted for display.
    const left =
      app.presentation_source?.left ||
      [definitions.hydrogen_trial, definitions.apply_gaussian_givens]
        .filter(Boolean)
        .join("\n");
    const right =
      app.presentation_source?.right ||
      [definitions.apply_pauli_layer, app.source, app.batch_source]
        .filter(Boolean)
        .join("\n");
    return `<div class="preparation-code">${[left, right].map((source, i) => fullCode({ code: source, lines_html: app.presentation_source?.[i ? "right_lines_html" : "left_lines_html"] }, i ? "Shadow Circuit and Batch" : "Trial State and Gaussian Rotations", { height: 490, width: 725 })).join("")}</div>`;
  }
  const phaseTime = (name) =>
    app.execution?.lifecycle?.find((e) => e.state === name)?.time_ms || 0;
  function quantumTime(progress) {
    const capture = app.execution;
    const breaks = [
      phaseTime("compiling"),
      qdmiStart(capture),
      capture.retrievals?.[0]?.time_ms || phaseTime("extracting-results"),
      totalTime(capture),
    ];
    const stops = [0, 0.2, 0.85, 1];
    const segment = progress < 0.2 ? 0 : progress < 0.85 ? 1 : 2;
    const fraction =
      (progress - stops[segment]) / (stops[segment + 1] - stops[segment]);
    return breaks[segment] + fraction * (breaks[segment + 1] - breaks[segment]);
  }
  function workflowClock(step, progress = 1) {
    const prep = phaseTime("compiling"),
      finish = totalTime(app.execution);

    const elapsed =
      step === 0
        ? 0
        : step === 1
          ? prep * progress
          : step === 2
            ? quantumTime(progress)
            : finish;
    const clock =
      step === 3
        ? `τ = ${fmt(progress * app.propagation.tau.at(-1), 2)} Ha⁻¹ · Imaginary-Time Replay`
        : step >= 4
          ? `${fmt(app.slurm.duration_ms / 1000, 2)} s · Complete Slurm Workflow`
          : `${fmt(elapsed / 1000, 2)} s · Recorded Application Clock`;
    const fraction =
      step < 2
        ? (elapsed / Math.max(1, finish)) * 0.6
        : step === 2
          ? (elapsed / finish) * 0.6
          : step === 3
            ? 0.6 + 0.4 * progress
            : 1;
    return `<div class="workflow-clock"><div class="clock-stages"><span class="${step <= 1 ? "active" : ""}">Prepare</span><span class="${step === 2 ? "active" : ""}">Compile · Run · Retrieve</span><span class="${step >= 3 ? "active" : ""}">Propagate on Slurm</span><b>${clock}</b></div><div class="clock-line"><div style="width:${100 * fraction}%"></div></div></div>`;
  }
  function hybridExecution(step, progress = 1) {
    const focused = step < 2 ? 0 : step === 2 ? 1 : step === 3 ? 2 : -1;
    const elapsed =
      step === 2 ? quantumTime(progress) : step > 2 ? Infinity : 0;
    const quantum = `<div class="quantum-batch"><div>${runtimeSequence(app.execution, elapsed)}<p class="panel-kicker">One QDMI Job · ${fmt(app.workload?.snapshots, 0)} Programs</p></div>${batchMatrix(elapsed)}</div>`;
    const workers = app.propagation?.workers || [],
      nodes = new Set(workers.map((w) => w.node)).size;
    const classical = `<div class="slurm-allocation">${logo("slurm")}<span>${nodes} Compute Nodes · ${workers.length} Slurm Ranks</span></div>${walkerScene(step === 3 ? progress : 1)}${energyChart(step === 3 ? progress : 1)}`;
    const bodies = [
      step === 0
        ? fullCode(
            {
              code:
                app.presentation_source?.overview ??
                app.source + "\n" + app.batch_source,
              lines_html: app.presentation_source?.overview_lines_html,
            },
            "Shadow Circuit and Batch",
            { height: 455, width: 1580, columns: 1 },
          )
        : step === 1
          ? preparationCode()
          : appBody(0),
      step >= 2 ? quantum : appBody(1),
      step >= 3 ? classical : appBody(2),
    ];
    return `<div class="app-panels execution focus-${focused}">${bodies.map((body, i) => appPanel(i, body)).join("")}</div>${workflowClock(step, progress)}<div class="workflow-foot"><span>${fmt(app.workload?.snapshots, 0)} Programs · ${fmt(app.workload?.total_shots, 0)} Shots${step >= 3 ? ` · ${app.propagation?.walkers} Walkers per Trial` : ""}</span>${step >= 2 ? `<span>Ideal DDSIM · ${app.execution?.simulator_threads ?? "Recorded"} Executor Slots · Time-Scaled Replay</span>` : ""}</div>`;
  }
  const compilerLabels = [
    "Input Program",
    "Device Specification",
    "Optimization",
    "Placement & Routing",
    "Native Synthesis",
    "Executable",
  ];
  function compilerFlow(active, content, { complete = false } = {}) {
    return `<div class="compiler-tiles">${compilerLabels.map((label, i) => `<div data-morph="compile-tile-${i}" class="compile-tile ${active === i ? "selected" : ""} ${complete || i < active ? "complete" : ""}"><span>${i + 1}</span><b>${label}</b></div>`).join("")}</div>${content}`;
  }
  function deviceCard(
    target,
    { layout = null, activeSites = [], blend = null, next = false } = {},
  ) {
    if (next)
      return `<div class="spec-card next-device" data-morph="device-card">${icon("chip")}<h3>Your Device Could Be Next</h3>${svg(viz.qdmiBadge(0, 0, 350, "Powered by"), "0 0 350 95", "QDMI interface")}<p>Capabilities · Constraints · One Interface</p></div>`;
    const i = targets.indexOf(target),
      brand = ["iqm", "ibm", "aws"][i];
    return `<div class="spec-card" data-morph="device-card" data-target="${esc(target.id)}"><div class="spec-title">${logo(brand)}<h3>${esc(target.label.replace(/^IQM\s*/i, ""))}</h3></div>${deviceView(target, activeSites, layout, blend)}<div class="spec-details"><b>${target.metadata.qubits} Qubits</b><span>${["R · CZ", "RZ · SX · X · CZ", "GPI · GPI2 · RZZ"][i]}</span><div class="spec-formats">${i === 0 ? logo("qir") + logo("iqm") + "-JSON" : i === 1 ? "QPY" : "OpenQASM 3"}</div></div><code class="device-id">${esc(target.id)}</code><div class="spec-badge">${svg(viz.qdmiBadge(0, 0, 350, "Powered by"), "0 0 350 95", "Powered by QDMI")}</div></div>`;
  }
  function routingFrame(target, progress) {
    const sweep = target.compilation.routing_trace?.find(
      (e) => e.phase === "final-routing",
    );
    const initial = target.compilation.layout?.initial || [];
    if (!sweep?.steps?.length)
      return {
        c: targetStage(target, "place-and-route")?.circuit,
        layout: initial,
        active: -1,
        sites: [],
        swaps: 0,
      };
    const steps = sweep.steps,
      weights = steps.map((e) => (e.operation === "qco.swap" ? 5 : 1)),
      total = weights.reduce((a, b) => a + b, 0);
    let consumed = 0,
      index = steps.length - 1,
      fraction = 1;
    for (let i = 0; i < steps.length; i++) {
      if (consumed + weights[i] > progress * total) {
        index = i;
        fraction = (progress * total - consumed) / weights[i];
        break;
      }
      consumed += weights[i];
    }
    let layout = sweep.before.slice(0, initial.length),
      before = [...layout],
      swaps = 0;
    steps.slice(0, index + 1).forEach((e, i) => {
      if (i === index) before = [...layout];
      if (e.operation === "qco.swap") {
        const [a, b] = e.sites;
        layout = layout.map((p) => (p === a ? b : p === b ? a : p));
        swaps++;
      }
    });
    const sites = [...new Set(steps.flatMap((e) => e.sites))].sort(
      (a, b) => a - b,
    );
    const c = {
      qubits: sites.map((id) => ({ id, site: id, label: `${id}` })),
      operations: steps.map((e, i) => ({
        id: `route-${i}`,
        name: e.controls
          ? "c".repeat(e.controls) + (e.base_gate || "").replace(/^qco\./, "")
          : (e.base_gate || e.operation).replace(/^qco\./, ""),
        qubits: e.sites,
        parameters: e.parameters || [],
        num_controls: e.controls || 0,
      })),
    };
    return {
      c,
      layout,
      active: progress >= 1 ? -1 : index,
      reveal: progress >= 1 ? Infinity : index + 1,
      sites: progress >= 1 ? [] : steps[index].sites,
      blend: { before, fraction: progress >= 1 ? 1 : fraction },
      swaps,
    };
  }
  function stageCard(target, id, title, options = {}) {
    return `<section class="compiled-stage" data-morph="stage-${id}"><h3>${title}</h3>${options.detail ?? circuit(options.c || targetStage(target, id)?.circuit, { full: true, key: id, ...options })}${programMetrics(target, id)}</section>`;
  }
  function compilationScene(step, progress = 1) {
    const targetIndex = step >= 9 ? 2 : step >= 7 ? 1 : 0,
      target = targets[targetIndex];
    const replay = step === 8 || step === 10;
    const stage = replay
      ? Math.min(6, 2 + Math.floor(progress * 5))
      : step <= 6
        ? step
        : 1;
    if (step === 12)
      return compilerFlow(
        5,
        `<div class="compiler-summary">${["circuit", "chip", "circuit", "chip", "circuit", "measurement"].map((n, i) => `<div data-morph="summary-${i}">${icon(n)}<h3>${compilerLabels[i]}</h3></div>`).join("")}</div><div class="abstract-compiler">${["pennylane", "qdmi", "mqt", "qir"].map((n) => logo(n)).join("<span>→</span>")}</div>`,
        { complete: true },
      );
    const localProgress = replay ? (progress * 5) % 1 : progress;
    const route =
      stage >= 3
        ? routingFrame(
            target,
            stage === 4 ? localProgress : stage === 3 ? 0 : 1,
          )
        : null;
    const fusion = target.compilation.synthesis_trace || [];
    const event =
      fusion[
        Math.min(
          fusion.length - 1,
          Math.floor(localProgress * Math.max(1, fusion.length)),
        )
      ];

    let body = "";
    if (stage >= 2) body += stageCard(target, "optimized", "3 · Optimization");
    if (stage >= 3)
      body += stageCard(
        target,
        "place-and-route",
        `4 · Placement & Routing · ${route.swaps} SWAP${route.swaps === 1 ? "" : "s"}`,
        {
          c: stage === 3 ? targetStage(target, "optimized")?.circuit : route.c,
          active: stage === 3 ? -1 : route.active,
          reveal: stage === 3 ? Infinity : route.reveal,
          layout: stage === 3 ? target.compilation.layout.initial : null,
        },
      );
    if (stage >= 5)
      body += stageCard(
        target,
        "target-native-synthesis",
        "5 · Native Synthesis",
        {
          detail:
            stage === 5 && event && localProgress < 0.85
              ? `<div class="fusion-detail" style="opacity:${Math.min(1, (0.85 - localProgress) / 0.1)}">${circuit(event.before.circuit, { full: true, active: event.input_indices, key: "fusion-before" })}<span>→</span>${circuit(event.after.circuit, { full: true, active: event.output_indices, key: "fusion-after" })}<small>Accepted Two-Qubit Fusion · Actual Before and After</small></div>`
              : null,
        },
      );
    if (stage >= 6)
      body = fullCode(
        targetStage(target, "qir-adaptive"),
        "6 · QIR 2.1 Adaptive Profile · LLVM IR",
        { columns: 2, width: 1200, height: 470 },
      );
    const input = circuit(targetStage(target, "source")?.circuit, {
      full: true,
      key: "compiler-input",
    });
    const activeSites =
      stage === 5 && localProgress < 1
        ? event
          ? [
              ...new Set(
                event.input_indices
                  .flatMap((i) => flatten(event.before.circuit)[i].qubits)
                  .map(
                    (q) =>
                      event.before.circuit.qubits.find((w) => w.id === q).site,
                  ),
              ),
            ]
          : []
        : route?.sites || [];
    return compilerFlow(
      stage <= 1
        ? stage
        : stage === 2
          ? 2
          : stage <= 4
            ? 3
            : stage === 5
              ? 4
              : 5,
      `<div class="compiler-workbench"><div class="compiler-input" data-morph="compiler-input"><h3>1 · Input Program</h3>${input}</div><div class="compiler-device ${step === 7 || step === 9 ? "rotate-card" : ""}">${stage >= 1 || step === 11 ? deviceCard(target, { layout: route?.layout, activeSites, blend: route?.blend, next: step === 11 }) : ""}</div><div class="compiler-stages count-${Math.min(3, Math.max(0, stage - 1))}">${body}</div></div>${stage === 0 ? programMetrics(target, "source") : ""}`,
    );
  }
  function searchScene(progress) {
    const sweep = progress * 4,
      turn = Math.floor(sweep),
      fraction = sweep - turn;
    const forward = turn % 2 === 0,
      position = forward ? fraction : 1 - fraction;
    let body = "";
    for (let row = 0; row < 4; row++) {
      const y = 100 + row * 128,
        x = 180 + position * 1020;
      body += `<g>${text(60, y + 8, `Trial ${row + 1}`, 25, "#526d88")}${line(145, y, 1310, y, "#bed1e3", 3)}${Array.from({ length: 13 }, (_, i) => `<rect x="${173 + i * 84}" y="${y - 25}" width="37" height="50" rx="5" fill="#eaf2fa" stroke="#7ea7cf"/>`).join("")}<circle cx="${x}" cy="${y}" r="35" fill="#2f70b8" opacity=".2"/><circle cx="${x}" cy="${y}" r="8" fill="#2f70b8"/>${text(1450, y + 8, forward ? "Forward →" : "← Backward", 24, "#087d92")}</g>`;
      for (let k = 0; k < 7; k++) {
        const xx = x + Math.cos(k * 2.4) * 44,
          yy = y + Math.sin(k * 2.4) * 44;
        body +=
          line(x, y, xx, yy, "#4aa9b9", 1) +
          `<circle cx="${xx}" cy="${yy}" r="4" fill="#087d92"/>`;
      }
    }
    return `<div class="search-caricature">${svg(body, "0 0 1600 610", "Illustration of parallel placement trials and A-star routing searches")}<div class="technical-caption"><b>Parallel Placement Trials</b><span>Bidirectional Traversals</span><span>A* Searches SWAP Sequences</span></div><p class="illustration-note">Algorithm Illustration · Not a Timing Trace</p></div>`;
  }
  function synthesisSketch(progress) {
    const phase = Math.min(3, Math.floor(progress * 4));
    let body = Array.from({ length: 2 }, (_, i) =>
      line(70, 180 + i * 110, 585, 180 + i * 110, "#adc4da", 3),
    ).join("");
    [160, 260, 390, 495].forEach((x, i) => {
      body += `<rect x="${x - 30}" y="${i % 2 ? 149 : 259}" width="60" height="62" rx="6" fill="#eaf2fa" stroke="#2f70b8"/>${text(x, i % 2 ? 190 : 300, i % 2 ? "R" : "H", 28)}`;
    });
    body += `<rect x="123" y="120" width="420" height="236" rx="18" fill="none" stroke="#087d92" stroke-width="${phase >= 1 ? 5 : 1}" stroke-dasharray="8 6"/>${text(340, 416, "Two-Qubit Block", 32)}`;
    if (phase >= 1) {
      body += line(585, 235, 680, 235) + text(816, 95, "Unitary Matrix", 30);
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
          body += `<rect x="${713 + c * 53}" y="${139 + r * 53}" width="46" height="46" rx="5" fill="${(r + c) % 3 === 0 ? "#2f70b8" : "#cfe3f4"}" opacity="${0.5 + 0.5 * progress}"/>`;
      body += text(816, 418, "U ∈ SU(4)", 32);
    }
    if (phase >= 2) {
      body += line(957, 235, 1030, 235);
      [0, 1].forEach((r) => {
        body += line(1040, 180 + r * 110, 1500, 180 + r * 110, "#adc4da", 3);
        [1090, 1260, 1440].forEach((x) => {
          body += `<rect x="${x - 27}" y="${155 + r * 110}" width="54" height="50" rx="5" fill="#eaf2fa" stroke="#2f70b8"/>${text(x, 188 + r * 110, "R", 28)}`;
        });
      });
      [1180, 1350].forEach(
        (x) =>
          (body +=
            line(x, 180, x, 290) +
            `<circle cx="${x}" cy="180" r="8" fill="#2f70b8"/><circle cx="${x}" cy="290" r="8" fill="#2f70b8"/>`),
      );
      body += text(1270, 418, "Target-Native Gates", 32);
    }
    return `<div class="synthesis-caricature">${svg(body, "0 0 1600 560", "Illustration of two-qubit peephole fusion and native synthesis")}<div class="technical-caption"><b>Fuse</b><span>Decompose</span><span>Synthesize</span></div><p class="illustration-note">Algorithm Illustration · R / CZ Example</p></div>`;
  }
  function displayedIR(a) {
    return a?.display_code
      ? { ...a, code: a.display_code, lines_html: a.display_lines_html }
      : a;
  }
  function underHood(step, progress = 1) {
    if (step === 0)
      return `<div class="ir-columns">${["source", "qc", "qco"].map((id, i) => `<section data-morph="ir-${id}"><h3>${["OpenQASM 3", "QC · Reference Semantics", "QCO · Value Semantics"][i]}</h3>${fullCode(displayedIR(appStage(id)), i ? `${["", "QC", "QCO"][i]} · Complete Program · Cleanup / CSE` : "Complete Input Program", { height: 570, width: 550, columns: 1, lens: i === 1 })}</section>`).join("")}<span class="ir-arrow first">Translation →</span><span class="ir-arrow second">Conversion →</span></div>`;
    if (step === 1) {
      const o = evidence.optimization || {};
      return `<div class="before-after">${["before", "after"].map((id, i) => `<section>${fullCode(o[id], i ? "After Canonicalization" : "Before Canonicalization", { height: 435, width: 825 })}${circuit(o[id]?.circuit, { full: true, key: "canonical-" + id })}</section>`).join("")}</div>`;
    }
    return step === 2 ? searchScene(progress) : synthesisSketch(progress);
  }
  function measuredGrowth(progress = 1) {
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
      const fraction =
        key === "unrolled_bytes"
          ? Math.min(1, progress * 2)
          : Math.max(0, progress * 2 - 1);
      const shown = points.slice(0, Math.ceil(fraction * points.length));
      body += `<polyline points="${shown.map((p) => `${x(p.qubits)},${y(p[key])}`).join(" ")}" fill="none" stroke="${color}" stroke-width="5"/>`;
      body += shown
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
  function paperFooter() {
    return `<div class="structure-footer"><div class="exchange-footer">${logo("mqt")}<span>⇄</span>${logo("jeff")}<span>⇄</span><div>Tools &amp; Frameworks<small>Structured Programs Across the Ecosystem</small></div></div>${qr("unrolling", "arxiv.org/abs/2609.16171", "Rovara, Haag, et al., 2026 · Why Are We Unrolling?")}</div>`;
  }
  function structureScene(step) {
    const rus = step >= 5,
      v = variant(rus ? "rus" : "parity"),
      source = v.stages.find((a) => a.id === "source"),
      qir = v.exports.find((a) => a.id === "qir-adaptive");
    const effect = rus ? null : [null, null, "reset", "branch", "loop"][step];
    const active = flatten(source.circuit).flatMap((o, i) =>
      (effect === "reset" && o.name === "reset") ||
      (effect === "branch" &&
        (o.name === "measure" || o.region.some((r) => r.name === "if_else"))) ||
      (effect === "loop" && o.region.some((r) => r.name === "for_loop"))
        ? [i]
        : [],
    );
    return `<div class="structured-program"><div class="structured-source" data-morph="structured-source">${fullCode(source, rus ? "Repeat Until Success" : "Syndrome Extraction", { height: 330, width: 570, hot: effect })}${circuit(source.circuit, { full: true, key: rus ? "rus" : "syndrome", active, hotRegion: effect === "loop" ? "for_loop" : effect === "branch" ? "if_else" : null })}${rus ? `<p class="rus-purpose">V₃ = (I + 2iZ) / √5 on |+⟩<br><span><a href="https://www.rintonpress.com/xxqic14/qic-14-1516/1277-1301.pdf">Paetznick &amp; Svore, 2014 · QIC 14</a></span></p>` : ""}</div><div class="structured-qir ${step === 0 || step === 5 ? "future" : ""}" data-morph="structured-qir">${fullCode(qir, "QIR 2.1 Adaptive Profile", { columns: 2, height: 530, width: 1130, hot: effect })}${effect ? `<div class="control-lens">${code(qir, { start: focus(qir, effect === "reset" ? "__reset__" : effect === "branch" ? "__read_result" : "phi ", 2), count: 6, title: "Same Structure · Native QIR", compact: true })}</div>` : ""}</div></div>${paperFooter()}`;
  }
  function growthScene(progress = 1) {
    const sketch = `${line(60, 104, 590, 104, "#a9c2d8", 3)}<rect x="87" y="64" width="83" height="81" rx="10" fill="#eaf2fa" stroke="#2f70b8"/>${text(129, 115, "H⊗n", 27)}<rect x="213" y="27" width="290" height="158" rx="18" fill="#eaf2fa" stroke="#2f70b8" stroke-width="3"/><rect x="241" y="63" width="103" height="83" rx="8" fill="white" stroke="#2f70b8"/><rect x="367" y="63" width="110" height="83" rx="8" fill="white" stroke="#2f70b8"/>${text(292, 112, "Oracle", 24)}${text(422, 112, "Diffusion", 24)}<path d="M476 208H240m10 -8l-10 8 10 8" fill="none" stroke="#2f70b8" stroke-width="3"/>${text(355, 247, "Repeat ⌊π√N / 4⌋ Times", 28)}${text(320, 307, "Grover’s Algorithm", 32)}`;
    return `<div class="grover-growth"><div>${svg(sketch, "0 0 630 340", "Illustrative Grover circuit loop")}</div>${measuredGrowth(progress)}</div>${paperFooter()}`;
  }
  function executionLog(capture, elapsed) {
    const events = [
      ...(capture.events || []),
      ...(capture.runtime_events || []),
    ].sort((a, b) => a.time_ms - b.time_ms);
    const rows = [];
    for (const event of events.filter(
      (e) => !Number.isInteger(e.program_index),
    )) {
      const previous = rows.at(-1);
      if (previous?.operation === event.operation) {
        previous.times.push(event.time_ms);
      } else rows.push({ ...event, times: [event.time_ms] });
    }
    return `<div class="execution-log" data-recorded-calls="${events.length}"><h3>Execution Log</h3>${rows.map((e, i) => `<div data-morph="log-${i}" class="${elapsed < e.time_ms ? "future" : ""}"><time>${fmt(Math.max(0, e.time_ms) / 1000, 3)} s</time><code>${esc(e.operation.replace(/^MQT_DDSIM_QDMI_/u, "").replace(/\(.*$/u, "()"))}</code>${e.times?.length > 1 ? `<b>×${e.times.filter((time) => time <= elapsed).length}</b>` : ""}</div>`).join("")}</div>`;
  }
  function adaptiveRuntime(step, progress = 1) {
    const elapsed =
        step === 0 ? qdmiStart(run) : interpolateTime(run, progress),
      source = variant("qpe").stages.find((a) => a.id === "source");
    return `<div class="qpe-banner"><div><h3>Iterative QPE · φ = 1/3 · 8 Bits · 4 Qubits</h3>${circuit(source.circuit, { key: "qpe", limit: 23 })}</div><div class="runtime-products">${logo("qdmi")}<span>→</span>${logo("mqt")}<b>DDSIM</b><span>→</span>${logo("llvm")}<b>JIT</b></div></div><div class="runtime-columns">${executionLog(run, elapsed)}<div><h3>QDMI → Runtime</h3>${runtimeSequence(run, elapsed)}</div><div><h3>Measurement Outcomes</h3>${histogram(run, elapsed)}</div></div><div class="runtime-clock"><span>${fmt((elapsed - qdmiStart(run)) / 1000, 3)} s Recorded Time</span><span>Adaptive QIR 2.1 · Replay Slowed for Inspection</span><div class="clock-line"><div style="width:${(100 * (elapsed - qdmiStart(run))) / (totalTime(run) - qdmiStart(run))}%"></div></div></div>`;
  }
  function featureOverview(step) {
    const groups = [
      [
        "circuit",
        "OpenQASM 3.1",
        "Modifiers · Reusable Gates · Classical Expressions",
        "if / switch / for / while · Loop-Indexed Qubits",
      ],
      [
        "measurement",
        "QIR 2.1",
        "Base + Adaptive Profiles",
        "Compile and Execute · Arrays",
      ],
      [
        "qiskit",
        "Qiskit 2.5",
        "Native C API Integration",
        "SamplerV2 · EstimatorV2 · QDMI Batches",
      ],
      [
        "pennylane",
        "PennyLane",
        "Hybrid Workflows Through QDMI",
        "Batch Programs · Samples · Expectation Values",
      ],
      [
        "chip",
        "Ready to Install",
        "Windows · macOS · Linux Wheels",
        "Python 3.15 · Portable LLVM · uv / Ruff / ty",
      ],
      [
        "developer",
        "Ready to Build On",
        "Tutorials · Guides · API Reference",
        "Contribution Guidelines · Agent Instructions",
      ],
    ];
    return `<div class="feature-core" data-morph="core-overview">${logo("mqt")}<b>Core</b><span>Quantum Programs → Compilation → Execution</span></div><div class="feature-grid">${groups.map(([art, title, first, second], i) => `<section data-morph="feature-${i}" class="feature-card ${i > step ? "future" : ""}">${["qiskit", "pennylane"].includes(art) ? logo(art) : icon(art)}<div><h3>${title}</h3><p>${first}</p><p>${second}</p></div></section>`).join("")}</div><div class="feature-resources">${qr("core-repo", "github.com/munich-quantum-toolkit/core", "Open Source · MQT and QDMI Communities")}${qr("core", "mqt.readthedocs.io/projects/core", "Tutorials · API · Guides")}</div>`;
  }
  function separator(title, subtitle, brand) {
    return `<div class="chapter-separator" data-morph="chapter"><div class="chapter-art">${logo(brand)}</div><p>${subtitle}</p><h1>${title}</h1><div class="chapter-rule"></div></div>`;
  }
  const slides = [];
  function add(id, section, title, builds, render, notes = "", playbacks = {}) {
    slides.push({ id, section, title, builds, render, notes, playbacks });
  }
  const opening = window.MQSF_OPENING;
  add(
    "title",
    "Munich Quantum Software Forum · October 2026",
    "System Software for Quantum Computing",
    0,
    opening.title,
    "Lukas Burgholzer, CTO and co-founder of MQSC; Senior Researcher at TUM CDA.",
  );
  for (const [mode, title] of [
    ["problem", "Connecting the Quantum Ecosystem"],
    ["stack", "A Shared Software Stack"],
    ["detail", "From the Metal to the User"],
  ]) {
    add(
      mode,
      "01 · System Software",
      title,
      opening.phases[mode].length - 1,
      (s, t) => opening.render(mode, s, t ?? 1),
      "A shared modular stack connects users, classical resources and quantum processors. QDMI and MQT are community projects; MQSC drives their use in practice.",
      opening.playbacks[mode],
    );
  }
  add(
    "hybrid-algorithm",
    "02 · A Practical Hybrid Application",
    appHeading,
    2,
    (s, t) => heading(appHeading) + applicationScene(s, t ?? 1),
    "H2/STO-3G is a bounded software demonstration adapted from the public Amazon Braket example. Quantum shadows supply trial overlaps; Slurm workers reuse them. This is not a quantum-advantage claim.",
    {
      0: { duration: 2200, transition: 900 },
      1: { duration: 3000, transition: 700 },
      2: { duration: 3500, transition: 700 },
    },
  );
  add(
    "afqmc-execution",
    "02 · A Practical Hybrid Application",
    appHeading,
    4,
    (s, t) => heading(appHeading) + hybridExecution(s, t ?? 1),
    "One QDMI job carries 4096 actual QIR programs. Compilation, native per-program completion and indexed retrieval timestamps are distinct. Four Slurm compute nodes propagate 128 independent walkers per trial. Bands show walker standard errors, not total statistical or systematic uncertainty.",
    {
      1: { duration: 2200, transition: 700 },
      2: { duration: 20000, transition: 850 },
      3: { duration: 8500, transition: 850 },
    },
  );
  add(
    "compile-chapter",
    "03 · Compilation",
    "From Quantum Program to Executable",
    0,
    () =>
      separator(
        "From Quantum Program<br>to Executable",
        "Targeting a Quantum Device",
        "mqt",
      ),
  );
  add(
    "routing",
    "03 · Compiling for a Device",
    "From Quantum Program to Executable",
    12,
    (s, t) =>
      heading("From Quantum Program to Executable") +
      compilationScene(s, t ?? 1),
    "Same AFQMC payload throughout. Optimization removes inverse gates. Placement assigns physical sites; routing inserts SWAPs only where connectivity requires them. Native synthesis follows. Provider metadata is queried then modeled locally; all verification execution uses DDSIM, no hardware jobs. Fusion highlights identify accepted real before/after events, not every individual lowering.",
    Object.fromEntries(
      [4, 5, 8, 10].map((s) => [
        s,
        { duration: s >= 8 ? 9500 : 6500, transition: 850 },
      ]),
    ),
  );
  add(
    "compiler-chapter",
    "03 · Compiler Infrastructure",
    "Inside the MQT Compiler Collection",
    0,
    () =>
      separator(
        "Inside the<br>MQT Compiler Collection",
        "Programs, Representations and Transformations",
        "mlir",
      ),
  );
  add(
    "compiler-stages",
    "03 · Inside the Compiler",
    "Inside the MQT Compiler Collection",
    3,
    (s, t) =>
      heading("Inside the MQT Compiler Collection") + underHood(s, t ?? 1),
    "Code is actual compiler output. Compact IR views use recorded cleanup passes and retain full original programs in the evidence download. Placement and synthesis caricatures illustrate algorithms; they are not measured timing traces.",
    {
      2: { duration: 5000, transition: 700, loop: true },
      3: { duration: 5000, transition: 700, loop: true },
    },
  );
  add(
    "structure-chapter",
    "04 · Structured Quantum Programs",
    "Why Are We Unrolling?",
    0,
    () =>
      separator(
        "Why Are We Unrolling?",
        "Preserving Structure Through Compilation",
        "jeff",
      ),
  );
  add(
    "grover-growth",
    "04 · Structured Quantum Programs",
    "Why Are We Unrolling?",
    0,
    (s, t) => heading("Why Are We Unrolling?") + growthScene(t ?? 1),
    "Measured Grover program sizes: first show unrolled output growth, then structure-preserving output. The circuit sketch explains the loop; it is intentionally illustrative.",
    { 0: { duration: 6500, transition: 700 } },
  );
  add(
    "structured",
    "04 · Structured Quantum Programs",
    "Structured Programs, All the Way Down",
    6,
    (s) => heading("Structured Programs, All the Way Down") + structureScene(s),
    "Syndrome extraction keeps local bit declarations, reset, feedback and loop structure in native QIR. RUS implements V3=(I+2iZ)/sqrt5 on |+>, with success 5/8 per attempt and a required Z correction on failure. Full entry points, declarations and module flags remain in the displayed QIR sheet and downloadable evidence.",
  );
  add(
    "execution-chapter",
    "05 · Runtime",
    "From Compilation to Execution",
    0,
    () =>
      separator(
        "From Compilation<br>to Execution",
        "A Runtime for Structured Quantum Programs",
        "qdmi",
      ),
  );
  add(
    "adaptive-execution",
    "05 · Adaptive QIR Execution",
    "A Runtime for Structured Quantum Programs",
    1,
    (s, t) =>
      heading("A Runtime for Structured Quantum Programs") +
      adaptiveRuntime(s, t ?? 1),
    "Iterative QPE estimates phase 1/3 with 8 output bits and 4 logical qubits. Call, JIT and shot timestamps share one clock. This is instrumented serial adaptive sampling; the plot is accumulated from actual ordered shots.",
    { 1: { duration: 9500, transition: 700 } },
  );
  add(
    "features",
    "06 · MQT Core",
    "Built for the Quantum Software Ecosystem",
    5,
    (s) =>
      heading("Built for the Quantum Software Ecosystem") + featureOverview(s),
    "Highlight verified capabilities. OpenQASM support is broad but excludes subroutines, extern/calibration and general arrays. Qiskit2.5 integrates through the C API with Python for unsupported API areas. All exports and execution traces come from the captured build.",
  );
  add(
    "ecosystem",
    "From the Metal to the User",
    "System Software for Quantum Computing",
    0,
    opening.conclusion,
    "Built with the MQT and QDMI communities — driven into practice by MQSC.",
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
      `${String(state.slide + 1)} / ${slides.length}<span class="build-dots" aria-label="Build ${state.step + 1} of ${slide.builds + 1}">${Array.from({ length: slide.builds + 1 }, (_, i) => (i === state.step ? "<b>•</b>" : "•")).join("")}</span>`;
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
