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
window.MQSF_CIRCUIT = (() => {
  const esc = (s) =>
    String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll('"', "&quot;");
  function flatten(c, region = []) {
    return (c?.operations || []).flatMap((o) =>
      o.blocks
        ? o.blocks.flatMap((block, branch) =>
            flatten({ operations: block }, [
              ...region,
              { ...o, blocks: undefined, branch },
            ]),
          )
        : [{ ...o, region }],
    );
  }
  function schedule(c) {
    const free = new Map(c.qubits.map((q) => [q.id, 0]));
    const rows = new Map(c.qubits.map((q, i) => [q.id, i]));
    const bits = new Map();
    let boundary = "",
      floor = 0;
    return flatten(c).map((o, index) => {
      const nextBoundary = o.region.map((r) => `${r.id}:${r.branch}`).join("/");
      // Never move an operation across a loop or branch boundary.
      if (nextBoundary !== boundary) floor = Math.max(...free.values(), floor);
      boundary = nextBoundary;
      const used = (
        o.qubits.length ? o.qubits : c.qubits.map((q) => q.id)
      ).filter((q) => rows.has(q));
      const lo = Math.min(...used.map((q) => rows.get(q))),
        hi = Math.max(...used.map((q) => rows.get(q)));
      const span = c.qubits
        .filter((_, row) => row >= lo && row <= hi)
        .map((q) => q.id);
      const dependencies = o.region.flatMap((r) => r.condition_bits || []);
      const column = Math.max(
        floor,
        ...span.map((q) => free.get(q)),
        ...dependencies.map((b) => bits.get(b) || 0),
      );
      span.forEach((q) => free.set(q, column + (o.name === "store" ? 2 : 1)));
      (o.clbits || []).forEach((b) => bits.set(b, column + 1));
      return { ...o, index, column };
    });
  }
  function gateLabel(name) {
    const rotation = /^r([xyz]{1,2})$/.exec(name);
    if (rotation)
      return `R<tspan baseline-shift="sub" font-size="70%">${rotation[1]}</tspan>`;
    return esc(
      { reset: "|0⟩", prx: "R", gpi: "GPI", gpi2: "GPI₂" }[name] ||
        name.toUpperCase(),
    );
  }
  function render(
    c,
    {
      active = -1,
      start = 0,
      limit = 16,
      label = "",
      key = "circuit",
      layout = null,
      camera = null,
      full = false,
      reveal = Infinity,
      barrier = null,
      hotRegion = null,
    } = {},
  ) {
    if (!c)
      return '<p class="note">No circuit representation for this compiler stage.</p>';
    const all = schedule(c),
      maximum = Math.max(0, ...all.map((o) => o.column));
    const highlighted = new Set(Array.isArray(active) ? active : [active]);
    const anchor =
      camera ??
      all[Array.isArray(active) ? active[0] : active >= 0 ? active : start]
        ?.column ??
      0;
    const firstColumn = full
      ? 0
      : Math.max(
          0,
          Math.min(
            anchor - (camera !== null || active >= 0 ? 5 : 0),
            Math.max(0, maximum - limit + 1),
          ),
        );
    const windowColumn = Math.floor(firstColumn);
    const operations = all.filter(
      (o) =>
        o.column >= windowColumn &&
        o.column < windowColumn + (full ? maximum + 1 : limit),
    );
    const columns = full
      ? maximum + 1
      : Math.min(limit, maximum - windowColumn + 1);
    const structured = all.some((o) => o.region.length),
      nested = Math.max(0, ...all.map((o) => o.region.length));
    // Gate bodies never exceed their column; a full circuit expands its SVG rather than overlapping.
    const pitch = full ? 76 : Math.max(76, Math.min(102, 900 / columns));
    const left = layout ? 250 : 130,
      first = left + 40;
    const width = Math.max(1100, first + (columns - 1) * pitch + 55),
      top = structured ? 100 + nested * 20 : 55,
      dy = 72;
    const height = Math.max(
      280,
      top + (c.qubits.length - 1) * dy + (structured ? 120 : 65),
    );
    const row = new Map(c.qubits.map((q, i) => [q.id, i])),
      y = (q) => top + row.get(q) * dy;
    const x = (o) => first + (o.column - windowColumn) * pitch;
    const txt = (xx, yy, t, size = 26, color = "#142b45", anchor = "middle") =>
      `<text x="${xx}" y="${yy}" font-size="${size}" fill="${color}" text-anchor="${anchor}">${esc(t)}</text>`;
    const ln = (x1, y1, x2, y2, color = "#2f70b8", w = 2) =>
      `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"/>`;
    const regions = new Map();
    operations.forEach((o) =>
      o.region.forEach((r, depth) => {
        const entry = regions.get(r.id) || {
          first: x(o),
          last: x(o),
          r,
          depth,
          qubits: new Set(),
        };
        entry.last = x(o) + (o.name === "store" ? 60 : 0);
        (o.qubits.length ? o.qubits : r.qubits || []).forEach((q) =>
          entry.qubits.add(q),
        );
        regions.set(r.id, entry);
      }),
    );
    let body = "";
    regions.forEach(({ first: a, last: b, r, depth, qubits }) => {
      const loop = ["for_loop", "while_loop"].includes(r.name);
      if (!loop && r.name !== "if_else") return;
      const ys = [...qubits].filter((q) => row.has(q)).map(y);
      if (!ys.length) return;
      const yy = loop ? top - 66 - depth * 18 : Math.min(...ys) - 30;
      const bottom = loop
        ? top + (c.qubits.length - 1) * dy + 36
        : Math.max(...ys) + 30;
      const pad = loop ? 32 : 28,
        selected = hotRegion === r.name;
      // shortcut: condition labels cover these captured bit guards; general expressions need a formatter.
      const caption =
        r.name === "for_loop"
          ? `${r.iterations ?? "n"} rounds`
          : r.name === "while_loop"
            ? "Repeat Until Success"
            : `if ${r.condition?.includes("BIT_XOR") ? "¬" : ""}${(r.condition_bits || []).map((bit) => `b${bit}`).join(", ")}`;
      body += `<g data-morph="${key}-region-${r.id}" data-control-flow="${r.name}"><rect x="${a - pad}" y="${yy}" width="${b - a + 2 * pad}" height="${bottom - yy}" rx="10" fill="${loop ? "#eaf2fa" : "#e9f5f3"}" fill-opacity="${selected ? ".9" : ".45"}" stroke="${selected ? "#086f83" : loop ? "#87acd0" : "#55a194"}" stroke-width="${selected ? 4 : 2}" ${loop ? "" : 'stroke-dasharray="5 4"'}/>${txt(a - pad + 12, loop ? yy - 10 : yy - 9, caption, 23, loop ? "#2f70b8" : "#087d92", "start")}${loop ? `<path d="M${b - 8} ${bottom + 11}H${a + 8}m10 -8l-10 8 10 8" fill="none" stroke="#6992bc" stroke-width="2"/>` : ""}</g>`;
    });
    body += c.qubits
      .map((q) => {
        const logical = row.get(q.id),
          physical = layout?.[logical] ?? q.site ?? q.id;
        const wireLabel = layout
          ? `<text x="${left - 20}" y="${y(q.id) + 8}" font-size="24" fill="#526d88" text-anchor="end" font-family="STIX Two Math,serif">q<tspan baseline-shift="sub" font-size="75%">${logical}</tspan><tspan baseline-shift="baseline"> ↦ ${physical}</tspan></text>`
          : txt(
              left - 18,
              y(q.id) + 8,
              String(q.label).replace(/^\$(.*)\$$/, "$1"),
              24,
              "#526d88",
              "end",
            );
        return `<g data-morph="${key}-wire-${q.id}">${ln(left, y(q.id), width - 28, y(q.id), "#b6c9dc", 2)}${wireLabel}</g>`;
      })
      .join("");
    if (barrier !== null) {
      const next = all[barrier];
      if (next) {
        const bx = x(next) - pitch / 2;
        body += `<path d="M${bx} ${top - 25}V${top + (c.qubits.length - 1) * dy + 25}" stroke="#718ba6" stroke-width="2" stroke-dasharray="5 5"/>${txt((left + bx) / 2, top - 32, "Trial State", 23, "#58718b")}${txt((bx + width) / 2, top - 32, "Classical Shadow", 23, "#58718b")}`;
      }
    }
    const measured = new Map(),
      feedback = new Set();
    operations.forEach((o) => {
      const xx = x(o),
        ys = o.qubits.filter((q) => row.has(q)).map(y),
        hot = highlighted.has(o.index);
      if (!ys.length) {
        if (o.name === "store" && o.region.some((r) => r.branch === 1)) {
          const bit = c.clbits?.find((b) => b.id === o.classical_writes?.[0]);
          const value = /^Value\(([01]), Bool\(\)\)$/.exec(
            o.parameters?.[1] || "",
          )?.[1];
          const label =
            value === undefined
              ? "Classical Store"
              : `else: ${bit?.label || "bit"} ← ${value}`;
          body += `<g data-morph="${key}-op-${o.id}" data-operation="store" data-column="${o.column}" transform="translate(${xx} 0)"><rect x="-35" y="${top + (c.qubits.length - 1) * dy - 15}" width="125" height="40" rx="6" fill="#e8f5f0" stroke="#55a194"/>${txt(37, top + (c.qubits.length - 1) * dy + 10, label, 18, "#087d92")}</g>`;
        }
        return;
      }
      const color = hot ? "#0756a8" : "#2f70b8",
        fill = hot ? "#cee6fc" : "white";
      const dot = (yy, i = 0) =>
        `<circle cx="0" cy="${yy}" r="7" fill="${o.control_state === undefined || o.control_state & (1 << i) ? color : "white"}" stroke="${color}" stroke-width="2"/>`;
      const box = (yy, name) =>
        `<rect x="-27" y="${yy - 25}" width="54" height="50" rx="5" fill="${fill}" stroke="${color}" stroke-width="${hot ? 4 : 2.3}"/><text x="0" y="${yy + 9}" text-anchor="middle" font-size="${name.length > 5 ? 19 : 27}" fill="${color}" font-family="STIX Two Math,serif">${gateLabel(name)}</text>`;
      const controlCount =
        o.controls ??
        o.num_controls ??
        (o.name.startsWith("cc")
          ? 2
          : o.name.startsWith("c") && o.name !== "ctrl"
            ? 1
            : 0);
      let gate = "";
      if (
        (controlCount > 0 && o.base_gate) ||
        [
          "cx",
          "cz",
          "ccx",
          "ccz",
          "cp",
          "crx",
          "cry",
          "crz",
          "ch",
          "csx",
          "cy",
          "cu",
        ].includes(o.name)
      ) {
        gate += ln(0, Math.min(...ys), 0, Math.max(...ys), color, 2.5);
        const controls = ys.slice(0, controlCount || 1),
          target = ys.at(-1);
        gate += controls.map((yy, i) => dot(yy, i)).join("");
        const base = o.base_gate || o.name.replace(/^c+/, "");
        if (base === "x")
          gate += `<circle cx="0" cy="${target}" r="21" fill="${fill}" stroke="${color}" stroke-width="2.5"/>${ln(-13, target, 13, target, color)}${ln(0, target - 13, 0, target + 13, color)}`;
        else if (base === "z" || base === "p")
          gate +=
            dot(target) +
            (base === "p"
              ? txt(17, (ys[0] + target) / 2 + 7, "P", 23, color, "start")
              : "");
        else gate += box(target, base);
      } else if (o.name === "swap") {
        gate +=
          ln(0, Math.min(...ys), 0, Math.max(...ys), color, 2.5) +
          ys
            .map(
              (yy) =>
                ln(-11, yy - 11, 11, yy + 11, color, 3) +
                ln(-11, yy + 11, 11, yy - 11, color, 3),
            )
            .join("");
      } else if (ys.length > 1) {
        const lo = Math.min(...ys),
          hi = Math.max(...ys);
        gate += `<rect x="-28" y="${lo - 25}" width="56" height="${hi - lo + 50}" rx="6" fill="${fill}" stroke="${color}" stroke-width="${hot ? 4 : 2.3}"/><text x="0" y="${(lo + hi) / 2 + 9}" text-anchor="middle" font-size="27" fill="${color}" font-family="STIX Two Math,serif">${gateLabel(o.name)}</text>`;
      } else if (o.name === "measure") {
        gate += `<rect x="-27" y="${ys[0] - 25}" width="54" height="50" rx="5" fill="${fill}" stroke="${color}" stroke-width="2.3"/><path d="M-18 ${ys[0] + 11}A18 18 0 0 1 18 ${ys[0] + 11}M0 ${ys[0] + 11}L13 ${ys[0] - 13}" fill="none" stroke="${color}" stroke-width="2.8"/>`;
      } else gate += box(ys[0], o.name);
      if (o.name === "measure")
        (o.clbits || []).forEach((b) => measured.set(b, { x: xx, y: ys[0] }));
      o.region
        .filter((r) => r.name === "if_else")
        .forEach((r, depth) => {
          (r.condition_bits || []).forEach((bit) => {
            const m = measured.get(bit),
              bounds = regions.get(r.id);
            if (!m || !bounds) return;
            const token = `${r.id}-${bit}`,
              bend = top + c.qubits.length * dy + depth * 14;
            const first = !feedback.has(token);
            feedback.add(token);
            body += `<path data-morph="${key}-classical-${o.id}-${bit}" data-condition-bit="${bit}" data-feedback-to="${o.id}" d="M${m.x} ${m.y + 25}V${bend}H${bounds.first - 28}V${Math.min(...ys)}" fill="none" stroke="#087d92" stroke-width="2.5" opacity="${first ? 1 : 0}"/>`;
          });
        });
      body += `<g data-morph="${key}-op-${o.id}" data-column="${o.column}" data-operation="${esc(o.name)}" transform="translate(${xx} 0)" class="${hot ? "gate-highlight" : ""}" opacity="${o.index < reveal ? 1 : 0}">${gate}<title>${esc(o.name)} ${esc((o.parameters || []).join(", "))}</title></g>`;
    });
    if (!full && firstColumn + columns <= maximum)
      body += txt(width - 15, top + 8, "…", 30);
    if (label) body += txt(width / 2, height - 5, label, 23, "#5b7088");
    return `<div class="circuit" data-op-count="${all.length}" data-first-column="${firstColumn}" data-left="${first}" data-right="${width - 28}"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Captured program circuit">${body}</svg></div>`;
  }
  return { flatten, schedule, render };
})();
