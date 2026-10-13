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

window.MQSF_OPENING = (() => {
  const blue = "#2F70B8",
    navy = "#142B45",
    orange = "#C66B18",
    muted = "#5B7088";
  const assets = window.MQSF_ASSETS;
  const { icon, qdmiBadge } = window.MQSF_VIZ;
  const esc = (value) =>
    String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll('"', "&quot;");
  const text = (x, y, value, size = 28, anchor = "middle", color = navy) =>
    `<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" fill="${color}">${esc(value)}</text>`;
  const logo = (name, x, y, w, h) =>
    `<image href="${assets[name]}" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet"/>`;
  const image = (name, cls = "", alt = name) =>
    `<img class="${cls}" src="${assets[name]}" alt="${esc(alt)}"/>`;
  const group = (
    key,
    body,
    visible = 1,
    transform = "translate(0 0) scale(1)",
  ) =>
    `<g data-morph="${key}" opacity="${visible}" transform="${transform}">${body}</g>`;
  const path = (d, color = blue, width = 3, extra = "") =>
    `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  const clamp = (v) => Math.max(0, Math.min(1, v));
  const phases = {
    problem: [
      "Quantum Processors",
      "Researchers, Workstation and Cloud Access",
      "A Second System",
      "Every Modality",
      "Cloud Compute",
      "HPC and Cloud Bursting",
      "Every User",
    ],
    stack: [
      "A Shared Stack",
      "Qiskit",
      "Program and Runtime",
      "One Interface, Many Devices",
      "PennyLane, OpenQASM and Your SDK",
    ],
    detail: [
      "Our Approach: From the Metal to the User",
      "QDMI and Its Devices",
      "Compiler Collection",
      "Orchestrator",
      "Our Principles",
    ],
  };
  const playbacks = {
    problem: {
      1: { duration: 2800 },
      2: { duration: 900 },
      3: { duration: 1800 },
      4: { duration: 3600 },
      5: { duration: 4800 },
      6: { duration: 3500 },
    },
    stack: {
      0: { duration: 2100, transition: 1800 },
      1: { duration: 1500 },
      2: { duration: 1100 },
      3: { duration: 2600 },
      4: { duration: 2100 },
    },
    detail: {
      1: { duration: 4400 },
      2: { duration: 2200 },
      3: { duration: 2200 },
      4: { duration: 1600 },
    },
  };
  const devices = [
    {
      id: "iqm",
      label: "QDMI on IQM",
      description: ["On-premise and Cloud Devices"],
      short: "iqm-finland/QDMI-on-IQM",
      qr: "qr-iqm",
      url: "https://github.com/iqm-finland/QDMI-on-IQM",
    },
    {
      id: "aws",
      label: "Amazon Braket",
      description: ["Cloud Service Integration"],
      short: "munich-quantum-software/amazon-braket-qdmi-device",
      qr: "qr-braket",
      url: "https://github.com/munich-quantum-software/amazon-braket-qdmi-device",
    },
    {
      id: "mqt",
      label: "MQT DDSIM",
      description: ["Classical Quantum", "Circuit Simulation"],
      short: "munich-quantum-toolkit/core",
      qr: "qr-core-repo",
      url: "https://github.com/munich-quantum-toolkit/core",
    },
    {
      id: "ibm",
      label: "IBM Quantum",
      description: ["Latest Addition"],
      short: "munich-quantum-software/ibm-qdmi-device",
      qr: "qr-ibm",
      url: "https://github.com/munich-quantum-software/ibm-qdmi-device",
    },
  ];
  // The badge straddles the incoming connection and the top edge of each card.
  function deviceCard(index, x = 0, y = 0, width = 398) {
    const d = devices[index];
    const identity =
      index === 0
        ? text(18, 43, "QDMI on", 25, "start") + logo("iqm", 132, 21, 78, 26)
        : index === 1
          ? logo("aws", 18, 22, 55, 30) +
            text(85, 44, "Amazon Braket", 24, "start")
          : index === 2
            ? logo("mqt", 18, 17, 105, 37) + text(140, 44, "DDSIM", 27, "start")
            : logo("ibm", 18, 22, 80, 30) +
              text(112, 44, "Quantum", 27, "start");
    return `<g transform="translate(${x} ${y}) scale(${width / 398})" role="img" aria-label="${d.label}"><rect width="398" height="129" rx="15" fill="#F7FAFD" stroke="#A9C6E1" stroke-width="1.7"/>${identity}${d.description.map((line, i) => text(18, 73 + i * 20, line, 17, "start", index === 3 ? orange : muted)).join("")}<a href="${d.url}" tabindex="-1">${logo(d.qr, 300, 10, 88, 88)}${logo("github", 18, 106, 16, 16)}${text(41, 118, d.short, d.short.length > 42 ? 11.1 : 12.5, "start", blue)}</a>${qdmiBadge(-33, -14, 163)}</g>`;
  }
  function compilerBadge(x = 0, y = 0, width = 350) {
    return `<g transform="translate(${x} ${y}) scale(${width / 350})">${logo("mqt", 21, 25, 129, 64)}${text(258, 48, "Compiler", 26)}${text(258, 82, "Collection", 26)}</g>`;
  }
  const hourglass = () =>
    `<g aria-label="Multiple inputs and outputs through one common interface">${[0, 25, 50].map((x) => path(`M${x} 0C${x} 14 25 17 25 27S${x} 40 ${x} 54`, blue, 2.4)).join("")}${[0, 25, 50].flatMap((x) => [0, 54].map((y) => `<circle cx="${x}" cy="${y}" r="3" fill="${blue}"/>`)).join("")}<circle cx="25" cy="27" r="4" fill="white" stroke="${blue}" stroke-width="2"/></g>`;
  function architecture(mode = "stack", phase = Infinity, progress = 1) {
    if (mode === "ecosystem" || mode === "core") {
      mode = "detail";
      phase = 3;
    }
    const problem = mode === "problem",
      detail = mode === "detail";
    const p = Math.min(phases[mode].length - 1, Math.max(0, Math.floor(phase)));
    const t = clamp(progress ?? 1);
    const reveal = (step, start = 0, end = 1) =>
      p > step ? 1 : p < step ? 0 : clamp((t - start) / (end - start));
    const on = (step) =>
      reveal(
        step,
        0,
        Math.min(1, 360 / (playbacks[mode]?.[step]?.duration || 1800)),
      );
    const s = (step, start = 0, end = 0.25) =>
      problem ? 0 : detail ? 1 : reveal(step, start, end);
    const d = (step, start = 0.18, end = 0.4) =>
      detail ? reveal(step, start, end) : 0;
    const retiring = (step) => 1 - (detail ? reveal(step, 0, 0.18) : 0);
    const principles = d(4);
    const route = (key, shape, amount, color = blue, arrow = false) =>
      path(
        shape,
        color,
        2.5,
        `data-morph="${key}" opacity="${amount > 0 ? 0.72 : 0}" pathLength="1" stroke-dasharray="${color === orange && amount === 1 ? ".014 .009" : "1"}" stroke-dashoffset="${1 - amount}" ${arrow ? `marker-end="url(#opening-arrow-${color === orange ? "orange" : "blue"})"` : ""}`,
      );
    const module = (key, x, y, w, h, content, visible = 1) =>
      group(
        key,
        `<rect width="${w}" height="${h}" rx="16" fill="#F4F8FC" stroke="#A9C6E1" stroke-width="1.7"/>${content}`,
        visible,
        `translate(${x} ${y}) scale(1)`,
      );
    const users = [
      { name: "Researchers", icon: "researcher", badge: "molecule", y: 95 },
      { name: "Experimentalists", icon: "researcher", badge: "chip", y: 235 },
      {
        name: "Software Developers",
        icon: "developer",
        badge: "circuit",
        y: 375,
      },
      { name: "End Users", icon: "user", badge: "measurement", y: 515 },
    ];
    const computes = [
      { name: "Workstation", kind: "workstation", y: 95 },
      { name: "Cloud Compute", kind: "cloud", y: 305 },
      { name: "HPC", kind: "server", y: 515 },
    ].map((c, i) => ({
      ...c,
      x: problem ? 750 : 450 + i * 350,
      y: problem ? c.y : 581,
    }));
    const qpus = [
      { kind: "cryostat", name: "Superconducting", y: 90 },
      { kind: "iontrap", name: "Trapped Ions", y: 204 },
      { kind: "neutralatom", name: "Neutral Atoms", y: 318 },
      { kind: "photonics", name: "Photonics", y: 432 },
      { kind: "spin", name: "Spin Qubits", y: 546 },
    ];
    let body = "";
    users.forEach((u, i) =>
      computes.forEach((c, j) => {
        const step = i ? 6 : [1, 4, 5][j];
        const order = i ? (i - 1) * 3 + j : 0;
        const count = i ? 9 : j === 2 ? 7 : 6;
        const amount =
          step === 1
            ? reveal(1, 0.18, 0.38)
            : reveal(step, order / count, (order + 1) / count);
        body += route(
          `world-user-compute-${i}-${j}`,
          `M220 ${u.y}C410 ${u.y} 490 ${c.y} ${c.x - 72} ${c.y}`,
          problem ? amount : 0,
          j === 2 ? orange : blue,
        );
      }),
    );
    computes.forEach((c, i) =>
      qpus.forEach((q, j) => {
        const step = i === 0 ? (j < 2 ? j + 1 : 3) : i === 1 ? 4 : 5;
        const order = i === 0 ? (j < 2 ? 0 : j - 2) : j + 1;
        const count = i === 0 ? (j < 2 ? 1 : 3) : i === 1 ? 6 : 7;
        body += route(
          `world-compute-qpu-${i}-${j}`,
          `M${c.x + 72} ${c.y}C1040 ${c.y} 1120 ${q.y} 1350 ${q.y}`,
          problem
            ? step === 1
              ? reveal(1, 0.55, 0.84)
              : reveal(step, order / count, (order + 1) / count)
            : 0,
          i === 2 ? orange : blue,
        );
      }),
    );
    body += route(
      "hpc-cloud-burst",
      "M822 515C985 515 927 403 752 403S550 305 678 305",
      problem ? reveal(5, 6 / 7, 1) : 0,
      orange,
      true,
    );
    body += group(
      "cloud-burst-label",
      '<rect x="610" y="409" width="191" height="30" rx="7" fill="white"/>' +
        text(705, 431, "Cloud Bursting", 21, "middle", orange),
      problem ? reveal(5, 6 / 7, 1) : 0,
    );
    users.forEach((u, i) => {
      body += route(
        `user-frontend-${i}`,
        `M220 ${u.y}H266V107H307`,
        s(1, i / 5, (i + 1) / 5) * retiring(4),
      );
    });
    const fronts = [
      { name: "Qiskit", brand: "qiskit", step: 1 },
      { name: "PennyLane", brand: "pennylane", step: 4 },
      { name: "OpenQASM 3", step: 4 },
      { name: "Your Favorite SDK", step: 4 },
    ];
    fronts.forEach((f, i) => {
      const y = 71 + i * 122;
      const start = i ? (i - 1) / 3 : 0;
      const visible = s(
        f.step,
        start + (i ? 0.12 : 0),
        start + (i ? 0.28 : 0.25),
      );
      body += route(
        `user-other-frontend-${i}`,
        `M266 107V${y + 36}H307`,
        i ? s(4, start, start + 0.12) * retiring(4) : 0,
      );
      body += module(
        `frontend-${i}`,
        307,
        y,
        221,
        72,
        f.brand === "qiskit"
          ? logo("qiskit", 14, 16, 40, 40) + text(137, 46, f.name, 26)
          : f.brand
            ? logo("pennylane", 18, 22, 185, 29)
            : i === 3
              ? path("M24 22L14 35L24 48M40 22L50 35L40 48", blue, 2.3) +
                text(145, 29, "Your Favorite", 18) +
                text(145, 58, "SDK", 22)
              : text(110, 46, f.name, 25),
        visible,
      );
      body += route(
        `frontend-junction-${i}`,
        `M528 ${y + 36}H571V279H605`,
        i ? s(4, start + 0.18, Math.min(1, start + 0.33)) : s(2, 0, 0.4),
      );
    });
    body += route(
      "frontend-program",
      "M605 279V158H658",
      s(2, 0.4, 0.75),
      blue,
      true,
    );
    body += route(
      "frontend-runtime",
      "M605 279V421H658",
      s(2, 0.4, 0.75),
      orange,
      true,
    );
    body += group(
      "lane-labels",
      text(610, 130, "Program", 19, "middle", blue) +
        text(607, 453, "Runtime", 19, "middle", orange),
      s(2, 0.6, 0.85),
    );
    body += group(
      "frontend-split",
      '<circle cx="605" cy="279" r="5" fill="white" stroke="#2F70B8" stroke-width="2"/>',
      s(2, 0.3, 0.4),
    );
    const productPulse = (step, width, height) =>
      group(
        `product-pulse-${step}`,
        `<rect x="-3" y="-3" width="${width + 6}" height="${height + 6}" rx="19" fill="none" stroke="${blue}" stroke-width="4"/>`,
        detail && p === step
          ? Math.sin(Math.PI * Math.min(1, t * (step === 1 ? 4 : 1))) * 0.75
          : 0,
      );
    body += module(
      "stack-compiler",
      658,
      65,
      350,
      180,
      group(
        "compiler-generic",
        text(175, 46, "Compiler", 29) +
          text(175, 83, "Infrastructure", 29) +
          text(175, 133, "Optimize · Map · Synthesize", 21, "middle", muted),
        retiring(2),
      ) +
        group(
          "compiler-product",
          compilerBadge() +
            logo("mlir", 30, 119, 64, 33) +
            logo("llvm", 108, 113, 65, 40) +
            logo("qir", 199, 116, 54, 38) +
            logo("jeff", 275, 113, 54, 40),
          d(2),
        ) +
        productPulse(2, 350, 180),
      s(2, 0.7, 1),
    );
    body += module(
      "stack-orchestrator",
      658,
      337,
      350,
      180,
      group(
        "orchestrator-generic",
        text(175, 47, "Resource Management", 25) +
          text(175, 84, "& Orchestration", 28) +
          text(175, 134, "Schedule · Execute · Collect", 21, "middle", muted),
        retiring(3),
      ) +
        group(
          "orchestrator-product",
          logo("mqsc", 22, 49, 115, 72) + text(244, 93, "Orchestrator", 27),
          d(3),
        ) +
        productPulse(3, 350, 180),
      s(2, 0.7, 1),
    );
    body += route(
      "compile-request",
      "M705 333V249",
      s(2, 0.8, 1),
      orange,
      true,
    );
    body += route("compiled-program", "M966 249V333", s(2, 0.8, 1), blue, true);
    body += group(
      "compile-labels",
      '<rect x="674" y="275" width="136" height="29" fill="white"/><rect x="883" y="294" width="142" height="27" fill="white"/>' +
        text(742, 295, "Compile Request", 16, "middle", orange) +
        text(954, 313, "Compiled Program", 16, "middle", blue),
      s(2, 0.8, 1),
    );
    body += module(
      "stack-backend",
      1118,
      65,
      170,
      452,
      `<g transform="translate(60 32)">${hourglass()}</g>` +
        group(
          "backend-generic",
          text(85, 211, "Backend", 27) + text(85, 247, "Interface", 27),
          retiring(1),
        ) +
        group(
          "backend-standard",
          text(85, 167, "Quantum", 24) +
            text(85, 201, "Device", 24) +
            text(85, 235, "Management", 22) +
            text(85, 269, "Interface", 24) +
            text(85, 310, "(QDMI)", 28),
          d(1, 0.18, 0.27),
        ) +
        productPulse(1, 170, 452),
      s(3, 0, 0.2),
    );
    body += route(
      "device-capabilities",
      detail && p >= 2 ? "M1114 164H1067V245H937" : "M1114 164H1012",
      s(3, 0.12, 0.3),
      blue,
      true,
    );
    body += route(
      "job-submission",
      detail && p >= 3 ? "M812 535V548H1100V408H1114" : "M1012 408H1114",
      s(3, 0.12, 0.3),
      orange,
      true,
    );
    body += route(
      "execution-results",
      detail && p >= 3 ? "M1114 476H1080V541H854V535" : "M1114 476H1012",
      s(3, 0.12, 0.3),
      orange,
      true,
    );
    body += group(
      "backend-labels",
      text(1061, 126, "Device", 16) +
        text(1061, 147, "Capabilities", 16) +
        text(1063, 396, "Submit", 17) +
        text(1063, 467, "Results", 17),
      s(3, 0.18, 0.35),
    );
    body += group(
      "compiler-qdmi-connector",
      qdmiBadge(729, 227, 208, "Integrated with"),
      d(2),
    );
    body += group(
      "orchestrator-qdmi-connector",
      qdmiBadge(729, 499, 208, "Integrated with"),
      d(3),
    );
    body += group(
      "orchestrator-slurm",
      '<rect x="753" y="319" width="160" height="38" rx="19" fill="white" stroke="#A9C6E1" stroke-width="1.5"/>' +
        text(768, 343, "Works with", 15, "start", blue) +
        logo("slurm", 848, 322, 54, 30),
      d(3),
    );
    qpus.forEach((q, i) => {
      body += route(
        `backend-qpu-${i}`,
        `M1288 290C1310 290 1310 ${q.y} 1350 ${q.y}`,
        s(3, 0.25 + i * 0.13, 0.38 + i * 0.13) * retiring(1),
      );
    });
    devices.forEach((_, i) => {
      const y = 34 + i * 143,
        start = 0.18 + i * 0.2;
      body += route(
        `backend-device-${i}`,
        `M1288 290H1304V${y + 2}H1317`,
        d(1, start, start + 0.1),
      );
      body += group(
        `world-device-${i}`,
        deviceCard(i, 1350, y),
        d(1, start + 0.1, start + 0.2),
      );
    });
    body += group(
      "hosting-underpass",
      path("M307 539V552H1288V539", muted, 2, 'stroke-dasharray="7 5"') +
        [450, 800, 1150].map((x) => path(`M${x} 552V570`, muted, 2)).join(""),
      problem ? 0 : retiring(4),
    );
    body += group(
      "stack-principles",
      text(1000, 25, "Modular · Efficient · Extensible", 27, "middle", blue),
      problem ? 0 : 1,
    );
    body += group(
      "users-heading",
      text(118, 15, "Users", 27),
      (problem ? on(1) : 1) * retiring(4),
    );
    body += group(
      "compute-heading",
      text(750, 15, "Classical Compute", 27),
      problem ? reveal(1, 0.38, 0.55) : 0,
    );
    body += group(
      "qpu-heading",
      text(1510, 15, "Quantum Processors", 27),
      retiring(1),
    );
    body += group(
      "cloud-access",
      text(1067, 52, "Cloud Access", 23, "middle", blue),
      problem ? reveal(1, 0.75, 1) : 0,
    );
    body += group(
      "on-premise-access",
      text(1072, 626, "On-premise Access", 23, "middle", orange),
      problem ? on(5) : 0,
    );
    users.forEach((u, i) => {
      body += group(
        `world-user-${i}`,
        icon(u.icon, 48, 0, 77) +
          icon(u.badge, 121, 37, 32) +
          text(99, 103, u.name, 21),
        (problem ? (i ? on(6) : reveal(1, 0, 0.18)) : 1) * retiring(4),
        `translate(18 ${u.y - 37}) scale(1)`,
      );
    });
    computes.forEach((c, i) => {
      body += group(
        `world-compute-${i}`,
        (i === 0
          ? icon("workstation")
          : i === 1
            ? icon("cloud", 0, 0, 102)
            : icon("server", -7, 0, 82) + icon("server", 40, 0, 82)) +
          text(51, 116, c.name, problem ? 27 : 34),
        (problem ? (i ? on(i === 1 ? 4 : 5) : reveal(1, 0.38, 0.55)) : 1) *
          retiring(4),
        `translate(${c.x - (problem ? 51 : 25)} ${c.y - (problem ? 45 : 25)}) scale(${problem ? 1 : 0.5})`,
      );
    });
    qpus.forEach((q, i) => {
      body += group(
        `world-qpu-${i}`,
        icon(q.kind, 0, 0, 80) + text(96, 46, q.name, 26, "start"),
        retiring(1),
        `translate(1360 ${q.y - 40}) scale(1)`,
      );
    });
    body += group(
      "future-device",
      text(1549, 626, "Your Device Could Be Next", 24, "middle", blue),
      d(1, 0.9, 1),
    );
    body += group("principles-reveal", principleScene(), principles);
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1760 650" role="img" aria-label="Users connect through frontends, compilation, orchestration and a shared device interface; the full stack runs on classical compute" style="font-family:Inter,Arial,sans-serif;fill:${navy}"><defs>${[
      ["blue", blue],
      ["orange", orange],
    ]
      .map(
        ([id, color]) =>
          `<marker id="opening-arrow-${id}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L6 3.5L0 7" fill="${color}"/></marker>`,
      )
      .join("")}</defs>${body}</svg>`;
  }

  function principleIcon(name, x, y, size = 68) {
    const uri = assets[`principle-${name}`];
    // Inline source geometry keeps website pictograms editable and animatable offline.
    const raw = atob(uri.split(",")[1]);
    const inner = raw.slice(
      raw.indexOf(">", raw.indexOf("<svg")) + 1,
      raw.lastIndexOf("</svg>"),
    );
    return `<g class="opening-principle-icon" transform="translate(${x} ${y}) scale(${size / 24})" fill="none" stroke="${blue}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${inner}</g>`;
  }
  function principleScene() {
    return group(
      "principles",
      [
        ["open", ["Open by Design"]],
        ["community", ["Community-Driven"]],
        ["unlocked", ["No Lock-Ins"]],
        ["practice", ["From Research", "to Practice"]],
        ["performance", ["Performance", "at Every Layer"]],
      ]
        .map(([name, lines], i) => {
          const y = 58 + i * 105;
          return (
            principleIcon(name, 9, y, 46) +
            lines
              .map((line, j) => text(72, y + 26 + j * 27, line, 22, "start"))
              .join("")
          );
        })
        .join(""),
    );
  }

  function projectReference(kind) {
    const qdmi = kind === "qdmi";
    const credits = qdmi
      ? ["tum-mark", "lrz-mark", "mqv", "mqsc"]
      : ["mqsc", "tum-mark"];
    return `<div class="opening-project" data-morph="opening-reference-${kind}">${image(qdmi ? "qr-qdmi-repo" : "qr-core-repo", "opening-project-qr", `${kind} repository QR code`)}<div><div class="opening-project-title"><b>${qdmi ? "QDMI" : "MQT Core"}</b><span>${credits.map((n) => image(n)).join("")}</span></div><p>${qdmi ? "Created by TUM CDA, TUM CAPS & LRZ · Maintained by MQV gGmbH & MQSC" : "Created and Maintained by MQSC · TUM CDA · MQT Community"}</p><a href="${qdmi ? "https://github.com/Munich-Quantum-Software-Stack/QDMI" : "https://github.com/munich-quantum-toolkit/core"}">${image("github", "opening-github", "GitHub")}${qdmi ? "Munich-Quantum-Software-Stack/QDMI" : "munich-quantum-toolkit/core"}</a></div></div>`;
  }
  function render(mode, phase, progress = 1) {
    const title = {
      problem: "Software that connects the quantum ecosystem",
      stack: "A shared software stack",
      detail: "Our Approach: From the Metal to the User",
    }[mode];
    const refs =
      mode === "stack"
        ? `<div class="opening-paper" data-morph="opening-paper">${image("mqss")}<div>Burgholzer, Echavarria, et al., 2026<br><a href="https://doi.org/10.1145/3773656.3773669">doi.org/10.1145/3773656.3773669</a></div>${image("qr-mqss", "opening-paper-qr", "Paper QR code")}</div>`
        : mode === "detail"
          ? `<div class="opening-references">${phase >= 2 ? projectReference("core") : "<div></div>"}${phase >= 1 ? projectReference("qdmi") : ""}</div>`
          : "";
    return `<div class="opening-frame"><h2 data-morph="heading">${title}</h2><div class="opening-scene" data-morph="world">${architecture(mode, phase, progress)}</div>${refs}</div>`;
  }
  const social = () =>
    `<div class="opening-social"><a href="https://github.com/burgholzer">${image("github", "", "GitHub")} @burgholzer</a><a href="https://www.linkedin.com/in/lukas-burgholzer-7a1741a7/">${image("linkedin", "", "LinkedIn")} Lukas Burgholzer</a></div>`;
  function title() {
    return `<div class="opening-title"><p class="talk-date">MQSF 2026 · Munich · 14–15 October</p><h1>System Software<br>for Quantum Computing</h1><p class="opening-slogan">From the Metal to the User</p><div class="opening-speaker"><b>Lukas Burgholzer</b><div class="opening-affiliations"><div>${image("mqsc")}<p>CTO &amp; Co-founder<span>Munich Quantum Software Company</span></p></div><div>${image("tum-cda")}<p>Senior Researcher<span>Chair for Design Automation · TUM</span></p></div></div>${social()}</div></div>`;
  }
  function conclusion() {
    return `<div class="opening-conclusion"><h1>From the Metal<br>To the User</h1><p class="opening-closing-credit">Built with the <b>MQT and QDMI communities</b><br>Driven into practice by <b>MQSC</b>.</p><div class="opening-closing-products"><div>${image("qdmi")}<span>-powered Solutions</span></div><div>${image("mqt")}<span>Compiler Collection</span></div><div>${image("mqsc")}<span>Orchestrator</span></div></div><div class="opening-closing-bottom"><div><div class="opening-community-logos">${["tum-mark", "lrz-mark", "mqv", "mqss"].map((n) => image(n)).join("")}</div>${social()}</div><a class="opening-contact" href="https://mq.sc">${image("qr-company", "", "MQSC website QR code")}<span>mq.sc</span></a></div></div>`;
  }
  window.MQSF_VIZ.architecturePhases = phases;
  return {
    title,
    render,
    conclusion,
    architecture,
    phases,
    playbacks,
    qdmiBadge,
    deviceCard,
    compilerBadge,
    devices,
  };
})();
