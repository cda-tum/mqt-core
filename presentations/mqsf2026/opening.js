/* Copyright (c) 2026 Munich Quantum Software Company GmbH
 * SPDX-License-Identifier: MIT */
"use strict";

window.MQSF_OPENING = (() => {
  const blue = "#2F70B8",
    navy = "#142B45",
    teal = "#087D92",
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
      "Researchers",
      "Workstation",
      "Cloud Access",
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
      "One Interface",
      "Quantum Devices",
      "Classical Resources",
      "PennyLane",
      "OpenQASM and Your SDK",
    ],
    detail: [
      "From the Metal to the User",
      "Quantum Device Management Interface",
      "QDMI on IQM",
      "Amazon Braket",
      "MQT DDSIM",
      "IBM Quantum",
      "Compiler Collection",
      "Orchestrator",
      "Our Principles",
    ],
  };
  const playbacks = {
    problem: {
      3: { duration: 3200 },
      4: { duration: 1800 },
      5: { duration: 3600 },
      6: { duration: 3600 },
      7: { duration: 4800 },
      8: { duration: 3500 },
    },
    stack: {
      0: { duration: 2400, transition: 1800 },
      1: { duration: 1500 },
      2: { duration: 2200 },
      3: { duration: 2200 },
      4: { duration: 3000 },
      5: { duration: 2500 },
      6: { duration: 1500 },
      7: { duration: 2000 },
    },
    detail: {
      1: { duration: 1800 },
      2: { duration: 1800 },
      3: { duration: 1800 },
      4: { duration: 1800 },
      5: { duration: 1800 },
      6: { duration: 2200 },
      7: { duration: 2200 },
      8: { duration: 2000 },
    },
  };
  const devices = [
    {
      id: "iqm",
      label: "QDMI on IQM",
      description: ["On-premise and Cloud Devices"],
      short: "github.com/iqm-finland/QDMI-on-IQM",
      qr: "qr-iqm",
      url: "https://github.com/iqm-finland/QDMI-on-IQM",
    },
    {
      id: "aws",
      label: "Amazon Braket",
      description: ["Cloud Service Integration"],
      short: "amazon-braket-qdmi-device.readthedocs.io",
      qr: "qr-braket",
      url: "https://amazon-braket-qdmi-device.readthedocs.io/en/stable/",
    },
    {
      id: "mqt",
      label: "MQT DDSIM",
      description: ["Classical Quantum", "Circuit Simulation"],
      short: "github.com/munich-quantum-toolkit/core",
      qr: "qr-core-repo",
      url: "https://github.com/munich-quantum-toolkit/core",
    },
    {
      id: "ibm",
      label: "IBM Quantum",
      description: ["Latest Addition"],
      short: "github.com/…/ibm-qdmi-device",
      qr: "qr-ibm",
      url: "https://github.com/munich-quantum-software/ibm-qdmi-device",
    },
  ];
  // Keep identity, QR and badge placement consistent when expanding device cards.
  function deviceCard(index, x = 0, y = 0, width = 398) {
    const d = devices[index];
    const descriptionY = d.description.length > 1 ? 64 : 70;
    const identity =
      index === 0
        ? text(18, 37, "QDMI on", 25, "start") + logo("iqm", 129, 13, 95, 31)
        : index === 1
          ? logo("aws", 18, 13, 59, 35) +
            text(90, 37, "Amazon Braket", 24, "start")
          : index === 2
            ? logo("mqt", 18, 10, 110, 40) + text(144, 39, "DDSIM", 27, "start")
            : logo("ibm", 18, 12, 84, 33) +
              text(117, 38, "Quantum", 27, "start");
    return `<g transform="translate(${x} ${y}) scale(${width / 398})" role="img" aria-label="${d.label}"><rect width="398" height="129" rx="15" fill="#F7FAFD" stroke="#A9C6E1" stroke-width="1.7"/>${identity}${d.description.map((line, i) => text(18, descriptionY + i * 24, line, 17, "start", index === 3 ? teal : muted)).join("")}<a href="${d.url}" tabindex="-1">${logo(d.qr, 298, 8, 91, 91)}${text(18, 105, d.short, 12.5, "start", blue)}</a>${qdmiBadge(217, 108, 169)}</g>`;
  }
  const hourglass = () =>
    `<g aria-label="Multiple inputs and outputs through one common interface">${[0, 25, 50].map((x) => path(`M${x} 0C${x} 14 25 17 25 27S${x} 40 ${x} 54`, blue, 2.4)).join("")}${[0, 25, 50].flatMap((x) => [0, 54].map((y) => `<circle cx="${x}" cy="${y}" r="3" fill="${blue}"/>`)).join("")}<circle cx="25" cy="27" r="4" fill="white" stroke="${blue}" stroke-width="2"/></g>`;
  function architecture(
    mode = "stack",
    phase = mode === "detail" ? 7 : Infinity,
    progress = 1,
  ) {
    if (mode === "ecosystem" || mode === "core") {
      mode = "detail";
      phase = 7;
    }
    const problem = mode === "problem",
      detail = mode === "detail";
    const p = Math.min(phases[mode].length - 1, Math.max(0, Math.floor(phase)));
    const t = clamp(progress ?? 1);
    const reveal = (step, index = 0, count = 1) =>
      p > step ? 1 : p < step ? 0 : clamp(t * count - index);
    const fade = clamp((t * (playbacks[mode]?.[p]?.duration || 1800)) / 450);
    const on = (step) => (p === step ? fade : p > step ? 1 : 0);
    const s = (step) => (problem ? 0 : detail ? 1 : on(step));
    // Retire old labels before their replacements enter the same space.
    const replacement = (step, incoming) => {
      if (!detail) return incoming ? 0 : 1;
      if (p !== step) return incoming ? Number(p > step) : Number(p < step);
      const elapsed = t * playbacks.detail[step].duration;
      return incoming ? clamp((elapsed - 450) / 450) : 1 - clamp(elapsed / 450);
    };
    const d = (step) => replacement(step, true);
    const retiring = (step) => replacement(step, false);
    const principles = d(8);
    const route = (key, shape, amount, color = blue, arrow = false) =>
      path(
        shape,
        color,
        2.5,
        `data-morph="${key}" opacity="${amount > 0 ? 0.65 : 0}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${1 - amount}" ${arrow ? 'marker-end="url(#opening-arrow)"' : ""}`,
      );
    const module = (key, x, y, w, h, body, visible = 1) =>
      group(
        key,
        `<rect width="${w}" height="${h}" rx="16" fill="#F4F8FC" stroke="#A9C6E1" stroke-width="1.7"/>${body}`,
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
      x: problem ? 750 : 420 + i * 330,
      y: problem ? c.y : 569,
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
        const step = i ? 8 : [3, 6, 7][j],
          order = i ? (i - 1) * 3 + j : 0,
          count = i ? 9 : j === 0 ? 2 : j === 2 ? 7 : 6;
        body += route(
          `world-user-compute-${i}-${j}`,
          `M220 ${u.y}C410 ${u.y} 490 ${c.y} ${c.x - 72} ${c.y}`,
          problem ? reveal(step, order, count) : 0,
          j === 2 ? teal : blue,
        );
      }),
    );
    computes.forEach((c, i) =>
      qpus.forEach((q, j) => {
        const step = i === 0 ? (j < 2 ? j + 3 : 5) : i === 1 ? 6 : 7;
        const index = i === 0 ? (j === 0 ? 1 : j === 1 ? 0 : j - 2) : j + 1;
        const count =
          i === 0 ? (j === 0 ? 2 : j === 1 ? 1 : 3) : i === 1 ? 6 : 7;
        body += route(
          `world-compute-qpu-${i}-${j}`,
          `M${c.x + 72} ${c.y}C1040 ${c.y} 1120 ${q.y} 1350 ${q.y}`,
          problem ? reveal(step, index, count) : 0,
          i === 2 ? teal : blue,
        );
      }),
    );
    body += route(
      "hpc-cloud-burst",
      "M815 509C936 509 936 305 815 305",
      problem ? reveal(7, 6, 7) : 0,
      teal,
    );
    body += group(
      "cloud-burst-label",
      '<rect x="943" y="391" width="164" height="29" rx="7" fill="white"/>' +
        text(952, 412, "Cloud Bursting", 20, "start", teal),
      problem && p >= 7 ? reveal(7, 6, 7) : 0,
    );
    users.forEach((u, i) => {
      body += route(
        `user-frontend-${i}`,
        `M220 ${u.y}C260 ${u.y} 250 57 303 57`,
        s(1) * retiring(8) * (detail ? 1 : reveal(1, i, 4)),
      );
    });
    qpus.forEach((q, i) => {
      body += route(
        `backend-qpu-${i}`,
        `M1160 464C1260 464 1230 ${q.y} 1350 ${q.y}`,
        s(4) * retiring(2) * (detail ? 1 : reveal(4, i, 5)),
      );
    });
    devices.forEach((_, i) => {
      body += route(
        `backend-device-${i}`,
        `M1160 464C1258 464 1250 ${67 + i * 147} 1346 ${67 + i * 147}`,
        d(i + 2) * retiring(8) * (p === i + 2 ? reveal(i + 2) : 1),
      );
    });
    body += group(
      "hosting-underpass",
      path("M420 552V534H1263V269H1210", "white", 10) +
        path(
          "M420 552V534H1263V269H1210",
          muted,
          2.5,
          'stroke-dasharray="7 5"',
        ) +
        path("M750 552V534M1080 552V534", muted, 2.5),
      s(5) * retiring(8) * (detail ? 1 : reveal(5)),
    );
    body += group(
      "users-heading",
      text(118, 15, "Users", 27),
      (problem ? on(1) : 1) * retiring(8),
    );
    body += group(
      "compute-heading",
      (problem
        ? ""
        : '<rect x="639" y="517" width="222" height="30" fill="white"/>') +
        text(750, problem ? 15 : 539, "Classical Compute", problem ? 27 : 22),
      (problem ? on(2) : s(0)) * retiring(8),
    );
    body += group(
      "qpu-heading",
      text(1510, 15, "Quantum Processors", 27),
      retiring(2) * retiring(8),
    );
    body += group(
      "cloud-access",
      text(1067, 52, "Cloud Access", 23, "middle", blue),
      problem ? on(3) : 0,
    );
    body += group(
      "on-premise-access",
      text(1072, 626, "On-premise Access", 23, "middle", teal),
      problem ? on(7) : 0,
    );
    users.forEach((u, i) => {
      body += group(
        `world-user-${i}`,
        icon(u.icon, 48, 0, 77) +
          icon(u.badge, 121, 37, 32) +
          text(99, 103, u.name, 21),
        (problem ? on(i ? 8 : 1) : 1) * retiring(8),
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
          text(51, 116, c.name, problem ? 27 : 32),
        (problem ? on([2, 6, 7][i]) : 1) * retiring(8),
        `translate(${c.x - (problem ? 51 : 32)} ${c.y - (problem ? 45 : 25)}) scale(${problem ? 1 : 0.62})`,
      );
    });
    qpus.forEach((q, i) => {
      body += group(
        `world-qpu-${i}`,
        icon(q.kind, 0, 0, 80) + text(96, 46, q.name, 26, "start"),
        retiring(2) * retiring(8),
        `translate(1360 ${q.y - 40}) scale(1)`,
      );
    });
    const fronts = [
      { name: "Qiskit", brand: "qiskit", step: 1 },
      { name: "PennyLane", brand: "pennylane", step: 6 },
      { name: "OpenQASM 3", step: 7 },
      { name: "Your Favorite SDK", step: 7 },
    ];
    fronts.forEach((f, i) => {
      const x = 303 + i * 235;
      body += module(
        `frontend-${i}`,
        x,
        21,
        214,
        72,
        f.brand === "qiskit"
          ? logo("qiskit", 13, 16, 40, 40) + text(130, 46, f.name, 26)
          : f.brand
            ? logo("pennylane", 15, 23, 184, 28)
            : i === 3
              ? path("M24 22L14 35L24 48M40 22L50 35L40 48", blue, 2.3) +
                text(137, 29, "Your Favorite", 18) +
                text(137, 59, "SDK", 22)
              : text(107, 46, f.name, 25),
        s(f.step),
      );
      body += route(
        `frontend-junction-${i}`,
        `M${x + 107} 93V130H750`,
        s(i ? f.step : 2) * (detail ? 1 : reveal(i ? f.step : 2)),
        blue,
      );
    });
    body += route(
      "frontend-program",
      "M750 130V151H515V197",
      s(2) * (detail ? 1 : reveal(2)),
      blue,
    );
    body += route(
      "frontend-runtime",
      "M750 130V151H1015V197",
      s(2) * (detail ? 1 : reveal(2)),
      teal,
    );
    body += group(
      "lane-labels",
      `<rect x="459" y="160" width="112" height="28" fill="white"/><rect x="958" y="160" width="114" height="28" fill="white"/>${text(515, 182, "Program", 23, "middle", blue)}${text(1015, 182, "Runtime", 23, "middle", teal)}`,
      s(2),
    );
    const productPulse = (step) =>
      group(
        `product-pulse-${step}`,
        `<rect x="-3" y="-3" width="396" height="164" rx="19" fill="none" stroke="${blue}" stroke-width="4"/>`,
        detail && p === step ? Math.sin(Math.PI * t) * 0.75 : 0,
      );
    body += module(
      "stack-compiler",
      320,
      201,
      390,
      158,
      group(
        "compiler-generic",
        text(195, 43, "Compiler", 29) +
          text(195, 80, "Infrastructure", 29) +
          text(195, 127, "Optimize · Map · Synthesize", 22, "middle", muted),
        retiring(6),
      ) +
        group(
          "compiler-product",
          logo("mqt", 19, 18, 143, 65) +
            text(276, 42, "Compiler", 27) +
            text(276, 77, "Collection", 27) +
            logo("mlir", 26, 105, 77, 35) +
            logo("llvm", 114, 103, 70, 39) +
            logo("qir", 218, 103, 58, 40) +
            logo("jeff", 301, 102, 59, 43),
          d(6),
        ) +
        productPulse(6),
      s(2),
    );
    body += module(
      "stack-orchestrator",
      820,
      201,
      390,
      158,
      group(
        "orchestrator-generic",
        text(195, 43, "Resource Management", 27) +
          text(195, 80, "& Orchestration", 29) +
          text(195, 127, "Schedule · Execute · Collect", 22, "middle", muted),
        retiring(7),
      ) +
        group(
          "orchestrator-product",
          logo("mqsc", 18, 18, 122, 77) +
            text(267, 65, "Orchestrator", 28) +
            logo("slurm", 31, 113, 77, 39) +
            qdmiBadge(189, 125, 182, "Built on"),
          d(7),
        ) +
        productPulse(7),
      s(2),
    );
    body += route("compile-request", "M816 247H716", s(2), teal, true);
    body += route("compiled-program", "M714 319H815", s(2), blue, true);
    body += module(
      "stack-backend",
      320,
      427,
      840,
      76,
      `<g transform="translate(25 11)">${hourglass()}</g>` +
        group(
          "backend-generic",
          text(458, 48, "Backend Interface", 30),
          retiring(1),
        ) +
        group(
          "backend-standard",
          text(463, 46, "Quantum Device Management Interface", 27),
          d(1),
        ),
      s(3),
    );
    body += route("device-capabilities", "M471 423V365", s(3), blue, true);
    body += route("job-submission", "M918 363V421", s(3), teal, true);
    body += route("execution-results", "M1100 423V365", s(3), teal, true);
    body += group(
      "backend-labels",
      `<rect x="379" y="378" width="186" height="29" fill="white"/><rect x="875" y="378" width="86" height="29" fill="white"/><rect x="1058" y="378" width="87" height="29" fill="white"/>${text(471, 399, "Device Capabilities", 19)}${text(918, 399, "Submit", 19)}${text(1100, 399, "Results", 19)}`,
      s(3),
    );
    devices.forEach((_, i) => {
      body += group(
        `world-device-${i}`,
        deviceCard(i, 1348, i * 147),
        d(i + 2) * retiring(8),
      );
    });
    body += group(
      "future-device",
      text(1550, 616, "Your Device Could Be Next", 24, "middle", blue),
      d(5) * retiring(8),
    );
    if (detail && p === 8)
      body += group("principles-reveal", principleScene(), principles);
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1760 650" role="img" aria-label="Users, Classical Compute and Quantum Processors connected through a shared software stack" style="font-family:Inter,Arial,sans-serif;fill:${navy}"><defs><marker id="opening-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L5 3L0 6" fill="${blue}"/></marker></defs>${body}</svg>`;
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
        ["open", 28, 102, ["Open by", "Design"]],
        ["community", 28, 355, ["Community-", "Driven"]],
        ["unlocked", 1367, 102, ["No Lock-Ins"]],
        ["practice", 1367, 355, ["From Research", "to Practice"]],
      ]
        .map(
          ([name, x, y, lines]) =>
            principleIcon(name, x, y) +
            lines
              .map((value, i) => text(x, y + 113 + i * 36, value, 31, "start"))
              .join(""),
        )
        .join("") +
        principleIcon("performance", 599, 548, 57) +
        text(681, 569, "Performance, Every Time", 29, "start") +
        text(681, 601, "At Every Layer of the Stack", 21, "start", muted),
    );
  }
  function projectReference(kind) {
    const qdmi = kind === "qdmi";
    const credits = qdmi
      ? ["tum-mark", "lrz-mark", "mqv", "mqsc"]
      : ["mqsc", "tum-mark"];
    return `<div class="opening-project" data-morph="opening-reference-${kind}">${image(qdmi ? "qr-qdmi-repo" : "qr-core-repo", "opening-project-qr", `${kind} repository QR code`)}<div><div class="opening-project-title"><b>${qdmi ? "QDMI" : "MQT Core"}</b><span>${credits.map((n) => image(n)).join("")}</span></div><p>${qdmi ? "Created by TUM CDA, TUM CAPS & LRZ · Maintained by MQV & MQSC" : "MQSC · TUM CDA · MQT Community"}</p><a href="${qdmi ? "https://github.com/Munich-Quantum-Software-Stack/QDMI" : "https://github.com/munich-quantum-toolkit/core"}">${qdmi ? "github.com/Munich-Quantum-Software-Stack/QDMI" : "github.com/munich-quantum-toolkit/core"}</a></div></div>`;
  }
  function render(mode, phase, progress = 1) {
    const title = {
      problem: "Software that connects the quantum ecosystem",
      stack: "A shared software stack",
      detail: "From the Metal to the User",
    }[mode];
    const refs =
      mode === "stack"
        ? `<div class="opening-paper" data-morph="opening-paper">${image("mqss")}<div>Burgholzer, Echavarria, et al., 2026<br><a href="https://doi.org/10.1145/3773656.3773669">doi.org/10.1145/3773656.3773669</a></div>${image("qr-mqss", "opening-paper-qr", "Paper QR code")}</div>`
        : mode === "detail"
          ? `<div class="opening-references">${phase >= 6 ? projectReference("core") : "<div></div>"}${phase >= 1 ? projectReference("qdmi") : ""}</div>`
          : "";
    return `<div class="opening-frame"><h2 data-morph="heading">${title}</h2><div class="opening-scene" data-morph="world">${architecture(mode, phase, progress)}</div>${refs}</div>`;
  }
  const social = () =>
    `<div class="opening-social"><a href="https://github.com/burgholzer">${image("github", "", "GitHub")} @burgholzer</a><a href="https://www.linkedin.com/in/lukas-burgholzer-7a1741a7/">${image("linkedin", "", "LinkedIn")} Lukas Burgholzer</a></div>`;
  function title() {
    return `<div class="opening-title"><p class="talk-date">MQSF 2026 · Munich · 14–15 October</p><h1>System Software<br>for Quantum Computing</h1><p class="opening-slogan">From the Metal to the User</p><div class="opening-speaker"><b>Lukas Burgholzer</b><div class="opening-affiliations"><div>${image("mqsc")}<p>CTO &amp; Co-founder<span>Munich Quantum Software Company</span></p></div><div>${image("tum-cda")}<p>Senior Researcher<span>Chair for Design Automation · TUM</span></p></div></div>${social()}</div></div>`;
  }
  function conclusion() {
    return `<div class="opening-conclusion"><h1>From the Metal<br>to the User</h1><p class="opening-closing-credit">Built with the <b>MQT and QDMI communities</b><br>driven into practice by <b>MQSC</b>.</p><div class="opening-closing-products"><div>${image("mqt")}<span>Compiler Collection</span></div><div>${image("qdmi")}<span>Device Interface</span></div><div>${image("mqsc")}<span>Orchestrator</span></div></div><div class="opening-closing-bottom"><div><div class="opening-community-logos">${["tum-mark", "lrz-mark", "mqv", "mqss"].map((n) => image(n)).join("")}</div>${social()}</div><a class="opening-contact" href="https://mq.sc">${image("qr-company", "", "MQSC website QR code")}<span>mq.sc</span></a></div></div>`;
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
    devices,
  };
})();
