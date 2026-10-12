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
  const blue = "#2F70B8";
  const navy = "#142B45";
  const pale = "#E8F1FA";
  const cyan = "#60B3CB";
  const gold = "#C58B42";
  const text = (x, y, value, size = 34, anchor = "middle", extra = "") =>
    `<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" ${extra}>${value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</text>`;
  const path = (d, color = blue, width = 5, extra = "") =>
    `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  const logo = (name, x, y, width, height) =>
    window.MQSF_ASSETS?.[name]
      ? `<image href="${window.MQSF_ASSETS[name]}" x="${x}" y="${y}" width="${width}" height="${height}" preserveAspectRatio="xMidYMid meet"/>`
      : "";

  function icon(name, x = 0, y = 0, size = 100) {
    let body = "";
    switch (name) {
      case "researcher":
        body = `<ellipse cx="50" cy="95" rx="40" ry="4" fill="${pale}"/>
          <path d="M24 87L28 58Q31 48 43 46H59Q73 51 76 67L80 87" fill="${pale}" stroke="${navy}" stroke-width="2"/>
          <path d="M43 46L50 63L59 46M50 63V88M36 57L32 80M68 59L74 80" fill="none" stroke="${blue}" stroke-width="2"/>
          <path d="M40 48L42 40H58L60 48L50 60Z" fill="#F4D9C4"/>
          <ellipse cx="50" cy="28" rx="15" ry="19" fill="#F4D9C4" stroke="${navy}" stroke-width="2"/>
          <path d="M35 28Q30 7 48 7Q70 5 66 31L61 20Q48 23 40 16L38 31" fill="${navy}"/>
          <path d="M36 29H46M54 29H64M46 29H54" stroke="${navy}" stroke-width="1.5"/>
          <rect x="38" y="26" width="10" height="8" rx="3" fill="none" stroke="${navy}"/>
          <rect x="52" y="26" width="10" height="8" rx="3" fill="none" stroke="${navy}"/>
          <path d="M46 39Q50 42 55 38" fill="none" stroke="${navy}"/>
          <path d="M26 71L34 83L45 83M73 72L68 82L59 83" fill="none" stroke="#E7BFA1" stroke-width="7" stroke-linecap="round"/>
          <path d="M30 70H69L65 89H35Z" fill="white" stroke="${blue}" stroke-width="2"/>
          <circle cx="50" cy="79" r="3" fill="${cyan}"/><path d="M29 91H71" stroke="${navy}" stroke-width="3" stroke-linecap="round"/>`;
        break;
      case "developer":
      case "user": {
        const skin = name === "developer" ? "#B87D5F" : "#F4D9C4";
        body = `<ellipse cx="50" cy="95" rx="40" ry="4" fill="${pale}"/>
          <path d="M22 89L27 61Q31 47 47 47H57Q73 49 77 63L82 89Z" fill="${name === "developer" ? "#60B3CB" : "#2F70B8"}" stroke="${navy}" stroke-width="2"/>
          <path d="M42 42V50Q50 58 58 50V41" fill="${skin}"/>
          <ellipse cx="50" cy="29" rx="15" ry="19" fill="${skin}" stroke="${navy}" stroke-width="1.5"/>
          ${name === "developer" ? `<path d="M35 30Q24 8 48 7Q72 5 67 36L63 56L57 46L62 29L58 18Q47 28 37 24L37 48L30 56Z" fill="${navy}"/>` : '<path d="M35 23Q33 7 49 8Q66 7 65 23L58 16L46 20L37 18Z" fill="#9BAAB8"/>'}
          <path d="M41 30H44M56 30H59M46 39Q51 43 56 38" fill="none" stroke="${navy}" stroke-width="1.5"/>
          <path d="M29 66L34 83H44M73 66L67 83H58" fill="none" stroke="${skin}" stroke-width="7" stroke-linecap="round"/>
          <path d="M30 69H70L66 89H34Z" fill="white" stroke="${navy}" stroke-width="2"/>
          <circle cx="50" cy="79" r="3" fill="${blue}"/><path d="M28 91H72" stroke="${navy}" stroke-width="3" stroke-linecap="round"/>`;
        break;
      }
      case "server":
        body = `<path d="M8 14L24 5H91L77 14Z" fill="#BBD2E9" stroke="${navy}" stroke-width="2"/>
          <path d="M77 14L91 5V87L77 96Z" fill="#BBD2E9" stroke="${navy}" stroke-width="2"/>
          <rect x="8" y="14" width="69" height="82" rx="2" fill="${pale}" stroke="${navy}" stroke-width="2"/>
          ${[23, 40, 57, 74].map((sy) => `<rect x="14" y="${sy}" width="57" height="13" rx="2" fill="white" stroke="${blue}" stroke-width="1.5"/><path d="M21 ${sy + 4}H46M21 ${sy + 8}H46" stroke="${navy}" stroke-width="1.5"/><circle cx="61" cy="${sy + 6}" r="2" fill="${cyan}"/>`).join("")}
          <path d="M18 91H66" stroke="${navy}" stroke-width="2"/>`;
        break;
      case "cryostat":
        body = `<path d="M19 10H81M26 10V79M74 10V79" fill="none" stroke="#91A7BB" stroke-width="2"/>
          <ellipse cx="50" cy="12" rx="35" ry="8" fill="#DDE7EE" stroke="${navy}" stroke-width="2"/>
          <path d="M15 12V18Q50 32 85 18V12" fill="#B8C9D6" stroke="${navy}" stroke-width="2"/>
          ${[
            [34, 28],
            [53, 22],
            [71, 16],
          ]
            .map(
              ([sy, rx]) =>
                `<path d="M${50 - rx} ${sy - 5}V${sy + 4}Q50 ${sy + 13} ${50 + rx} ${sy + 4}V${sy - 5}" fill="#C99857" stroke="${gold}" stroke-width="1.5"/><ellipse cx="50" cy="${sy - 5}" rx="${rx}" ry="6" fill="#F2D8AE" stroke="${gold}" stroke-width="1.5"/>`,
            )
            .join("")}
          ${[-12, -4, 4, 12].map((dx) => `<path d="M${50 + dx} 20C${42 + dx} 34 ${58 + dx} 43 ${50 + dx} 57S${42 + dx} 76 ${50 + dx} 87" fill="none" stroke="${gold}" stroke-width="1.7"/>`).join("")}
          <rect x="43" y="85" width="14" height="10" rx="2" fill="${navy}"/><path d="M47 90H53" stroke="${cyan}" stroke-width="2"/>`;
        break;
      case "iontrap":
        body = `<path d="M9 48L36 15L94 48L66 82Z" fill="#D3E2EF" stroke="${navy}" stroke-width="2"/>
          <path d="M9 48V57L66 90L94 57V48M66 82V90" fill="#AFC7DD" stroke="${navy}" stroke-width="2"/>
          ${[0, 1, 2, 3, 4, 5].map((n) => `<path d="M${24 + n * 8} ${34 + n * 4.6}L${35 + n * 8} ${21 + n * 4.6}M${20 + n * 8} ${46 + n * 4.6}L${10 + n * 8} ${59 + n * 4.6}" stroke="${gold}" stroke-width="4"/>`).join("")}
          <path d="M23 40L79 73" stroke="white" stroke-width="7"/>
          ${[0, 1, 2, 3, 4].map((n) => `<circle cx="${31 + n * 9}" cy="${45 + n * 5.3}" r="5" fill="${blue}" opacity=".2"/><circle cx="${31 + n * 9}" cy="${45 + n * 5.3}" r="2.2" fill="${blue}"/>`).join("")}
          <path d="M2 25L84 75" stroke="${cyan}" stroke-width="2" opacity=".65"/>`;
        break;
      case "neutralatom":
        body = `<path d="M12 75L49 96L90 72L53 51Z" fill="${pale}" stroke="${navy}" stroke-width="1.5"/>
          <ellipse cx="51" cy="22" rx="34" ry="12" fill="${pale}" stroke="${blue}" stroke-width="2"/>
          <ellipse cx="51" cy="22" rx="23" ry="7" fill="white" stroke="${blue}"/>
          <path d="M22 29L31 53M80 29L71 53" stroke="${cyan}" stroke-width="1.5" stroke-dasharray="3 3"/>
          ${[0, 1, 2, 3].flatMap((row) => [0, 1, 2, 3].map((col) => `<circle cx="${26 + col * 11 + row * 5}" cy="${63 + row * 6 - col * 3}" r="5" fill="${blue}" opacity=".14"/><circle cx="${26 + col * 11 + row * 5}" cy="${63 + row * 6 - col * 3}" r="2.5" fill="${blue}"/>`)).join("")}`;
        break;
      case "chip":
        body = `<rect x="24" y="24" width="52" height="52" rx="7" fill="${pale}" stroke="${navy}" stroke-width="3"/>
          ${[32, 44, 56, 68].map((p) => `<path d="M${p} 13V24M${p} 76V87M13 ${p}H24M76 ${p}H87" stroke="${navy}" stroke-width="3"/>`).join("")}
          <ellipse cx="50" cy="50" rx="22" ry="9" fill="none" stroke="${blue}" stroke-width="2"/>
          <ellipse cx="50" cy="50" rx="22" ry="9" transform="rotate(60 50 50)" fill="none" stroke="${blue}" stroke-width="2"/>
          <ellipse cx="50" cy="50" rx="22" ry="9" transform="rotate(120 50 50)" fill="none" stroke="${blue}" stroke-width="2"/><circle cx="50" cy="50" r="4" fill="${blue}"/>`;
        break;
      case "photonics":
        body = `<path d="M9 38L42 17L93 47L61 70Z" fill="${pale}" stroke="${navy}" stroke-width="2"/>
          <path d="M9 38V50L61 82L93 59V47M61 70V82" fill="#BED6EB" stroke="${navy}" stroke-width="2"/>
          <path d="M19 38C34 29 45 42 55 45S72 43 84 49M17 47C29 39 41 52 52 55S69 54 82 60" fill="none" stroke="${blue}" stroke-width="3"/>
          <path d="M4 25L30 41M4 41L22 51M71 57L97 74" stroke="${cyan}" stroke-width="3"/>
          <circle cx="42" cy="42" r="4" fill="${cyan}"/><circle cx="66" cy="53" r="4" fill="${cyan}"/>`;
        break;
      case "spin":
        body = `<path d="M9 36L41 17L92 47L60 68Z" fill="${pale}" stroke="${navy}" stroke-width="2"/>
          <path d="M9 36V51L60 82L92 62V47M60 68V82" fill="#BED6EB" stroke="${navy}" stroke-width="2"/>
          ${[0, 1, 2].map((n) => `<path d="M${28 + n * 13} ${28 + n * 8}L${48 + n * 13} ${40 + n * 8}L${36 + n * 13} ${48 + n * 8}L${16 + n * 13} ${36 + n * 8}Z" fill="${gold}"/><circle cx="${38 + n * 11}" cy="${42 + n * 6}" r="4" fill="${blue}"/>`).join("")}
          <path d="M40 38V17M36 23L40 17L44 23M65 52V30M61 36L65 30L69 36" fill="none" stroke="${blue}" stroke-width="2"/>`;
        break;
      case "cloud":
        body = `<path d="M22 77C-1 76-2 42 21 40C16 11 55 2 67 28C96 21 111 63 86 76Z" fill="${pale}" stroke="${blue}" stroke-width="2"/>`;
        break;
      case "measurement":
        body = `<path d="M13 74A38 38 0 0 1 87 74" fill="${pale}" stroke="${navy}" stroke-width="3"/>
          <path d="M50 72L74 36M22 61L29 64M34 39L38 46M59 35L57 43M79 57L72 61" stroke="${blue}" stroke-width="3" stroke-linecap="round"/><circle cx="50" cy="72" r="5" fill="${blue}"/>`;
        break;
      case "circuit":
        body = `<path d="M6 29H94M6 51H94M6 73H94M63 29V74" fill="none" stroke="${navy}" stroke-width="2"/><rect x="22" y="17" width="23" height="23" rx="3" fill="${pale}" stroke="${blue}" stroke-width="2"/><path d="M28 22V35M39 22V35M28 28H39" stroke="${blue}" stroke-width="2"/><circle cx="63" cy="29" r="4" fill="${blue}"/><circle cx="63" cy="51" r="9" fill="white" stroke="${blue}" stroke-width="2"/><path d="M57 51H69M63 45V57" stroke="${blue}" stroke-width="2"/>`;
        break;
      case "molecule":
      case "orbitals":
        body = `<ellipse cx="35" cy="47" rx="29" ry="34" transform="rotate(-25 35 47)" fill="${blue}" opacity=".12"/><ellipse cx="65" cy="47" rx="29" ry="34" transform="rotate(25 65 47)" fill="${cyan}" opacity=".2"/><path d="M33 52L67 39" stroke="${navy}" stroke-width="5"/><circle cx="33" cy="52" r="18" fill="${blue}"/><circle cx="67" cy="39" r="11" fill="${cyan}"/><circle cx="29" cy="47" r="4" fill="white" opacity=".7"/><circle cx="63" cy="34" r="4" fill="white" opacity=".7"/>`;
        break;
      default:
        throw new Error(`Unknown MQSF illustration: ${name}`);
    }
    return `<g transform="translate(${x} ${y}) scale(${size / 100})" aria-hidden="true">${body}</g>`;
  }

  function scene(name, label, body) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 650" role="img" aria-label="${label}" class="keynote-illustration" style="font-family:Inter,Arial,sans-serif;fill:${navy}">
      <defs><marker id="${name}-arrow" markerWidth="11" markerHeight="9" refX="9" refY="4.5" orient="auto"><path d="M0 0L10 4.5L0 9" fill="${blue}"/></marker></defs>
      <rect width="1600" height="650" fill="white"/>${body}</svg>`;
  }

  const keyed = (
    key,
    body,
    transform = "translate(0 0) scale(1)",
    opacity = 1,
  ) =>
    `<g data-morph="${key}" transform="${transform}" opacity="${opacity}">${body}</g>`;

  const architecturePhases = {
    problem: [
      "Quantum processors",
      "Researchers",
      "Classical compute",
      "One quantum system",
      "A second system",
      "Every modality",
      "Cloud compute",
      "HPC",
      "Every user",
    ],
    stack: [
      "A shared stack",
      "Qiskit",
      "Program and runtime",
      "One interface",
      "Quantum devices",
      "Classical resources",
      "OpenQASM",
      "PennyLane and more",
    ],
    detail: [
      "The QDMI standard",
      "QDMI on IQM",
      "Amazon Braket",
      "MQT DDSIM",
      "IBM Quantum and more",
      "Compiler Collection",
      "Orchestrator",
      "Our principles",
    ],
  };

  function qdmiBadge(x = 0, y = 0, width = 190, label = "Powered by") {
    return `<g transform="translate(${x} ${y}) scale(${width / 190})" role="img" aria-label="${label} QDMI"><rect width="190" height="39" rx="19.5" fill="white" stroke="#8DB4D8" stroke-width="1.5"/>${text(13, 25, label, 15, "start", `fill="${blue}"`)}${logo("qdmi", 100, 7, 78, 26)}</g>`;
  }

  // Click phases change targets only; the presentation owns all transition timing.
  function architecture(mode = "stack", phase = Infinity) {
    const problem = mode === "problem";
    const detail = ["detail", "ecosystem", "core"].includes(mode);
    const closing = mode === "ecosystem" || mode === "core";
    const phases = architecturePhases[mode] || architecturePhases.detail;
    const p = Math.max(
      0,
      Math.min(phases.length - 1, Math.floor(phase ?? Infinity)),
    );
    const on = (step) => Number(p >= step);
    const stack = (step) => Number(!problem && (detail || p >= step));
    const product = (step) => Number(detail && (closing || p >= step));
    const physical = 1 - product(1);
    const users = [
      { name: "Researchers", icon: "researcher", badge: "molecule", y: 114 },
      { name: "Experimentalists", icon: "researcher", badge: "chip", y: 267 },
      {
        name: "Software developers",
        icon: "developer",
        badge: "circuit",
        y: 420,
      },
      { name: "End users", icon: "user", badge: "measurement", y: 573 },
    ];
    const computes = [
      { name: "Workstation", kind: "workstation", y: 128 },
      { name: "Cloud compute", kind: "cloud", y: 332 },
      { name: "HPC", kind: "server", y: 536 },
    ].map((compute, i) => ({
      ...compute,
      x: problem ? 650 : 443 + i * 278,
      y: problem ? compute.y : 599,
    }));
    const qpus = [
      { kind: "cryostat", name: "Superconducting", y: 130 },
      { kind: "iontrap", name: "Trapped ions", y: 232 },
      { kind: "neutralatom", name: "Neutral atoms", y: 334 },
      { kind: "photonics", name: "Photonics", y: 436 },
      { kind: "spin", name: "Spin qubits", y: 538 },
    ];
    const arrow = 'marker-end="url(#architecture-arrow)"';
    const route = (key, d, visible, extra = "", color = blue, width = 3) =>
      path(
        d,
        color,
        width,
        `data-morph="${key}" opacity="${visible}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${visible ? 0 : 1}" ${extra}`,
      );
    const module = (key, x, y, width, height, body, visible) =>
      keyed(
        key,
        `<rect width="${width}" height="${height}" rx="16" fill="#F4F8FC" stroke="#A9C6E1" stroke-width="2"/>${body}`,
        `translate(${x} ${y}) scale(1)`,
        visible,
      );
    const workstation = `<rect x="5" y="6" width="92" height="65" rx="5" fill="${pale}" stroke="${navy}" stroke-width="2"/><rect x="13" y="14" width="76" height="47" rx="2" fill="white" stroke="${blue}"/><path d="M41 72V85H62V72M29 87H74" fill="none" stroke="${navy}" stroke-width="3"/><path d="M23 49L38 32L51 43L71 24" fill="none" stroke="${cyan}" stroke-width="3"/>`;

    const rawRoutes =
      users
        .flatMap((user, u) =>
          computes.map((compute, c) => {
            const visible = Number(problem && p >= (u ? 8 : [3, 6, 7][c]));
            return route(
              `world-user-compute-${u}-${c}`,
              `M236 ${user.y}C350 ${user.y} 462 ${compute.y} ${compute.x - 72} ${compute.y}`,
              visible * 0.6,
              `data-route="user-classical" data-from="user-${u}" data-to="compute-${c}"`,
              c === 2 ? "#16899D" : blue,
            );
          }),
        )
        .join("") +
      computes
        .flatMap((compute, c) =>
          qpus.map((qpu, q) => {
            const visible = Number(
              problem && p >= (c === 0 ? (q < 2 ? 3 + q : 5) : c === 1 ? 6 : 7),
            );
            return route(
              `world-compute-qpu-${c}-${q}`,
              `M${compute.x + 72} ${compute.y}C860 ${compute.y} 1055 ${qpu.y} 1203 ${qpu.y}`,
              visible * 0.58,
              `data-route="classical-qpu" data-from="compute-${c}" data-to="qpu-${q}" data-hosting="${c === 2 ? "cloud-or-on-premises" : "cloud"}"`,
              c === 2 ? "#16899D" : blue,
            );
          }),
        )
        .join("");

    const people = users
      .map((user, i) =>
        keyed(
          `world-user-${i}`,
          icon(user.icon, 61, 0, 87) +
            icon(user.badge, 139, 38, 36) +
            text(118, 115, user.name, 25),
          `translate(10 ${user.y - 56}) scale(1)`,
          problem ? on(i ? 8 : 1) : 1,
        ),
      )
      .join("");
    const classical = computes
      .map((compute, i) =>
        keyed(
          `world-compute-${i}`,
          (i === 0
            ? keyed("world-workstation-symbol", workstation)
            : i === 1
              ? icon("cloud", 0, 0, 102)
              : icon("server", -9, 0, 88) + icon("server", 42, 0, 88)) +
            text(51, 129, compute.name, problem ? 28 : 39),
          `translate(${compute.x - (problem ? 51 : 30.6)} ${compute.y - (problem ? 45 : 27)}) scale(${problem ? 1 : 0.6})`,
          problem ? on(2) : 1,
        ),
      )
      .join("");
    const devices = qpus
      .map((qpu, i) =>
        keyed(
          `world-qpu-${i}`,
          icon(qpu.kind, 0, 0, 80) + text(98, 48, qpu.name, 27, "start"),
          `translate(1209 ${qpu.y - 37}) scale(1)`,
          physical,
        ),
      )
      .join("");
    const headings =
      keyed(
        "world-column-labels",
        text(128, 25, "Users", 28) + text(650, 25, "Classical compute", 28),
        "translate(0 0) scale(1)",
        Number(problem),
      ) +
      keyed(
        "world-device-label",
        text(1389, 25, "Quantum processors", 28) +
          text(
            1389,
            65,
            "Cloud-hosted or on-premises",
            21,
            "middle",
            'fill="#5B7088"',
          ),
        "translate(0 0) scale(1)",
        physical,
      ) +
      keyed(
        "world-cloud-access",
        text(858, 76, "Cloud access", 23, "middle", `fill="${blue}"`),
        "translate(0 0) scale(1)",
        Number(problem && p >= 3),
      ) +
      keyed(
        "world-hpc-access",
        text(
          960,
          626,
          "HPC also reaches on-premises QPUs",
          22,
          "middle",
          'fill="#16899D"',
        ),
        "translate(0 0) scale(1)",
        Number(problem && p >= 7),
      );
    const userInput = users
      .map((user, i) =>
        route(
          `world-user-frontend-${i}`,
          `M236 ${user.y}C275 ${user.y} 272 81 305 81`,
          stack(1) * 0.45,
          `${arrow} data-route="user-classical"`,
          "#739BBE",
          2.5,
        ),
      )
      .join("");
    const deviceOutput = qpus
      .map((qpu, i) =>
        route(
          `world-backend-qpu-${i}`,
          `M1148 503C1186 503 1162 ${qpu.y} 1203 ${qpu.y}`,
          stack(4) * physical * 0.65,
          `${arrow} data-route="classical-qpu"`,
        ),
      )
      .join("");
    const hosting = route(
      "hosting-bus",
      "M443 576V563H1127V409M721 576V563M999 576V563",
      stack(5) * 0.7,
      `${arrow} data-flow="classical-hosting"`,
      "#739BBE",
      2.5,
    );

    const frontendCards =
      [
        { name: "Qiskit", brand: "qiskit", x: 305, reveal: 1 },
        { name: "OpenQASM 3", x: 594, reveal: 6 },
        { name: "PennyLane", brand: "pennylane", x: 883, reveal: 7 },
      ]
        .map((frontend, i) =>
          keyed(
            `frontend-${i}`,
            `<rect width="244" height="76" rx="12" fill="white" stroke="#A9C6E1" stroke-width="2"/>` +
              (frontend.brand === "pennylane"
                ? logo(frontend.brand, 22, 25, 200, 26)
                : frontend.brand
                  ? logo(frontend.brand, 19, 16, 43, 43) +
                    text(155, 48, frontend.name, 29)
                  : text(122, 48, frontend.name, 29)),
            `translate(${frontend.x} 43) scale(1)`,
            stack(frontend.reveal),
          ),
        )
        .join("") +
      keyed(
        "frontend-more",
        text(1166, 87, "…", 43),
        "translate(0 0) scale(1)",
        stack(7),
      );
    const frontendFlow =
      [427, 716, 1005]
        .map((x, i) =>
          route(
            `frontend-junction-${i}`,
            `M${x} 119V150H716`,
            stack([2, 6, 7][i]),
            `data-lane="program-runtime"`,
          ),
        )
        .join("") +
      route(
        "frontend-program",
        "M716 150V166H497V213",
        stack(2),
        `${arrow} data-lane="program"`,
      ) +
      route(
        "frontend-runtime",
        "M716 150V166H942V213",
        stack(2),
        `${arrow} data-lane="runtime"`,
        "#16899D",
      ) +
      keyed(
        "frontend-lane-labels",
        `<rect x="443" y="181" width="108" height="28" fill="white"/><rect x="886" y="181" width="112" height="28" fill="white"/>` +
          text(497, 203, "Program", 24, "middle", `fill="${blue}"`) +
          text(942, 203, "Runtime", 24, "middle", 'fill="#16899D"'),
        "translate(0 0) scale(1)",
        stack(2),
      );

    const compilerPanel = module(
      "stack-compiler",
      305,
      220,
      399,
      178,
      keyed(
        "stack-compiler-generic",
        text(199.5, 45, "Compiler", 30) +
          text(199.5, 82, "Infrastructure", 30) +
          text(
            199.5,
            133,
            "Optimize · Map · Synthesize",
            25,
            "middle",
            'fill="#5B7088"',
          ),
        "translate(0 0) scale(1)",
        1 - product(5),
      ) +
        keyed(
          "stack-compiler-product",
          logo("mqt", 20, 25, 126, 58) +
            text(272, 50, "Compiler", 29) +
            text(272, 86, "Collection", 29) +
            logo("mlir", 23, 122, 73, 39) +
            logo("llvm", 113, 120, 72, 43) +
            logo("qir", 206, 120, 64, 43) +
            logo("jeff", 291, 116, 83, 47),
          "translate(0 0) scale(1)",
          product(5),
        ),
      stack(2),
    );
    const orchestratorPanel = module(
      "stack-orchestrator",
      755,
      220,
      393,
      178,
      keyed(
        "stack-orchestrator-generic",
        text(196.5, 45, "Resource Management", 28) +
          text(196.5, 82, "& Orchestration", 30) +
          text(
            196.5,
            133,
            "Schedule · Execute · Collect",
            25,
            "middle",
            'fill="#5B7088"',
          ),
        "translate(0 0) scale(1)",
        1 - product(6),
      ) +
        keyed(
          "stack-orchestrator-product",
          logo("mqsc", 21, 25, 106, 67) +
            text(263, 65, "Orchestrator", 29) +
            logo("slurm", 38, 115, 82, 46) +
            qdmiBadge(172, 118, 194, "Built on"),
          "translate(0 0) scale(1)",
          product(6),
        ),
      stack(2),
    );
    const compilerFlow =
      route(
        "compile-request",
        "M755 272H709",
        stack(2),
        `${arrow} data-flow="compile-request"`,
        "#16899D",
        2.5,
      ) +
      route(
        "compiled-program",
        "M704 348H750",
        stack(2),
        `${arrow} data-flow="compiled-program"`,
        blue,
        2.5,
      );
    const backend = module(
      "stack-backends",
      305,
      468,
      843,
      70,
      keyed(
        "stack-backend-symbol",
        path(
          "M22 18L43 35L22 52M43 35L64 18M43 35L64 52M43 8V35M43 35V62",
          blue,
          2.5,
        ) +
          [
            [22, 18],
            [22, 52],
            [64, 18],
            [64, 52],
            [43, 8],
            [43, 62],
          ]
            .map(
              ([x, y]) =>
                `<circle cx="${x}" cy="${y}" r="3.5" fill="${blue}"/>`,
            )
            .join(""),
      ) +
        keyed(
          "stack-backend-generic",
          text(468, 45, "Backend Interface", 30),
          "translate(0 0) scale(1)",
          1 - product(0),
        ) +
        keyed(
          "stack-backend-standard",
          text(468, 44, "Quantum Device Management Interface (QDMI)", 28),
          "translate(0 0) scale(1)",
          product(0),
        ),
      stack(3),
    );
    const backendFlow =
      route(
        "device-capabilities",
        "M408 468V402",
        stack(3),
        `${arrow} data-flow="device-capabilities"`,
        blue,
        2.5,
      ) +
      route(
        "job-submission",
        "M856 398V462",
        stack(3),
        `${arrow} data-flow="submission"`,
        "#16899D",
        2.5,
      ) +
      route(
        "execution-results",
        "M1060 468V402",
        stack(3),
        `${arrow} data-flow="results"`,
        "#16899D",
        2.5,
      ) +
      keyed(
        "stack-backend-labels",
        `<rect x="316" y="422" width="184" height="26" fill="white"/><rect x="808" y="422" width="96" height="26" fill="white"/><rect x="1010" y="422" width="100" height="26" fill="white"/>` +
          text(408, 442, "Device capabilities", 21) +
          text(856, 442, "Submit", 21) +
          text(1060, 442, "Results", 21),
        "translate(0 0) scale(1)",
        stack(3),
      );

    const adapters = [
      ["iqm", "QDMI on IQM", "IQM quantum computers"],
      ["aws", "Amazon Braket", "Cloud quantum devices"],
      ["mqt", "MQT DDSIM", "Classical quantum circuit simulation"],
      ["ibm", "IBM Quantum", "IBM quantum devices"],
    ];
    const adapterPaths = adapters
      .map((_, i) =>
        route(
          `backend-adapter-${i}`,
          `M1148 503C1188 503 1173 ${99 + i * 130} 1212 ${99 + i * 130}`,
          product(i + 1) * 0.6,
          `${arrow} data-route="classical-qpu"`,
        ),
      )
      .join("");
    const adapterCards = adapters
      .map(([brand, name, description], i) =>
        keyed(
          `world-adapter-${i}`,
          `<rect width="372" height="112" rx="14" fill="#F7FAFD" stroke="#A9C6E1" stroke-width="2"/>` +
            logo(brand, 15, 14, 88, 42) +
            text(235, 43, name, 26) +
            text(186, 75, description, i === 2 ? 20 : 22) +
            qdmiBadge(104, 85, 164, "Powered by"),
          `translate(1215 ${40 + i * 130}) scale(1)`,
          product(i + 1),
        ),
      )
      .join("");
    const adapterFuture = keyed(
      "world-adapter-future",
      `<rect x="1215" y="570" width="372" height="60" rx="14" fill="white" stroke="#A9C6E1" stroke-width="2" stroke-dasharray="5 7"/>` +
        text(1401, 608, "Your device could be next", 24),
      "translate(0 0) scale(1)",
      product(4),
    );
    const principles = keyed(
      "stack-principles",
      text(
        726,
        20,
        "Open by Design · Community-Driven · No Lock-Ins",
        24,
        "middle",
        `fill="${blue}"`,
      ),
      "translate(0 0) scale(1)",
      product(7),
    );
    return scene(
      "architecture",
      "Users and classical resources connect to quantum systems through compiler infrastructure, resource orchestration, and the open Quantum Device Management Interface standard. Workstations access cloud-hosted systems; HPC also reaches on-premises devices.",
      rawRoutes +
        userInput +
        deviceOutput +
        adapterPaths +
        hosting +
        headings +
        people +
        classical +
        devices +
        frontendCards +
        frontendFlow +
        compilerPanel +
        orchestratorPanel +
        compilerFlow +
        backend +
        backendFlow +
        adapterCards +
        adapterFuture +
        principles,
    );
  }

  window.MQSF_VIZ = {
    architecture,
    architecturePhases,
    qdmiBadge,
  };
})();
