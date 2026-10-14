# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Measure missing main commits on a dedicated Linux machine and publish results."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import operator
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from subprocess import run as run_command
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

from benchmarks.dashboard import METRICS, read_results

GIT = shutil.which("git") or "git"
LOGGER = logging.getLogger(__name__)
TIMING = ("rounds=4", "repeat=5", "warmup_time=0.1")


def git(*args: str) -> str:
    """Run a local Git operation without interpreting arguments as shell code.

    Returns:
        The command's standard output.
    """
    return subprocess.check_output([GIT, *args], text=True).strip()  # ruff: ignore[subprocess-without-shell-equals-true] -- argument list, no shell


def fetch_results(destination: Path) -> None:
    """Import data-only history, preserving the snapshot before its first run.

    Raises:
        ValueError: If the data branch contains files other than regular JSON.
    """
    if not git("ls-remote", "--heads", "origin", "refs/heads/benchmark_results"):
        LOGGER.info("No benchmark_results branch yet; keeping the saved snapshot.")
        return
    git("fetch", "origin", "refs/heads/benchmark_results")
    archive = subprocess.check_output([GIT, "archive", "FETCH_HEAD", "results/"])  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed Git arguments
    with tempfile.TemporaryDirectory() as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as data:
            if any(
                not member.isdir() and (not member.isfile() or not member.name.endswith(".json")) for member in data
            ):
                msg = "Benchmark history must contain only regular JSON files."
                raise ValueError(msg)
            data.extractall(directory, filter="data")
        source = Path(directory) / "results"
        read_results(source)
        for path in source.rglob("*.json"):
            target = destination / path.relative_to(source)
            if target.exists() and path.parent != source and path.read_bytes() != target.read_bytes():
                msg = f"Imported history conflicts with local measurement: {path.relative_to(source)}"
                raise ValueError(msg)
        shutil.copytree(source, destination, dirs_exist_ok=True)
    LOGGER.info("Imported benchmark history into %s", destination)


def select_revisions(results: Path, baseline: str | None, candidate: str = "HEAD") -> tuple[str, str]:
    """Resolve a candidate and an ancestor baseline from this checkout.

    Returns:
        Full baseline and candidate commit IDs. Repeated candidates reuse the
        previous pair, so a manual rerun still compares distinct revisions.

    Raises:
        ValueError: If the baseline equals or is not an ancestor of the candidate.
    """
    candidate = git("rev-parse", "--verify", "--end-of-options", f"{candidate}^{{commit}}")
    if not baseline:
        latest = results / "latest.json"
        previous = json.loads(latest.read_text()) if latest.exists() else {}
        baseline = previous.get("candidate", "HEAD^")
        if baseline == candidate:
            baseline = previous["baseline"]
    baseline = git("rev-parse", "--verify", "--end-of-options", f"{baseline}^{{commit}}")
    if baseline == candidate or subprocess.call([GIT, "merge-base", "--is-ancestor", baseline, candidate]):  # ruff: ignore[subprocess-without-shell-equals-true] -- resolved commit IDs
        msg = "The baseline must be a distinct ancestor of the candidate."
        raise ValueError(msg)
    return baseline, candidate


def configuration(host: str, cpu: int) -> dict[str, Any]:
    """Identify the machine, toolchain, benchmark suite, and measurement controls.

    Returns:
        The stable configuration used to separate trend series.

    Raises:
        RuntimeError: If Linux affinity is unavailable.
        ValueError: If the machine label or CPU is invalid.
    """
    from asv.machine import Machine  # ruff: ignore[import-outside-top-level] -- RTD import needs only stdlib

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None:
        msg = "Dedicated benchmarks require Linux CPU affinity support."
        raise RuntimeError(msg)
    if cpu not in affinity(0):
        msg = f"CPU {cpu} is not available to this process."
        raise ValueError(msg)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", host):
        msg = "Use a machine label of up to 64 letters, digits, dots, underscores, or hyphens."
        raise ValueError(msg)
    llvm_config = Path(os.environ["MLIR_DIR"]).resolve().parents[2] / "bin/llvm-config"
    config = json.loads(Path("asv.conf.json").read_text(encoding="utf-8"))
    hardware = Machine.get_defaults()
    power = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    settings = {}
    for name in (
        "scaling_driver",
        "scaling_governor",
        "scaling_min_freq",
        "scaling_max_freq",
        "energy_performance_preference",
    ):
        path = power / name
        if path.is_file():
            settings[name] = path.read_text().strip()
    return {
        "host": host,
        "cpu_affinity": [cpu],
        "cpu_settings": settings,
        "hardware": {key: hardware[key] for key in ("cpu", "num_cpu", "ram")},
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "llvm": version(str(llvm_config)),
        "compiler": version(os.environ["CXX"]),
        "c_compiler": version(os.environ["CC"]),
        "asv": importlib.metadata.version("asv"),
        "uv": version(shutil.which("uv") or "uv"),
        "benchmark_version": hashlib.sha256(Path(__file__).with_name("compiler.py").read_bytes()).hexdigest(),
        "build": {key: config.get(key) for key in ("matrix", "build_command", "environment_type", "launch_method")},
        "build_environment": {
            key: os.environ.get(key, "")
            for key in (
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "CMAKE_ARGS",
                "SKBUILD_CMAKE_ARGS",
            )
        },
        "timing": TIMING,
    }


def series_id(config: dict[str, Any], reference: str) -> str:
    """Return a stable series identity without timestamps or candidate revisions."""
    return hashlib.sha256(json.dumps([config, reference], sort_keys=True).encode()).hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace a checkpoint atomically so interruptions cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare(
    results: Path,
    baseline: str | None,
    host: str,
    cpu: int,
    *,
    candidate: str = "HEAD",
    reference_only: bool = False,
) -> dict[str, Any]:
    """Prepare a unique comparison or reference-only observation.

    Returns:
        Provenance, including the stable series and generator identities.
    """
    from asv.machine import Machine, MachineCollection  # ruff: ignore[import-outside-top-level] -- RTD import needs only stdlib

    config = configuration(host, cpu)
    if reference_only:
        base = candidate = git("rev-parse", "--verify", "--end-of-options", f"{baseline}^{{commit}}")
    else:
        base, candidate = select_revisions(results, baseline, candidate)
    created = datetime.now(UTC)
    run_id = f"{created:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    machine = f"{host}-{run_id}"
    inputs = {
        commit: hashlib.sha256(
            git("ls-tree", "-r", commit, "--", "include/mqt-core/bench", "src/bench", "bindings/bench").encode()
        ).hexdigest()
        for commit in {base, candidate}
    }
    metadata: dict[str, Any] = config | {
        "machine": machine,
        "baseline": base,
        "candidate": candidate,
        "run_id": run_id,
        "created_at": created.isoformat(),
        "kind": "reference" if reference_only else "comparison",
        "series": series_id(config, base),
        "input_versions": inputs,
        "generator_sources_changed": inputs[base] != inputs[candidate],
        "harness_commit": git("rev-parse", "HEAD"),
    }
    MachineCollection.save(machine, Machine.get_defaults() | {"machine": machine})
    state = Path(".asv")
    write_json(state / "run.json", metadata)
    (state / "commits.txt").write_text("\n".join(dict.fromkeys((base, candidate))) + "\n")
    asv_config = json.loads(Path("asv.conf.json").read_text(encoding="utf-8"))
    for name in ("repo", "benchmark_dir", "env_dir", "html_dir"):
        asv_config[name] = str(Path(asv_config[name]).resolve())
    # asv caches built wheels inside each environment; toolchain changes need fresh ones.
    asv_config["env_dir"] = str(Path(asv_config["env_dir"]) / metadata["series"])
    asv_config.update(results_dir=str(results.resolve()), pythons=[platform.python_version()])
    write_json(state / "runner.conf.json", asv_config)
    return metadata


def version(executable: str) -> str:
    """Return the first version line from a locally configured tool."""
    return subprocess.check_output([executable, "--version"], text=True).splitlines()[0]  # ruff: ignore[subprocess-without-shell-equals-true] -- local tool, no shell


def finalize(results: Path, metadata: dict[str, Any], *, save: bool = True) -> str:
    """Validate a complete comparison before making it the published baseline.

    Returns:
        A Markdown summary with relative timings and circuit quality changes.

    Raises:
        ValueError: If revisions, environments, cases, metrics, or samples do not match.
    """
    machine = metadata["machine"]
    runs = [run for run in read_results(results)["runs"] if run["machine"] == machine]
    by_commit = {run["commit"]: run for run in runs}
    expected_commits = {metadata["baseline"], metadata["candidate"]}
    expected_count = 1 if metadata.get("kind") == "reference" else 2
    if len(runs) != expected_count or set(by_commit) != expected_commits:
        msg = "Expected exactly the baseline and candidate in this measurement group."
        raise ValueError(msg)
    base, candidate = by_commit[metadata["baseline"]], by_commit[metadata["candidate"]]
    case = operator.itemgetter("circuit", "qubits", "topology")
    before = {case(row): row for row in base["rows"]}
    definitions = results / machine / "benchmarks.json"
    if not definitions.exists():
        definitions = results / "benchmarks.json"
    definition = json.loads(definitions.read_text())[METRICS[0]["benchmark"]]
    expected = math.prod(map(len, definition["params"]))
    if (
        base["environment"] != candidate["environment"]
        or len(before) != expected
        or set(before) != {case(row) for row in candidate["rows"]}
    ):
        msg = "The comparison must use the same environment and circuit cases."
        raise ValueError(msg)
    lines = [
        "## Compiler benchmark comparison",
        "",
        f"Baseline `{base['commit']}` → candidate `{candidate['commit']}`.",
        (
            "Both revisions were measured together on the same machine. "
            "Timing changes are informational; no regression gate is applied."
        ),
        "",
        "| Circuit | Qubits | Target | Baseline ms | Candidate ms | Time change | 2Q change | Depth change |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidate["rows"]:
        old = before[case(row)]
        for metric in METRICS:
            left, right = old[metric["id"]], row[metric["id"]]
            if (
                left["status"] != "ok"
                or right["status"] != "ok"
                or not left["version"]
                or left["version"] != right["version"]
                or not left["current_suite"]
                or not right["current_suite"]
            ):
                msg = "Incomplete measurements or different benchmark suites; refusing to publish."
                raise ValueError(msg)
        for timing in (old["time"], row["time"]):
            samples = timing["samples"] or []
            if len(samples) != 20 or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
                for value in samples
            ):
                msg = "Expected 20 finite positive timing samples from four rounds of five repeats."
                raise ValueError(msg)
        a, b = old["time"]["value"], row["time"]["value"]
        delta = f"{(b / a - 1) * 100:+.1f}%" if a else "unavailable"
        gates = f"{row['two_qubit_count']['value'] - old['two_qubit_count']['value']:+g}"
        depth = f"{row['depth']['value'] - old['depth']['value']:+g}"
        if metadata["generator_sources_changed"]:
            delta = gates = depth = "different inputs"
        lines.append(
            f"| {row['circuit']} | {row['qubits']} | {row['topology']} "
            f"| {a * 1000:.3f} | {b * 1000:.3f} | {delta} | {gates} | {depth} |"
        )
    if metadata["generator_sources_changed"]:
        lines.extend(["", "Core benchmark generator sources differ; changes can include different compiler inputs."])
    if save:
        machine_path = results / machine / "machine.json"
        info = json.loads(machine_path.read_text())
        info["measurement"] = metadata
        snapshot = machine_path.with_name("benchmarks.json")
        if not snapshot.exists():
            shutil.copyfile(results / "benchmarks.json", snapshot)
        write_json(machine_path, info)
        write_json(results / "latest.json", metadata)
    return "\n".join(lines) + "\n"


def completed_measurements(results: Path) -> dict[str, dict[str, Any]]:
    """Return durable completion records, excluding partial asv output."""
    return {
        path.parent.name: info["measurement"]
        for path in results.glob("*/machine.json")
        if (info := json.loads(path.read_text())).get("measurement")
    }


def publish(results: Path) -> None:
    """Append all complete comparisons and finished batch records, then refresh RTD.

    Raises:
        ValueError: If an existing measurement or batch would change.
    """
    groups = completed_measurements(results)
    if not groups:
        msg = "No complete measurements to publish."
        raise ValueError(msg)
    for machine, metadata in groups.items():
        if metadata["machine"] != machine or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", machine):
            msg = "Invalid measurement group."
            raise ValueError(msg)
        finalize(results, metadata, save=False)
    remote = git("remote", "get-url", "--push", "origin")
    with tempfile.TemporaryDirectory() as directory:
        history = Path(directory)

        def data_git(*args: str) -> str:
            return git("-C", directory, *args)

        def copy_immutable(source: Path, target: Path) -> None:
            if target.exists() and source.read_bytes() != target.read_bytes():
                msg = f"Already published with different samples or metadata: {target.name}"
                raise ValueError(msg)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        data_git("init")
        data_git("remote", "add", "origin", remote)
        if data_git("ls-remote", "--heads", "origin", "refs/heads/benchmark_results"):
            data_git("fetch", "--depth=1", "origin", "refs/heads/benchmark_results")
            data_git("checkout", "-b", "benchmark_results", "FETCH_HEAD")
        else:
            data_git("checkout", "--orphan", "benchmark_results")
        destination = history / "results"
        for machine in groups:
            for source in (results / machine).glob("*.json"):
                copy_immutable(source, destination / machine / source.name)
        for source in (results / "batches").glob("*.json"):
            copy_immutable(source, destination / "batches" / source.name)
        latest = max(groups.values(), key=lambda item: (item.get("kind") != "reference", item["created_at"]))
        previous = destination / "latest.json"
        old = json.loads(previous.read_text()) if previous.exists() else {}
        if not old or old["created_at"] <= latest["created_at"]:
            write_json(previous, latest)
            definitions = results / latest["machine"] / "benchmarks.json"
            shutil.copyfile(definitions, destination / "benchmarks.json")
        data_git("add", "results")
        if data_git("diff", "--cached", "--name-only"):
            for key in ("user.name", "user.email"):
                data_git("config", key, git("config", key))
            data_git("commit", "-m", "📊 Append compiler benchmark measurements")
            data_git("push", "origin", "HEAD:refs/heads/benchmark_results")
        LOGGER.info("Published %s complete measurement groups on benchmark_results.", len(groups))
    if token := os.environ.get("RTD_API_TOKEN"):
        request = urllib.request.Request(
            "https://app.readthedocs.org/api/v3/projects/mqt-core/versions/latest/builds/",
            data=b"",
            headers={"Authorization": f"Token {token}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30):  # ruff: ignore[suspicious-url-open-usage] -- fixed HTTPS endpoint
            LOGGER.info("Requested an RTD latest build.")
    else:
        LOGGER.info("Set RTD_API_TOKEN to request a rebuild; the next RTD latest build will import these results.")


@contextlib.contextmanager
def locked_checkout() -> Generator[None, None, None]:
    """Hold one measurement lock and require a clean, reviewed harness checkout.

    Yields:
        Control while the lock is held.

    Raises:
        RuntimeError: If a run is active or the checkout has unsaved changes.
    """
    import fcntl  # ruff: ignore[import-outside-top-level] -- Linux runner only

    state = Path(".asv")
    state.mkdir(exist_ok=True)
    with (state / "runner.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            msg = "A benchmark run is already active in this checkout."
            raise RuntimeError(msg) from error
        if git("status", "--porcelain", "--untracked-files=normal"):
            msg = "Benchmark from a clean dedicated checkout so the suite matches the recorded revision."
            raise RuntimeError(msg)
        yield


def measure(results: Path, metadata: dict[str, Any]) -> None:
    """Run the prepared asv comparison and checkpoint only fully validated results."""
    state = Path(".asv")
    write_json(state / "run.json", metadata)
    cpu = metadata["cpu_affinity"][0]
    environment = os.environ | {"MQT_BENCHMARK_CPU": str(cpu), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    command = [
        sys.executable,
        "-m",
        "asv",
        "--config",
        str(state / "runner.conf.json"),
        "run",
        f"HASHFILE:{state / 'commits.txt'}",
        "--machine",
        metadata["machine"],
        "--cpu-affinity",
        str(cpu),
        "--interleave-rounds",
        "--record-samples",
        "--show-stderr",
    ]
    for attribute in TIMING:
        command.extend(("--attribute", attribute))
    run_command(command, env=environment, check=True)  # ruff: ignore[subprocess-without-shell-equals-true] -- argument list
    summary = finalize(results, metadata)
    (state / "summary.md").write_text(summary)
    sys.stdout.write(summary)


def build_dashboard(results: Path) -> None:
    """Build the saved history without running measurements."""
    run_command(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed module, no shell
        [sys.executable, "-m", "benchmarks.dashboard", "build", "--results", str(results)], check=True
    )


def run(results: Path, baseline: str | None, host: str, cpu: int, *, upload: bool) -> None:
    """Measure one clean checkout and retain complete results before publication."""
    with locked_checkout():
        fetch_results(results)
        metadata = prepare(results, baseline, host, cpu)
        measure(results, metadata)
        build_dashboard(results)
        if upload:
            publish(results)


def main_history(reference: str, start: str) -> list[str]:
    """Select main-line revisions after an inclusive, immutable start commit.

    Returns:
        First-parent revisions from start through the fetched main head.

    Raises:
        ValueError: If either boundary is not a full SHA on main's first-parent history.
    """
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (reference, start)):
        msg = "reference and start must be full lowercase commit SHAs."
        raise ValueError(msg)
    commits = git("rev-list", "--first-parent", "--reverse", "refs/remotes/origin/main").splitlines()
    if reference not in commits or start not in commits or commits.index(reference) > commits.index(start):
        msg = "reference and start must lie on main's first-parent history, with reference at or before start."
        raise ValueError(msg)
    return commits[commits.index(start) :]


def batch(results: Path, settings: dict[str, Any], *, upload: bool) -> None:
    """Measure every missing main commit, retaining successes across retries.

    Raises:
        RuntimeError: If one or more attempts fail after the remaining revisions finish.
    """
    host, cpu, reference = settings["machine"], settings["cpu"], settings["reference"]
    start = settings.get("start", reference)
    with locked_checkout():
        git("fetch", "origin", "refs/heads/main:refs/remotes/origin/main")
        commits = main_history(reference, start)
        profile = configuration(host, cpu)
        series = series_id(profile, reference)
        fetch_results(results)
        done = {
            item["candidate"]
            for item in completed_measurements(results).values()
            if item.get("series") == series and item.get("kind") != "reference"
        }
        first_position = int(git("rev-list", "--first-parent", "--count", start))
        reference_position = int(git("rev-list", "--first-parent", "--count", reference))
        pending = [
            (first_position + index, commit)
            for index, commit in enumerate(commits)
            if commit != reference and commit not in done
        ]
        created = datetime.now(UTC)
        identity = f"{created:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        record: dict[str, Any] = {
            "schema": 1,
            "id": identity,
            "series": series,
            "host": host,
            "reference": reference,
            "start": start,
            "head": commits[-1],
            "created_at": created.isoformat(),
            "skipped": len(done.intersection(commits)),
            "commits": [{"commit": commit, "position": first_position + index} for index, commit in enumerate(commits)],
            "attempts": [],
            "status": "interrupted",
        }
        try:
            for position, commit in [(-1, reference), *pending]:
                attempt: dict[str, Any] = {"commit": commit, "role": "reference" if position == -1 else "candidate"}
                record["attempts"].append(attempt)
                metadata = prepare(results, reference, host, cpu, candidate=commit, reference_only=position == -1)
                if metadata["series"] != series:
                    msg = "Machine or toolchain configuration changed during the batch."
                    raise RuntimeError(msg)
                metadata.update(
                    batch=identity,
                    position=reference_position if position == -1 else position,
                    reference_position=reference_position,
                )
                attempt["machine"] = metadata["machine"]
                try:
                    measure(results, metadata)
                except (subprocess.CalledProcessError, ValueError, RuntimeError) as error:
                    attempt.update(status="failed", error=str(error))
                    LOGGER.exception("Measurement failed for %s", commit)
                    if position == -1:
                        break
                else:
                    attempt["status"] = "ok"
            record["status"] = "partial" if any(item["status"] != "ok" for item in record["attempts"]) else "complete"
        finally:
            for attempt in record["attempts"]:
                attempt.setdefault("status", "interrupted")
            record["finished_at"] = datetime.now(UTC).isoformat()
            write_json(results / "batches" / f"{identity}.json", record)
        if completed_measurements(results):
            build_dashboard(results)
            if upload:
                publish(results)
        if record["status"] != "complete":
            msg = "Some measurements failed; completed results are saved and missing commits will be retried."
            raise RuntimeError(msg)


def main() -> None:
    """Run a local comparison, publish its data, or import history for RTD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "batch", "publish", "fetch"))
    parser.add_argument("--results", type=Path, default=Path(".asv/results"))
    parser.add_argument("--baseline")
    parser.add_argument("--settings", type=Path, help="Batch JSON with machine, cpu, reference, and optional start")
    parser.add_argument("--machine", help="Stable physical-machine label, e.g. dgx-spark")
    parser.add_argument("--cpu", type=int, help="Logical CPU for both revisions; choose the same CPU each run")
    parser.add_argument("--publish", action="store_true", help="Upload complete results and optionally rebuild RTD")
    args = parser.parse_args()
    if args.action == "run" and (args.machine is None or args.cpu is None):
        parser.error("run requires --machine and --cpu")
    if args.action == "batch" and args.settings is None:
        parser.error("batch requires --settings")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.action == "fetch":
        fetch_results(args.results)
    elif args.action == "batch":
        batch(args.results, json.loads(args.settings.read_text()), upload=args.publish)
    elif args.action == "publish":
        with locked_checkout():
            publish(args.results)
    else:
        run(args.results, args.baseline, args.machine, args.cpu, upload=args.publish)


if __name__ == "__main__":
    main()
