# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Protect dedicated-machine measurements, history imports, and baseline selection."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from asv.machine import MachineCollection
from benchmarks import runner
from benchmarks.dashboard import METRICS, read_results
from benchmarks.dashboard import __main__ as dashboard_command

from compiler_benchmarks.test_dashboard import write_results

if TYPE_CHECKING:
    from typing import Any
    from urllib.request import Request


@pytest.fixture
def history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    """Create three commits without touching a real repository.

    Returns:
        The repository and its commits in chronological order.
    """
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.chdir(root)
    runner.git("init", "-b", "main")
    runner.git("config", "user.name", "Benchmark test")
    runner.git("config", "user.email", "benchmark@example.invalid")
    (root / ".gitignore").write_text(".asv/\n")
    runner.git("add", ".gitignore")
    commits = []
    for number in range(3):
        (root / "source.txt").write_text(str(number))
        runner.git("add", "source.txt")
        runner.git("commit", "-m", f"Source {number}")
        commits.append(runner.git("rev-parse", "HEAD"))
    return root, commits


def test_selects_previous_candidate_and_bootstraps_first_parent(history: tuple[Path, list[str]]) -> None:
    """Paired measurements cover changes since the previous published candidate."""
    root, commits = history
    results = root / "results"
    results.mkdir()
    assert runner.select_revisions(results, None) == (commits[1], commits[2])
    (results / "latest.json").write_text(json.dumps({"baseline": commits[0], "candidate": commits[1]}))
    assert runner.select_revisions(results, None) == (commits[1], commits[2])
    assert runner.select_revisions(results, commits[0]) == (commits[0], commits[2])
    (results / "latest.json").write_text(json.dumps({"baseline": commits[0], "candidate": commits[2]}))
    assert runner.select_revisions(results, None) == (commits[0], commits[2])


def test_rejects_same_commit_and_unrelated_baselines(history: tuple[Path, list[str]]) -> None:
    """A comparison needs two distinct revisions on the candidate's history."""
    root, commits = history
    with pytest.raises(ValueError, match="distinct ancestor"):
        runner.select_revisions(root / "results", "HEAD")
    runner.git("checkout", "-b", "other", commits[0])
    runner.git("commit", "--allow-empty", "-m", "Other history")
    unrelated = runner.git("rev-parse", "HEAD")
    runner.git("checkout", "main")
    with pytest.raises(ValueError, match="distinct ancestor"):
        runner.select_revisions(root / "results", unrelated)


def test_preparation_records_toolchain_and_keeps_runs_separate(
    history: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated commits must retain separate samples and the real toolchain versions."""
    root, commits = history
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 3}, raising=False)
    monkeypatch.setattr(MachineCollection, "get_machine_file_path", lambda: str(root / "machines.json"))
    monkeypatch.setattr(runner, "version", lambda executable: f"version of {executable}")
    monkeypatch.setenv("MLIR_DIR", str(root / "llvm/lib/cmake/mlir"))
    monkeypatch.setenv("CC", "clang")
    monkeypatch.setenv("CXX", "clang++")
    (root / "asv.conf.json").write_text(
        json.dumps({
            "repo": ".",
            "benchmark_dir": "benchmarks",
            "env_dir": "build/asv/env",
            "html_dir": ".asv/html/asv",
        })
    )
    results = root / "results"
    metadata = runner.prepare(results, None, "dgx-spark", 2)
    assert metadata["cpu_affinity"] == [2]
    assert metadata["host"] == "dgx-spark"
    assert metadata["llvm"] == f"version of {root / 'llvm/bin/llvm-config'}"
    assert metadata["compiler"] == "version of clang++"
    assert metadata["candidate"] == commits[2]
    assert (root / ".asv/commits.txt").read_text().splitlines() == commits[1:]
    (results / metadata["machine"]).mkdir(parents=True)
    original_config = json.loads((root / ".asv/runner.conf.json").read_text())
    second = runner.prepare(results, None, "dgx-spark", 2)
    assert metadata["series"] == second["series"]
    monkeypatch.setenv("CXX", "another-clang++")
    changed = runner.prepare(results, None, "dgx-spark", 2)
    changed_config = json.loads((root / ".asv/runner.conf.json").read_text())
    assert changed["series"] != second["series"]
    assert changed_config["env_dir"] != original_config["env_dir"]
    assert second["machine"] != metadata["machine"]
    assert (results / metadata["machine"]).is_dir()
    with pytest.raises(ValueError, match="not available"):
        runner.prepare(results, None, "dgx-spark", 4)
    with pytest.raises(ValueError, match="machine label"):
        runner.prepare(results, None, "../escape", 2)


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Create a complete pair with a deliberately slower candidate.

    Returns:
        Raw asv results and machine provenance for that pair.
    """
    results = tmp_path / "results"
    base = write_results(results, values=(1, 2))
    for metric in METRICS:
        base["results"][metric["benchmark"]][4] = [[1.0] * 20, [2.0] * 20]
    candidate = copy.deepcopy(base)
    candidate.update(commit_hash="b" * 40, date=4000)
    candidate["results"][METRICS[0]["benchmark"]][2] = [1.5, 3.0]
    candidate["results"][METRICS[0]["benchmark"]][4] = [[1.5] * 20, [3.0] * 20]
    (results / "test-machine/run.json").write_text(json.dumps(base))
    (results / "test-machine/candidate.json").write_text(json.dumps(candidate))
    return results, {
        "machine": "test-machine",
        "baseline": base["commit_hash"],
        "candidate": candidate["commit_hash"],
        "created_at": "2026-09-17T00:00:00+00:00",
        "generator_sources_changed": False,
        "cpu_affinity": [2],
    }


def test_valid_pair_reports_regressions_without_failing(pair: tuple[Path, dict[str, Any]]) -> None:
    """Slower results are published as measurements, with their samples and provenance."""
    results, metadata = pair
    summary = runner.finalize(results, metadata)
    assert "+50.0%" in summary
    assert json.loads((results / "latest.json").read_text()) == metadata
    assert all(run["measurement"] == metadata for run in read_results(results)["runs"])


@pytest.mark.parametrize("failure", ["missing_revision", "environment", "missing_metric", "suite", "samples", "cases"])
def test_incomplete_pairs_do_not_advance_published_baseline(pair: tuple[Path, dict[str, Any]], failure: str) -> None:
    """Build failures and partial measurements must not become a future baseline."""
    results, metadata = pair
    path = results / "test-machine/candidate.json"
    result = json.loads(path.read_text())
    if failure == "missing_revision":
        result["commit_hash"] = "c" * 40
    elif failure == "environment":
        result["env_name"] = "different-python"
    elif failure == "missing_metric":
        del result["results"][METRICS[0]["benchmark"]]
    elif failure == "suite":
        result["results"][METRICS[0]["benchmark"]][1] = "different-suite"
    elif failure == "samples":
        result["results"][METRICS[0]["benchmark"]][4][0] = [1.0]
    else:
        for metric in METRICS:
            result["results"][metric["benchmark"]][0][1] = ["4"]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=r"Expected|comparison must|Incomplete"):
        runner.finalize(results, metadata)
    assert not (results / "latest.json").exists()
    assert "measurement" not in json.loads((results / "test-machine/machine.json").read_text())


def test_same_commits_in_separate_runs_remain_separate(pair: tuple[Path, dict[str, Any]]) -> None:
    """The dashboard's machine/environment selector cannot mix measurement pairs."""
    results, metadata = pair
    runner.finalize(results, metadata)
    shutil.copytree(results / "test-machine", results / "second-run")
    machine = json.loads((results / "second-run/machine.json").read_text())
    machine["machine"] = "second-run"
    machine["measurement"] = metadata | {"machine": "second-run", "created_at": "2026-09-18T00:00:00+00:00"}
    (results / "second-run/machine.json").write_text(json.dumps(machine))
    runs = read_results(results)["runs"]
    assert len(runs) == 4
    assert [run["machine"] for run in runs] == ["second-run", "second-run", "test-machine", "test-machine"]
    assert len({run["machine"] for run in runs if run["commit"] == metadata["candidate"]}) == 2


@pytest.mark.parametrize("publish", [False, True])
def test_history_import_preserves_snapshot_and_full_source_history(
    history: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, publish: bool
) -> None:
    """RTD can bootstrap without a data branch and later import it without executing files."""
    source, commits = history
    if publish:
        runner.git("checkout", "-b", "benchmark_results")
        write_results(source / "results")
        runner.git("add", "results")
        runner.git("commit", "-m", "Results")
        runner.git("checkout", "main")
    checkout = tmp_path / "checkout"
    runner.git("clone", "--branch", "main", str(source), str(checkout))
    monkeypatch.chdir(checkout)
    snapshot = checkout / "snapshot"
    snapshot.mkdir()
    (snapshot / "retained.json").write_text("{}")
    runner.fetch_results(snapshot)
    assert (snapshot / "retained.json").exists()
    assert (snapshot / "test-machine/run.json").exists() == publish
    assert runner.git("rev-parse", "HEAD") == commits[2]
    assert runner.git("rev-parse", "--is-shallow-repository") == "false"


def test_history_import_rejects_executable_content(
    history: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The results branch must never introduce code into the benchmark suite."""
    source, _ = history
    runner.git("checkout", "-b", "benchmark_results")
    write_results(source / "results")
    (source / "results/unwanted.py").write_text("raise RuntimeError('must not run')")
    runner.git("add", "results")
    runner.git("commit", "-m", "Invalid result content")
    runner.git("checkout", "main")
    checkout = tmp_path / "checkout"
    runner.git("clone", "--branch", "main", str(source), str(checkout))
    monkeypatch.chdir(checkout)
    with pytest.raises(ValueError, match="regular JSON"):
        runner.fetch_results(checkout / "snapshot")
    assert not (checkout / "snapshot").exists()


@pytest.fixture
def publication_remote(history: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a local bare remote for real publication without network writes.

    Returns:
        The bare repository receiving benchmark results.
    """
    monkeypatch.delenv("RTD_API_TOKEN", raising=False)
    root, _ = history
    remote = root.parent / "remote.git"
    runner.git("init", "--bare", str(remote))
    runner.git("remote", "add", "origin", str(remote))
    return remote


def test_publication_is_append_only_and_retryable(pair: tuple[Path, dict[str, Any]], publication_remote: Path) -> None:
    """Two uploads retain history, exclude partial runs, and retry without extra commits."""
    results, metadata = pair
    runner.finalize(results, metadata)
    partial = results / "incomplete-run"
    partial.mkdir()
    (partial / "machine.json").write_text(json.dumps({"machine": "incomplete-run"}))
    runner.publish(results)
    head = runner.git("--git-dir", str(publication_remote), "rev-parse", "benchmark_results")
    runner.publish(results)
    assert runner.git("--git-dir", str(publication_remote), "rev-parse", "benchmark_results") == head
    paths = runner.git("--git-dir", str(publication_remote), "ls-tree", "-r", "--name-only", "benchmark_results")
    assert "incomplete-run" not in paths
    shutil.copytree(results / "test-machine", results / "second-run")
    path = results / "second-run/machine.json"
    info = json.loads(path.read_text())
    info["machine"] = "second-run"
    path.write_text(json.dumps(info))
    runner.finalize(results, metadata | {"machine": "second-run", "created_at": "2026-09-18T00:00:00+00:00"})
    runner.publish(results)
    paths = runner.git("--git-dir", str(publication_remote), "ls-tree", "-r", "--name-only", "benchmark_results")
    assert "results/test-machine/run.json" in paths
    assert "results/second-run/run.json" in paths
    runner.finalize(results, metadata)
    runner.publish(results)
    latest = json.loads(
        runner.git("--git-dir", str(publication_remote), "show", "benchmark_results:results/latest.json")
    )
    assert latest["machine"] == "second-run"


@pytest.mark.usefixtures("publication_remote")
def test_publication_never_replaces_saved_samples(pair: tuple[Path, dict[str, Any]]) -> None:
    """A reused group name cannot silently change the historical measurements."""
    results, metadata = pair
    runner.finalize(results, metadata)
    runner.publish(results)
    path = results / "test-machine/candidate.json"
    result = json.loads(path.read_text())
    result["results"][METRICS[1]["benchmark"]][2][0] += 1
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="different samples"):
        runner.publish(results)


def test_rtd_failure_can_be_retried_after_successful_publication(
    pair: tuple[Path, dict[str, Any]], publication_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A documentation service failure must not lose an already uploaded measurement."""
    results, metadata = pair
    runner.finalize(results, metadata)
    monkeypatch.setenv("RTD_API_TOKEN", "test-token")

    def unavailable(*_args: object, **_kwargs: object) -> None:
        msg = "RTD unavailable"
        raise OSError(msg)

    monkeypatch.setattr(runner.urllib.request, "urlopen", unavailable)
    with pytest.raises(OSError, match="RTD unavailable"):
        runner.publish(results)
    head = runner.git("--git-dir", str(publication_remote), "rev-parse", "benchmark_results")
    requests = []

    def capture(request: Request, *, timeout: int) -> contextlib.AbstractContextManager[None]:
        requests.append((request, timeout))
        return contextlib.nullcontext()

    monkeypatch.setattr(runner.urllib.request, "urlopen", capture)
    runner.publish(results)
    request, timeout = requests[0]
    assert request.full_url == "https://app.readthedocs.org/api/v3/projects/mqt-core/versions/latest/builds/"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Token test-token"
    assert timeout == 30
    assert runner.git("--git-dir", str(publication_remote), "rev-parse", "benchmark_results") == head


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.usefixtures("publication_remote")
@pytest.mark.skipif(os.name != "posix", reason="The runner uses a Unix process lock")
def test_local_command_only_publishes_completed_measurements(
    pair: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, *, fails: bool
) -> None:
    """The command applies CPU controls and stops before publication on measurement failure."""
    results, metadata = pair
    monkeypatch.setattr(runner, "prepare", lambda *_: metadata)
    commands = []

    def measure(
        command: list[str], *, env: dict[str, str] | None = None, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert check
        commands.append((command, env))
        if fails:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner, "run_command", measure)
    if fails:
        with pytest.raises(subprocess.CalledProcessError):
            runner.run(results, None, "dgx-spark", 2, upload=True)
        assert not (results / "latest.json").exists()
        assert not runner.git("ls-remote", "--heads", "origin", "refs/heads/benchmark_results")
    else:
        runner.run(results, None, "dgx-spark", 2, upload=True)
        assert runner.git("ls-remote", "--heads", "origin", "refs/heads/benchmark_results")
        assert len(commands) == 2
    command, environment = commands[0]
    assert "--interleave-rounds" in command
    assert "rounds=4" in command
    assert "repeat=5" in command
    assert environment is not None
    assert environment["MQT_BENCHMARK_CPU"] == "2"
    assert environment["OMP_NUM_THREADS"] == "1"


@pytest.mark.skipif(os.name != "posix", reason="The runner uses a Unix process lock")
def test_local_command_refuses_dirty_checkout(history: tuple[Path, list[str]]) -> None:
    """Results cannot claim a revision while running an uncommitted benchmark suite."""
    root, _ = history
    (root / "source.txt").write_text("uncommitted change")
    with pytest.raises(RuntimeError, match="clean dedicated checkout"):
        runner.run(root / "results", None, "dgx-spark", 2, upload=False)


def test_local_command_refuses_overlapping_runs(history: tuple[Path, list[str]]) -> None:
    """A timer and a manual invocation cannot benchmark this checkout together."""
    fcntl = pytest.importorskip("fcntl")

    root, _ = history
    (root / ".asv").mkdir()
    with (root / ".asv/runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already active"):
            runner.run(root / "results", None, "dgx-spark", 2, upload=False)


def test_release_boundary_selects_only_main_first_parents(
    history: tuple[Path, list[str]],
    publication_remote: Path,
) -> None:
    """The inclusive release boundary excludes older and internal branch commits."""
    _, commits = history
    runner.git("checkout", "-b", "feature", commits[1])
    runner.git("commit", "--allow-empty", "-m", "Internal feature commit")
    internal = runner.git("rev-parse", "HEAD")
    runner.git("checkout", "main")
    runner.git("merge", "--no-ff", "feature", "-m", "Merge feature")
    merged = runner.git("rev-parse", "HEAD")
    runner.git("push", str(publication_remote), "main")
    runner.git("fetch", "origin", "main:refs/remotes/origin/main")
    assert runner.main_history(commits[1], commits[1]) == [commits[1], commits[2], merged]
    for reference, start in [("v4.0.0", commits[1]), (internal, internal), (commits[2], commits[1])]:
        with pytest.raises(ValueError, match=r"full lowercase|first-parent"):
            runner.main_history(reference, start)


@pytest.fixture
def worker(
    history: tuple[Path, list[str]],
    pair: tuple[Path, dict[str, Any]],
    publication_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], list[str], list[str]]:
    """Replace only expensive machine measurement with validated sample fixtures.

    Returns:
        Results path, settings, commits and the mutable list of commits to fail.
    """
    if os.name != "posix":
        pytest.skip("The worker uses a Unix process lock")
    root, commits = history
    template, _ = pair
    runner.git("push", str(publication_remote), "main")
    results = root / ".asv/results"
    results.mkdir(parents=True)
    shutil.copyfile(template / "benchmarks.json", results / "benchmarks.json")
    settings = {"machine": "test-dgx", "cpu": 2, "reference": commits[0], "start": commits[0]}
    profile = {"host": "test-dgx", "cpu_affinity": [2], "benchmark_version": "suite-1"}
    monkeypatch.setattr(runner, "configuration", lambda *_: profile)
    monkeypatch.setattr(runner, "build_dashboard", lambda _: None)
    failures: list[str] = []

    def prepare(
        _results: Path,
        baseline: str,
        _host: str,
        _cpu: int,
        *,
        candidate: str,
        reference_only: bool,
    ) -> dict[str, Any]:
        number = len(list(results.glob("*/machine.json")))
        machine = f"observation-{number}"
        folder = results / machine
        folder.mkdir()
        (folder / "machine.json").write_text(json.dumps({"machine": machine, "version": 1}))
        return profile | {
            "machine": machine,
            "baseline": baseline,
            "candidate": candidate,
            "series": runner.series_id(profile, baseline),
            "kind": "reference" if reference_only else "comparison",
            "created_at": f"2026-09-17T00:{number:02d}:00+00:00",
            "generator_sources_changed": False,
            "input_versions": {baseline: "input-1", candidate: "input-1"},
        }

    def measure(destination: Path, metadata: dict[str, Any]) -> None:
        if metadata["candidate"] in failures:
            raise subprocess.CalledProcessError(1, ["asv", "run"])
        for commit in {metadata["baseline"], metadata["candidate"]}:
            result = json.loads((template / "test-machine/run.json").read_text())
            result["commit_hash"] = commit
            (destination / metadata["machine"] / f"{commit}.json").write_text(json.dumps(result))
        runner.finalize(destination, metadata)

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "measure", measure)
    return results, settings, commits, failures


def test_failed_candidates_retry_and_empty_batches_still_measure_reference(
    worker: tuple[Path, dict[str, Any], list[str], list[str]],
    publication_remote: Path,
) -> None:
    """Successes survive failure, import, upload retry, and a subsequent empty batch."""
    results, settings, commits, failures = worker
    failures.append(commits[1])
    with pytest.raises(RuntimeError, match="Some measurements failed"):
        runner.batch(results, settings, upload=True)
    complete = runner.completed_measurements(results)
    assert {item["candidate"] for item in complete.values()} == {commits[0], commits[2]}
    manifest = json.loads(next((results / "batches").glob("*.json")).read_text())
    assert manifest["status"] == "partial"
    assert [attempt["status"] for attempt in manifest["attempts"]] == ["ok", "failed", "ok"]
    saved = {path: path.read_bytes() for group in complete for path in (results / group).glob("*.json")}
    failures.clear()
    runner.batch(results, settings, upload=True)
    assert len(runner.completed_measurements(results)) == 4
    assert all(path.read_bytes() == value for path, value in saved.items())
    runner.batch(results, settings, upload=True)
    complete = runner.completed_measurements(results)
    assert len(complete) == 5
    assert sum(item["kind"] == "reference" for item in complete.values()) == 3
    runner.publish(results)
    paths = runner.git("--git-dir", str(publication_remote), "ls-tree", "-r", "--name-only", "benchmark_results")
    assert paths.count("/machine.json") == 5
    assert paths.count("results/batches/") == 3
    report = read_results(results)
    assert len(report["batches"]) == 3
    assert all(row["time"]["input_version"] == "input-1" for run in report["runs"] for row in run["rows"])


def test_initial_reference_failure_stops_candidates(
    worker: tuple[Path, dict[str, Any], list[str], list[str]],
) -> None:
    """An unusable reference must stop the batch before it measures candidates."""
    results, settings, commits, failures = worker
    failures.append(commits[0])
    with pytest.raises(RuntimeError, match="Some measurements failed"):
        runner.batch(results, settings, upload=False)
    assert not runner.completed_measurements(results)
    record = json.loads(next((results / "batches").glob("*.json")).read_text())
    assert len(record["attempts"]) == 1
    assert record["attempts"][0]["role"] == "reference"


def test_series_change_remeasures_history(worker: tuple[Path, dict[str, Any], list[str], list[str]]) -> None:
    """Changed benchmark conditions must not count previous measurements as complete."""
    results, settings, _, _ = worker
    runner.batch(results, settings, upload=False)
    runner.configuration("test-dgx", 2)["benchmark_version"] = "suite-2"
    runner.batch(results, settings, upload=False)
    complete = runner.completed_measurements(results)
    assert len(complete) == 6
    assert len({item["series"] for item in complete.values()}) == 2


def test_upload_failure_preserves_unpublished_history_on_next_import(
    worker: tuple[Path, dict[str, Any], list[str], list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote history imports must preserve successes collected after the last upload."""
    results, settings, commits, failures = worker
    failures.append(commits[1])
    with pytest.raises(RuntimeError, match="Some measurements failed"):
        runner.batch(results, settings, upload=True)
    failures.clear()
    publish = runner.publish

    def unavailable(_: Path) -> None:
        msg = "Upload unavailable"
        raise OSError(msg)

    monkeypatch.setattr(runner, "publish", unavailable)
    with pytest.raises(OSError, match="Upload unavailable"):
        runner.batch(results, settings, upload=True)
    assert len(runner.completed_measurements(results)) == 4
    monkeypatch.setattr(runner, "publish", publish)
    runner.batch(results, settings, upload=True)
    assert len(runner.completed_measurements(results)) == 5


def test_interruption_keeps_successes_and_records_unfinished_attempt(
    worker: tuple[Path, dict[str, Any], list[str], list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled batch checkpoints its reference and retries its unfinished candidate."""
    results, settings, commits, _ = worker
    measure = runner.measure

    def interrupted(destination: Path, metadata: dict[str, Any]) -> None:
        if metadata["candidate"] == commits[1]:
            raise KeyboardInterrupt
        measure(destination, metadata)

    monkeypatch.setattr(runner, "measure", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.batch(results, settings, upload=False)
    assert len(runner.completed_measurements(results)) == 1
    manifest = json.loads(next((results / "batches").glob("*.json")).read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["attempts"][-1]["status"] == "interrupted"
    monkeypatch.setattr(runner, "measure", measure)
    runner.batch(results, settings, upload=False)
    assert len(runner.completed_measurements(results)) == 4


def test_old_suite_snapshots_remain_publishable(pair: tuple[Path, dict[str, Any]], publication_remote: Path) -> None:
    """Updating global asv definitions must not invalidate immutable old observations."""
    results, metadata = pair
    runner.finalize(results, metadata)
    definitions = json.loads((results / "benchmarks.json").read_text())
    updated = copy.deepcopy(definitions)
    for metric in METRICS:
        updated[metric["benchmark"]]["version"] = "new-suite"
    (results / "benchmarks.json").write_text(json.dumps(updated))
    runner.publish(results)
    assert all(row["time"]["current_suite"] for run in read_results(results)["runs"] for row in run["rows"])
    remote = json.loads(
        runner.git("--git-dir", str(publication_remote), "show", "benchmark_results:results/benchmarks.json")
    )
    assert remote == definitions


def test_changed_inputs_suppress_summary_deltas(pair: tuple[Path, dict[str, Any]]) -> None:
    """Input-source changes must not be presented as compiler-only regressions."""
    results, metadata = pair
    summary = runner.finalize(results, metadata | {"generator_sources_changed": True})
    assert "different inputs" in summary
    assert "+50.0%" not in summary


def test_asv_publication_includes_main_beyond_fixed_harness(
    history: tuple[Path, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The asv explorer must retain results newer than the reviewed harness checkout."""
    root, commits = history
    runner.git("update-ref", "refs/remotes/origin/main", commits[-1])
    runner.git("checkout", "--detach", commits[0])
    (root / "asv.conf.json").write_text(
        json.dumps({
            "branches": ["HEAD"],
            "html_dir": ".asv/html/asv",
            "results_dir": ".asv/results",
        })
    )
    monkeypatch.setattr(dashboard_command, "__file__", str(root / "benchmarks/dashboard/__main__.py"))
    monkeypatch.setattr(dashboard_command.sys, "argv", ["dashboard", "build"])
    monkeypatch.setattr(dashboard_command, "build_dashboard", lambda *_args, **_kwargs: None)
    configs = []

    def capture(command: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        assert "asv" in command
        assert cwd == root
        assert check
        configs.append(json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(command, 0)

    with monkeypatch.context() as commands:
        commands.setattr(dashboard_command.subprocess, "run", capture)
        dashboard_command.main()
    assert configs[0]["branches"] == ["HEAD", "refs/remotes/origin/main"]
    assert runner.git("rev-parse", "HEAD") == commits[0]
