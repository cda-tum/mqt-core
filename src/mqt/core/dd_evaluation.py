# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Evaluating the json file generated by the benchmarking script."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pandas as pd

if TYPE_CHECKING:
    from os import PathLike

# Avoid output truncation
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

__sort_options = ["ratio", "algorithm"]
__higher_better_metrics = ["hits", "hit_ratio"]


class _BColors:
    """Class for colored output in the terminal."""

    OKGREEN = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def __flatten_dict(d: dict[Any, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary. Every value only has one key which is the path to the value.

    Returns:
        A dictionary with the flattened keys and the values.
    """
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(__flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


class _BenchmarkDict(TypedDict, total=False):
    """A dictionary containing the data of one particular benchmark."""

    algo: str
    task: str
    n: int
    component: str
    metric: str


def __post_processing(key: str) -> _BenchmarkDict:
    """Postprocess the key of a flattened dictionary to get the metrics for the DataFrame columns.

    Returns:
        A dictionary containing the algorithm, task, number of qubits, component, and metric.

    Raises:
        ValueError: If the key is missing the algorithm, task, number of qubits, or metric.
    """
    metrics_divided = key.split(".")
    result_metrics = _BenchmarkDict()
    if len(metrics_divided) < 4:
        raise ValueError("Benchmark " + key + " is missing algorithm, task, number of qubits or metric!")
    result_metrics["algo"] = metrics_divided.pop(0)
    result_metrics["task"] = metrics_divided.pop(0)
    result_metrics["n"] = int(metrics_divided.pop(0))
    num_remaining_benchmarks = len(metrics_divided)
    if num_remaining_benchmarks == 1:
        result_metrics["component"] = ""
        result_metrics["metric"] = metrics_divided.pop(0)
    elif num_remaining_benchmarks == 2:
        if metrics_divided[0] == "dd":
            result_metrics["component"] = "" if metrics_divided[0] == "dd" else metrics_divided.pop(0)
            result_metrics["metric"] = metrics_divided[-1]
    else:
        separator = "_"
        # if the second-to-last element is not "total" then only the last element is the metric and the rest component
        if metrics_divided[-2] == "total":
            metric = separator.join(metrics_divided[-2:])
            result_metrics["metric"] = metric
            component = separator.join(metrics_divided[:-2])
            component = component.removeprefix("dd_")
            result_metrics["component"] = component
        else:
            result_metrics["metric"] = metrics_divided[-1]
            component = separator.join(metrics_divided[:-1])
            component = component.removeprefix("dd_")
            result_metrics["component"] = component

    return result_metrics


class _DataDict(TypedDict):
    """A dictionary containing the data for one entry in the DataFrame."""

    before: float
    after: float
    ratio: float
    algo: str
    task: str
    n: int
    component: str
    metric: str


def __aggregate(baseline_filepath: str | PathLike[str], feature_filepath: str | PathLike[str]) -> pd.DataFrame:
    """Aggregate the data from the baseline and feature json files into one DataFrame for visualization.

    Returns:
        A DataFrame containing the aggregated data.
    """
    base_path = Path(baseline_filepath)
    with base_path.open(mode="r", encoding="utf-8") as f:
        d = json.load(f)
    flattened_data = __flatten_dict(d)
    feature_path = Path(feature_filepath)
    with feature_path.open(mode="r", encoding="utf-8") as f:
        d_feature = json.load(f)
    flattened_feature = __flatten_dict(d_feature)

    for k, v in flattened_data.items():
        value = v
        if value == "unused":
            value = float("nan")
        if k in flattened_feature:
            ls = [value, flattened_feature[k]]
            flattened_data[k] = ls
            del flattened_feature[k]
        else:
            ls = [value, float("nan")]
            flattened_data[k] = ls
    # If a benchmark is in the feature file but not in the baseline file, it should be added with baseline marked as
    # "skipped"
    for k, v in flattened_feature.items():
        value = v
        if value == "unused":
            value = float("nan")
        ls = [float("nan"), value]
        flattened_data[k] = ls

    df_all_entries = []  # records individual entries

    for k, v in flattened_data.items():
        before = v[0]
        after = v[1]
        if math.isnan(before) or math.isnan(after):
            ratio = float("nan")
        else:
            ratio = after / before if before != 0 else 1 if after == 0 else math.inf
        key = k
        if k.endswith(tuple(__higher_better_metrics)):
            ratio = 1 / ratio if ratio != 0 else math.inf
            key += "*"
        before = round(before, 3) if isinstance(before, float) else before
        after = round(after, 3) if isinstance(after, float) else after
        ratio = round(ratio, 3)
        # postprocessing
        result_metrics = __post_processing(key)

        df_all_entries.append(
            _DataDict(
                before=before,
                after=after,
                ratio=ratio,
                algo=result_metrics["algo"],
                task=result_metrics["task"],
                n=result_metrics["n"],
                component=result_metrics["component"],
                metric=result_metrics["metric"],
            ),
        )

    df_all = pd.DataFrame(df_all_entries)
    df_all.index = pd.Index([""] * len(df_all.index))

    return df_all


def __print_results(
    *,
    df: pd.DataFrame,
    sort_indices: list[str],
    factor: float,
    no_split: bool,
    only_changed: bool,
) -> None:
    """Print the results in a nice table."""
    # after significantly smaller than before
    m1 = df["ratio"] < 1 - factor
    # after significantly larger than before
    m2 = df["ratio"] > 1 + factor
    # after is nan or before is nan or after is close to before
    m3 = (df["ratio"] != df["ratio"]) | ((1 - factor <= df["ratio"]) & (df["ratio"] <= 1 + factor))

    if no_split:
        if only_changed:
            print(df[m1 | m2].sort_values(by=sort_indices).to_markdown(index=False, stralign="right"))
        print(df.sort_values(by=sort_indices).to_markdown(index=False, stralign="right"))
        return

    print(f"\n{_BColors.OKGREEN}Benchmarks that have improved:{_BColors.ENDC}\n")
    print(df[m1].sort_values(by=sort_indices).to_markdown(index=False, stralign="right"))

    print(f"\n{_BColors.FAIL}Benchmarks that have worsened:{_BColors.ENDC}\n")
    print(df[m2].sort_values(by=sort_indices, ascending=False).to_markdown(index=False, stralign="right"))

    if only_changed:
        return

    print("\nBenchmarks that have stayed the same:\n")
    print(df[m3].sort_values(by=sort_indices).to_markdown(index=False, stralign="right"))


def compare(
    baseline_filepath: str | PathLike[str],
    feature_filepath: str | PathLike[str],
    *,
    factor: float = 0.1,
    sort: str = "ratio",
    dd: bool = False,
    only_changed: bool = False,
    no_split: bool = False,
    algorithm: str | None = None,
    task: str | None = None,
    num_qubits: int | None = None,
) -> None:
    """Compare the results of two benchmarking runs from the generated json file.

    Args:
        baseline_filepath: Path to the baseline json file.
        feature_filepath: Path to the feature json file.
        factor: How much a result has to change to be considered significant.
        sort: Sort the table by this column. Valid options are "ratio" and "algorithm".
        dd: Whether to show the detailed DD benchmark results.
        only_changed: Whether to only show results that changed significantly.
        no_split: Whether to merge all results together in one table or to separate the results
                  into benchmarks that improved, stayed the same, or worsened.
        algorithm: Only show results for this algorithm.
        task: Only show results for this task.
        num_qubits: Only show results for this number of qubits. Can only be used if algorithm is also specified.

    Raises:
        ValueError: If factor is negative or sort is invalid or if num_qubits is specified while algorithm is not.
    """
    if factor < 0:
        msg = "Factor must be positive!"
        raise ValueError(msg)
    if sort not in __sort_options:
        msg = "Invalid sort option! Valid options are 'ratio' and 'algorithm'."
        raise ValueError(msg)
    if algorithm is None and num_qubits is not None:
        msg = "num_qubits can only be specified if algorithm is also specified!"
        raise ValueError(msg)

    df_all = __aggregate(baseline_filepath, feature_filepath)

    if task is not None:
        df_all = df_all[df_all["task"].str.contains(task, case=False)]
    if algorithm is not None:
        df_all = df_all[df_all["algo"].str.contains(algorithm, case=False)]
    if num_qubits is not None:
        df_all = df_all[df_all["n"] == num_qubits]

    df_runtime = df_all[df_all["metric"] == "runtime"]
    df_runtime = df_runtime.drop(columns=["component", "metric"])
    print("\nRuntime:")
    sort_indices = ["ratio"] if sort == "ratio" else ["algo", "task", "n"]
    __print_results(
        df=df_runtime, sort_indices=sort_indices, factor=factor, no_split=no_split, only_changed=only_changed
    )

    if not dd:
        return

    print("\nDD Package details:")
    df_dd = df_all[df_all["metric"] != "runtime"]
    sort_indices = ["ratio"] if sort == "ratio" else ["algo", "task", "n", "component", "metric"]
    __print_results(df=df_dd, sort_indices=sort_indices, factor=factor, no_split=no_split, only_changed=only_changed)


def main() -> None:
    """Main function for the command line interface.

    This function is called when running the `mqt-core-compare` CLI command.

    .. code-block:: bash

        mqt-core-compare baseline.json feature.json [options]

    In addition to the mandatory filepath arguments, it provides the following optional command line options:

    - :code:`--factor <float>`: How much a result has to change to be considered significant.
    - :code:`--sort`: Sort the table by this column. Valid options are 'ratio' and 'algorithm'.
    - :code:`--dd`: Whether to show the detailed DD benchmark results.
    - :code:`--only_changed`: Whether to only show results that changed significantly.
    - :code:`--no_split`: Whether to merge all results together in one table or to separate the results into benchmarks
      that improved, stayed the same, or worsened.
    - :code:`--algorithm <str>`: Only show results for this algorithm.
    - :code:`--task <str>`: Only show results for this task.
    - :code:`--num_qubits <int>`: Only show results for this number of qubits.
      Can only be used if algorithm is also specified.
    """
    parser = argparse.ArgumentParser(
        description="Compare the results of two benchmarking runs from the generated json files.",
    )
    parser.add_argument("baseline_filepath", type=str, help="Path to the baseline json file.")
    parser.add_argument("feature_filepath", type=str, help="Path to the feature json file.")
    parser.add_argument(
        "--factor",
        type=float,
        default=0.1,
        help="How much a result has to change to be considered significant.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="ratio",
        help="Sort the table by this column. Valid options are 'ratio' and 'algorithm'.",
    )
    parser.add_argument("--dd", action="store_true", help="Whether to show the detailed DD benchmark results.")
    parser.add_argument(
        "--only_changed",
        action="store_true",
        help="Whether to only show results that changed significantly.",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Whether to merge all results together in one table or to separate the results into "
        "benchmarks that improved, stayed the same, or worsened.",
    )
    parser.add_argument("--algorithm", type=str, help="Only show results for this algorithm.")
    parser.add_argument("--task", type=str, help="Only show results for this task.")
    parser.add_argument(
        "--num_qubits",
        type=int,
        help="Only show results for this number of qubits. Can only be used if algorithm is also specified.",
    )
    args = parser.parse_args()
    assert args is not None
    compare(
        args.baseline_filepath,
        args.feature_filepath,
        factor=args.factor,
        sort=args.sort,
        dd=args.dd,
        only_changed=args.only_changed,
        no_split=args.no_split,
        algorithm=args.algorithm,
        task=args.task,
        num_qubits=args.num_qubits,
    )
