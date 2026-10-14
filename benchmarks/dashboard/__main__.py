# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Publish the local dashboard or serve its existing static output."""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks.dashboard import build_dashboard


def main() -> None:
    """Build both dashboards or serve the last successful build on localhost."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "serve"), nargs="?", default="build")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--results", type=Path, help="Read a saved asv database instead of local results")
    parser.add_argument("--output", type=Path, help="Directory for the MQT site and nested asv explorer")
    parser.add_argument("--embed", type=Path, help="Write the Sphinx HTML fragment outside the public output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "asv.conf.json").read_text())
    output = (args.output or (root / config["html_dir"]).parent).resolve()
    results = (args.results or root / config["results_dir"]).resolve()
    if args.action == "build":
        config.update(repo=str(root), results_dir=str(results), html_dir=str(output / "asv"))
        # The fixed harness checkout can precede commits measured by a weekend batch.
        main_ref = "refs/remotes/origin/main"
        if (
            subprocess.call(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local Git arguments
                [shutil.which("git") or "git", "-C", str(root), "show-ref", "--verify", "--quiet", main_ref]
            )
            == 0
        ):
            config["branches"] = list(dict.fromkeys([*config["branches"], main_ref]))
        with tempfile.TemporaryDirectory() as temporary:
            config_file = Path(temporary) / "asv.conf.json"
            config_file.write_text(json.dumps(config))
            subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local module and generated config; no shell
                [sys.executable, "-m", "asv", "--config", str(config_file), "publish"], cwd=root, check=True
            )
        build_dashboard(results, output, embed_path=args.embed)
        logger.info("Dashboard built at %s", output)
        return
    if not (output / "measurements.json").is_file():
        parser.error("No dashboard build. Run python -m benchmarks.dashboard build first.")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(output))
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        parser.error(f"Cannot serve on port {args.port}: {error}. Select another --port if it is occupied.")
    with server, contextlib.suppress(KeyboardInterrupt):
        logger.info("MQT dashboard: http://127.0.0.1:%s/", server.server_port)
        server.serve_forever()


if __name__ == "__main__":
    main()
