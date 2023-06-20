"""Nox sessions."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from nox.sessions import Session


@nox.session(reuse_venv=True)
def docs(session: Session) -> None:
    """
    Build the docs. Pass "--serve" to serve.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    session.install(".[docs]")
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs
        )
        return

    session.run(
        "sphinx-build",
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        "-b",
        args.builder,
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if args.serve:
        print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
        session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
