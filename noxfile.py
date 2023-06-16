"""Nox sessions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from nox.sessions import Session


@nox.session
def docs(session: Session) -> None:
    """Build the documentation.

    Simply execute `nox -rs docs -- serve` to locally build and serve the docs.
    """
    session.install("-r", "docs/requirements.txt")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")
