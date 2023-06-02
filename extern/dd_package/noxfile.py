from __future__ import annotations

import os

import nox
from nox.sessions import Session

nox.options.sessions = ["lint"]

PYTHON_ALL_VERSIONS = ["3.7", "3.8", "3.9", "3.10", "3.11"]

if os.environ.get("CI", None):
    nox.options.error_on_missing_interpreters = True


@nox.session
def lint(session: Session) -> None:
    """
    Lint the Python part of the codebase using pre-commit.
    Simply execute `nox -rs lint` to run all configured hooks.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)
