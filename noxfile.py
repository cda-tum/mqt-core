from __future__ import annotations

import nox
from nox.sessions import Session

nox.options.sessions = ["lint"]


@nox.session
def lint(session: Session) -> None:
    """
    Lint the Python part of the codebase using pre-commit.
    Simply execute `nox -rs lint` to run all configured hooks.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def pylint(session: Session) -> None:
    """
    Run pylint.
    Simply execute `nox -rs pylint` to run pylint.
    Run as `nox -rs pylint -- skip-install` to skip installing the package and its dependencies.
    """
    session.install("pylint")
    run_install = True
    if session.posargs and "skip-install" in session.posargs:
        run_install = False
        session.posargs.remove("skip-install")
    if run_install:
        session.install("-e", ".")
    session.run("pylint", "mqt.qfr", "--extension-pkg-allow-list=mqt.qfr.pyqfr", *session.posargs)


@nox.session
def mypy(session: Session) -> None:
    """
    Run mypy.
    Simply execute `nox -rs mypy` to run mypy.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--hook-stage", "manual", "mypy", *session.posargs)
