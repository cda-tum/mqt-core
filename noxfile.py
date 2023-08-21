"""Nox sessions."""

from __future__ import annotations

import os

import nox

nox.options.sessions = ["lint", "pylint", "tests"]

PYTHON_ALL_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]

if os.environ.get("CI", None):
    nox.options.error_on_missing_interpreters = True


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(reuse_venv=True)
def pylint(session: nox.Session) -> None:
    """Run PyLint."""
    session.install("nanobind", "scikit-build-core[pyproject]", "setuptools_scm", "pybind11")
    session.install("--no-build-isolation", "-ve.[dev]", "pylint")
    session.run("pylint", "mqt.core", *session.posargs)


@nox.session(reuse_venv=True, python=PYTHON_ALL_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    posargs = list(session.posargs)
    env = {"PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    install_arg = "-ve.[coverage]" if "--cov" in posargs else "-ve.[test]"
    if "--cov" in posargs:
        posargs.append("--cov-config=pyproject.toml")

    session.install("nanobind", "scikit-build-core[pyproject]", "setuptools_scm", "pybind11")
    session.install("--no-build-isolation", install_arg)
    session.run("pytest", *posargs, env=env)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs."""
    session.install("sphinx-autobuild")
    session.install("nanobind", "scikit-build-core[pyproject]", "setuptools_scm", "pybind11")
    session.install("--no-build-isolation", "-ve.[docs]")

    session.run("sphinx-autobuild", "docs", "docs/_build/html", "--open-browser")
