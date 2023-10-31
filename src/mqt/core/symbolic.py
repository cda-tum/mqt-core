"""Symbolic expressions and terms."""

from __future__ import annotations

from ._core.symbolic import Expression, Term, Variable

__all__ = ("Variable", "Term", "Expression")

for cls in (Variable, Term, Expression):
    cls.__module__ = "mqt.core.symbolic"
del cls
