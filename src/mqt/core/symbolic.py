"""Symbolic expressions and terms."""

from __future__ import annotations

from ._core.symbolic import Expression, Term, Variable

__all__ = ("Expression", "Term", "Variable")

for cls in (Variable, Term, Expression):
    cls.__module__ = "mqt.core.symbolic"
del cls
