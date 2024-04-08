from collections.abc import Iterator, Mapping, Sequence
from typing import overload

from .._compat.typing import Self

class Variable:
    """
    A symbolic variable.

    Args:
        name: The name of the variable.

    Note:
        Variables are uniquely identified by their name, so if a variable with the same name already exists, the existing variable will be returned.
    """
    def __eq__(self: Self, arg0: object) -> bool: ...
    def __gt__(self: Self, arg0: Variable) -> bool: ...
    def __hash__(self: Self) -> int: ...
    def __init__(self: Self, name: str = "") -> None: ...
    def __lt__(self: Self, arg0: Variable) -> bool: ...
    def __ne__(self: Self, arg0: object) -> bool: ...
    @property
    def name(self: Self) -> str:
        """
        The name of the variable.
        """

class Term:
    """
    A symbolic term which consists of a variable with a given coefficient.

    Args:
        variable: The variable of the term.
        coefficient: The coefficient of the term.
    """
    def __eq__(self: Self, arg0: object) -> bool: ...
    def __hash__(self: Self) -> int: ...
    def __init__(self: Self, variable: Variable, coefficient: float = 1.0) -> None: ...
    def __mul__(self: Self, arg0: float) -> Term: ...
    def __ne__(self: Self, arg0: object) -> bool: ...
    def __rmul__(self: Self, arg0: float) -> Term: ...
    def __rtruediv__(self: Self, arg0: float) -> Term: ...
    def __truediv__(self: Self, arg0: float) -> Term: ...
    def add_coefficient(self: Self, coeff: float) -> None:
        """
        Add a coefficient to the coefficient of this term.

        Args:
            coeff: The coefficient to add.
        """
    def evaluate(self: Self, assignment: Mapping[Variable, float]) -> float:
        """
        Evaluate the term with a given variable assignment.

        Args:
            assignment: The variable assignment.

        Returns:
            The evaluated value of the term.
        """
    def has_zero_coefficient(self: Self) -> bool:
        """
        Check if the coefficient of the term is zero.
        """
    @property
    def coefficient(self: Self) -> float:
        """
        The coefficient of the term.
        """
    @property
    def variable(self: Self) -> Variable:
        """
        The variable of the term.
        """

class Expression:
    """
    A symbolic expression which consists of a sum of terms and a constant.
    The expression is of the form :math:`constant + term_1 + term_2 + \dots + term_n`.

    Args:
        terms: The list of terms.
        constant: The constant.

    Alternatively, an expression can be created with a single term and a constant or just a constant.
    """

    constant: float
    """
    The constant of the expression.
    """

    @overload
    def __add__(self: Self, arg0: Expression) -> Expression: ...
    @overload
    def __add__(self: Self, arg0: Term) -> Expression: ...
    @overload
    def __add__(self: Self, arg0: float) -> Expression: ...
    def __eq__(self: Self, arg0: object) -> bool: ...
    def __getitem__(self: Self, idx: int) -> Term: ...
    def __hash__(self: Self) -> int: ...
    @overload
    def __init__(self: Self, terms: Sequence[Term], constant: float = 0.0) -> None:
        """
        Create an expression with a given list of terms and a constant.

        Args:
            terms: The list of terms.
            constant: The constant.
        """
    @overload
    def __init__(self: Self, term: Term, constant: float = 0.0) -> None:
        """
        Create an expression with a given term and a constant.

        Args:
            term: The term.
            constant: The constant.
        """
    @overload
    def __init__(self: Self, constant: float = 0.0) -> None:
        """
        Create an expression with a given constant.

        Args:
            constant: The constant.
        """
    def __iter__(self: Self) -> Iterator[Term]: ...
    def __len__(self: Self) -> int: ...
    def __mul__(self: Self, arg0: float) -> Expression: ...
    def __ne__(self: Self, arg0: object) -> bool: ...
    @overload
    def __radd__(self: Self, arg0: Term) -> Expression: ...
    @overload
    def __radd__(self: Self, arg0: float) -> Expression: ...
    def __rmul__(self: Self, arg0: float) -> Expression: ...
    @overload
    def __rsub__(self: Self, arg0: Term) -> Expression: ...
    @overload
    def __rsub__(self: Self, arg0: float) -> Expression: ...
    def __rtruediv__(self: Self, arg0: float) -> Expression: ...
    @overload
    def __sub__(self: Self, arg0: Expression) -> Expression: ...
    @overload
    def __sub__(self: Self, arg0: Term) -> Expression: ...
    @overload
    def __sub__(self: Self, arg0: float) -> Expression: ...
    def __truediv__(self: Self, arg0: float) -> Expression: ...
    def evaluate(self: Self, assignment: Mapping[Variable, float]) -> float:
        """
        Evaluate the expression with a given variable assignment.

        Args:
            assignment: The variable assignment.

        Returns:
            The evaluated value of the expression.
        """
    def is_constant(self: Self) -> bool:
        """
        Check if the expression is a constant.
        """
    def is_zero(self: Self) -> bool:
        """
        Check if the expression is zero.
        """
    def num_terms(self: Self) -> int:
        """
        The number of terms in the expression.
        """
    @property
    def terms(self: Self) -> list[Term]:
        """
        The terms of the expression.
        """
    @property
    def variables(self: Self) -> set[Variable]:
        """
        The variables in the expression.
        """
