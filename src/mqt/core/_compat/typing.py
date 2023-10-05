from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if sys.version_info < (3, 11):
    if TYPE_CHECKING:
        from typing_extensions import Self, assert_never
    else:
        Self = object

        def assert_never(_: Any) -> None:  # noqa: ANN401
            msg = "Expected code to be unreachable"
            raise AssertionError(msg)

else:
    from typing import Self, assert_never

__all__ = ["Self", "assert_never"]


def __dir__() -> list[str]:
    return __all__
