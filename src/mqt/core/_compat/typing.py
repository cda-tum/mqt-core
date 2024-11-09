# noqa: A005
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import Self, assert_never
elif TYPE_CHECKING:
    from typing_extensions import Self, assert_never
else:
    Self = object

    def assert_never(_: object) -> None:
        msg = "Expected code to be unreachable"
        raise AssertionError(msg)


__all__ = ["Self", "assert_never"]


def __dir__() -> list[str]:
    return __all__
