"""Command line interface for mqt-core."""

from __future__ import annotations

import argparse
import sys

from ._version import version as __version__
from .commands import cmake_dir, include_dir


def main() -> None:
    """Entry point for the mqt-core command line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__, help="Print version and exit.")

    parser.add_argument(
        "--include_dir", action="store_true", help="Print the path to the mqt-core C++ include directory."
    )
    parser.add_argument(
        "--cmake_dir", action="store_true", help="Print the path to the mqt-core CMake module directory."
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.include_dir:
        print(include_dir())
    if args.cmake_dir:
        print(cmake_dir())


if __name__ == "__main__":
    main()
