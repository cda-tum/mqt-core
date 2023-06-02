![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/cda-tum/zx/ci.yml?branch=main&style=flat-square&logo=github&label=c%2B%2B)](https://github.com/cda-tum/zx/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/cda-tum/zx?style=flat-square&logo=codecov)](https://codecov.io/gh/cda-tum/zx)

# MQT ZX - A library for working with ZX-diagrams

A library for working with ZX-diagrams developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) as part of the Munich Quantum Toolkit (MQT).

If you have any questions, feel free to contact us via [quantum.cda@xcit.tum.de](mailto:quantum.cda@xcit.tum.de) or by creating an issue on [GitHub](https://github.com/cda-tum/zx/issues).

## System Requirements and Building

The implementation is compatible with any C++17 compiler and a minimum CMake version of 3.19.

To get the most out of this library it is recommended to have the GMP library installed.

### Building tests

From the project root, run:

Configuration: `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_ZX_TESTS=ON -S . -B build`

Compiling: `cmake --build build --config Release --target zx_test`
