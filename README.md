[![PyPI](https://img.shields.io/pypi/v/mqt.core?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.core/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/cda-tum/mqt-core/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/cda-tum/mqt-core/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/cda-tum/mqt-core/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/cda-tum/mqt-core/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/mqt-core?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/core)
[![codecov](https://img.shields.io/codecov/c/github/cda-tum/mqt-core?style=flat-square&logo=codecov)](https://codecov.io/gh/cda-tum/mqt-core)

<p align="center">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt-core/main/extern/mqt_light.png" width="60%">
   <img src="https://raw.githubusercontent.com/cda-tum/mqt-core/main/extern/mqt_dark.png" width="60%">
 </picture>
</p>

# MQT Core - The Backbone of the Munich Quantum Toolkit (MQT)

The MQT Core library forms the backbone of the quantum software tools developed as part of the _Munich Quantum Toolkit_ (_MQT_)[^1] by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/). This includes the following tools:

- [MQT DDSIM](https://github.com/cda-tum/mqt-ddsim): A Tool for Classical Quantum Circuit Simulation based on Decision Diagrams.
- [MQT QMAP](https://github.com/cda-tum/mqt-qmap): A Tool for Quantum Circuit Mapping.
- [MQT QCEC](https://github.com/cda-tum/mqt-qcec): A Tool for Quantum Circuit Equivalence Checking.
- [MQT QECC](https://github.com/cda-tum/mqt-qecc): A Tool for Quantum Error Correcting Codes.
- [MQT DDVis](https://github.com/cda-tum/mqt-ddvis): A Web-Application visualizing Decision Diagrams for Quantum Computing.
- [MQT SyReC](https://github.com/cda-tum/mqt-syrec): A Tool for Synthesis of Reversible Circuits/Quanutm Computing Oracles.

For a full list of tools and libraries, please visit the [MQT website](https://mqt.readthedocs.io/).

<!-- SPHINX-START -->

MQT Core encompasses:

- `MQT::Core`: An intermediate representation (IR) for quantum computations including means to import and export circuits in various formats.
- `MQT::CoreDD`: A fully-fledged decision diagram (DD) library for quantum computing.
- `MQT::CoreZX`: A library for working with ZX-diagrams and the ZX-calculus.
- `MQT::CoreECC`: A library for working with error correcting codes (ECCs) for quantum computing.
- `MQT::CorePython`: A Python interface for the MQT Core library (e.g., facilitating the import of Qiskit QuantumCircuits).

If you have any questions, feel free to create a [discussion](https://github.com/cda-tum/mqt-core/discussions) or an [issue](https://github.com/cda-tum/mqt-core/issues) on [GitHub](https://github.com/cda-tum/mqt-core).

## System Requirements

Building (and running) is continuously tested under Linux, MacOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/runner-images).
However, the implementation should be compatible with any current C++ compiler supporting C++17 and a minimum CMake version of 3.19.

_Disclaimer_: We noticed some issues when compiling with Microsoft's `MSVC` compiler toolchain. If you are developing under Windows, consider using the `clang` compiler toolchain.
A detailed description of how to set this up can be found [here](https://learn.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-160).

It is recommended (although not required) to have [GraphViz](https://www.graphviz.org) installed for visualization purposes.

If you want to use the ZX library, it is recommended (although not strictly necessary) to have [GMP](https://gmplib.org/) installed in your system.

[^1]: The Munich Quantum Toolkit was formerly known under the acronym _JKQ_ and developed by the [Institute for Integrated Circuits](https://iic.jku.at/eda/) at the [Johannes Kepler University Linz](https://jku.at)).
