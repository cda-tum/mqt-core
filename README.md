[![PyPI](https://img.shields.io/pypi/v/mqt.qfr?logo=pypi&style=plastic)](https://pypi.org/project/mqt.qfr/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/cda-tum/qfr/ci.yml?branch=main&logo=github&style=plastic)](https://github.com/cda-tum/qfr/actions?query=workflow%3A%22CI%22)
[![Codecov branch](https://img.shields.io/codecov/c/github/cda-tum/qfr/master?label=codecov&logo=codecov&style=plastic)](https://codecov.io/gh/cda-tum/qfr)
![GitHub](https://img.shields.io/github/license/cda-tum/qfr?style=plastic)
[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2103.08281&color=inactive&style=plastic)](https://arxiv.org/abs/2103.08281)

# MQT QFR - A Library for Quantum Functionality Representation Written in C++

A library for the representation of quantum functionality by the [Institute for Integrated Circuits](http://iic.jku.at/eda/) at the [Johannes Kepler University Linz](https://jku.at).
If you have any questions feel free to contact us using [iic-quantum@jku.at](mailto:iic-quantum@jku.at) or by creating an issue on [GitHub](https://github.com/cda-tum/qfr/issues).

## Efficient Construction of Functional Representations for Quantum Algorithms

The QFR library provides the means for constructing the functionality of a given quantum circuit using [decision diagrams](https://iic.jku.at/eda/research/quantum_dd) in the form of the `mqt.qfr` Python package. It includes a traditional,
sequential approach (`qfr.ConstructionMethod.sequential`) and the efficient, recursive method proposed in [[1]](https://arxiv.org/abs/2103.08281) (`qfr.ConstructionMethod.recursive`).

[[1]](https://arxiv.org/abs/2103.08281) L. Burgholzer, R. Raymond, I. Sengupta, and R. Wille. **"Efficient Construction of Functional Representations for Quantum Algorithms"**. [arXiv:2103.08281](https://arxiv.org/abs/2103.08281), 2021

In order to make the library as easy to use as possible (without compilation), we provide pre-built wheels for most common platforms (64-bit Linux, MacOS, Windows). These can be installed using

```bash
pip install mqt.qfr
```

However, in order to get the best performance out of the QFR, it is recommended to build it locally from the source distribution (see [system requirements](#system-requirements)) via

```bash
pip install mqt.qfr --no-binary mqt.qfr
```

This enables platform specific compiler optimizations that cannot be enabled on portable wheels.

Once installed, in Python, the functionality of a given circuit (provided, e.g., as Qiskit QuantumCircuit) can be constructed with:

```python
from mqt import qfr
from qiskit import QuantumCircuit

# create your quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# construct the functionality of the circuit
results = qfr.construct(qc)

# print the results
print(results)
```

The `construct` function additionally provides the options `store_decision_diagram` and `store_matrix` that allow to store the resulting decision diagram or matrix, respectively. Note that storing the resulting matrix takes considerable amounts of memory in comparison to the typical memory footprint incurred by the corresponding decision diagram. At least `2^(2n+1)*sizeof(float)` byte are required for storing the matrix representing an n-qubit quantum circuit.

Special routines are available for constructing the functionality of the Quantum Fourier Transform (`construct_qft(nqubits, ...)`) or Grover's algorithm (`construct_grover(nqubits, seed, ...)`). For details on the method employed for Grover's search we refer to [[1, Section 4.2]](https://arxiv.org/abs/2103.08281).

## MQT Toolset

The QFR library is the backbone of the quantum software tools in:

- [MQT DDSIM](https://github.com/cda-tum/ddsim): a decision diagram-based simulator for quantum circuits.
- [MQT QMAP](https://github.com/cda-tum/qmap): a tool for mapping/compiling quantum circuits to real quantum architectures.
- [MQT QCEC](https://github.com/cda-tum/qcec): a decision diagram-based equivalence checking tool for quantum circuits.
- [MQT DDVis](http://github.com/cda-tum/ddvis): a visualization tool for how decision diagrams are used in simulation and verification.
  - You can find an online instance of this tool at http://iic.jku.at/eda/research/quantum_dd/tool/

It acts as an intermediate representation and provides the facilitites to

- **Obtain intermediate representations from circuit descriptions.**

  Currently available file formats are:

  - `OpenQASM` (e.g. used by [Qiskit](https://github.com/Qiskit/qiskit))
  - `Real` (e.g. from [RevLib](http://revlib.org))
  - `GRCS` Google Random Circuit Sampling Benchmarks (see [GRCS](https://github.com/sboixo/GRCS))
  - `TFC` (e.g. from [Reversible Logic Synthesis Benchmarks Page](http://webhome.cs.uvic.ca/~dmaslov/mach-read.html))
  - `QC` (e.g. from [Feynman](https://github.com/meamy/feynman))

  Importing a circuit from a file in either of those formats is done via:

  ```c++
  std::string filename = "PATH_TO_FILE";
  qc::QuantumComputation qc(filename);
  ```

  or by calling

  ```c++
  qc.import(filename);
  ```

  which first resets the `qc` object before importing the new file.

- **Generate circuit representations for important quantum algorithms.**

  Currently available algorithms are:

  - Entanglement

    ```c++
    dd::QubitCount n = 2;
    qc::Entanglement entanglement(n); // generates bell circuit
    ```

    Generates the circuit for preparing an _n_ qubit GHZ state. Primarily used as a simple test case.

  - Bernstein-Vazirani

    ```c++
    unsigned long long hiddenInteger = 16777215ull;
    qc::BernsteinVazirani bv(hiddenInteger); // generates Bernstein-Vazirani circuit for given hiddenInteger
    ```

    Generates the circuit for the Berstein-Vazirani algorithm using the provided _hiddenInteger_

  - Quantum Fourier Transform (QFT)

    ```c++
    dd::QubitCount n = 3;
    qc::QFT qft(n); // generates the QFT circuit for n qubits
    ```

    Generates the circuit for the _n_ qubit Quantum Fourier Transform.

  - Grover's search algorithm

    ```c++
    dd::QubitCount n = 2;
    qc::Grover grover(n); // generates Grover's algorithm for a random n-bit oracle
    ```

    The algorithm performs ~ &#960;/4 &#8730;2&#8319; Grover iterations. An optional `unsigned int` parameter controls the _seed_ of the random number generation for the oracle generation.

  - (Iterative) Quantum Phase Estimation
    ```c++
    dd::QubitCount n = 3;
    bool exact = true; // whether to generate an exactly representable phase or not
    qc::QPE qpe(n, exact);
    qc::IQPE iqpe(n, exact);
    ```
    Generates a random bitstring of length `n` (`n+1` in the inexact case) and constructs the corresponding (iterative) Quantum Phase Estimation algorithm using _n+1_ qubits. Alternatively, a specific `phase` and `precision` might be
    passed to the constructor.

- **Generate circuit descriptions from intermediate representations.**

  The library also supports the output of circuits in various formats by calling

  ```c++
  std::string filename = "PATH_TO_DESTINATION_FILE.qasm";
  qc.dump(filename);
  ```

  Currently available file formats are:

  - `OpenQASM` (.qasm)

## Development

### System Requirements

Building (and running) is continuously tested under Linux, MacOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/virtual-environments). However, the implementation should be compatible
with any current C++ compiler supporting C++17 and a minimum CMake version of 3.19.

_Disclaimer_: We noticed some issues when compiling with Microsoft's `MSVC` compiler toolchain. If you are developing under Windows, consider using the `clang` compiler toolchain. A detailed description of how to set this up can be
found [here](https://docs.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-160).

It is recommended (although not required) to have [GraphViz](https://www.graphviz.org) installed for visualization purposes.

The QFR library also provides functionality to represent functionality of quantum circuits in the form of [ZX-diagrams](https://github.com/cda-tum/zx). At this point this feature is only used by [MQT QCEC](https://github.com/cda-tum/qcec) but is going to be extended further in future releases. If you want to use this feature it is recommended (although not strictly necessary) to have [GMP](https://gmplib.org/) installed in your system.

### Configure, Build, and Install

To start off, clone this repository using

```shell
git clone --recurse-submodules -j8 https://github.com/cda-tum/qfr
```

Note the `--recurse-submodules` flag. It is required to also clone all the required submodules. If you happen to forget passing the flag on your initial clone, you can initialize all the submodules by
executing `git submodule update --init --recursive` in the main project directory.

Our projects use CMake as the main build configuration tool. Building a project using CMake is a two-stage process. First, CMake needs to be _configured_ by calling

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

This tells CMake to search the current directory `.` (passed via `-S`) for a _CMakeLists.txt_ file and process it into a directory `build` (passed via `-B`).
The flag `-DCMAKE_BUILD_TYPE=Release` tells CMake to configure a _Release_ build (as opposed to, e.g., a _Debug_ build).

After configuring with CMake, the project can be built by calling

```shell
 cmake --build build --config Release
```

This tries to build the project in the `build` directory (passed via `--build`).
Some operating systems and developer environments explicitly require a configuration to be set, which is why the `--config` flag is also passed to the build command. The flag `--parallel <NUMBER_OF_THREADS>` may be added to trigger a parallel build.

Building the project this way generates

- the core library `libqfr.a` (Unix) / `qfr.lib` (Windows) in the `build/src` folder
- the DD and ZX libraries `libqfr_dd.a` (Unix) / `qfr_dd.lib` (Windows) and `libqfr_zx.a` (Unix) / `qfr_zx.lib` (Windows) in the `build/src/` folder
- test executables `qfr_test`, `qfr_test_dd`, and `qfr_test_zx` containing a small set of unit tests in the `build/test` folder

You can link against the library built by this project in other CMake project using the `MQT::qfr` target (or any of the other targets `MQT::qfr_dd` and `MQT::qfr_zx`).

### Extending the Python Package

To extend the Python package you can locally install the package in edit mode, so that changes in the Python code are instantly available.
The following example assumes you have a [virtual environment](https://docs.python.org/3/library/venv.html) set up and activated.

```commandline
(venv) $ pip install cmake
(venv) $ pip install --editable .
```

If you change parts of the C++ code, you have to run the second line to make the changes visible in Python.
