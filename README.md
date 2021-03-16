[![PyPI](https://img.shields.io/pypi/v/jkq.qfr?logo=pypi&style=plastic)](https://pypi.org/project/jkq.qfr/)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/iic-jku/qfr/CI?logo=github&style=plastic)](https://github.com/iic-jku/qfr/actions?query=workflow%3A%22CI%22)
[![Codecov branch](https://img.shields.io/codecov/c/github/iic-jku/qfr/master?label=codecov&logo=codecov&style=plastic)](https://codecov.io/gh/iic-jku/qfr)
![GitHub](https://img.shields.io/github/license/iic-jku/qcec?style=plastic)
[![toolset: JKQ](https://img.shields.io/badge/toolset-JKQ-blue?style=plastic)](https://github.com/iic-jku/jkq)
[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2103.08281&color=inactive&style=plastic)](https://arxiv.org/abs/2103.08281)

# QFR - A JKQ Library for Quantum Functionality Representation Written in C++

A JKQ library for the representation of quantum functionality by the [Institute for Integrated Circuits](http://iic.jku.at/eda/) at the [Johannes Kepler University Linz](https://jku.at). This package is part of the [JKQ toolset](https://github.com/iic-jku/jkq).

Developers: Lukas Burgholzer, Hartwig Bauer, Stefan Hillmich, Thomas Grurl and Robert Wille.

If you have any questions feel free to contact us using [iic-quantum@jku.at](mailto:iic-quantum@jku.at) or by creating an issue on [GitHub](https://github.com/iic-jku/qfr/issues).

## Efficient Construction of Functional Representations for Quantum Algorithms
The QFR library provides the means for constructing the functionality of a given quantum circuit using [decision diagrams](https://iic.jku.at/eda/research/quantum_dd) in the form of the `jkq.qfr` Python package. 
It includes a traditional, sequential approach (`qfr.ConstructionMethod.sequential`) and the efficient, recursive method proposed in [[1]](https://arxiv.org/abs/2103.08281) (`qfr.ConstructionMethod.recursive`).

[[1]](https://arxiv.org/abs/2103.08281) L. Burgholzer, R. Raymond, I. Sengupta, and R. Wille. **"Efficient Construction of Functional Representations for Quantum Algorithms"**. [arXiv:2103.08281](https://arxiv.org/abs/2103.08281), 2021

In order to start using it, install the package using
```bash
pip install jkq.qfr
```
Then, in Python, the functionality of a given circuit (provided, e.g., as Qiskit QuantumCircuit) can be constructed with:
```python
from jkq import qfr
from qiskit import QuantumCircuit

# create your quantum circuit
qc = <...>

# construct the functionality of the circuit
results = qfr.construct(qc)

# print the results
print(results)
```
The `construct` function additionally provides the options `store_decision_diagram` and `store_matrix` that allow to store the resulting decision diagram or matrix, respectively. Note that storing the resulting matrix takes considerable amounts of memory in comparison to the typical memory footprint incurred by the corresponding decision diagram. At least `2^(2n+1)*sizeof(float)` byte are required for storing the matrix representing an n-qubit quantum circuit.

Special routines are available for constructing the functionality of the Quantum Fourier Transform (`construct_qft(nqubits, ...)`) or Grover's algorithm (`construct_grover(nqubits, seed, ...)`). For details on the method employed for Grover's search we refer to [[1, Section 4.2]](https://arxiv.org/abs/2103.08281).

## JKQ Toolset

The QFR library is the backbone of the quantum software tools in [JKQ: JKU Tools for Quantum Computing](https://iic.jku.at/files/eda/2020_iccad_jku_tools_for_quantum_computing.pdf):
- [JKQ DDSIM](https://github.com/iic-jku/ddsim): a decision diagram-based simulator for quantum circuits.
- [JKQ QMAP](https://github.com/iic-jku/qmap): a tool for mapping/compiling quantum circuits to real quantum architectures.
- [JKQ QCEC](https://github.com/iic-jku/qcec): a decision diagram-based equivalence checking tool for quantum circuits.
- [JKQ DDVis](http://github.com/iic-jku/ddvis): a visualization tool for how decision diagrams are used in simulation and verification.
    - You can find an online instance of this tool at http://iic.jku.at/eda/research/quantum_dd/tool/

It acts as an intermediate representation and provides the facilitites to
* **Obtain intermediate representations from circuit descriptions.**

  Currently available file formats are:

    * `OpenQASM` (e.g. used by [Qiskit](https://github.com/Qiskit/qiskit))
    * `Real` (e.g. from [RevLib](http://revlib.org))
    * `GRCS` Google Random Circuit Sampling Benchmarks (see [GRCS](https://github.com/sboixo/GRCS))
    * `TFC` (e.g. from [Reversible Logic Synthesis Benchmarks Page](http://webhome.cs.uvic.ca/~dmaslov/mach-read.html))
    * `QC` (e.g. from [Feynman](https://github.com/meamy/feynman))
    
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
   
* **Generate circuit representations for important quantum algorithms.** 

    Currently available algorithms are:
    * Entanglement
    
        ```c++
      unsigned short n = 2;
      qc::Entanglement entanglement(n); // generates bell circuit
      ```
    
      Generates the circuit for preparing an *n* qubit GHZ state. Primarily used as a simple test case. 
    
    * Bernstein-Vazirani
        ```c++
      unsigned long long hiddenInteger = 16777215ull;
      qc::BernsteinVazirani bv(hiddenInteger); // generates Bernstein-Vazirani circuit for given hiddenInteger
         ```
    
        Generates the circuit for the Berstein-Vazirani algorithm using the provided *hiddenInteger*
    
    * Quantum Fourier Transform (QFT)
  
        ```c++
      unsigned short n = 3;
      qc::QFT qft(n); // generates the QFT circuit for n qubits
      ```
  
        Generates the circuit for the *n* qubit Quantum Fourier Transform.
  * Grover's search algorithm
  
       ```c++
      unsigned short n = 2;
      qc::Grover grover(n); // generates Grover's algorithm for a random n-bit oracle
      ```
    
       The algorithm performs ~ &#960;/4 &#8730;2&#8319; Grover iterations. An optional `unsigned int` parameter controls the *seed* of the random number generation for the oracle generation.
    
* **Generate circuit descriptions from intermediate representations.**
   
  The library also supports the output of circuits in various formats by calling
    
  ```c++
  std::string filename = "PATH_TO_DESTINATION_FILE.qasm";
  qc.dump(filename);
  ```
  
  Currently available file formats are:
    * `OpenQASM` (.qasm)

## Development

### System Requirements

Building (and running) is continuously tested under Linux, MacOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/virtual-environments).
However, the implementation should be compatible with any current C++ compiler supporting C++17 and a minimum CMake version of 3.14.

It is recommended (although not required) to have [GraphViz](https://www.graphviz.org) installed for visualization purposes.

### Configure, Build, and Install

To start off, clone this repository using
```shell
git clone --recurse-submodules -j8 https://github.com/iic-jku/qfr 
```
Note the `--recurse-submodules` flag. It is required to also clone all the required submodules. If you happen to forget passing the flag on your initial clone, you can initialize all the submodules by executing `git submodule update --init --recursive` in the main project directory.

Our projects use CMake as the main build configuration tool. Building a project using CMake is a two-stage process. First, CMake needs to be *configured* by calling
```shell 
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```
This tells CMake to search the current directory `.` (passed via `-S`) for a *CMakeLists.txt* file and process it into a directory `build` (passed via `-B`).
The flag `-DCMAKE_BUILD_TYPE=Release` tells CMake to configure a *Release* build (as opposed to, e.g., a *Debug* build).

After configuring with CMake, the project can be built by calling
```shell
 cmake --build build --config Release
```
This tries to build the project in the `build` directory (passed via `--build`).
Some operating systems and developer environments explicitly require a configuration to be set, which is why the `--config` flag is also passed to the build command. The flag `--parallel <NUMBER_OF_THREADS>` may be added to trigger a parallel build.

Building the project this way generates
- the library `libqfr.a` (Unix) / `qfr.lib` (Windows) in the `build/src` folder
- a test executable `qfr_test` containing a small set of unit tests in the `build/test` folder
- a small demo example executable `qfr_example` in the `build/test` directory.

You can link against the library built by this project in other CMake project using the `JKQ::qfr` target.

### Extending the Python Package

To extend the Python package you can locally install the package in edit mode, so that changes in the Python code are instantly available.
The following example assumes you have a [virtual environment](https://docs.python.org/3/library/venv.html) set up and activated.

```commandline
(venv) $ pip install cmake
(venv) $ pip install --editable .
```

If you change parts of the C++ code, you have to run the second line to make the changes visible in Python.
