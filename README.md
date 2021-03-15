[![CI](https://github.com/iic-jku/qfr/workflows/CI/badge.svg)](https://github.com/iic-jku/qfr/actions?query=workflow%3A%22CI%22)
[![codecov](https://codecov.io/gh/iic-jku/qfr/branch/master/graph/badge.svg)](https://codecov.io/gh/iic-jku/qfr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![toolset: JKQ](https://img.shields.io/badge/toolset-JKQ-blue)](https://github.com/iic-jku/jkq)

# QFR - A JKQ Library for Quantum Functionality Representation Written in C++

A JKQ library for the representation of quantum functionality by the [Institute for Integrated Circuits](http://iic.jku.at/eda/) at the [Johannes Kepler University Linz](https://jku.at). This package is part of the [JKQ toolset](https://github.com/iic-jku/jkq).

Developers: Lukas Burgholzer, Hartwig Bauer, Stefan Hillmich, Thomas Grurl and Robert Wille.

If you have any questions feel free to contact us using [iic-quantum@jku.at](mailto:iic-quantum@jku.at) or by creating an issue on [GitHub](https://github.com/iic-jku/qfr/issues).

## Usage

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
