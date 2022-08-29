[![PyPI](https://img.shields.io/pypi/v/mqt.qfr?logo=pypi&style=plastic)](https://pypi.org/project/mqt.qfr/)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/cda-tum/qfr/CI?logo=github&style=plastic)](https://github.com/cda-tum/qfr/actions?query=workflow%3A%22CI%22)
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
qc = < ... >

# construct the functionality of the circuit
results = qfr.construct(qc)

# print the results
print(results)
```
The `construct` function additionally provides the options `store_decision_diagram` and `store_matrix` that allow to store the resulting decision diagram or matrix, respectively. Note that storing the resulting matrix takes considerable amounts of memory in comparison to the typical memory footprint incurred by the corresponding decision diagram. At least `2^(2n+1)*sizeof(float)` byte are required for storing the matrix representing an n-qubit quantum circuit.

Special routines are available for constructing the functionality of the Quantum Fourier Transform (`construct_qft(nqubits, ...)`) or Grover's algorithm (`construct_grover(nqubits, seed, ...)`). For details on the method employed for Grover's search we refer to [[1, Section 4.2]](https://arxiv.org/abs/2103.08281).

## ECC Framework: Automatic Implementation and Evaluation of Error-Correcting Codes for Quantum Computing
The QFR library offers means for automatic implementation and evaluation of error-correcting codes for quantum computing. More precisely, the library allows to automatically apply different error correction schemes to quantum circuits provided as openqasm files. The "protected" quantum circuits can then be exported again in the form of openqasm files or can be directly used for noise-aware quantum circuit simulation. For the latter case, we also provide a wrapper script which makes use of the provided framework to apply error correction schemes to circuits and directly simulate those circuits using qiskit.

**Note: The ECC framework is only available within the current branch and can only be installed directly from source**

### Installation

If you have not done so already, clone the repository using:
```bash
git clone --recurse-submodules -j8 https://github.com/pichristoph/qfr.git
```

Make sure you are in the main project directory for the next steps. Switch to the branch feature/ecc,
```bash
cd qfr
git switch feature/ecc
```
and (if necessary), update the submodules.

```bash
git submodule update --init --recursive
```

Then, the ECC framework can be installed using pip
```bash
(venv) pip install --editable .
```

If you want to use Qiskit for quantum circuit simulation, you need to install it as well

```bash
(venv) pip install qiskit
```

### Usage

Having the Python module installed, error correcting codes can be applied using apply_ecc of module qfr, like so

```python
from mqt import qfr

file = "path/to/qasm/file.qasm" # Path to the openqasm file the quantum circuit shall be loaded from
ecc = "Q7Steane" # Error correction code that shall be applied to the quantum circuit
ecc_frequency = 100 # After how many times a qubit is used, error correction is applied
ecc_mc = False # Only allow single controlled gates in the created quantum circuit
ecc_cf = False # Only allow single clifford gates in the created quantum circuit

result = qfr.apply_ecc(file, ecc, ecc_frequency, ecc_mc, ecc_cf)

# print the resulting circuit
print(result["circ"])
```
Currently, the error correction schemes Q3Shor, Q5Laflamme, Q7Steane, Q9Shor, Q9Surface, and Q18Surface are supported. 

We provide a wrapper script for applying error correction to quantum circuits (provided as openQasm) and followed by a noise-aware quantum circuit simulation (using qiskit). The script can be used like this:

```bash
$ /venv/ecc_qiskit_wrapper -ecc Q7Steane -fq 100 -m D -p 0.0001 -n 2000 -fs aer_simulator_stabilizer -s 0 -f  ent_simple1000_n2.qasm
_____Trying to simulate with D(prob=0.0001, shots=2000, n_qubits=17) Error______
State |00> probability 0.515
State |01> probability 0.0055
State |10> probability 0.0025
State |11> probability 0.477
```

The script offers a help function, which displays available parameters:

```bash
$ /venv/ecc_qiskit_wrapper --help
usage: ecc_qiskit_wrapper [-h] [-m M] [-p P] [-n N] [-s S] -f F [-e E] [-fs FS] [-ecc ECC] [-fq FQ] [-mc MC] [-cf CF]

QiskitWrapper interface with error correction support!

optional arguments:
  -h, --help  show this help message and exit
  -m M        Define the error_channels (e.g., -m APD), available errors channels are amplitude damping (A), phase flip (P), bit flip (B), and depolarization (D) (Default="D")
  -p P        Set the noise probability (Default=0.001)
  -n N        Set the number of shots. 0 for deterministic simulation (Default=2000)
  -s S        Set a seed (Default=0)
  -f F        Path to a openqasm file
  -e E        Export circuit, with error correcting code applied, as openqasm circuit instead of simulation it (e.g., -e "/path/to/new/openqasm_file") (Default=None)
  -fs FS      Specify a simulator (Default: "statevector_simulator" for simulation without noise, "aer_simulator_density_matrix", for deterministic noise-aware simulation"aer_simulator_statevector", for stochastic noise-
              aware simulation). Available: [AerSimulator('aer_simulator'), AerSimulator('aer_simulator_statevector'), AerSimulator('aer_simulator_density_matrix'), AerSimulator('aer_simulator_stabilizer'),
              AerSimulator('aer_simulator_matrix_product_state'), AerSimulator('aer_simulator_extended_stabilizer'), AerSimulator('aer_simulator_unitary'), AerSimulator('aer_simulator_superop'),
              QasmSimulator('qasm_simulator'), StatevectorSimulator('statevector_simulator'), UnitarySimulator('unitary_simulator'), PulseSimulator('pulse_simulator')]
  -ecc ECC    Specify a ecc to be applied to the circuit. Currently available are Q3Shor, Q5Laflamme, Q7Steane, Q9Shor, Q9Surface, and Q18Surface (Default=none)
  -fq FQ      Specify after how many qubit usages error correction is applied to it (Default=100)
  -mc MC      Only allow single controlled gates (Default=False)
  -cf CF      Only allow clifford operations (Default=False)
```

### Available error-correcting codes and operations
| Operation | Q3Shor | Q5Laflamme | Q7Steane | Q9Shor | Q9Surface | Q18Surface |
| --- | --- | --- | --- | --- | --- | --- |
| Pauli (X,Y,Z) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| controlled Pauli (CX,CY,CZ) | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_check_mark: | :heavy_check_mark: | ? | :heavy_multiplication_x: |
| Hadamard      | :warning: | :heavy_multiplication_x: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_check_mark: |
| S, S&dagger;      | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: |
| T, T&dagger;     | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: |

:warning: = operation is applied without the scheme of the error-correcting code (i.e. decoding and encoding is performed before/afterwards, respectively, and the operation is encoded as-is)

### Properties of the implemented error-correcting codes
|  | Q3Shor | Q5Laflamme | Q7Steane | Q9Shor | Q9Surface | Q18Surface |
| --- | --- | --- | --- | --- | --- | --- |
| able to detect bit flips | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| able to detect phase flips | :heavy_multiplication_x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| #physical data qubits per logical qubit | 3 | 5 | 7 | 9 | 9 | 18 |
| #ancilla qubits (total) | 2 | 4 | 3 | 8 | 8 | 18 per qubit |
| #qubits for n logical qubits | 3n+2 | 5n+4 | 7n+3 | 9n+8 | 9n+8 | 36n |
| #classical bits (total) | 2 | 5 | 3 | 8 | 8 | 16 |

More-detailed information about the error-correcting codes can be found in the README information [here](./include/eccs).

## MQT Toolset

The QFR library is the backbone of the quantum software tools in:
- [MQT DDSIM](https://github.com/cda-tum/ddsim): a decision diagram-based simulator for quantum circuits.
- [MQT QMAP](https://github.com/cda-tum/qmap): a tool for mapping/compiling quantum circuits to real quantum architectures.
- [MQT QCEC](https://github.com/cda-tum/qcec): a decision diagram-based equivalence checking tool for quantum circuits.
- [MQT DDVis](http://github.com/cda-tum/ddvis): a visualization tool for how decision diagrams are used in simulation and verification.
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
      dd::QubitCount n = 2;
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
      dd::QubitCount n = 3;
      qc::QFT qft(n); // generates the QFT circuit for n qubits
      ```

      Generates the circuit for the *n* qubit Quantum Fourier Transform.

    * Grover's search algorithm

         ```c++
      dd::QubitCount n = 2;
      qc::Grover grover(n); // generates Grover's algorithm for a random n-bit oracle
      ```

      The algorithm performs ~ &#960;/4 &#8730;2&#8319; Grover iterations. An optional `unsigned int` parameter controls the *seed* of the random number generation for the oracle generation.

    * (Iterative) Quantum Phase Estimation
        ```c++
      dd::QubitCount n = 3;
      bool exact = true; // whether to generate an exactly representable phase or not
      qc::QPE qpe(n, exact);
      qc::IQPE iqpe(n, exact);
      ```
      Generates a random bitstring of length `n` (`n+1` in the inexact case) and constructs the corresponding (iterative) Quantum Phase Estimation algorithm using *n+1* qubits. Alternatively, a specific `phase` and `precision` might be
      passed to the constructor.

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

Building (and running) is continuously tested under Linux, MacOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/virtual-environments). However, the implementation should be compatible
with any current C++ compiler supporting C++17 and a minimum CMake version of 3.14.

*Disclaimer*: We noticed some issues when compiling with Microsoft's `MSVC` compiler toolchain. If you are developing under Windows, consider using the `clang` compiler toolchain. A detailed description of how to set this up can be
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

You can link against the library built by this project in other CMake project using the `MQT::qfr` target.

### Extending the Python Package

To extend the Python package you can locally install the package in edit mode, so that changes in the Python code are instantly available.
The following example assumes you have a [virtual environment](https://docs.python.org/3/library/venv.html) set up and activated.

```commandline
(venv) $ pip install cmake
(venv) $ pip install --editable .
```

If you change parts of the C++ code, you have to run the second line to make the changes visible in Python.
