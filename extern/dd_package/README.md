[![Build Status](https://travis-ci.com/iic-jku/dd_package.svg?branch=master)](https://travis-ci.com/iic-jku/dd_package)
[![codecov](https://codecov.io/gh/iic-jku/dd_package/branch/master/graph/badge.svg)](https://codecov.io/gh/iic-jku/dd_package)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# A Package for Decision Diagrams Written in C++

A DD package tailored to quantum computing by the [Institute for Integrated Circuits](http://iic.jku.at/eda/) at the [Johannes Kepler University Linz](https://jku.at).

Developers: Alwin Zulehner, Stefan Hillmich, Lukas Burgholzer, and Robert Wille

With code from the QMDD implementation provided by Michael Miller (University of Victoria, Canada)
and Philipp Niemann (University of Bremen, Germany).

For more information, please visit [iic.jku.at/eda/research/quantum_dd](http://iic.jku.at/eda/research/quantum_dd).

If you have any questions feel free to contact us using [iic_quantum@jku.at](mailto:iic_quantum@jku.at) or opening an issue.

The old version of this package which does not use namespaces or classes can be found in the branch `non-oop`.

## Usage

This package caters primarily to our requirements regarding quantum-related functionality and, hence, may not be straightforward to use for other purposes.

A small example shows how to create set a single qubit in superposition.

```c++
auto* dd = std::make_new dd::Package; // Create new package instance
dd::Edge zero_state = dd->makeZeroState(1) ; // zero_state = |0>

/* Creating a DD requires three inputs:
 * 1. A 2x2 matrix describing a single qubits operation (here: the Hadamard matrix)
 * 2. The number of qubits the DD will operated on (here: one qubit)
 * 3. An int array the length of the the number of qubits where the index is the qubit and the value is either
 *    -1 -> don't care; 0 -> negative control; 1 -> positive control; 2 -> target qubit
 *    In this example we only have a target.
 */
int line[1] = {2};
dd::Edge h_op = dd->makeGateDD(Hmat, 1, line);

// Multiplying the operation and the state results in a new state, here a single qubit in super position
dd::Edge superposition = dd->multiply(h_op, zero_state); 

delete dd; // alternatively use smart pointers ;)
```

For implementing more complex functionality which requires garbage collection, be advised that you have to do the reference counting by hand. 

### System Requirements

Building (and running) is continuously tested under Linux (Ubuntu 18.04) using gcc-7.4, gcc-9 and clang-9, MacOS (Mojave 10.14) using AppleClang and gcc-9, and Windows using MSVC 15.9. 
However, the implementation should be compatible with any current C++ compiler supporting C++11 and a minimum CMake version of 3.10.

It is recommended (although not required) to have [GraphViz](https://www.graphviz.org) installed for visualization purposes.
  
### Build and Run 

To build the package and run a small demo execute the following 
(several *.dot files will be created in the working directory which will be automatically converted to SVG if GraphViz is installed).
```commandline
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ cmake --build . --config Release
$ ./test/dd_package_example
Circuits are equal!
00: √½
01: 0
10: 0
11: √½
Bell states have a fidelity of 1
Bell state and zero state have a fidelity of 0.5
DD of my gate has size 2
Identity function for 4 qubits has trace: 16

DD statistics:
  Current # nodes in UniqueTable: 22
  Total compute table lookups: 37
  Number of operations:
    add:  44
    mult: 86
    kron: 0
  Compute table hit ratios (hits/looks/ratio):
    adds: 2 / 4 / 0.5
    mult: 14 / 28 / 0.5
    kron: 0 / 0 / 0
  UniqueTable:
    Collisions: 5
    Matches:    17
```
**Windows users** using Visual Studio and the MSVC compiler need to build the project with 
```commandline
$ mkdir build && cd build
$ cmake .. -G "Visual Studio 15 2017" -A x64 -DCMAKE_BUILD_TYPE=Release
$ cmake --build . --config Release
```

As of now, a small set of unit tests is included as well which you can execute as follows (after performing the build steps described above).

```
$ ./test/dd_package_test
Running main() from [...]/extern/googletest/googletest/src/gtest_main.cc
[==========] Running 4 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 1 test from DDComplexTest
[ RUN      ] DDComplexTest.TrivialTest
[       OK ] DDComplexTest.TrivialTest (1 ms)
[----------] 1 test from DDComplexTest (1 ms total)

[----------] 3 tests from DDPackageTest
[ RUN      ] DDPackageTest.TrivialTest
[       OK ] DDPackageTest.TrivialTest (31 ms)
[ RUN      ] DDPackageTest.BellState
[       OK ] DDPackageTest.BellState (21 ms)
[ RUN      ] DDPackageTest.IdentifyTrace
[       OK ] DDPackageTest.IdentifyTrace (18 ms)
[----------] 3 tests from DDPackageTest (70 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 2 test suites ran. (71 ms total)
[  PASSED  ] 4 tests.
```

The DD package may be installed on the system by executing

```commandline
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ cmake --build . --config Release --target install
```

It can then be included in other projects using the following CMake snippet

```cmake
find_package(DDpackage)
target_link_libraries(${TARGET_NAME} PRIVATE DD::DDpackage)
```

## Reference

If you use the DD package for your research, we will be thankful if you refer to it by citing the following publication:

```
@article{zulehner2019package,
    title={How to Efficiently Handle Complex Values? Implementing Decision Diagrams for Quantum Computing},
    author={Zulehner, Alwin and Hillmich, Stefan and Wille, Robert},
    journal={International Conference on Computer Aided Design (ICCAD)},
    year={2019}
}
```

## Further Information

The following papers provide further information on different aspects of representing states and operation in the quantum realm.

- For the representation of unitary matrices and state vectors (with a particular focus on simulation and measurement):  
A. Zulehner and R. Wille. Advanced Simulation of Quantum Computations. IEEE Transactions on Computer Aided Design of Integrated Circuits and Systems (TCAD), 2018.
- For the representation and manipulation of unitary matrices (including proof of canonicy, multi-qubit systems, etc):  
P. Niemann, R. Wille, D. M. Miller, M. A. Thornton, and R. Drechsler. QMDDs: Efficient Quantum Function Representation and Manipulation. IEEE Transactions on Computer Aided Design of Integrated Circuits and Systems (TCAD), 35(1):86-99, 2016.
- The paper describing this decision diagram package (with a special focus on the representation of complex numbers):  
A. Zulehner, S. Hillmich and R. Wille. How to Efficiently Handle Complex Values? Implementing Decision Diagrams for Quantum Computing. The IEEE/ACM International Conference on Computer-Aided Design (ICCAD). 2019
