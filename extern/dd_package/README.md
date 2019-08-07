# A Package for Decision Diagrams Written in C++

DD package by JKU Linz, Austria

Developers: Alwin Zulehner, Stefan Hillmich, Robert Wille

With code from the QMDD implementation provided by Michael Miller (University of Victoria, Canada)
and Philipp Niemann (University of Bremen, Germany).

For more information, please visit [iic.jku.at/eda/research/quantum_dd](http://iic.jku.at/eda/research/quantum_dd).

If you have any questions feel free to contact us using [iic_quantum@jku.at](mailto:iic_quantum@jku.at).

## Usage

### System Requirements

The package has been tested under Linux (Ubuntu 18.04, 64-bit) and should be compatible with any current version of g++/cmake.
  
### Build and Run 

To build the package and run a small demo type:
```
$ mkdir build
$ cd build 
$ cmake ..
$ make
$ ./dd_example
Circuits are equal!
00: 0.707107 + 0i
01: 0 + 0i
10: 0 + 0i
11: 0.707107 + 0i
Bell states have a fidelity of 1
Bell state and zero state have a fidelity of 0.707107

DD statistics:
  Current # nodes in UniqueTable: 18
  Total compute table lookups: 33
  Number of operations:
    add:  20
    mult: 95
    kron: 0
  Compute table hit ratios (hits/looks/ratio):
    adds: 0 / 4 / 0
    mult: 14 / 29 / 0.482759
    kron: 0 / 0 / -nan
  UniqueTable:
    Collisions: 2
    Matches:    20
```

## Reference

If you use the DD package for your research, we will be thankful if you refer to it by citing the following publication:

```
@article{zulehner2019package,
    title={How to Efficiently Handle Complex Values? Implementing Decision Diagrams for Quantum Computing},
    author={Zulehner, Alwin and Hillmich, Stefan and Wille, Robert},
    journal={{IEEE/ACM} International Conference on Computer-Aided Design},
    year={2019},
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
