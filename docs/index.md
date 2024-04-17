# MQT Core - The Backbone of the Munich Quantum Toolkit (MQT)

```{raw} latex
\begin{abstract}
```

MQT Core is an open-source C++17 and Python library for quantum computing that forms the backbone of the quantum software tools developed as part of the _{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`_ by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).
It consists of multiple components that are used throughout the MQT. Specifically, it includes

1. a fully fledged intermediate representation (IR) for quantum computations that is used
   to represent and manipulate quantum circuits as well as to interface with other quantum software tools (such as Qiskit), e.g., in our quantum circuit compiler {doc}`MQT QMAP <qmap:index>`.
2. a state-of-the-art decision diagram (DD) package for quantum computing that is used for classical quantum circuit simulation (as part of {doc}`MQT DDSIM <ddsim:index>`) and verification (as part of {doc}`MQT QCEC <qcec:index>`).
3. a state-of-the-art ZX-diagram package for working with the ZX-calculus, e.g., for verification as part of {doc}`MQT QCEC <qcec:index>`.

This documentation provides a comprehensive guide to the MQT Core library, including installation instructions, a quickstart guide, and detailed API documentation.
The source code of MQT Core is publicly available on GitHub at [cda-tum/mqt-core](https://github.com/cda-tum/mqt-core), while pre-built binaries are available via [PyPI](https://pypi.org/project/mqt.core/) for all major operating systems and all modern Python versions.
MQT Core is fully compatible with Qiskit 1.0 and above.

````{only} latex
```{note}
A live version of this document is available at [mqt.readthedocs.io/projects/core](https://mqt.readthedocs.io/projects/core).
```
````

```{raw} latex
\end{abstract}

\sphinxtableofcontents
```

```{toctree}
:hidden:

self
```

```{toctree}
:maxdepth: 2
:caption: User Guide

quickstart
```

````{only} not latex
```{toctree}
:maxdepth: 2
:caption: DD Package Evaluation

dd_package_evaluation
```

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Developers
:glob:

contributing
support
DevelopmentGuide
```
````

```{toctree}
:maxdepth: 3
:caption: API Reference

api/mqt/core/index
```
