# MQT Core - The Backbone of the Munich Quantum Toolkit (MQT)

```{raw} latex
\begin{abstract}
```

MQT Core is an open-source C++17 and Python library for quantum computing that forms the backbone of the quantum software tools developed as part of the _{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`_ by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).
To this end, it consists of multiple components that are used throughout the MQT, including a fully fledged intermediate representation (IR) for quantum computations, a state-of-the-art decision diagram (DD) package for quantum computing, and a state-of-the-art ZX-diagram package for working with the ZX-calculus.
This documentation provides a comprehensive guide to the MQT Core library, including {doc}`installation instructions <installation>`, a {doc}`quickstart guide <quickstart>`, and detailed {doc}`API documentation <api/mqt/core/index>`.
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

installation
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
:hidden:
:caption: API Reference

api/mqt/core/index
```
