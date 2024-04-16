# MQT Core - The Backbone of the Munich Quantum Toolkit (MQT)

The MQT Core library forms the backbone of the quantum software tools developed as part of the _{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`_ by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/). This includes the following tools:

- {doc}`MQT DDSIM <ddsim:index>`: A Tool for Classical Quantum Circuit Simulation based on Decision Diagrams.
- {doc}`MQT QMAP <qmap:index>`: A Tool for Quantum Circuit Mapping.
- {doc}`MQT QCEC <qcec:index>`: A Tool for Quantum Circuit Equivalence Checking.
- {doc}`MQT QECC <qecc:index>`: A Tool for Quantum Error Correcting Codes.
- [MQT DDVis](https://github.com/cda-tum/mqt-ddvis): A Web-Application visualizing Decision Diagrams for Quantum Computing.
- {doc}`MQT SyReC <syrec:index>`: A Tool for Synthesis of Reversible Circuits/Quantum Computing Oracles.

For a full list of tools and libraries, please visit the {doc}`MQT website <mqt:index>`.

```{include} ../README.md
:start-after: <!-- SPHINX-START -->
:end-before: <!-- SPHINX-END -->
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

```{toctree}
:maxdepth: 2
:caption: DD Package Evaluation

dd_package_evaluation
```

````{only} not latex
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
