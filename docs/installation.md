# Installation

MQT Core is mainly developed as a C++17 library with Python bindings.
The resulting Python package is available on [PyPI](https://pypi.org/project/mqt.core/) and can be installed via `pip` for all major operating systems and all modern Python versions.

```console
(.venv) $ pip install mqt.core
```

In most practical cases (under 64-bit Linux, MacOS incl. Apple Silicon, and Windows), this requires no compilation and merely downloads and installs a platform-specific pre-built wheel.

:::{attention}
As of version 2.7.0, support for Python 3.8 has been officially dropped.
We strongly recommend that users upgrade to a more recent version of Python to ensure compatibility and continue receiving updates and support.
Thank you for your understanding.
:::

## Building from source for performance

In order to get the best performance and enable platform-specific optimizations that cannot be enabled on portable wheels, it is recommended to build the library from source via:

```console
(.venv) $ pip install mqt.core --no-binary mqt.core
```

This requires a [C++ compiler supporting C++17](https://en.wikipedia.org/wiki/List_of_compilers#C++_compilers) and a minimum [CMake](https://cmake.org/) version of 3.19.
The library is continuously tested under Linux, MacOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/virtual-environments).
In order to access the latest build logs, visit the [GitHub Actions page](https://github.com/cda-tum/mqt-core/actions/workflows/ci.yml).

## Integrating MQT Core into your project

If you want to use the MQT Core Python package in your own project, you can simply add it as a dependency in your `pyproject.toml` or `setup.py` file.
This will automatically install the MQT Core package when your project is installed.

::::{tab-set}
:::{tab-item} pyproject.toml

```toml
[project]
# ...
dependencies = ["mqt.core>=2.4.0"]
# ...
```

:::

:::{tab-item} setup.py

```python
from setuptools import setup

setup(
    # ...
    install_requires=["mqt.core>=2.4.0"],
    # ...
)
```

:::
::::

If you want to integrate the C++ library directly into your project, you can either

- add it as a git submodule and build it as part of your project, or
- install MQT Core on your system and use CMakes `find_package` command to locate it, or
- use CMake's [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) module to combine both approaches.

::::{tab-set}
:::{tab-item} FetchContent

This is the recommended approach for projects because it allows to detect installed versions of MQT Core and only downloads the library if it is not available on the system.
Furthermore, CMake's [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) module allows for lots of flexibility in how the library is integrated into the project.

```cmake
include(FetchContent)
set(FETCH_PACKAGES "")

set(MQT_CORE_VERSION 2.4.0 CACHE STRING "MQT Core version")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(
    mqt-core
    GIT_REPOSITORY https://github.com/cda-tum/mqt-core.git
    GIT_TAG v${MQT_CORE_VERSION}
    FIND_PACKAGE_ARGS ${MQT_CORE_VERSION})
  list(APPEND FETCH_PACKAGES mqt-core)
else()
  find_package(mqt-core ${MQT_CORE_VERSION} QUIET)
  if(NOT mqt-core_FOUND)
    FetchContent_Declare(
      mqt-core
      GIT_REPOSITORY https://github.com/cda-tum/mqt-core.git
      GIT_TAG v${MQT_CORE_VERSION})
    list(APPEND FETCH_PACKAGES mqt-core)
  endif()
endif()

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
```

:::

:::{tab-item} git submodule

Integrating the library as a git submodule is the simplest approach.
However, handling git submodules can be cumbersome, especially when working with multiple branches or versions of the library.
First, add the submodule to your project (e.g., in the `external` directory) via:

```bash
git submodule add https://github.com/cda-tum/mqt-core.git external/mqt-core
```

Then, add the following lines to your `CMakeLists.txt` to make the library's targets available in your project:

```cmake
add_subdirectory(external/mqt-core)
```

:::

:::{tab-item} find_package

MQT Core can be installed on your system after building it from source.

```bash
git clone https://github.com/cda-tum/mqt-core.git
cd mqt-core
cmake -S . -B build
cmake --build build
cmake --install build
```

Then, in your project's `CMakeLists.txt`, you can use the `find_package` command to locate the installed library:

```cmake
find_package(mqt-core 2.4.0 REQUIRED)
```

::::
