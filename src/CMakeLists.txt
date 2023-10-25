if(NOT TARGET nlohmann_json)
  # cmake-lint: disable=C0103
  set(JSON_BuildTests
      OFF
      CACHE INTERNAL "")
  # cmake-lint: disable=C0103
  set(JSON_MultipleHeaders
      OFF
      CACHE INTERNAL "")
  # add nlohmann::json library
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/json" "extern/json" EXCLUDE_FROM_ALL)

  # the following sets the SYSTEM flag for the include dirs of the json libs to suppress warnings
  set_target_properties(
    nlohmann_json PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                             $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
  )# cmake-lint: disable=C0307
endif()

if(NOT TARGET ${PROJECT_NAME})
  # main project library
  add_library(
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/include/algorithms/BernsteinVazirani.hpp
    ${PROJECT_SOURCE_DIR}/include/algorithms/Entanglement.hpp
    ${PROJECT_SOURCE_DIR}/include/algorithms/GoogleRandomCircuitSampling.hpp
    ${PROJECT_SOURCE_DIR}/include/algorithms/Grover.hpp
    ${PROJECT_SOURCE_DIR}/include/algorithms/QFT.hpp
    ${PROJECT_SOURCE_DIR}/include/algorithms/QPE.hpp
    ${PROJECT_SOURCE_DIR}/include/algorithms/RandomCliffordCircuit.hpp
    ${PROJECT_SOURCE_DIR}/include/algorithms/WState.hpp
    ${PROJECT_SOURCE_DIR}/include/CircuitOptimizer.hpp
    ${PROJECT_SOURCE_DIR}/include/Definitions.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Expression.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/ClassicControlledOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/CompoundOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Control.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/NonUnitaryOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/Operation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/StandardOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/operations/SymbolicOperation.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Parser.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Scanner.hpp
    ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Token.hpp
    ${PROJECT_SOURCE_DIR}/include/Permutation.hpp
    ${PROJECT_SOURCE_DIR}/include/QuantumComputation.hpp
    algorithms/BernsteinVazirani.cpp
    algorithms/Entanglement.cpp
    algorithms/GoogleRandomCircuitSampling.cpp
    algorithms/Grover.cpp
    algorithms/QFT.cpp
    algorithms/QPE.cpp
    algorithms/RandomCliffordCircuit.cpp
    algorithms/WState.cpp
    CircuitOptimizer.cpp
    operations/Expression.cpp
    operations/NonUnitaryOperation.cpp
    operations/Operation.cpp
    operations/StandardOperation.cpp
    operations/SymbolicOperation.cpp
    parsers/GRCSParser.cpp
    parsers/QASMParser.cpp
    parsers/QCParser.cpp
    parsers/RealParser.cpp
    parsers/TFCParser.cpp
    parsers/qasm_parser/Parser.cpp
    parsers/qasm_parser/Scanner.cpp
    QuantumComputation.cpp)

  # set include directories
  target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
                                                    ${PROJECT_BINARY_DIR}/include)

  target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json)

  # add options and warnings to the library
  target_link_libraries(${PROJECT_NAME} PUBLIC project_options project_warnings)

  # add MQT alias
  add_library(MQT::Core ALIAS ${PROJECT_NAME})
  add_library(MQT::${OLD_PROJECT_NAME} ALIAS ${PROJECT_NAME})
endif()

# add DD package library
add_subdirectory(dd)

# add ZX package library
add_subdirectory(zx)

# add ECC library
add_subdirectory(ecc)

# ** Note ** The following target will soon be removed from the project. All top-level projects
# should switch to nanobind. After that, the pybind submodules will be removed.
if(BINDINGS AND NOT TARGET ${PROJECT_NAME}-python)
  if(NOT SKBUILD)
    # Manually detect the installed pybind11 package and import it into CMake.
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE pybind11_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
  endif()

  # add pybind11 library
  find_package(pybind11 CONFIG REQUIRED)

  if(NOT TARGET pybind11_json)
    # add pybind11_json library
    add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11_json" "extern/pybind11_json"
                     EXCLUDE_FROM_ALL)
  endif()

  # add Python interface library
  add_library(${PROJECT_NAME}-python ${PROJECT_SOURCE_DIR}/include/python/qiskit/QuantumCircuit.hpp
                                     python/qiskit/QuantumCircuit.cpp)

  # link with main project library and pybind11 libraries
  target_link_libraries(${PROJECT_NAME}-python PUBLIC ${PROJECT_NAME} pybind11::pybind11
                                                      pybind11_json)

  # the following sets the SYSTEM flag for the include dirs of the pybind11_json libs to suppress
  # warnings
  set_target_properties(
    pybind11_json PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                             $<TARGET_PROPERTY:pybind11_json,INTERFACE_INCLUDE_DIRECTORIES>
  )# cmake-lint: disable=C0307

  # add MQT alias
  add_library(MQT::CorePython ALIAS ${PROJECT_NAME}-python)
  add_library(MQT::${OLD_PROJECT_NAME}_python ALIAS ${PROJECT_NAME}-python)
endif()

if(BUILD_MQT_CORE_BINDINGS)
  add_subdirectory(python)
endif()
