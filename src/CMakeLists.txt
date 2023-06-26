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

if(NOT TARGET ${PROJECT_NAME}-dd)
  # add DD Package library
  add_library(
    ${PROJECT_NAME}-dd
    ${PROJECT_SOURCE_DIR}/include/dd/Complex.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/ComplexCache.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/ComplexNumbers.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/ComplexTable.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/ComplexValue.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/ComputeTable.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Control.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/DDpackageConfig.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Definitions.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Edge.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Export.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/GateMatrixDefinitions.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Node.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Package.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Package_fwd.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/ToffoliTable.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/UnaryComputeTable.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/DensityNoiseTable.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/StochasticNoiseOperationTable.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/UniqueTable.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/FunctionalityConstruction.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/NoiseFunctionality.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Operations.hpp
    ${PROJECT_SOURCE_DIR}/include/dd/Simulation.hpp
    dd/Edge.cpp
    dd/FunctionalityConstruction.cpp
    dd/Node.cpp
    dd/NoiseFunctionality.cpp
    dd/Operations.cpp
    dd/Simulation.cpp)
  target_link_libraries(${PROJECT_NAME}-dd PUBLIC ${PROJECT_NAME})
  add_library(MQT::CoreDD ALIAS ${PROJECT_NAME}-dd)
  add_library(MQT::${OLD_PROJECT_NAME}_dd ALIAS ${PROJECT_NAME}-dd)
endif()

if(NOT TARGET ${PROJECT_NAME}-zx)
  # add ZX package library
  add_library(
    ${PROJECT_NAME}-zx
    ${PROJECT_SOURCE_DIR}/include/zx/Rational.hpp
    ${PROJECT_SOURCE_DIR}/include/zx/ZXDiagram.hpp
    ${PROJECT_SOURCE_DIR}/include/zx/Definitions.hpp
    ${PROJECT_SOURCE_DIR}/include/zx/Rules.hpp
    ${PROJECT_SOURCE_DIR}/include/zx/Simplify.hpp
    ${PROJECT_SOURCE_DIR}/include/zx/Utils.hpp
    ${PROJECT_SOURCE_DIR}/include/zx/FunctionalityConstruction.hpp
    zx/Rational.cpp
    zx/ZXDiagram.cpp
    zx/Rules.cpp
    zx/Simplify.cpp
    zx/Utils.cpp
    zx/FunctionalityConstruction.cpp)
  target_link_libraries(${PROJECT_NAME}-zx PUBLIC ${PROJECT_NAME})

  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/boost/config" "extern/boost/config"
                   EXCLUDE_FROM_ALL)
  target_link_libraries(${PROJECT_NAME}-zx PUBLIC Boost::config)

  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/boost/multiprecision"
                   "extern/boost/multiprecision" EXCLUDE_FROM_ALL)
  target_link_libraries(${PROJECT_NAME}-zx PUBLIC Boost::multiprecision)
  # the following sets the SYSTEM flag for the include dirs of the boost libs to suppress warnings
  # cmake-lint: disable=C0307
  set_target_properties(
    boost_config PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                            $<TARGET_PROPERTY:boost_config,INTERFACE_INCLUDE_DIRECTORIES>)
  # cmake-lint: disable=C0307
  set_target_properties(
    boost_multiprecision
    PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
               $<TARGET_PROPERTY:boost_multiprecision,INTERFACE_INCLUDE_DIRECTORIES>)

  # # link to GMP libraries if present
  if(GMP_FOUND)
    target_compile_definitions(${PROJECT_NAME}-zx PUBLIC GMP)
    target_link_libraries(${PROJECT_NAME}-zx PUBLIC GMP::gmp GMP::gmpxx)
  endif()

  # add MQT alias
  add_library(MQT::CoreZX ALIAS ${PROJECT_NAME}-zx)
  add_library(MQT::${OLD_PROJECT_NAME}_zx ALIAS ${PROJECT_NAME}-zx)
endif()

if(NOT TARGET ${PROJECT_NAME}-ecc)
  # add ECC package library
  add_library(
    ${PROJECT_NAME}-ecc
    ${PROJECT_SOURCE_DIR}/include/ecc/Ecc.hpp
    ${PROJECT_SOURCE_DIR}/include/ecc/Id.hpp
    ${PROJECT_SOURCE_DIR}/include/ecc/Q3Shor.hpp
    ${PROJECT_SOURCE_DIR}/include/ecc/Q5Laflamme.hpp
    ${PROJECT_SOURCE_DIR}/include/ecc/Q7Steane.hpp
    ${PROJECT_SOURCE_DIR}/include/ecc/Q9Shor.hpp
    ${PROJECT_SOURCE_DIR}/include/ecc/Q9Surface.hpp
    ${PROJECT_SOURCE_DIR}/include/ecc/Q18Surface.hpp
    ecc/Ecc.cpp
    ecc/Q3Shor.cpp
    ecc/Q5Laflamme.cpp
    ecc/Q7Steane.cpp
    ecc/Q9Shor.cpp
    ecc/Q9Surface.cpp
    ecc/Q18Surface.cpp)

  target_link_libraries(${PROJECT_NAME}-ecc PUBLIC ${PROJECT_NAME})

  # add MQT alias
  add_library(MQT::CoreECC ALIAS ${PROJECT_NAME}-ecc)
  add_library(MQT::${OLD_PROJECT_NAME}_ecc ALIAS ${PROJECT_NAME}-ecc)
endif()

if(NOT TARGET ${PROJECT_NAME}-python)
  # add pybind11 library
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11" "extern/pybind11" EXCLUDE_FROM_ALL)

  if(NOT TARGET pybind11_json)
    # add pybind11_json library
    add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11_json" "extern/pybind11_json"
                     EXCLUDE_FROM_ALL)
  endif()

  # add Python interface library
  add_library(
    ${PROJECT_NAME}-python
    ${PROJECT_SOURCE_DIR}/include/python/qiskit/QuantumCircuit.hpp
    ${PROJECT_SOURCE_DIR}/include/python/qiskit/QasmQobjExperiment.hpp
    python/qiskit/QuantumCircuit.cpp python/qiskit/QasmQobjExperiment.cpp)

  # link with main project library and pybind11 libraries
  target_link_libraries(${PROJECT_NAME}-python PRIVATE ${PROJECT_NAME} pybind11::pybind11
                                                       pybind11_json)

  # add MQT alias
  add_library(MQT::CorePython ALIAS ${PROJECT_NAME}-python)
  add_library(MQT::${OLD_PROJECT_NAME}_python ALIAS ${PROJECT_NAME}-python)
endif()

if(BINDINGS)
  add_subdirectory(python)
endif()
