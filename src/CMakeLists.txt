# add MQT::DDPackage library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/dd_package" "extern/dd_package" EXCLUDE_FROM_ALL)

# add MQT::zx library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/zx" "extern/zx" EXCLUDE_FROM_ALL)

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
# the following sets the SYSTEM flag for the include dirs of the json libs to suppress warnings
# cmake-lint: disable=C0307
set_target_properties(
  nlohmann_json PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                           $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)

# add options and warnings to the library
target_link_libraries(${PROJECT_NAME} PUBLIC project_options project_warnings)

# Temporary: The ZX package is not yet clearly separated and some of its definitions are used
# throughout the library. Therefore, the ZX package always needs to be linked into the main QFR
# library.
target_link_libraries(${PROJECT_NAME} PUBLIC MQT::zx)

# add MQT alias
add_library(MQT::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# add DD Package library
add_library(
  ${PROJECT_NAME}_dd
  ${PROJECT_SOURCE_DIR}/include/dd/FunctionalityConstruction.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/NoiseFunctionality.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Operations.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Simulation.hpp
  dd/FunctionalityConstruction.cpp
  dd/NoiseFunctionality.cpp
  dd/Operations.cpp
  dd/Simulation.cpp)
target_link_libraries(${PROJECT_NAME}_dd PUBLIC MQT::DDPackage MQT::${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_dd PUBLIC ${PROJECT_SOURCE_DIR}/include/dd
                                                     ${PROJECT_BINARY_DIR}/include/dd)
add_library(MQT::${PROJECT_NAME}_dd ALIAS ${PROJECT_NAME}_dd)

# add ZX package library
add_library(${PROJECT_NAME}_zx ${PROJECT_SOURCE_DIR}/include/zx/FunctionalityConstruction.hpp
                               zx/FunctionalityConstruction.cpp)
target_link_libraries(${PROJECT_NAME}_zx PUBLIC MQT::zx MQT::${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_zx PUBLIC ${PROJECT_SOURCE_DIR}/include/zx
                                                     ${PROJECT_BINARY_DIR}/include/zx)
add_library(MQT::${PROJECT_NAME}_zx ALIAS ${PROJECT_NAME}_zx)

# add ECC package library
add_library(
  ${PROJECT_NAME}_ecc
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

target_link_libraries(${PROJECT_NAME}_ecc PUBLIC MQT::${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_ecc PUBLIC ${PROJECT_SOURCE_DIR}/include/ecc
                                                      ${PROJECT_BINARY_DIR}/include/ecc)
add_library(MQT::${PROJECT_NAME}_ecc ALIAS ${PROJECT_NAME}_ecc)
