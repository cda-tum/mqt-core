set(MQT_CORE_TARGET_NAME "mqt-core")
set(MQT_CORE_TARGETS "")

if(MQT_CORE_INSTALL)
  include(GNUInstallDirs)
  set(MQT_CORE_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/mqt-core")
  set(MQT_CORE_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/mqt-core")
  set(MQT_CORE_TARGETS_EXPORT_NAME "mqt-core-targets")
  set(MQT_CORE_CMAKE_CONFIG_TEMPLATE "${PROJECT_SOURCE_DIR}/cmake/mqt-core-config.cmake.in")
  set(MQT_CORE_CMAKE_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(MQT_CORE_CMAKE_VERSION_CONFIG_FILE
      "${MQT_CORE_CMAKE_CONFIG_DIR}/mqt-core-config-version.cmake")
  set(MQT_CORE_CMAKE_PROJECT_CONFIG_FILE "${MQT_CORE_CMAKE_CONFIG_DIR}/mqt-core-config.cmake")
  set(MQT_CORE_CMAKE_PROJECT_TARGETS_FILE "${MQT_CORE_CMAKE_CONFIG_DIR}/mqt-core-targets.cmake")

  if(APPLE)
    set(BASEPOINT @loader_path)
  else()
    set(BASEPOINT $ORIGIN)
  endif()
  set(CMAKE_INSTALL_RPATH ${BASEPOINT} ${BASEPOINT}/${CMAKE_INSTALL_LIBDIR})
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(NOT TARGET project_warnings)
  # Use the warnings specified in CompilerWarnings.cmake
  add_library(project_warnings INTERFACE)

  # Standard compiler warnings
  include(${PROJECT_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
  set_project_warnings(project_warnings)

  # Add MQT alias
  add_library(MQT::ProjectWarnings ALIAS project_warnings)
  set_target_properties(project_warnings PROPERTIES EXPORT_NAME ProjectWarnings)
endif()

if(NOT TARGET project_options)
  # Use the options specified in CompilerOptions.cmake
  add_library(project_options INTERFACE)

  # Standard compiler options
  include(${PROJECT_SOURCE_DIR}/cmake/CompilerOptions.cmake)
  enable_project_options(project_options)

  # Sanitizer options if supported by compiler
  include(${PROJECT_SOURCE_DIR}/cmake/Sanitizers.cmake)
  enable_sanitizers(project_options)

  # Add MQT alias
  add_library(MQT::ProjectOptions ALIAS project_options)
  set_target_properties(project_options PROPERTIES EXPORT_NAME ProjectOptions)
endif()

if(NOT TARGET ${MQT_CORE_TARGET_NAME})
  # main project library
  add_library(
    ${MQT_CORE_TARGET_NAME}
    ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/BernsteinVazirani.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/Entanglement.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/GoogleRandomCircuitSampling.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/Grover.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/QFT.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/QPE.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/RandomCliffordCircuit.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/WState.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/CircuitOptimizer.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/Definitions.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/operations/Expression.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/operations/ClassicControlledOperation.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/operations/CompoundOperation.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/operations/Control.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/operations/NonUnitaryOperation.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/operations/Operation.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/operations/StandardOperation.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/operations/SymbolicOperation.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/Scanner.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/Token.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/Parser.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/Statement.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/Types.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/InstVisitor.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/NestedEnvironment.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/Exception.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/Gate.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/StdGates.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/passes/CompilerPass.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/passes/ConstEvalPass.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/parsers/qasm3_parser/passes/TypeCheckPass.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/Permutation.hpp
    ${MQT_CORE_INCLUDE_BUILD_DIR}/QuantumComputation.hpp
    algorithms/BernsteinVazirani.cpp
    algorithms/Entanglement.cpp
    algorithms/GoogleRandomCircuitSampling.cpp
    algorithms/Grover.cpp
    algorithms/QFT.cpp
    algorithms/QPE.cpp
    algorithms/RandomCliffordCircuit.cpp
    algorithms/WState.cpp
    CircuitOptimizer.cpp
    operations/ClassicControlledOperation.cpp
    operations/CompoundOperation.cpp
    operations/Expression.cpp
    operations/NonUnitaryOperation.cpp
    operations/Operation.cpp
    operations/StandardOperation.cpp
    operations/SymbolicOperation.cpp
    parsers/GRCSParser.cpp
    parsers/QASM3Parser.cpp
    parsers/QCParser.cpp
    parsers/RealParser.cpp
    parsers/TFCParser.cpp
    parsers/qasm3_parser/Parser.cpp
    parsers/qasm3_parser/Scanner.cpp
    parsers/qasm3_parser/Types.cpp
    parsers/qasm3_parser/Statement.cpp
    parsers/qasm3_parser/passes/ConstEvalPass.cpp
    parsers/qasm3_parser/passes/TypeCheckPass.cpp
    QuantumComputation.cpp)

  # set include directories
  target_include_directories(
    ${MQT_CORE_TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${MQT_CORE_INCLUDE_BUILD_DIR}>
                                   $<INSTALL_INTERFACE:${MQT_CORE_INCLUDE_INSTALL_DIR}>)

  # add options and warnings to the library
  target_link_libraries(${MQT_CORE_TARGET_NAME} PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  # add MQT alias
  add_library(MQT::Core ALIAS ${MQT_CORE_TARGET_NAME})
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME Core)
  list(APPEND MQT_CORE_TARGETS ${MQT_CORE_TARGET_NAME})
endif()

# add datastructures package
add_subdirectory(datastructures)

# add DD package library
add_subdirectory(dd)

# add ZX package library
add_subdirectory(zx)

# add ECC library
add_subdirectory(ecc)

# add NA library
add_subdirectory(na)

# ** Note ** The following target will soon be removed from the project. All top-level projects
# should switch to using the mqt-core Python package.
if(BINDINGS AND NOT TARGET mqt-core-python)
  # add Python interface library
  add_library(
    ${MQT_CORE_TARGET_NAME}-python ${MQT_CORE_INCLUDE_BUILD_DIR}/python/qiskit/QuantumCircuit.hpp
                                   python/qiskit/QuantumCircuit.cpp)

  # link with main project library and pybind11 libraries
  target_link_libraries(${MQT_CORE_TARGET_NAME}-python PUBLIC MQT::Core pybind11::pybind11)
  target_link_libraries(${MQT_CORE_TARGET_NAME}-python PRIVATE MQT::ProjectOptions
                                                               MQT::ProjectWarnings)

  # add MQT alias
  add_library(MQT::CorePython ALIAS ${MQT_CORE_TARGET_NAME}-python)
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-python
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CorePython)
  list(APPEND MQT_CORE_TARGETS ${MQT_CORE_TARGET_NAME}-python)
endif()

if(BUILD_MQT_CORE_BINDINGS)
  add_subdirectory(python)
endif()

# Installation instructions for the main library
if(MQT_CORE_INSTALL)
  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    ${MQT_CORE_CMAKE_CONFIG_TEMPLATE} ${MQT_CORE_CMAKE_PROJECT_CONFIG_FILE}
    INSTALL_DESTINATION ${MQT_CORE_CONFIG_INSTALL_DIR}
    PATH_VARS MQT_CORE_ZX_WITH_GMP
    NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO)
  write_basic_package_version_file(
    ${MQT_CORE_CMAKE_VERSION_CONFIG_FILE}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

  install(FILES ${MQT_CORE_CMAKE_PROJECT_CONFIG_FILE} ${MQT_CORE_CMAKE_VERSION_CONFIG_FILE}
          DESTINATION ${MQT_CORE_CONFIG_INSTALL_DIR})

  install(
    TARGETS ${MQT_CORE_TARGETS} project_warnings project_options
    EXPORT ${MQT_CORE_TARGETS_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
            NAMELINK_COMPONENT ${MQT_CORE_TARGET_NAME}_Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Development
    INCLUDES
    DESTINATION ${MQT_CORE_INCLUDE_INSTALL_DIR})

  install(DIRECTORY ${MQT_CORE_INCLUDE_BUILD_DIR}/ DESTINATION ${MQT_CORE_INCLUDE_INSTALL_DIR})

  install(
    EXPORT ${MQT_CORE_TARGETS_EXPORT_NAME}
    FILE ${MQT_CORE_PROJECT_TARGETS_FILE}
    NAMESPACE MQT::
    DESTINATION ${MQT_CORE_CONFIG_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Cache.cmake
          ${PROJECT_SOURCE_DIR}/cmake/CompilerOptions.cmake
          ${PROJECT_SOURCE_DIR}/cmake/CompilerWarnings.cmake
          ${PROJECT_SOURCE_DIR}/cmake/FindGMP.cmake
          ${PROJECT_SOURCE_DIR}/cmake/PackageAddTest.cmake
          ${PROJECT_SOURCE_DIR}/cmake/PreventInSourceBuilds.cmake
          ${PROJECT_SOURCE_DIR}/cmake/Sanitizers.cmake
          ${PROJECT_SOURCE_DIR}/cmake/StandardProjectSettings.cmake
    DESTINATION ${MQT_CORE_CONFIG_INSTALL_DIR})
endif()
