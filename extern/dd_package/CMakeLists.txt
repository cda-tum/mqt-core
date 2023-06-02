# project definition
project(
  DDPackage
  LANGUAGES CXX
  VERSION 2.1.0
  DESCRIPTION "MQT decision diagram package tailored to quantum computing")

# add main library code
if(NOT TARGET ${PROJECT_NAME})
  add_library(
    ${PROJECT_NAME}
    INTERFACE
    include/dd/Complex.hpp
    include/dd/ComplexCache.hpp
    include/dd/ComplexNumbers.hpp
    include/dd/ComplexTable.hpp
    include/dd/ComplexValue.hpp
    include/dd/ComputeTable.hpp
    include/dd/Control.hpp
    include/dd/Definitions.hpp
    include/dd/Edge.hpp
    include/dd/Export.hpp
    include/dd/GateMatrixDefinitions.hpp
    include/dd/Node.hpp
    include/dd/Package.hpp
    include/dd/ToffoliTable.hpp
    include/dd/UnaryComputeTable.hpp
    include/dd/DensityNoiseTable.hpp
    include/dd/StochasticNoiseOperationTable.hpp
    include/dd/UniqueTable.hpp)

  # add options and warnings to library
  target_link_libraries(${PROJECT_NAME} INTERFACE project_options project_warnings)

  # set include directories
  target_include_directories(${PROJECT_NAME} INTERFACE include ${PROJECT_BINARY_DIR}/include)

  # add MQT alias
  add_library(MQT::DDPackage ALIAS ${PROJECT_NAME})
endif()

# add test code
option(BUILD_DD_PACKAGE_TESTS "Also build tests for DD package")
if(BUILD_DD_PACKAGE_TESTS)
  check_submodule_present(googletest)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
