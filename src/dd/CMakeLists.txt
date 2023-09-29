if(NOT TARGET ${PROJECT_NAME}-dd)
  file(GLOB_RECURSE DD_HEADERS ${PROJECT_SOURCE_DIR}/include/dd/*.hpp)

  # add DD Package library
  add_library(
    ${PROJECT_NAME}-dd
    ${DD_HEADERS}
    CachedEdge.cpp
    Complex.cpp
    ComplexNumbers.cpp
    ComplexValue.cpp
    Edge.cpp
    FunctionalityConstruction.cpp
    MemoryManager.cpp
    Node.cpp
    NoiseFunctionality.cpp
    Operations.cpp
    RealNumber.cpp
    RealNumberUniqueTable.cpp
    Simulation.cpp
    statistics/MemoryManagerStatistics.cpp
    statistics/Statistics.cpp
    statistics/TableStatistics.cpp
    statistics/UniqueTableStatistics.cpp)
  target_link_libraries(${PROJECT_NAME}-dd PUBLIC ${PROJECT_NAME})
  add_library(MQT::CoreDD ALIAS ${PROJECT_NAME}-dd)
  add_library(MQT::${OLD_PROJECT_NAME}_dd ALIAS ${PROJECT_NAME}-dd)
endif()
