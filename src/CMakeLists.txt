# allow loader to find lib
if(APPLE)
	set(basePoint @loader_path)
else()
	set(basePoint $ORIGIN)
endif()
include(GNUInstallDirs)
file(RELATIVE_PATH relDir
     ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
     ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
     )
set(CMAKE_INSTALL_RPATH ${basePoint} ${basePoint}/${relDir})

add_library(${PROJECT_NAME}
            parsers/qasm_parser/Scanner.cpp
            parsers/qasm_parser/Parser.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/operations/Operation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/operations/StandardOperation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/operations/NonUnitaryOperation.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/QuantumComputation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/CircuitOptimizer.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/QFT.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/Grover.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/BernsteinVazirani.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/GoogleRandomCircuitSampling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/Entanglement.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/RandomCliffordCircuit.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/parsers/RealParser.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/parsers/GRCSParser.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/parsers/QASMParser.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/parsers/TFCParser.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/parsers/QCParser.cpp

            ${${PROJECT_NAME}_SOURCE_DIR}/include/parsers/qasm_parser/Token.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/parsers/qasm_parser/Scanner.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/parsers/qasm_parser/Parser.hpp

            ${${PROJECT_NAME}_SOURCE_DIR}/include/operations/Operation.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/operations/StandardOperation.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/operations/NonUnitaryOperation.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/operations/CompoundOperation.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/operations/ClassicControlledOperation.hpp

            ${${PROJECT_NAME}_SOURCE_DIR}/include/QuantumComputation.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/CircuitOptimizer.hpp

            ${${PROJECT_NAME}_SOURCE_DIR}/include/algorithms/QFT.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/algorithms/Grover.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/algorithms/BernsteinVazirani.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/algorithms/GoogleRandomCircuitSampling.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/algorithms/Entanglement.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/algorithms/RandomCliffordCircuit.hpp)
add_library(JKQ::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

option(TRY_USING_INSTALLED_DD_PACKAGE "Attempt searching for installed version of DD package and use it if found" OFF)

if (TRY_USING_INSTALLED_DD_PACKAGE)
	find_package(DDpackage QUIET)
	if (DDpackage_FOUND)
		message(STATUS "using installed DD package")
	else()
		message(STATUS "using DD package submodule")
		add_subdirectory("${PROJECT_SOURCE_DIR}/extern/dd_package" "extern/dd_package")
	endif()
else()
	message(STATUS "using DD package submodule")
	add_subdirectory("${PROJECT_SOURCE_DIR}/extern/dd_package" "extern/dd_package")
endif ()

target_link_libraries(${PROJECT_NAME} PUBLIC JKQ::DDpackage)

# add nlohmann::json library
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/json" "extern/json" EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json)

# add coverage compiler and linker flag if COVERAGE is set
if (COVERAGE)
	target_compile_options(${PROJECT_NAME} PRIVATE --coverage)
	target_link_libraries(${PROJECT_NAME} PRIVATE --coverage)
endif()

# check if interprocedural optimization is supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
	set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# set compiler flags depending on compiler
if(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE "/W4" "/GR" "$<$<CONFIG:RELEASE>:/O2 /GL>")
	target_link_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:/LTCG>")
	target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall $<$<CONFIG:DEBUG>:-g3 -Og -Wextra -Wpedantic -pedantic> $<$<CONFIG:RELEASE>:-O3 -mtune=native -march=native -DNDEBUG -g0>)
endif()

# installation instructions

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} nlohmann_json
        EXPORT qfr-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME qfr)

install(DIRECTORY ${${PROJECT_NAME}_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT qfr-targets
        FILE qfrTargets.cmake
        NAMESPACE JKQ::
        DESTINATION ${INSTALL_CONFIGDIR}
        )

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
		${CMAKE_CURRENT_BINARY_DIR}/qfrConfigVersion.cmake
		VERSION ${PROJECT_VERSION}
		COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${${PROJECT_NAME}_SOURCE_DIR}/cmake/qfrConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/qfrConfig.cmake
                              INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
                              )

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/qfrConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/qfrConfigVersion.cmake
        DESTINATION ${INSTALL_CONFIGDIR}
        )

# exporting from the build tree
export(EXPORT qfr-targets FILE ${CMAKE_CURRENT_BINARY_DIR}/qfrTargets.cmake NAMESPACE JKQ::)

# Register package in user's package registry
export(PACKAGE qfr)
