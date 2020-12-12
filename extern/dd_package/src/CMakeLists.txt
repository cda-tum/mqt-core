add_library(${PROJECT_NAME}
            ${CMAKE_CURRENT_SOURCE_DIR}/DDpackage.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DDdebug.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DDcomplex.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DDexport.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DDreorder.cpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/DDpackage.h
            ${${PROJECT_NAME}_SOURCE_DIR}/include/DDcomplex.h
            ${${PROJECT_NAME}_SOURCE_DIR}/include/DDexport.h)
add_library(JKQ::DDpackage ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)

# check if interprocedural optimization is supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
	set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# add coverage compiler and linker flag if COVERAGE is set
if (COVERAGE)
	target_compile_options(${PROJECT_NAME} PRIVATE --coverage)
	target_link_libraries(${PROJECT_NAME} PRIVATE --coverage)
endif()

# set compiler flags depending on compiler
if(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE "/W4" "/utf-8" "/std:c++17" "$<$<CONFIG:RELEASE>:/O2 /GL>")
	target_link_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall $<$<CONFIG:DEBUG>:-g3 -Og -Wextra -Wpedantic -pedantic> $<$<CONFIG:RELEASE>:-O3 -mtune=native -march=native -DNDEBUG -g0>)
endif()

# installation instructions

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/DDpackage)

install(TARGETS ${PROJECT_NAME}
        EXPORT ddpackage-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME DDpackage)

install(DIRECTORY ${${PROJECT_NAME}_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ddpackage-targets
        FILE DDpackageTargets.cmake
        NAMESPACE JKQ::
        DESTINATION ${INSTALL_CONFIGDIR}
        )

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
		${CMAKE_CURRENT_BINARY_DIR}/DDpackageConfigVersion.cmake
		VERSION ${PROJECT_VERSION}
		COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${${PROJECT_NAME}_SOURCE_DIR}/cmake/DDpackageConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/DDpackageConfig.cmake
                              INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
                              )

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/DDpackageConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/DDpackageConfigVersion.cmake
        DESTINATION ${INSTALL_CONFIGDIR}
        )

# exporting from the build tree
export(EXPORT ddpackage-targets FILE ${CMAKE_CURRENT_BINARY_DIR}/DDpackageTargets.cmake NAMESPACE JKQ::)

#Register package in user's package registry
export(PACKAGE DDpackage)
