get_filename_component(QFR_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(CMakeFindDependencyMacro)

find_dependency(DDpackage 1.1 REQUIRED CONFIG)

if(NOT TARGET JKQ::qfr)
	include("${QFR_CMAKE_DIR}/qfrTargets.cmake")
endif()

set(qfr_LIBRARIES JKQ::qfr)
