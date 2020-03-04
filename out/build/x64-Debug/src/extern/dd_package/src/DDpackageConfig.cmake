get_filename_component(DDpackage_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

list(APPEND CMAKE_MODULE_PATH ${DDpackage_CMAKE_DIR})

if(NOT TARGET JKQ::DDpackage)
	include("${DDpackage_CMAKE_DIR}/DDpackageTargets.cmake")
endif()

set(DDpackage_LIBRARIES JKQ::DDpackage)
