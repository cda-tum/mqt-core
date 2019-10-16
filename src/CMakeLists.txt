project(IntermediateRepresentationLib)

file(GLOB_RECURSE SOURCES "StandardOperation.cpp" "DD-package-oop/*.cpp")
file(GLOB_RECURSE HEADERS "StandardOperation.h" "DD-package-oop/*.h")

add_library(IntermediateRepresentationLib SHARED ${SOURCES} ${HEADERS})
target_include_directories(IntermediateRepresentationLib PUBLIC ./DD-package-oop/ ./)

