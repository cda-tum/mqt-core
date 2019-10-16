project(IntermediateRepresentationLib)

add_library(DDpackage STATIC ./DD-package/DDcomplex.cpp ./DD-package/DDpackage.cpp)
set_property(TARGET DDpackage PROPERTY CXX_STANDARD 11)
set_property(TARGET DDpackage PROPERTY CXX_EXTENSIONS OFF)
target_compile_options(DDpackage PRIVATE -DNDEBUG -g -mtune=native -march=native -pipe -flto)
target_include_directories(DDpackage PUBLIC DD-package)

add_library(IntermediateRepresentationLib STATIC QuantumComputation.cpp)
set_property(TARGET IntermediateRepresentationLib PROPERTY CXX_STANDARD 14)
set_property(TARGET IntermediateRepresentationLib PROPERTY CXX_EXTENSIONS OFF)
target_compile_options(IntermediateRepresentationLib PRIVATE -mtune=native -march=native -flto)
target_include_directories(IntermediateRepresentationLib PUBLIC ./ QASMParser Operations)
target_link_libraries(IntermediateRepresentationLib DDpackage)

