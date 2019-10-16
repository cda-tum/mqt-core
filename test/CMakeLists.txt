project(IntermediateRepresentationTest)

add_executable(IntermediateRepresentation main.cpp)
target_compile_options(IntermediateRepresentation PUBLIC -mtune=native -march=native -pipe -Wall -Wextra -Wnon-virtual-dtor -pedantic -flto)
target_link_libraries(IntermediateRepresentation IntermediateRepresentationLib)
set_property(TARGET IntermediateRepresentation PROPERTY CXX_STANDARD 14)
set_property(TARGET IntermediateRepresentation PROPERTY CXX_EXTENSIONS OFF)

