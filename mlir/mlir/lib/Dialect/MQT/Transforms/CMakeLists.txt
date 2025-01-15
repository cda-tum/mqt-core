get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} ${extension_libs})

add_mlir_library(MLIRMQTThePattern ThePattern.cpp LINK_LIBS ${LIBS})
target_include_directories(MLIRMQTThePattern PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
