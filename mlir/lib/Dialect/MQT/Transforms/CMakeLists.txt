get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} ${extension_libs} MLIRMQT)

add_mlir_library(MLIRMQTThePass ThePass.cpp LINK_LIBS ${LIBS})
target_include_directories(MLIRMQTThePass PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
