get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

find_library(
  MLIRQuantum_LIB
  NAMES MLIRQuantum
  PATHS ${MLIR_BUILD_DIR}/lib
  NO_DEFAULT_PATH)

add_compile_options(-fexceptions)

file(GLOB CONVERSION_SOURCES *.cpp)

set(LIBRARIES ${conversion_libs} MLIRIR MLIRPass MQT::CoreIR ${MLIRQuantum_LIB})

add_mlir_library(QuantumToMQTOpt ${CONVERSION_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS
                 QuantumToMQTOptIncGen)
