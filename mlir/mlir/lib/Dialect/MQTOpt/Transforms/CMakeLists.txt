get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBRARIES ${dialect_libs} MLIRMQTOpt)
add_compile_options(-fexceptions)

file(GLOB TRANSFORMS_SOURCES *.cpp)

add_mlir_library(MLIRMQTOptTransforms ${TRANSFORMS_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS
                 MLIRMQTOptTransformsIncGen)
