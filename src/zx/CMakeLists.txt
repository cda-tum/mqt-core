if(NOT TARGET ${MQT_CORE_TARGET_NAME}-zx)
  # make sure the MQT::multiprecision target is available
  if(NOT TARGET MQT::Multiprecision)
    # handle the case for a complete Boost installation
    if(NOT TARGET Boost::multiprecision)
      add_library(multiprecision INTERFACE)
      target_link_libraries(multiprecision INTERFACE Boost::headers)
      add_library(MQT::Multiprecision ALIAS multiprecision)
      set(MQT_CORE_ZX_SYSTEM_BOOST
          TRUE
          CACHE BOOL "Whether a system version of Boost was used")
    else()
      # Boost::multiprecision does not provide its own installation instructions. As a consequence,
      # this needs a workaround.

      # first, we get the include directory of the multiprecision library
      get_target_property(MULTIPRECISION_INCLUDE_DIR Boost::multiprecision
                          INTERFACE_INCLUDE_DIRECTORIES)

      # then, we create a target that will be used to export the include directory
      add_library(multiprecision INTERFACE)
      target_include_directories(
        multiprecision SYSTEM INTERFACE $<BUILD_INTERFACE:${MULTIPRECISION_INCLUDE_DIR}>
                                        $<INSTALL_INTERFACE:${MQT_CORE_INCLUDE_INSTALL_DIR}>)
      add_library(MQT::Multiprecision ALIAS multiprecision)

      if(MQT_CORE_INSTALL)
        # finally, we create install instructions for the respective header files
        install(DIRECTORY ${MULTIPRECISION_INCLUDE_DIR}/boost
                DESTINATION ${MQT_CORE_INCLUDE_INSTALL_DIR})
      endif()

      set(MQT_CORE_ZX_SYSTEM_BOOST
          FALSE
          CACHE BOOL "Whether a system version of Boost was used")
    endif()
  endif()

  file(GLOB_RECURSE ZX_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/zx/*.hpp)

  # add ZX package library
  add_library(
    ${MQT_CORE_TARGET_NAME}-zx
    ${ZX_HEADERS}
    Rational.cpp
    ZXDiagram.cpp
    Rules.cpp
    Simplify.cpp
    Utils.cpp
    FunctionalityConstruction.cpp)
  target_link_libraries(${MQT_CORE_TARGET_NAME}-zx PUBLIC MQT::Core MQT::Multiprecision)
  target_link_libraries(${MQT_CORE_TARGET_NAME}-zx PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  find_package(GMP)
  if(NOT GMP_FOUND)
    message(NOTICE "Did not find GMP. Using Boost multiprecision library instead.")
  endif()
  # # link to GMP libraries if present
  if(GMP_FOUND)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-zx PUBLIC GMP)
    target_link_libraries(${MQT_CORE_TARGET_NAME}-zx PUBLIC GMP::gmp GMP::gmpxx)
  endif()
  set(MQT_CORE_ZX_WITH_GMP
      ${GMP_FOUND}
      CACHE BOOL "Whether to use GMP for multiprecision arithmetic")

  # add MQT alias
  add_library(MQT::CoreZX ALIAS ${MQT_CORE_TARGET_NAME}-zx)
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-zx
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreZX)
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-zx multiprecision
      PARENT_SCOPE)
endif()
