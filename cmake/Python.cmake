# a function that sets up some cache variables and tries to find Python
function(find_python)
  # Some common settings for finding Python
  set(Python_FIND_VIRTUALENV
      FIRST
      CACHE STRING "Give precedence to virtualenvs when searching for Python")
  set(Python_FIND_FRAMEWORK
      LAST
      CACHE STRING "Prefer Brew/Conda to Apple framework Python")
  set(Python_ARTIFACTS_INTERACTIVE
      ON
      CACHE BOOL "Prevent multiple searches for Python and instead cache the results.")

  # top-level call to find Python
  find_package(
    Python 3.8 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

  if(BUILD_MQT_CORE_BINDINGS)
    if(NOT SKBUILD)
      # Manually detect the installed pybind11 package and import it into CMake.
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE pybind11_DIR)
      list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
    endif()

    message(STATUS "Python executable: ${Python_EXECUTABLE}")

    # add pybind11 library
    find_package(pybind11 CONFIG REQUIRED)
  endif()
endfunction()
