cmake_minimum_required(VERSION 3.24)

project(FDFD VERSION 1.0.0 LANGUAGES C CXX)

if(DEFINED SKBUILD)
    set(BUILD_TESTING OFF CACHE BOOL "Do not compile C++ test programs into Python builds" FORCE)
endif()
include(CTest)

set(FDFD_BUILD_PYTHON_DEFAULT OFF)
if(DEFINED SKBUILD)
    set(FDFD_BUILD_PYTHON_DEFAULT ON)
endif()
option(FDFD_BUILD_PYTHON
    "Build the Cython extension for the active Python interpreter"
    ${FDFD_BUILD_PYTHON_DEFAULT})
if(FDFD_BUILD_PYTHON)
    # scikit-build-core provides the active uv interpreter and makes build
    # requirements such as cython-cmake visible through CMAKE_PREFIX_PATH.
    find_package(Python 3.11...<3.14 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(Cython 3 MODULE REQUIRED)
    include(UseCython)

    set(FDFD_CYTHON_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/libraries/periodic_eigensolver/src/periodic_eigensolver/_cython_kernels.pyx")
    cython_transpile(
        "${FDFD_CYTHON_SOURCE}"
        LANGUAGE C
        MODULE_NAME periodic_eigensolver._cython_kernels
        CYTHON_ARGS -X language_level=3
        OUTPUT_VARIABLE FDFD_CYTHON_GENERATED_SOURCE
    )
    Python_add_library(fdfd-cython-kernels MODULE
        "${FDFD_CYTHON_GENERATED_SOURCE}"
        WITH_SOABI
    )
    set_target_properties(fdfd-cython-kernels PROPERTIES
        OUTPUT_NAME "_cython_kernels"
    )
    install(TARGETS fdfd-cython-kernels
        LIBRARY DESTINATION periodic_eigensolver
        RUNTIME DESTINATION periodic_eigensolver
    )
endif()

if(SKBUILD)
    set(FDFD_NATIVE_INSTALL_DIR "fdfd/native/bin")
    set(FDFD_NATIVE_BUNDLE_INSTALL_DIR "fdfd/native")
    set(FDFD_NATIVE_LICENSE_INSTALL_DIR "fdfd/native/licenses")
else()
    set(FDFD_NATIVE_INSTALL_DIR "bin")
    set(FDFD_NATIVE_BUNDLE_INSTALL_DIR ".")
    set(FDFD_NATIVE_LICENSE_INSTALL_DIR "share/licenses")
endif()

if(SKBUILD AND MINGW)
    get_filename_component(FDFD_NATIVE_RUNTIME_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/fdfd-native-runtime.txt"
        CONTENT "${FDFD_NATIVE_RUNTIME_DIR}\n"
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/fdfd-native-runtime.txt"
        DESTINATION "${FDFD_NATIVE_INSTALL_DIR}"
    )
endif()

include(cmake/NativeRuntime.cmake)

# Release builds may reuse an already qualified, self-contained runtime bundle.
# Ordinary source installs compile the applications in this same CMake build.
set(FDFD_NATIVE_BUNDLE "$ENV{FDFD_NATIVE_BUNDLE}" CACHE PATH
    "Qualified native runtime bundle to install instead of compiling the apps")

option(CEM_BUILD_TRANSMISSION_LINE_CALCULATOR
    "Build the native transmission-line calculator" ON)
option(CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER
    "Build the FEM waveguide scattering viewer" ON)
option(CEM_BUILD_FEM_PERIODIC_MODE_VIEWER
    "Build the standalone periodic-FEM HDF5 viewer and inspector" ON)

if(FDFD_NATIVE_BUNDLE)
    if(NOT EXISTS "${FDFD_NATIVE_BUNDLE}/bin")
        message(FATAL_ERROR "FDFD_NATIVE_BUNDLE is incomplete: ${FDFD_NATIVE_BUNDLE}")
    endif()
    install(DIRECTORY "${FDFD_NATIVE_BUNDLE}/"
        DESTINATION "fdfd/native"
    )
elseif(CEM_BUILD_TRANSMISSION_LINE_CALCULATOR)
    add_subdirectory(apps/transmission_line_calculator)
endif()

if(NOT FDFD_NATIVE_BUNDLE AND CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER)
    add_subdirectory(apps/fem_waveguide_scattering_viewer)
endif()

if(NOT FDFD_NATIVE_BUNDLE AND CEM_BUILD_FEM_PERIODIC_MODE_VIEWER)
    add_subdirectory(apps/fem_periodic_mode_viewer)
endif()

if(NOT FDFD_NATIVE_BUNDLE
   AND NOT CEM_BUILD_TRANSMISSION_LINE_CALCULATOR
   AND NOT CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER
   AND NOT CEM_BUILD_FEM_PERIODIC_MODE_VIEWER)
    message(WARNING "All native applications are disabled; there is nothing to build.")
endif()
