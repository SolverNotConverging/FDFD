cmake_minimum_required(VERSION 3.24)

project(FDFD VERSION 1.0.0 LANGUAGES C CXX)

include(CTest)

option(CEM_BUILD_TRANSMISSION_LINE_CALCULATOR
    "Build the native transmission-line calculator" ON)
option(CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER
    "Build the FEM waveguide scattering viewer" ON)
option(CEM_BUILD_FEM_PERIODIC_MODE_VIEWER
    "Build the standalone periodic-FEM HDF5 viewer and inspector" ON)

if(CEM_BUILD_TRANSMISSION_LINE_CALCULATOR)
    add_subdirectory(apps/transmission_line_calculator)
endif()

if(CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER)
    add_subdirectory(apps/fem_waveguide_scattering_viewer)
endif()

if(CEM_BUILD_FEM_PERIODIC_MODE_VIEWER)
    add_subdirectory(apps/fem_periodic_mode_viewer)
endif()

if(NOT CEM_BUILD_TRANSMISSION_LINE_CALCULATOR
   AND NOT CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER
   AND NOT CEM_BUILD_FEM_PERIODIC_MODE_VIEWER)
    message(WARNING "All native applications are disabled; there is nothing to build.")
endif()
