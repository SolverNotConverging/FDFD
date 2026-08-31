cmake_minimum_required(VERSION 3.24)

project(FDFDNativeApplications LANGUAGES C CXX)

include(CTest)

option(FDFD_BUILD_TRANSMISSION_LINE_CALCULATOR
    "Build the native transmission-line calculator" ON)
option(FDFD_BUILD_WAVEFEM_VIEWER
    "Build the native WaveFEM HDF5 viewer" ON)
option(FDFD_BUILD_FEM_PERIODIC_VIEWER
    "Build the standalone periodic-FEM HDF5 viewer and inspector" ON)

if(FDFD_BUILD_TRANSMISSION_LINE_CALCULATOR)
    add_subdirectory(TransmissionLineCalculator)
endif()

if(FDFD_BUILD_WAVEFEM_VIEWER)
    add_subdirectory(WaveFEMViewer)
endif()

if(FDFD_BUILD_FEM_PERIODIC_VIEWER)
    add_subdirectory(FEMPeriodicViewer)
endif()

if(NOT FDFD_BUILD_TRANSMISSION_LINE_CALCULATOR
   AND NOT FDFD_BUILD_WAVEFEM_VIEWER
   AND NOT FDFD_BUILD_FEM_PERIODIC_VIEWER)
    message(WARNING "All native applications are disabled; there is nothing to build.")
endif()
