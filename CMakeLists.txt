cmake_minimum_required(VERSION 3.24)

project(FDFDNativeApplications LANGUAGES C CXX)

include(CTest)

option(FDFD_BUILD_TRANSMISSION_LINE_CALCULATOR
    "Build the native transmission-line calculator" ON)
option(FDFD_BUILD_WAVEFEM_VIEWER
    "Build the native WaveFEM HDF5 viewer" ON)

if(FDFD_BUILD_TRANSMISSION_LINE_CALCULATOR)
    add_subdirectory(TransmissionLineCalculator)
endif()

if(FDFD_BUILD_WAVEFEM_VIEWER)
    add_subdirectory(WaveFEMViewer)
endif()

if(NOT FDFD_BUILD_TRANSMISSION_LINE_CALCULATOR
   AND NOT FDFD_BUILD_WAVEFEM_VIEWER)
    message(WARNING "Both native applications are disabled; there is nothing to build.")
endif()
