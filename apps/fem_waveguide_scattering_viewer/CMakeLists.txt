cmake_minimum_required(VERSION 3.24)

project(FEMWaveguideScatteringViewer VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Qt6 6.2 REQUIRED COMPONENTS Concurrent Widgets)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)

add_executable(fem-waveguide-scattering-viewer WIN32 MACOSX_BUNDLE
    native/main.cpp
    native/h5_reader.cpp
    native/main_window.cpp
    native/plot_widget.cpp
)

add_executable(fem-waveguide-scattering-viewer-inspect
    native/inspect.cpp
    native/h5_reader.cpp
)

target_include_directories(fem-waveguide-scattering-viewer PRIVATE native)
target_link_libraries(fem-waveguide-scattering-viewer PRIVATE
    Qt6::Concurrent
    Qt6::Widgets
    HDF5::HDF5
)
target_include_directories(fem-waveguide-scattering-viewer-inspect PRIVATE native)
target_link_libraries(fem-waveguide-scattering-viewer-inspect PRIVATE HDF5::HDF5)

target_compile_definitions(fem-waveguide-scattering-viewer PRIVATE
    QT_NO_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    CEM_SCATTERING_VIEWER_VERSION="${PROJECT_VERSION}"
)

function(cem_scattering_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

cem_scattering_enable_warnings(fem-waveguide-scattering-viewer)
cem_scattering_enable_warnings(fem-waveguide-scattering-viewer-inspect)

install(TARGETS fem-waveguide-scattering-viewer fem-waveguide-scattering-viewer-inspect
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION .
)
