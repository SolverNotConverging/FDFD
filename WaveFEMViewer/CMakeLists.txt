cmake_minimum_required(VERSION 3.24)

project(WaveFEMViewer VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Qt6 6.2 REQUIRED COMPONENTS Concurrent Widgets)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)

add_executable(wavefem-viewer WIN32 MACOSX_BUNDLE
    native/main.cpp
    native/h5_reader.cpp
    native/main_window.cpp
    native/plot_widget.cpp
)

add_executable(wavefem-viewer-inspect
    native/inspect.cpp
    native/h5_reader.cpp
)

target_include_directories(wavefem-viewer PRIVATE native)
target_link_libraries(wavefem-viewer PRIVATE
    Qt6::Concurrent
    Qt6::Widgets
    HDF5::HDF5
)
target_include_directories(wavefem-viewer-inspect PRIVATE native)
target_link_libraries(wavefem-viewer-inspect PRIVATE HDF5::HDF5)

target_compile_definitions(wavefem-viewer PRIVATE
    QT_NO_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    WAVEFEM_VIEWER_VERSION="${PROJECT_VERSION}"
)

function(wavefem_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

wavefem_enable_warnings(wavefem-viewer)
wavefem_enable_warnings(wavefem-viewer-inspect)

install(TARGETS wavefem-viewer wavefem-viewer-inspect
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION .
)
