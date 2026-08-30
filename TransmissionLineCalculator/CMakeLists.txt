cmake_minimum_required(VERSION 3.24)

project(TransmissionLineCalculator VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
if(DEFINED Eigen3_VERSION AND Eigen3_VERSION VERSION_LESS 3.4)
    message(FATAL_ERROR "Transmission Line Calculator requires Eigen 3.4 or newer")
endif()

# MSYS2's MinGW64 Gmsh 4.15.2-2 export references a DLL in lib/ even
# though the package installs it in bin/, and its interface contains paths
# from the package build machine. Predeclaring the one expected imported
# target makes gmshConfig.cmake accept it without loading that broken export.
# Keep the find_package call so a missing package/configuration still fails
# normally, and remove this compatibility block once the package is fixed.
if(MINGW)
    find_path(TL_GMSH_INCLUDE_DIR NAMES gmsh.h REQUIRED)
    find_library(TL_GMSH_IMPORT_LIBRARY NAMES gmsh libgmsh REQUIRED)
    find_file(TL_GMSH_RUNTIME_LIBRARY NAMES libgmsh.dll PATH_SUFFIXES bin REQUIRED)
    foreach(gmsh_artifact IN ITEMS
        TL_GMSH_INCLUDE_DIR
        TL_GMSH_IMPORT_LIBRARY
        TL_GMSH_RUNTIME_LIBRARY
    )
        if(NOT EXISTS "${${gmsh_artifact}}")
            message(FATAL_ERROR "Required Gmsh artifact is missing: ${${gmsh_artifact}}")
        endif()
    endforeach()
    if(NOT TARGET gmsh::shared)
        add_library(gmsh::shared SHARED IMPORTED GLOBAL)
        set_target_properties(gmsh::shared PROPERTIES
            IMPORTED_IMPLIB "${TL_GMSH_IMPORT_LIBRARY}"
            IMPORTED_LOCATION "${TL_GMSH_RUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TL_GMSH_INCLUDE_DIR}"
        )
    endif()
endif()

# Gmsh packages do not all export the same CMake target name. Prefer their
# package configuration, then fall back to the library and header installed by
# system package managers and the official SDK.
find_package(gmsh CONFIG QUIET)
foreach(gmsh_candidate IN ITEMS gmsh::shared gmsh::lib gmsh::gmsh gmsh)
    if(TARGET "${gmsh_candidate}")
        set(TL_GMSH_TARGET "${gmsh_candidate}")
        break()
    endif()
endforeach()

if(NOT TL_GMSH_TARGET)
    find_path(TL_GMSH_INCLUDE_DIR NAMES gmsh.h REQUIRED)
    find_library(TL_GMSH_LIBRARY NAMES gmsh libgmsh REQUIRED)
    add_library(tl-gmsh UNKNOWN IMPORTED GLOBAL)
    set_target_properties(tl-gmsh PROPERTIES
        IMPORTED_LOCATION "${TL_GMSH_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TL_GMSH_INCLUDE_DIR}"
    )
    set(TL_GMSH_TARGET tl-gmsh)
endif()

find_package(Qt6 6.2 REQUIRED COMPONENTS Concurrent Widgets)
find_package(ftxui CONFIG REQUIRED)

function(tl_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(tl-core STATIC
    native/solver.cpp
)
add_library(tl::core ALIAS tl-core)
target_include_directories(tl-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/native>
        $<INSTALL_INTERFACE:include/TransmissionLineCalculator>
)
target_link_libraries(tl-core PUBLIC Eigen3::Eigen "${TL_GMSH_TARGET}")
target_compile_features(tl-core PUBLIC cxx_std_20)
tl_enable_warnings(tl-core)

add_executable(transmission-line-calculator WIN32 MACOSX_BUNDLE
    native/main.cpp
    native/main_window.cpp
    native/field_plot.cpp
    native/field_view.cpp
)
set_target_properties(transmission-line-calculator PROPERTIES AUTOMOC ON)
target_link_libraries(transmission-line-calculator PRIVATE
    tl::core
    Qt6::Concurrent
    Qt6::Widgets
)
target_compile_definitions(transmission-line-calculator PRIVATE
    QT_NO_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    TL_CALCULATOR_VERSION="${PROJECT_VERSION}"
)
tl_enable_warnings(transmission-line-calculator)

add_executable(transmission-line-calculator-cli
    native/cli.cpp
    native/tui.cpp
)
target_link_libraries(transmission-line-calculator-cli PRIVATE
    tl::core
    Threads::Threads
    ftxui::component
    ftxui::dom
    ftxui::screen
)
target_compile_definitions(transmission-line-calculator-cli PRIVATE
    TL_CALCULATOR_VERSION="${PROJECT_VERSION}"
)
tl_enable_warnings(transmission-line-calculator-cli)

include(CTest)
if(BUILD_TESTING)
    add_executable(tl-solver-tests
        tests/test_solver.cpp
    )
    target_link_libraries(tl-solver-tests PRIVATE tl::core)
    tl_enable_warnings(tl-solver-tests)

    add_test(NAME tl-solver-tests COMMAND tl-solver-tests)
    set_tests_properties(tl-solver-tests PROPERTIES TIMEOUT 180)

    add_executable(tl-field-view-tests
        tests/test_field_view.cpp
        native/field_view.cpp
    )
    target_include_directories(tl-field-view-tests PRIVATE native)
    target_link_libraries(tl-field-view-tests PRIVATE tl::core)
    tl_enable_warnings(tl-field-view-tests)

    add_test(NAME tl-field-view-tests COMMAND tl-field-view-tests)
    set_tests_properties(tl-field-view-tests PROPERTIES TIMEOUT 30)

    add_test(
        NAME tl-gui-smoke
        COMMAND transmission-line-calculator --smoke-test
    )
    set_tests_properties(tl-gui-smoke PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        TIMEOUT 30
    )

    add_test(
        NAME tl-gui-calculate-smoke
        COMMAND transmission-line-calculator --calculate-smoke-test
    )
    set_tests_properties(tl-gui-calculate-smoke PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        TIMEOUT 120
    )

    add_test(
        NAME tl-tui-smoke
        COMMAND transmission-line-calculator-cli --smoke-test
    )
    set_tests_properties(tl-tui-smoke PROPERTIES TIMEOUT 30)

    add_test(
        NAME tl-tui-help
        COMMAND transmission-line-calculator-cli --help
    )
    set_tests_properties(tl-tui-help PROPERTIES
        PASS_REGULAR_EXPRESSION "interactive FTXUI"
        TIMEOUT 30
    )

    add_test(
        NAME tl-tui-version
        COMMAND transmission-line-calculator-cli --version
    )
    set_tests_properties(tl-tui-version PROPERTIES
        PASS_REGULAR_EXPRESSION "Transmission Line Calculator ${PROJECT_VERSION}"
        TIMEOUT 30
    )

    add_test(
        NAME tl-tui-rejects-legacy-flags
        COMMAND transmission-line-calculator-cli --type coaxial
    )
    set_tests_properties(tl-tui-rejects-legacy-flags PROPERTIES
        WILL_FAIL TRUE
        TIMEOUT 30
    )

    add_test(
        NAME tl-tui-requires-terminal
        COMMAND transmission-line-calculator-cli
    )
    set_tests_properties(tl-tui-requires-terminal PROPERTIES
        WILL_FAIL TRUE
        TIMEOUT 30
    )
endif()

install(TARGETS
    tl-core
    transmission-line-calculator
    transmission-line-calculator-cli
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION .
    ARCHIVE DESTINATION lib
)
install(FILES
    native/model.hpp
    native/solver.hpp
    DESTINATION include/TransmissionLineCalculator
)
