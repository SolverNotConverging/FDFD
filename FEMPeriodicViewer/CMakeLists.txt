cmake_minimum_required(VERSION 3.24)

project(FEMPeriodicViewer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

set(FEM_PERIODIC_VIEWER_WITH_VTK "AUTO" CACHE STRING
    "VTK-backed 3D viewport: AUTO, ON, or OFF")
set_property(CACHE FEM_PERIODIC_VIEWER_WITH_VTK PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${FEM_PERIODIC_VIEWER_WITH_VTK}" FEM_PERIODIC_VTK_MODE)
if(NOT FEM_PERIODIC_VTK_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
        "FEM_PERIODIC_VIEWER_WITH_VTK must be AUTO, ON, or OFF")
endif()

find_package(Qt6 6.2 REQUIRED COMPONENTS Concurrent Widgets)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)

set(FEM_PERIODIC_VTK_TARGETS)
set(FEM_PERIODIC_VTK_ENABLED OFF)
if(NOT FEM_PERIODIC_VTK_MODE STREQUAL "OFF")
    set(FEM_PERIODIC_VTK_COMPONENTS
        CommonCore
        CommonDataModel
        FiltersCore
        FiltersGeneral
        FiltersSources
        InteractionStyle
        InteractionWidgets
        RenderingAnnotation
        RenderingCore
        RenderingOpenGL2
        GUISupportQt
    )
    if(FEM_PERIODIC_VTK_MODE STREQUAL "ON")
        find_package(VTK 9.2 REQUIRED COMPONENTS ${FEM_PERIODIC_VTK_COMPONENTS})
    else()
        find_package(VTK 9.2 QUIET COMPONENTS ${FEM_PERIODIC_VTK_COMPONENTS})
    endif()
    if(VTK_FOUND AND DEFINED VTK_QT_VERSION AND NOT VTK_QT_VERSION STREQUAL "6")
        if(FEM_PERIODIC_VTK_MODE STREQUAL "ON")
            message(FATAL_ERROR
                "FEMPeriodicViewer requires VTK built against Qt 6; "
                "found VTK_QT_VERSION=${VTK_QT_VERSION}.")
        else()
            message(STATUS
                "Ignoring VTK built against Qt ${VTK_QT_VERSION}; "
                "the Qt 6 viewer will be built without 3D support")
            set(VTK_FOUND OFF)
        endif()
    endif()
    if(VTK_FOUND)
        set(FEM_PERIODIC_VTK_ENABLED ON)
    elseif(FEM_PERIODIC_VTK_MODE STREQUAL "AUTO")
        message(STATUS "VTK was not found; building the 2D-only FEM periodic viewer")
    endif()
endif()
if(FEM_PERIODIC_VTK_ENABLED)
    list(APPEND FEM_PERIODIC_VTK_TARGETS
        VTK::CommonCore
        VTK::CommonDataModel
        VTK::FiltersCore
        VTK::FiltersGeneral
        VTK::FiltersSources
        VTK::InteractionStyle
        VTK::InteractionWidgets
        VTK::RenderingAnnotation
        VTK::RenderingCore
        VTK::RenderingOpenGL2
        VTK::GUISupportQt
    )
endif()

add_library(fem-periodic-archive STATIC
    native/h5_reader.cpp
)
target_include_directories(fem-periodic-archive PUBLIC native)
target_link_libraries(fem-periodic-archive PUBLIC HDF5::HDF5)

add_executable(fem-periodic-viewer WIN32 MACOSX_BUNDLE
    native/main.cpp
    native/main_window.cpp
    native/field_plot_2d.cpp
    native/vtk_field_view.cpp
)
target_include_directories(fem-periodic-viewer PRIVATE native)
target_link_libraries(fem-periodic-viewer PRIVATE
    fem-periodic-archive
    Qt6::Concurrent
    Qt6::Widgets
    ${FEM_PERIODIC_VTK_TARGETS}
)
target_compile_definitions(fem-periodic-viewer PRIVATE
    QT_NO_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    FEM_PERIODIC_VIEWER_VERSION="${PROJECT_VERSION}"
    $<$<BOOL:${FEM_PERIODIC_VTK_ENABLED}>:FEM_PERIODIC_VIEWER_WITH_VTK=1>
)

add_executable(fem-periodic-inspect native/inspect.cpp)
target_link_libraries(fem-periodic-inspect PRIVATE fem-periodic-archive)
if(WIN32)
    target_link_libraries(fem-periodic-inspect PRIVATE shell32)
endif()

function(fem_periodic_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive- /utf-8)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

fem_periodic_enable_warnings(fem-periodic-archive)
fem_periodic_enable_warnings(fem-periodic-viewer)
fem_periodic_enable_warnings(fem-periodic-inspect)

if(FEM_PERIODIC_VTK_ENABLED)
    vtk_module_autoinit(
        TARGETS fem-periodic-viewer
        MODULES ${FEM_PERIODIC_VTK_TARGETS}
    )
endif()

install(TARGETS fem-periodic-viewer fem-periodic-inspect
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION .
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    DESTINATION share/licenses/FEMPeriodicViewer
)

if(BUILD_TESTING)
    add_executable(fem-periodic-fixture tests/create_fixture.cpp)
    target_link_libraries(fem-periodic-fixture PRIVATE HDF5::HDF5)
    target_include_directories(fem-periodic-fixture PRIVATE native)
    fem_periodic_enable_warnings(fem-periodic-fixture)

    add_executable(fem-periodic-reader-test tests/reader_test.cpp)
    target_link_libraries(fem-periodic-reader-test PRIVATE fem-periodic-archive)
    fem_periodic_enable_warnings(fem-periodic-reader-test)

    set(FEM_PERIODIC_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-2d.h5")
    set(FEM_PERIODIC_NATIVE_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-native.h5")
    set(FEM_PERIODIC_VLEN_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-vlen.h5")
    set(FEM_PERIODIC_UNICODE_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-µ.h5")
    set(FEM_PERIODIC_SWEEP_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-sweep.h5")
    set(FEM_PERIODIC_BAD_FILE "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-bad.h5")
    set(FEM_PERIODIC_BAD_ATTRIBUTE_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-bad-attribute.h5")
    set(FEM_PERIODIC_3D_FILE "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-3d.h5")
    set(FEM_PERIODIC_BAD_LINKS_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-bad-links.h5")
    set(FEM_PERIODIC_BAD_COEFFICIENT_SPACE_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-bad-coefficient-space.h5")
    set(FEM_PERIODIC_BAD_CELL_EDGE_ORDER_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-bad-cell-edge-order.h5")
    set(FEM_PERIODIC_BAD_PERIODIC_EDGE_SIGN_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-bad-periodic-edge-sign.h5")
    set(FEM_PERIODIC_BAD_VECTOR_RANK_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-bad-vector-rank.h5")
    set(FEM_PERIODIC_BAD_MODE_METADATA_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/test-periodic-bad-mode-metadata.h5")
    set(FEM_PERIODIC_DIRECTORY_SMOKE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/directory-smoke")
    set(FEM_PERIODIC_DIRECTORY_BAD_INDEX_FILE
        "${FEM_PERIODIC_DIRECTORY_SMOKE_DIR}/000000-invalid-index.h5")
    set(FEM_PERIODIC_DIRECTORY_BAD_LAZY_FILE
        "${FEM_PERIODIC_DIRECTORY_SMOKE_DIR}/000001-invalid-lazy.h5")
    set(FEM_PERIODIC_DIRECTORY_SMOKE_FILE
        "${FEM_PERIODIC_DIRECTORY_SMOKE_DIR}/000002-periodic.h5")
    set(FEM_PERIODIC_REMOVE_ON_EXIT_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/fem-periodic-view-temporary.h5")
    set(FEM_PERIODIC_IMMEDIATE_REMOVE_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/fem-periodic-view-immediate-temporary.h5")

    add_test(NAME fem-periodic-create-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_TEST_FILE}" compound)
    set_tests_properties(fem-periodic-create-fixture PROPERTIES FIXTURES_SETUP periodic_fixture)

    add_test(NAME fem-periodic-reader
        COMMAND fem-periodic-reader-test "${FEM_PERIODIC_TEST_FILE}")
    set_tests_properties(fem-periodic-reader PROPERTIES FIXTURES_REQUIRED periodic_fixture)

    add_test(NAME fem-periodic-copy-unicode-path-fixture
        COMMAND ${CMAKE_COMMAND} -E copy
            "${FEM_PERIODIC_TEST_FILE}" "${FEM_PERIODIC_UNICODE_TEST_FILE}")
    set_tests_properties(fem-periodic-copy-unicode-path-fixture PROPERTIES
        FIXTURES_REQUIRED periodic_fixture
        FIXTURES_SETUP periodic_unicode_path_fixture)
    add_test(NAME fem-periodic-inspector-unicode-path
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_UNICODE_TEST_FILE}" 0 0)
    set_tests_properties(fem-periodic-inspector-unicode-path PROPERTIES
        FIXTURES_REQUIRED periodic_unicode_path_fixture
        PASS_REGULAR_EXPRESSION "format=fem-periodic-modes")

    add_test(NAME fem-periodic-inspector
        COMMAND ${CMAKE_COMMAND}
            "-DINSPECTOR=$<TARGET_FILE:fem-periodic-inspect>"
            "-DFILE=${FEM_PERIODIC_TEST_FILE}"
            -DSCENARIO=single
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_inspector.cmake")
    set_tests_properties(fem-periodic-inspector PROPERTIES
        FIXTURES_REQUIRED periodic_fixture
    )

    add_test(NAME fem-periodic-create-vlen-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_VLEN_TEST_FILE}" vlen)
    set_tests_properties(fem-periodic-create-vlen-fixture PROPERTIES
        FIXTURES_SETUP periodic_vlen_fixture)
    add_test(NAME fem-periodic-reader-vlen-utf8
        COMMAND fem-periodic-reader-test "${FEM_PERIODIC_VLEN_TEST_FILE}")
    set_tests_properties(fem-periodic-reader-vlen-utf8 PROPERTIES
        FIXTURES_REQUIRED periodic_vlen_fixture)

    add_test(NAME fem-periodic-create-sweep-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_SWEEP_TEST_FILE}" sweep)
    set_tests_properties(fem-periodic-create-sweep-fixture PROPERTIES
        FIXTURES_SETUP periodic_sweep_fixture)
    add_test(NAME fem-periodic-inspector-sweep-second-case
        COMMAND ${CMAKE_COMMAND}
            "-DINSPECTOR=$<TARGET_FILE:fem-periodic-inspect>"
            "-DFILE=${FEM_PERIODIC_SWEEP_TEST_FILE}"
            -DSCENARIO=sweep
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_inspector.cmake")
    set_tests_properties(fem-periodic-inspector-sweep-second-case PROPERTIES
        FIXTURES_REQUIRED periodic_sweep_fixture)
    add_test(NAME fem-periodic-reader-sweep-second-case
        COMMAND fem-periodic-reader-test "${FEM_PERIODIC_SWEEP_TEST_FILE}")
    set_tests_properties(fem-periodic-reader-sweep-second-case PROPERTIES
        FIXTURES_REQUIRED periodic_sweep_fixture)

    if(HDF5_VERSION VERSION_GREATER_EQUAL 2.0)
        add_test(NAME fem-periodic-create-native-fixture
            COMMAND fem-periodic-fixture "${FEM_PERIODIC_NATIVE_TEST_FILE}" native)
        set_tests_properties(fem-periodic-create-native-fixture PROPERTIES
            FIXTURES_SETUP periodic_native_fixture)
        add_test(NAME fem-periodic-reader-native-complex
            COMMAND fem-periodic-reader-test "${FEM_PERIODIC_NATIVE_TEST_FILE}")
        set_tests_properties(fem-periodic-reader-native-complex PROPERTIES
            FIXTURES_REQUIRED periodic_native_fixture)
    endif()

    add_test(NAME fem-periodic-create-bad-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_BAD_FILE}" bad-schema)
    set_tests_properties(fem-periodic-create-bad-fixture PROPERTIES
        FIXTURES_SETUP periodic_bad_fixture)
    add_test(NAME fem-periodic-reject-bad-schema
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_BAD_FILE}")
    set_tests_properties(fem-periodic-reject-bad-schema PROPERTIES
        FIXTURES_REQUIRED periodic_bad_fixture
        WILL_FAIL TRUE
    )

    add_test(NAME fem-periodic-create-bad-attribute-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_BAD_ATTRIBUTE_FILE}" bad-attribute)
    set_tests_properties(fem-periodic-create-bad-attribute-fixture PROPERTIES
        FIXTURES_SETUP periodic_bad_attribute_fixture)
    add_test(NAME fem-periodic-reject-array-attribute
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_BAD_ATTRIBUTE_FILE}")
    set_tests_properties(fem-periodic-reject-array-attribute PROPERTIES
        FIXTURES_REQUIRED periodic_bad_attribute_fixture
        WILL_FAIL TRUE
    )

    add_test(NAME fem-periodic-create-bad-links-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_BAD_LINKS_FILE}" bad-links)
    set_tests_properties(fem-periodic-create-bad-links-fixture PROPERTIES
        FIXTURES_SETUP periodic_bad_links_fixture)
    add_test(NAME fem-periodic-inspector-reject-bad-links
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_BAD_LINKS_FILE}")
    set_tests_properties(fem-periodic-inspector-reject-bad-links PROPERTIES
        FIXTURES_REQUIRED periodic_bad_links_fixture
        WILL_FAIL TRUE)
    add_test(NAME fem-periodic-gui-reject-bad-links
        COMMAND fem-periodic-viewer --smoke-test "${FEM_PERIODIC_BAD_LINKS_FILE}")
    set_tests_properties(fem-periodic-gui-reject-bad-links PROPERTIES
        FIXTURES_REQUIRED periodic_bad_links_fixture
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        TIMEOUT 20
        WILL_FAIL TRUE)

    add_test(NAME fem-periodic-gui-offscreen
        COMMAND fem-periodic-viewer --smoke-test "${FEM_PERIODIC_TEST_FILE}")
    set_tests_properties(fem-periodic-gui-offscreen PROPERTIES
        FIXTURES_REQUIRED periodic_fixture
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        TIMEOUT 20
    )

    add_test(NAME fem-periodic-create-directory-smoke-dir
        COMMAND ${CMAKE_COMMAND} -E make_directory "${FEM_PERIODIC_DIRECTORY_SMOKE_DIR}")
    set_tests_properties(fem-periodic-create-directory-smoke-dir PROPERTIES
        FIXTURES_SETUP periodic_directory_smoke_dir)
    add_test(NAME fem-periodic-create-directory-bad-index-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_DIRECTORY_BAD_INDEX_FILE}" bad-schema)
    set_tests_properties(fem-periodic-create-directory-bad-index-fixture PROPERTIES
        FIXTURES_REQUIRED periodic_directory_smoke_dir
        FIXTURES_SETUP periodic_directory_bad_index_fixture)
    add_test(NAME fem-periodic-create-directory-bad-lazy-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_DIRECTORY_BAD_LAZY_FILE}" bad-links)
    set_tests_properties(fem-periodic-create-directory-bad-lazy-fixture PROPERTIES
        FIXTURES_REQUIRED periodic_directory_bad_index_fixture
        FIXTURES_SETUP periodic_directory_bad_lazy_fixture)
    add_test(NAME fem-periodic-create-directory-smoke-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_DIRECTORY_SMOKE_FILE}" compound)
    set_tests_properties(fem-periodic-create-directory-smoke-fixture PROPERTIES
        FIXTURES_REQUIRED periodic_directory_bad_lazy_fixture
        FIXTURES_SETUP periodic_directory_smoke_fixture)
    add_test(NAME fem-periodic-gui-directory-offscreen
        COMMAND fem-periodic-viewer --smoke-test "${FEM_PERIODIC_DIRECTORY_SMOKE_DIR}")
    set_tests_properties(fem-periodic-gui-directory-offscreen PROPERTIES
        FIXTURES_REQUIRED periodic_directory_smoke_fixture
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        TIMEOUT 20)

    add_test(NAME fem-periodic-copy-remove-on-exit-fixture
        COMMAND ${CMAKE_COMMAND} -E copy
            "${FEM_PERIODIC_TEST_FILE}" "${FEM_PERIODIC_REMOVE_ON_EXIT_FILE}")
    set_tests_properties(fem-periodic-copy-remove-on-exit-fixture PROPERTIES
        FIXTURES_REQUIRED periodic_fixture
        FIXTURES_SETUP periodic_remove_on_exit_fixture)
    add_test(NAME fem-periodic-gui-remove-source-on-exit
        COMMAND fem-periodic-viewer --smoke-test --remove-source-on-exit
            "${FEM_PERIODIC_REMOVE_ON_EXIT_FILE}")
    set_tests_properties(fem-periodic-gui-remove-source-on-exit PROPERTIES
        FIXTURES_REQUIRED periodic_remove_on_exit_fixture
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        TIMEOUT 20)
    add_test(NAME fem-periodic-copy-immediate-remove-fixture
        COMMAND ${CMAKE_COMMAND} -E copy
            "${FEM_PERIODIC_TEST_FILE}" "${FEM_PERIODIC_IMMEDIATE_REMOVE_FILE}")
    set_tests_properties(fem-periodic-copy-immediate-remove-fixture PROPERTIES
        FIXTURES_REQUIRED periodic_fixture
        FIXTURES_SETUP periodic_immediate_remove_fixture)
    add_test(NAME fem-periodic-gui-immediate-close-removes-source
        COMMAND fem-periodic-viewer --remove-source-on-exit --test-exit-immediately
            "${FEM_PERIODIC_IMMEDIATE_REMOVE_FILE}")
    set_tests_properties(fem-periodic-gui-immediate-close-removes-source PROPERTIES
        FIXTURES_REQUIRED periodic_immediate_remove_fixture
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        TIMEOUT 20)

    add_test(NAME fem-periodic-create-3d-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_3D_FILE}" tetra)
    set_tests_properties(fem-periodic-create-3d-fixture PROPERTIES
        FIXTURES_SETUP periodic_3d_fixture)
    add_test(NAME fem-periodic-inspector-3d
        COMMAND ${CMAKE_COMMAND}
            "-DINSPECTOR=$<TARGET_FILE:fem-periodic-inspect>"
            "-DFILE=${FEM_PERIODIC_3D_FILE}"
            -DSCENARIO=3d
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_inspector.cmake")
    set_tests_properties(fem-periodic-inspector-3d PROPERTIES
        FIXTURES_REQUIRED periodic_3d_fixture)

    add_test(NAME fem-periodic-create-bad-coefficient-space-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_BAD_COEFFICIENT_SPACE_FILE}"
            bad-coefficient-space)
    set_tests_properties(fem-periodic-create-bad-coefficient-space-fixture PROPERTIES
        FIXTURES_SETUP periodic_bad_coefficient_space_fixture)
    add_test(NAME fem-periodic-reject-3d-p1-coefficients
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_BAD_COEFFICIENT_SPACE_FILE}"
            0 0 --coefficients)
    set_tests_properties(fem-periodic-reject-3d-p1-coefficients PROPERTIES
        FIXTURES_REQUIRED periodic_bad_coefficient_space_fixture
        WILL_FAIL TRUE)
    add_test(NAME fem-periodic-create-bad-cell-edge-order-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_BAD_CELL_EDGE_ORDER_FILE}"
            bad-cell-edge-order)
    set_tests_properties(fem-periodic-create-bad-cell-edge-order-fixture PROPERTIES
        FIXTURES_SETUP periodic_bad_cell_edge_order_fixture)
    add_test(NAME fem-periodic-reject-bad-cell-edge-order
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_BAD_CELL_EDGE_ORDER_FILE}" 0 0)
    set_tests_properties(fem-periodic-reject-bad-cell-edge-order PROPERTIES
        FIXTURES_REQUIRED periodic_bad_cell_edge_order_fixture
        WILL_FAIL TRUE)
    add_test(NAME fem-periodic-create-bad-periodic-edge-sign-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_BAD_PERIODIC_EDGE_SIGN_FILE}"
            bad-periodic-edge-sign)
    set_tests_properties(fem-periodic-create-bad-periodic-edge-sign-fixture PROPERTIES
        FIXTURES_SETUP periodic_bad_periodic_edge_sign_fixture)
    add_test(NAME fem-periodic-reject-bad-periodic-edge-sign
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_BAD_PERIODIC_EDGE_SIGN_FILE}" 0 0)
    set_tests_properties(fem-periodic-reject-bad-periodic-edge-sign PROPERTIES
        FIXTURES_REQUIRED periodic_bad_periodic_edge_sign_fixture
        WILL_FAIL TRUE)
    add_test(NAME fem-periodic-create-bad-vector-rank-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_BAD_VECTOR_RANK_FILE}"
            bad-vector-rank)
    set_tests_properties(fem-periodic-create-bad-vector-rank-fixture PROPERTIES
        FIXTURES_SETUP periodic_bad_vector_rank_fixture)
    add_test(NAME fem-periodic-reject-bad-vector-rank
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_BAD_VECTOR_RANK_FILE}" 0 0)
    set_tests_properties(fem-periodic-reject-bad-vector-rank PROPERTIES
        FIXTURES_REQUIRED periodic_bad_vector_rank_fixture
        WILL_FAIL TRUE)
    add_test(NAME fem-periodic-create-bad-mode-metadata-fixture
        COMMAND fem-periodic-fixture "${FEM_PERIODIC_BAD_MODE_METADATA_FILE}"
            bad-mode-metadata)
    set_tests_properties(fem-periodic-create-bad-mode-metadata-fixture PROPERTIES
        FIXTURES_SETUP periodic_bad_mode_metadata_fixture)
    add_test(NAME fem-periodic-reject-bad-mode-metadata
        COMMAND fem-periodic-inspect "${FEM_PERIODIC_BAD_MODE_METADATA_FILE}" 0 0)
    set_tests_properties(fem-periodic-reject-bad-mode-metadata PROPERTIES
        FIXTURES_REQUIRED periodic_bad_mode_metadata_fixture
        WILL_FAIL TRUE)
    if(FEM_PERIODIC_VTK_ENABLED)
        add_test(NAME fem-periodic-gui-vtk-offscreen
            COMMAND fem-periodic-viewer --smoke-test "${FEM_PERIODIC_3D_FILE}")
        set_tests_properties(fem-periodic-gui-vtk-offscreen PROPERTIES
            FIXTURES_REQUIRED periodic_3d_fixture
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            TIMEOUT 20
        )
        add_test(NAME fem-periodic-gui-vtk-slice-offscreen
            COMMAND fem-periodic-viewer --smoke-test-slice "${FEM_PERIODIC_3D_FILE}")
        set_tests_properties(fem-periodic-gui-vtk-slice-offscreen PROPERTIES
            FIXTURES_REQUIRED periodic_3d_fixture
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            TIMEOUT 20
        )
    endif()
endif()
