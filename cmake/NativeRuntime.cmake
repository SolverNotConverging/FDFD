if(WIN32 AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "Windows builds require MSVC. MinGW is not supported. Use a fresh build directory and scripts/setup_msvc_windows.ps1.")
endif()
if(WIN32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Windows builds require an x64 MSVC toolchain (x64-windows dependencies).")
endif()

set(FDFD_QT_TEST_ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
if(WIN32)
    # vcpkg's Qt Base supplies qminimal rather than qoffscreen on Windows.
    set(FDFD_QT_TEST_ENVIRONMENT
        "QT_QPA_PLATFORM=minimal;QT_QPA_PLATFORM_PLUGIN_PATH=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/Qt6/plugins/platforms")
endif()

# Record the libraries selected by CMake, including multi-config Debug/Release
# locations. These local-build hints are not a redistributable runtime bundle.
if(NOT DEFINED FDFD_NATIVE_INSTALL_DIR)
    set(FDFD_NATIVE_INSTALL_DIR bin)
    set(FDFD_NATIVE_BUNDLE_INSTALL_DIR .)
    set(FDFD_NATIVE_LICENSE_INSTALL_DIR share/licenses)
endif()

function(fdfd_native_runtime target)
    if(NOT WIN32)
        # Local source installs continue using the dependency installation
        # selected by CMake (e.g. Homebrew or a non-system Linux prefix).
        set_property(TARGET ${target} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
        return()
    endif()
    get_property(runtime_configured GLOBAL PROPERTY FDFD_MSVC_RUNTIME_CONFIGURED)
    if(NOT runtime_configured)
        set_property(GLOBAL PROPERTY FDFD_MSVC_RUNTIME_CONFIGURED TRUE)
        set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${FDFD_NATIVE_INSTALL_DIR}")
        include(InstallRequiredSystemLibraries)
    endif()
    get_filename_component(compiler_directory "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(runtime_files "$<JOIN:$<TARGET_RUNTIME_DLLS:${target}>,\n>")
    set(runtime_directories "${compiler_directory}")
    if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
        string(APPEND runtime_directories
            "\n${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/$<$<CONFIG:Debug>:debug/>bin")
    endif()
    # Qt's imported plugin target also works for installer and vcpkg layouts.
    set(plugin_file "")
    if(TARGET Qt6::QWindowsIntegrationPlugin)
        set(plugin_file "$<TARGET_FILE:Qt6::QWindowsIntegrationPlugin>")
    endif()
    set(hint "$<TARGET_FILE_DIR:${target}>/${target}.runtime.txt")
    file(GENERATE OUTPUT "${hint}" CONTENT
        "compiler=${CMAKE_CXX_COMPILER_ID}\n[dlls]\n${runtime_files}\n[directories]\n${runtime_directories}\n[platform-plugin]\n${plugin_file}\n")
    install(FILES "${hint}" DESTINATION "${FDFD_NATIVE_INSTALL_DIR}")

    # Resolve transitive imports too (Gmsh/OpenCASCADE, HDF5, VTK, Qt).
    # Installing from the same build works for standalone apps and uv wheels.
    set(deploy_plugins "")
    foreach(plugin IN ITEMS QWindowsIntegrationPlugin QMinimalIntegrationPlugin QOffscreenIntegrationPlugin)
        if(TARGET Qt6::${plugin})
            string(APPEND deploy_plugins "\"$<TARGET_FILE:Qt6::${plugin}>\"\n")
        endif()
    endforeach()
    set(deploy_executable "$<TARGET_FILE:${target}>")
    set(deploy_bin "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/$<$<CONFIG:Debug>:debug/>bin")
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DeployWindows.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${target}-deploy.in" @ONLY)
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${target}-deploy-$<CONFIG>.cmake"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/${target}-deploy.in")
    install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${target}-deploy-$<CONFIG>.cmake")
endfunction()

# A wheel cannot rely on the Qt installation used by the build machine.  The
# shared libraries are relocated by delocate after the wheel is built, while
# Qt's dynamically discovered plugins must first be installed explicitly.
function(fdfd_macos_qt_plugins target)
    if(NOT APPLE OR NOT SKBUILD)
        return()
    endif()

    set(qt_plugins
        "Qt6::QCocoaIntegrationPlugin|platforms"
        "Qt6::QOffscreenIntegrationPlugin|platforms"
        "Qt6::QMacStylePlugin|styles"
        "Qt6::QGifPlugin|imageformats"
        "Qt6::QICOPlugin|imageformats"
        "Qt6::QJpegPlugin|imageformats"
    )
    foreach(plugin IN LISTS qt_plugins)
        string(REPLACE "|" ";" plugin_fields "${plugin}")
        list(GET plugin_fields 0 plugin_target)
        list(GET plugin_fields 1 plugin_directory)
        if(TARGET "${plugin_target}")
            install(FILES "$<TARGET_FILE:${plugin_target}>"
                DESTINATION
                    "${FDFD_NATIVE_BUNDLE_INSTALL_DIR}/${target}.app/Contents/PlugIns/${plugin_directory}"
            )
        elseif(plugin_target STREQUAL "Qt6::QCocoaIntegrationPlugin")
            message(FATAL_ERROR "The macOS Qt platform plugin is required for wheel builds")
        endif()
    endforeach()
endfunction()
