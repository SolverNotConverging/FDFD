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
endfunction()
