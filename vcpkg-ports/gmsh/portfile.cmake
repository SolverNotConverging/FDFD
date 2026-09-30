vcpkg_download_distfile(ARCHIVE
    URLS "https://gmsh.info/src/gmsh-${VERSION}-source.tgz"
    FILENAME "gmsh-${VERSION}-source.tgz"
    SHA512 9e05267e3d2bae54eeab21b384ab5cc85383e8040ed622ff859485fabd3dc9fc4b5a87d9d58acf3fc5573471d9c3c8e0831df4f44ae196d1a26ecbe9c613753c
)
vcpkg_extract_source_archive(
    SOURCE_PATH
    ARCHIVE "${ARCHIVE}"
    PATCHES
        installdirs.diff
        linking-and-naming.diff
        opencascade.diff
)
# OpenCASCADE 8 requires C++17; Gmsh 4.15 still hard-codes C++14.
vcpkg_replace_string("${SOURCE_PATH}/CMakeLists.txt"
    "set(CMAKE_CXX_STANDARD 14)"
    "set(CMAKE_CXX_STANDARD 17)\nif(MSVC)\n  add_compile_options(/bigobj)\nendif()")
# OpenCASCADE 8 folded iterator headers into their container headers and
# renamed the 2D curve-properties alias without changing its operations.
vcpkg_replace_string("${SOURCE_PATH}/src/geo/OCCEdge.cpp"
    "#include <Geom2dLProp_CLProps2d.hxx>"
    "#include <GeomLProp_CLProps.hxx>")
vcpkg_replace_string("${SOURCE_PATH}/src/geo/OCCEdge.cpp"
    "Geom2dLProp_CLProps2d aCLProps"
    "GeomLProp_CLProps2d aCLProps")
vcpkg_replace_string("${SOURCE_PATH}/src/geo/GModelIO_OCC.cpp"
    "#include <TopTools_DataMapIteratorOfDataMapOfIntegerShape.hxx>"
    "#include <TopTools_DataMapOfIntegerShape.hxx>")
vcpkg_replace_string("${SOURCE_PATH}/src/geo/GModelIO_OCC.cpp"
    "#include <TopTools_DataMapIteratorOfDataMapOfShapeInteger.hxx>"
    "#include <TopTools_DataMapOfShapeInteger.hxx>")
vcpkg_replace_string("${SOURCE_PATH}/src/geo/GModelIO_OCC.cpp"
    "#include <TopTools_ListIteratorOfListOfShape.hxx>"
    "#include <TopTools_ListOfShape.hxx>\n#include <TopTools_MapOfShape.hxx>\n#include <Interface_HArray1OfHAsciiString.hxx>\n#include <TCollection_HAsciiString.hxx>\n#include <TDF_LabelSequence.hxx>")
# OpenCASCADE 8 removed the Handle_Class typedefs; Handle(Class) is supported
# by both the current and earlier OpenCASCADE releases.
foreach(occ_class IN ITEMS
    HeaderSection_FileDescription HeaderSection_FileName HeaderSection_FileSchema
    Interface_HArray1OfHAsciiString Standard_Transient StepData_StepModel
    TCollection_HAsciiString TDocStd_Document XCAFApp_Application
    XCAFDoc_ColorTool XCAFDoc_LayerTool XCAFDoc_MaterialTool XCAFDoc_ShapeTool)
    vcpkg_replace_string("${SOURCE_PATH}/src/geo/GModelIO_OCC.cpp"
        "Handle_${occ_class}" "Handle(${occ_class})")
endforeach()
# The upstream Windows process helper passes a narrow command line even when
# vcpkg enables UNICODE, so select the matching Win32 API explicitly.
vcpkg_replace_string("${SOURCE_PATH}/src/mesh/Field.cpp"
    "STARTUPINFO siStartInfo;" "STARTUPINFOA siStartInfo;")
vcpkg_replace_string("${SOURCE_PATH}/src/mesh/Field.cpp"
    "sizeof(STARTUPINFO)" "sizeof(STARTUPINFOA)")
vcpkg_replace_string("${SOURCE_PATH}/src/mesh/Field.cpp"
    "bSuccess = CreateProcess(" "bSuccess = CreateProcessA(")

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" BUILD_LIB)
string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" BUILD_SHARED)
string(COMPARE EQUAL "${VCPKG_CRT_LINKAGE}" "static" STATIC_RUNTIME)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        mpi         ENABLE_MPI
        occ         ENABLE_OCC
        occ         ENABLE_OCC_CAF
        zipper      ENABLE_ZIPPER
)


vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        ${FEATURE_OPTIONS}
        -DENABLE_BUILD_LIB=${BUILD_LIB}
        -DENABLE_BUILD_SHARED=${BUILD_SHARED}
        -DENABLE_MSVC_STATIC_RUNTIME=${STATIC_RUNTIME}
        -DENABLE_OS_SPECIFIC_INSTALL=OFF
        -DGMSH_PACKAGER=vcpkg
        -DGMSH_RELEASE=ON
        -DENABLE_PACKAGE_STRIP=ON
        -DENABLE_SYSTEM_CONTRIB=ON
        # Not implemented
        -DENABLE_GRAPHICS=OFF # Requires mesh, post, plugins and onelab
        -DENABLE_POST=OFF
        -DENABLE_PLUGINS=OFF
        # FDFD requires native mesh generation, not just the CAD API.
        -DENABLE_MESH=ON
        -DENABLE_PARSER=OFF
        -DENABLE_PROFILE=OFF
        -DENABLE_PRIVATE_API=OFF
        -DENABLE_QUADMESHINGTOOLS=OFF
        -DENABLE_PRO=OFF
        -DENABLE_TOUCHBAR=OFF
        -DENABLE_VISUDEV=OFF
        -DENABLE_BLAS_LAPACK=OFF
        -DENABLE_CAIRO=OFF
        -DENABLE_CGNS=OFF
        -DENABLE_CGNS_CPEX0045=OFF
        -DENABLE_EIGEN=ON
        -DENABLE_GMP=OFF
        -DENABLE_OPENMP=OFF
        -DENABLE_POPPLER=OFF
        -DENABLE_WRAP_JAVA=OFF
        -DENABLE_WRAP_PYTHON=OFF
        # Requies dependencies which not included in vcpkg yet
        -DENABLE_3M=OFF
        -DENABLE_ALGLIB=OFF
        -DENABLE_ANN=OFF
        -DENABLE_BAMG=OFF
        -DENABLE_BLOSSOM=OFF
        -DENABLE_BUILD_DYNAMIC=OFF # Needs gfortran
        -DENABLE_FLTK=OFF # Needs executable fltk-config
        -DENABLE_DINTEGRATION=OFF
        -DENABLE_GEOMETRYCENTRAL=OFF
        -DENABLE_DOMHEX=OFF
        -DENABLE_GETDP=OFF
        -DENABLE_GMM=OFF
        -DENABLE_HXT=OFF
        -DENABLE_KBIPACK=OFF
        -DENABLE_MATHEX=OFF
        -DENABLE_MED=OFF
        -DENABLE_METIS=OFF
        -DENABLE_MMG=OFF
        -DENABLE_MPEG_ENCODE=OFF
        -DENABLE_MUMPS=OFF
        -DENABLE_NUMPY=OFF
        -DENABLE_NETGEN=OFF
        -DENABLE_PETSC4PY=OFF
        -DENABLE_ONELAB_METAMODEL=OFF
        -DENABLE_ONELAB=OFF
        -DENABLE_OPENACC=OFF
        -DENABLE_OPTHOM=OFF
        -DENABLE_OSMESA=OFF
        -DENABLE_P4EST=OFF
        -DENABLE_PETSC=OFF
        -DENABLE_QUADTRI=OFF
        -DENABLE_REVOROPT=OFF
        -DENABLE_SLEPC=OFF
        -DENABLE_SOLVER=OFF
        -DENABLE_TCMALLOC=OFF
        -DENABLE_VOROPP=OFF
        -DENABLE_WINSLOWUNTANGLER=OFF
        # experimental
        -DENABLE_BUILD_ANDROID=OFF
        -DENABLE_BUILD_IOS=OFF

        -DENABLE_OS_SPECIFIC_INSTALL=OFF # Needs system permission
        -DENABLE_RPATH=OFF # Should use dependencies in vcpkg
        -DENABLE_TESTS=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()
file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/fdfd-meshing-enabled.txt"
    "ENABLE_MESH=ON\n")

vcpkg_copy_tools(TOOL_NAMES gmsh AUTO_CLEAN)

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
