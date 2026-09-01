#==================================================================
#==== REQUIRED CMAKE VERSION ======================================
#==================================================================
cmake_minimum_required(VERSION 3.28)

#==================================================================
#==== PROJECT NAME AND LANGS ======================================
#==================================================================
project(
    PARALLEL_CFD
    VERSION 0.1.0
    DESCRIPTION "Parallel unstructured-mesh CFD preprocessing toolkit"
    LANGUAGES C CXX
)

#==================================================================
#==== TESTING MANAGEMENT ==========================================
#==================================================================
include(CTest)

#==================================================================
#==== MPI LIBRARY =================================================
#==================================================================
find_package(MPI 3.1 REQUIRED COMPONENTS C CXX)

#==================================================================
#==== CGNS/HDF5 LIBRARY ===========================================
#==================================================================
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)
if(NOT HDF5_IS_PARALLEL)
    message(
        FATAL_ERROR
        "Serial HDF5 was found, but the project requires Parallel HDF5"
    )
endif()

# CGNS targets reference "hdf5::hdf5-static", which is only defined by the
# HDF5 project's own CMake config — map that name to CMake's FindHDF5 target.
if(NOT TARGET hdf5::hdf5-static)
    add_library(hdf5::hdf5-static INTERFACE IMPORTED GLOBAL)

    set_target_properties(
        hdf5::hdf5-static
        PROPERTIES
            INTERFACE_LINK_LIBRARIES HDF5::HDF5
    )
endif()

find_package(cgns CONFIG QUIET)
if(NOT TARGET CGNS::cgns_static)
    # CGNS 4.4.0 exports its targets without a package configuration file
    # (cgnsConfig.cmake appears only in 4.5+) — pick the targets file directly.
    find_path(
        CGNS_CMAKE_DIR
        NAMES cgns-targets.cmake
        PATH_SUFFIXES lib/cmake/cgns
        REQUIRED
    )
    include("${CGNS_CMAKE_DIR}/cgns-targets.cmake")
endif()
if(NOT TARGET CGNS::cgns_static)
    message(
        FATAL_ERROR
        "CGNS::cgns_static target was not found"
    )
endif()

set_property(TARGET CGNS::cgns_static PROPERTY INTERFACE_LINK_LIBRARIES "")
target_link_libraries(CGNS::cgns_static INTERFACE HDF5::HDF5)


#==================================================================
#==== DKAMINPAR LIBRARY ===========================================
#==================================================================
find_package(
    KaMinPar 3.7.3
    CONFIG
    REQUIRED
)
find_package(TBB CONFIG REQUIRED)

#==================================================================
#==== CXX MINIMUM VERSION REQUIRED ================================
#==================================================================
add_library(cfd_project_options INTERFACE)
target_compile_features(cfd_project_options INTERFACE cxx_std_20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) 

#==================================================================
#==== CXX WARNINGS LIBRARY ========================================
#==================================================================
add_library(cfd_project_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        cfd_project_warnings
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
    )
endif()

#==================================================================
#==== GLOBAL INDEX DATATYPE =======================================
#==================================================================
option(CFD_INDEX_64BIT "Enable 64-bit global mesh indices" OFF)

if(CFD_INDEX_64BIT)
    add_compile_definitions(CFD_INDEX_64BIT)
    message(STATUS "Global indexing: 64-bit (int64_t)")
else()
    message(STATUS "Global indexing: 32-bit (int32_t)")
endif()

#==================================================================
#==== LIBRARIES ===================================================
#==================================================================
    #================
    #=== CFD CORE ===
    #================

    #===================
    #=== MPI WRAPPER ===
    #===================
add_library(
    cfd_mpi
    STATIC
        src/mpi/log.cpp
        src/mpi/mpi_util.cpp
)

target_include_directories(
    cfd_mpi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    cfd_mpi
    PUBLIC
        MPI::MPI_CXX
        cfd_project_options
    PRIVATE
        cfd_project_warnings
)

    #=================
    #=== MESH DATA ===
    #=================
add_library(
    cfd_mesh
    STATIC
        src/mesh/geometry.cpp
        src/mesh/faces.cpp
        src/mesh/localmesh.cpp
        src/mesh/reorder.cpp
        src/mesh/bc_config.cpp
        src/mesh/validate.cpp
)

target_include_directories(
    cfd_mesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    cfd_mesh
    PUBLIC
        cfd_project_options
        cfd_mpi
    PRIVATE
        cfd_project_warnings
)

    #=================
    #=== PARTITION ===
    #=================
add_library(
    cfd_partition
    STATIC
        src/partition/partition.cpp
)

target_include_directories(
    cfd_partition
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    cfd_partition
    PUBLIC
        cfd_project_options
        cfd_mesh
        cfd_mpi
    PRIVATE
        KaMinPar::dKaMinPar
        TBB::tbb
        cfd_project_warnings
)

    #=========================
    #=== CGNS INPUT/OUTPUT ===
    #=========================
add_library(
    cfd_cgns_io
    STATIC
        src/io/cgns/cgns_reader.cpp
)

target_include_directories(
    cfd_cgns_io
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    cfd_cgns_io
    PUBLIC
        CGNS::cgns_static
        cfd_project_options
        cfd_mpi
    PRIVATE
        cfd_project_warnings
)

    #==============================================
    #=== SOLVER MESH (PARTITIONED) INPUT/OUTPUT ===
    #==============================================
add_library(
    cfd_solver_mesh_io
    STATIC
        src/io/solver_mesh/hdf5_writer.cpp
        src/io/solver_mesh/hdf5_reader.cpp
        src/io/vtk/vtu.cpp
)

target_include_directories(
    cfd_solver_mesh_io
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    cfd_solver_mesh_io
    PUBLIC
        MPI::MPI_CXX
        HDF5::HDF5
        cfd_project_options
        cfd_mesh
        cfd_partition
    PRIVATE
        cfd_project_warnings
)

#==================================================================
#==== EXECUTABLES =================================================
#==================================================================
    #===============================================
    #=== MESH PARTITONING (PREPARE SOLVER INPUT) ===
    #===============================================
add_executable(mesh_partition apps/mesh_partition.cpp)

target_link_libraries(
    mesh_partition
    PRIVATE
        cfd_project_warnings
        cfd_project_options
        cfd_cgns_io
          
        cfd_mpi      
        cfd_mesh      
        cfd_partition  
        cfd_solver_mesh_io          
)

    #=============================
    #=== PARTITIONED MESH READ ===
    #=============================
add_executable(solver_mesh_check apps/solver_mesh_check.cpp)

target_link_libraries(
    solver_mesh_check
    PRIVATE
    cfd_project_warnings
    cfd_project_options

    cfd_mpi      
    cfd_mesh      
    cfd_solver_mesh_io 
)

#==================================================================
#==== TESTS =======================================================
#==================================================================
if(BUILD_TESTING)
    #=== DKAMINPAR DISTRIBUTED API SMOKE TEST ===
    add_executable(cfd_kaminpar_api tests/kaminpar_api.cpp)

    target_link_libraries(
        cfd_kaminpar_api
        PRIVATE
            KaMinPar::dKaMinPar
            TBB::tbb
            MPI::MPI_CXX
            cfd_project_options
            cfd_project_warnings
    )

    add_test(
        NAME kaminpar.api.2
        COMMAND
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:cfd_kaminpar_api>
    )

    set_tests_properties(
        kaminpar.api.2
        PROPERTIES
            PROCESSORS 2
            TIMEOUT 60
    )
endif()
