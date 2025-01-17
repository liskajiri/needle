set(CMAKE_CXX_COMPILER $ENV{CONDA_PREFIX}/bin/clang)
message(STATUS "CMake prefix path: ${CMAKE_PREFIX_PATH}")

cmake_minimum_required(VERSION 3.28..3.31)
project(needle LANGUAGES CXX)
cmake_policy(SET CMP0146 OLD)

option(USE_CLANG "Build with clang" ON)
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}: ${CMAKE_CXX_COMPILER_VERSION}")

# Build system setup
set(CMAKE_GENERATOR "Ninja")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Build performance options
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include_directories(
    $ENV{CONDA_PREFIX}/include
)

# Library paths
link_directories(
    $ENV{CONDA_PREFIX}/lib
)

# Compiler flags
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "-std=c++20 -O3 -march=native -mavx2 -stdlib=libc++ ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "/std:c++20 /O2 /arch:AVX2 ${CMAKE_CXX_FLAGS}")
endif()


#--- Python setup
execute_process(COMMAND python3-config --prefix OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# Detect the installed nanobind package and import it into CMake
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

###################
### CPU BACKEND ###
###################
# Module interface files

# Implementation files
set(SOURCE_FILES
    src/cpu/ndarray_backend_cpu.cc
    src/cpu/ndarray.cc
    src/cpu/elementwise.cc
    src/cpu/scalar_ops.cc
    src/cpu/matmul.cc
    src/cpu/reductions.cc
)

nanobind_add_module(ndarray_backend_cpu ${SOURCE_FILES})
target_sources(ndarray_backend_cpu
    PUBLIC
        FILE_SET modules TYPE CXX_MODULES FILES
        ${SOURCE_FILES}
)

# Solves linker issues
add_library(core INTERFACE)
target_include_directories(core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ndarray_backend_cpu PRIVATE core)
target_link_libraries(ndarray_backend_cpu PRIVATE
    c++
    c++abi
    c++experimental
)
# target_compile_options(ndarray_backend_cpu PRIVATE -stdlib=libc++)
target_compile_features(ndarray_backend_cpu PRIVATE cxx_std_20)

# OpenMP
set(OpenMP_ROOT "$ENV{CONDA_PREFIX}")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "$ENV{CONDA_PREFIX}/lib/libomp.so")
set(OpenMP_CXX_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/lib/clang/19/include")

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    include_directories(
        ${OpenMP_CXX_INCLUDE_DIRS}
        $ENV{CONDA_PREFIX}/include
    )
    link_directories($ENV{CONDA_PREFIX}/lib)
    target_link_libraries(ndarray_backend_cpu PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()

# Output properties
set_target_properties(
    ndarray_backend_cpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
)

####################
### CUDA BACKEND ###
####################
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

    execute_process(COMMAND "nvidia-smi" ERROR_QUIET RESULT_VARIABLE NV_RET)
    if(NV_RET EQUAL "0")
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
    else()
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
    endif()

    CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})
    target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
    pybind11_extension(ndarray_backend_cuda)
    pybind11_strip(ndarray_backend_cuda)

    set_target_properties(ndarray_backend_cuda PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
    )
endif()

message(STATUS "Final compile options: ${CMAKE_CXX_FLAGS}")
