cmake_minimum_required(VERSION 3.31)
project(needle LANGUAGES CXX)
cmake_policy(SET CMP0146 OLD)

# Build system setup
set(CMAKE_GENERATOR "Ninja")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

#--- Python setup
execute_process(COMMAND python3-config --prefix OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

#--- pybind11 setup
execute_process(COMMAND python3 -m pybind11 --cmakedir
    RESULT_VARIABLE __pybind_exit_code
    OUTPUT_VARIABLE __pybind_path
    OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})

# Compiler flags
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "-std=c++20 -O3 -march=native -mavx2 ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "/std:c++20 /O2 /arch:AVX2 ${CMAKE_CXX_FLAGS}")
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})

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

add_library(ndarray_backend_cpu MODULE src/cpu/ndarray_backend_cpu.cc)
target_sources(ndarray_backend_cpu
    PUBLIC
        FILE_SET modules TYPE CXX_MODULES FILES
        ${SOURCE_FILES}
)

# Add modules
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ndarray_backend_cpu PRIVATE -fmodules)
    set(CMAKE_CXX_COMPILER clang++)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ndarray_backend_cpu PRIVATE -fmodules-ts)
    set(CMAKE_CXX_COMPILER g++)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(ndarray_backend_cpu PRIVATE /experimental:module)
    set(CMAKE_CXX_COMPILER MSVC)
endif()
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
target_compile_features(ndarray_backend_cpu PRIVATE cxx_std_20)

target_link_libraries(ndarray_backend_cpu
    PUBLIC
        ${LINKER_LIBS}
        pybind11::module
)

pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)

# OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(ndarray_backend_cpu PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found")
endif()

# Output properties
set_target_properties(ndarray_backend_cpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()

####################
### CUDA BACKEND ###
####################
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "Found CUDA, building CUDA backend")
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
