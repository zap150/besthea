cmake_minimum_required(VERSION 3.13)
project(besthea)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/macros.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

setup_compiler()

if("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
  message(FATAL_ERROR "In-source build is disabled. Remove CMakeCache.txt and"
    " CMakFiles and start again from a dedicated build directory.")
endif()

add_compile_definitions(DATA_WIDTH=8)

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

enable_OpenMP()
enable_Boost()
enable_Eigen()

file(GLOB besthea_SOURCES src/*.cpp)

add_library(besthea STATIC ${besthea_SOURCES})

add_subdirectory(test/uniform_tensor_dirichlet)
add_subdirectory(test/uniform_tensor_neumann)
