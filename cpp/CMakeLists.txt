cmake_minimum_required(VERSION 3.13)
project(besthea)

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3)
    message(FATAL_ERROR "GCC compiler is too old, besthea can be"
      " compiled only with g++-8.3 or higher.")
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/macros.cmake)

setup_compiler()

if("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
  message(FATAL_ERROR "In-source build is disabled. Remove CMakeCache.txt and"
    " CMakFiles and start again from a dedicated build directory.")
endif()

add_compile_definitions(DATA_WIDTH=4)

enable_OpenMP()

#find_package(MKL REQUIRED)
#include_directories(SYSTEM ${MKL_INCLUDE_DIRS})

include_directories($ENV{MKLROOT}/include)
link_directories($ENV{MKLROOT}/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_INT=long")

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

include_directories(include external third_party/eigen)

file(GLOB besthea_SOURCES src/*.cpp)

add_library(besthea STATIC ${besthea_SOURCES})

add_subdirectory(test/uniform_tensor_dirichlet)
add_subdirectory(test/uniform_tensor_neumann)
