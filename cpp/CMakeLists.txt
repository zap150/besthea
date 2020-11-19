cmake_minimum_required(VERSION 3.5)
project(besthea)

include(cmake/macros.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

check_insource()
setup_compiler()

enable_MKL()
enable_MPI()
enable_OpenMP()
enable_Boost()
enable_Eigen()
enable_catch()

file(GLOB besthea_SOURCES src/*.cpp)
include_directories(include)
add_library(besthea STATIC ${besthea_SOURCES})

option(BUILD_TEST "Build tests using the library" OFF)

if(BUILD_TEST)
  add_subdirectory(test)
endif()
