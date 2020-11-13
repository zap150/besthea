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

file(GLOB besthea_SOURCES src/*.cpp)

add_library(besthea STATIC ${besthea_SOURCES})

add_subdirectory(test/uniform_tensor_dirichlet)
add_subdirectory(test/uniform_tensor_neumann)
