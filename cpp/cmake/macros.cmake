macro(check_insource)
  if("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    message(FATAL_ERROR "In source build is not allowed. "
    "Remove CMakeCache.txt and CMakFiles and use a dedicated build directory.")
  endif()
endmacro()

macro(setup_compiler)
  if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -pedantic-errors")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3)
      message(FATAL_ERROR "GCC compiler is too old, besthea can be"
        " compiled only with g++-8.3 or higher.")
    endif()
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -w2")
  else()
    message(FATAL_ERROR "Unknown C++ compiler: ${CMAKE_CXX_COMPILER_ID}.")
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3")
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

  add_compile_definitions(DATA_WIDTH=8)
endmacro()

macro(enable_OpenMP)
  find_package(OpenMP REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endmacro()

macro(enable_Boost)
  include_directories(include third_party/boost/align/include)
  include_directories(include third_party/boost/config/include)
  include_directories(include third_party/boost/assert/include)
  include_directories(include third_party/boost/static_assert/include)
  include_directories(include third_party/boost/core/include)

endmacro()

macro(enable_Eigen)
  include_directories(include third_party/eigen)
endmacro()

macro(enable_MKL)
  find_package(MKL REQUIRED)
  include_directories(${MKL_INCLUDE_DIRS})
endmacro()

macro(enable_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_INCLUDE_PATH})
endmacro()
