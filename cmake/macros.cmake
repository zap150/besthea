macro(check_insource)
  if("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    message(FATAL_ERROR
    "Remove CMakeCache.txt and CMakeFiles and use a dedicated build directory, "
    "e.g., mkdir build && cd build && cmake .. && make")
  endif()
endmacro()

macro(setup_compiler)
  if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    message(STATUS "Using GNU ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3)
      message(FATAL_ERROR "GCC compiler is too old, besthea can be"
        " compiled only with g++ 8.3.0 or higher")
    endif()

    # cant use add_compile_options, because cmake adds the options
    # to the cuda compiler as well, which does not understand them
    string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic-errors")
    # add_compile_options(-Wall -Wextra -pedantic-errors)

    # GNU cannot vectorise complicated loops
    #add_compile_options(-fopt-info-omp-vec-optimized-missed)

  elseif (CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    message(STATUS "Using AppleClang ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
      message(FATAL_ERROR "Clang compiler is too old, besthea can be"
        " compiled only with AppleClang 12 or higher")
    endif()

    add_compile_options(-Wall -Wextra -pedantic-errors)
    # Clang cannot vectorise complicated loops
    add_compile_options(-Wno-pass-failed)

  elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang
    AND NOT CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    message(STATUS "Using Clang ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
      message(FATAL_ERROR "Clang compiler is too old, besthea can be"
        " compiled only with clang++ 7 or higher")
    endif()

    add_compile_options(-Wall -Wextra -pedantic-errors)
    # Clang cannot vectorise complicated loops
    add_compile_options(-Wno-pass-failed)
    #add_compile_options(
    #  -Rpass="vect" -Rpass-missed="vect" -Rpass-analysis="vect")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
      # TODO: get rid of this warning (don't understand it)
      add_compile_options(-Wno-dtor-name)
    endif()

  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    message(STATUS "Using Intel ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0.1)
      message(FATAL_ERROR "Intel compiler is too old, besthea can be"
        " compiled only with icpc 19.0.1 or higher")
    endif()

    # cant use add_compile_options, because cmake adds the options
    # to the cuda compiler as well, which does not understand them
    string(APPEND CMAKE_CXX_FLAGS " -w3")
    #add_compile_options(-w3)
    #add_compile_options(-qopt-report=5 -qopt-report-phase=vec)
    # attribute appears more than once
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 2620")
    # parameter was never referenced
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 869")
    # declaration hides variable
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 1599")
    # value copied to temporary, reference to temporary used
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 383")
    # inlining inhibited by limit max-total-size
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 11074")
    # to get full report use -qopt-report=4 -qopt-report-phase ipo
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 11076")
    # specified as both a system and non-system include directory
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 2547")
    # unrecognised GCC pragma
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 2282")
    # floating-point equality and inequality comparisons
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 1572")
    # external function definition with no prior declaration
    string(APPEND CMAKE_CXX_FLAGS " -diag-disable 1418")

  else()
    message(FATAL_ERROR "Unknown C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)

  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
  endif()
endmacro()

macro(enable_filesystem)
  # try without -lstd++fs
  try_run(RUNS_WITH_STDFS COMPILES_WITH_STDFS
    "${CMAKE_BINARY_DIR}/try"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/has_stdfs.cpp"
    CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
    RUN_OUTPUT_VARIABLE TRY_OUTPUT)

  if (NOT COMPILES_WITH_STDFS OR RUNS_WITH_STDFS STREQUAL "FAILED_TO_RUN")
    # try with -lstd++fs
    try_run(RUNS_WITH_STDFS COMPILES_WITH_STDFS
      "${CMAKE_BINARY_DIR}/try"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/has_stdfs.cpp"
      CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
      LINK_LIBRARIES stdc++fs
      RUN_OUTPUT_VARIABLE TRY_OUTPUT)

      if (NOT COMPILES_WITH_STDFS OR RUNS_WITH_STDFS STREQUAL "FAILED_TO_RUN")
        message(FATAL_ERROR "No std::filesystem support")
      else()
        message(STATUS "Found std::filesystem: libstdc++fs")
        set(STDFS_LIBRARIES stdc++fs)
      endif()
  else()
    message(STATUS "Found std::filesystem: libstdc++")
  endif()
endmacro()

macro(enable_OpenMP)
  if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    string(APPEND CMAKE_CXX_FLAGS " -fopenmp")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    string(APPEND CMAKE_CXX_FLAGS " -qopenmp")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    string(APPEND CMAKE_CXX_FLAGS " -Xclang -fopenmp")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang AND NOT CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    string(APPEND CMAKE_CXX_FLAGS " -fopenmp")
  endif()

  if(USE_CUDA)
    # now cuda host compiler is g++, so -fopenmp is enough
    # in the future, when other cuda host compilers are supported, similar else-ifs will need to be there as above
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -fopenmp")
  endif()
endmacro()

macro(enable_Boost)
  set(Boost_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/align/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/config/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/assert/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/static_assert/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/core/include)
endmacro()

macro(enable_Eigen)
  set(Eigen_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
endmacro()

macro(enable_MKL)
  find_package(MKL REQUIRED)
endmacro()

macro(enable_MPI)
  find_package(MPI REQUIRED)
  set(MPI_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
endmacro()

macro(enable_Lyra)
  set(Lyra_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Lyra/include)
endmacro()

macro(setup_CUDA)
  # intel host compiler in combination with eigen 3.3.90 causes problems
  # specifically https://gitlab.com/libeigen/eigen/-/issues/2180
  # therefore we stick with the default host compiler, which is on linux g++
  # and do not change anything. after intel is working, this should be reworked

  enable_language(CUDA)

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
    cmake_policy(SET CMP0074 NEW)
  endif()

  find_package(CUDA REQUIRED)

  if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    string(APPEND CMAKE_CUDA_FLAGS " -std=c++17")
  else()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
  endif()

  string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wall")
  string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wextra")

  if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    string(APPEND CMAKE_CUDA_FLAGS " -arch=compute_60")
  else()
    cmake_policy(SET CMP0104 NEW)
    set(CMAKE_CUDA_ARCHITECTURES 60-virtual)
  endif()
endmacro()