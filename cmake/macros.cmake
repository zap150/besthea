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

    add_compile_options(-Wall -Wextra)
    # can't use add_compile_options here, need to pass this only to c++ compiler
    # gcc, as nvcc's host compiler, produces some of such errors, which originate
    # from some nvcc preprocessing
    string(APPEND CMAKE_CXX_FLAGS " -pedantic-errors")

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

    add_compile_options(-Wno-overlength-strings)

  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    message(STATUS "Using Intel ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0.1)
      message(FATAL_ERROR "Intel compiler is too old, besthea can be"
        " compiled only with icpc 19.0.1 or higher")
    endif()

    add_compile_options(-w3)
    #add_compile_options(-qopt-report=5 -qopt-report-phase=vec)
    # zero used for undefined preprocessing identifier
    add_compile_options(-diag-disable=193)
    # attribute appears more than once
    add_compile_options(-diag-disable=2620)
    # parameter was never referenced
    add_compile_options(-diag-disable=869)
    # declaration hides variable
    add_compile_options(-diag-disable=1599)
    # value copied to temporary, reference to temporary used
    add_compile_options(-diag-disable=383)
    # inlining inhibited by limit max-total-size
    add_compile_options(-diag-disable=11074)
    # to get full report use -qopt-report=4 -qopt-report-phase ipo
    add_compile_options(-diag-disable=11076)
    # specified as both a system and non-system include directory
    add_compile_options(-diag-disable=2547)
    # unrecognised GCC pragma
    add_compile_options(-diag-disable=2282)
    # floating-point equality and inequality comparisons
    add_compile_options(-diag-disable=1572)
    # external function definition with no prior declaration
    add_compile_options(-diag-disable=1418)
    # selector expression is constant
    add_compile_options(-diag-disable=280)

  else()
    message(FATAL_ERROR "Unknown C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)

  if (NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE RelWithDebInfo)
    set(CMAKE_BUILD_TYPE Release)
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
    add_compile_options(-fopenmp)
  elseif (CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    #add_compile_options(-I/opt/local/include/libomp -Xclang -fopenmp)
    add_compile_options(-Xclang -fopenmp)
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang
    AND NOT CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    add_compile_options(-fopenmp)
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    add_compile_options(-qopenmp)
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
  string(TOUPPER "${BESTHEA_USE_CUDA}" BESTHEA_USE_CUDA )
  if("${BESTHEA_USE_CUDA}" STREQUAL "REQUIRE"
      OR "${BESTHEA_USE_CUDA}" STREQUAL "REQUIRED")
    set(BESTHEA_USE_CUDA "REQUIRE")
  elseif("${BESTHEA_USE_CUDA}" STREQUAL "FORBID")
    set(BESTHEA_USE_CUDA "FORBID")
  elseif("${BESTHEA_USE_CUDA}" STREQUAL ""
      OR "${BESTHEA_USE_CUDA}" STREQUAL "AUTO")
    set(BESTHEA_USE_CUDA "AUTO")
  else()
    message(WARNING "Invalid value of variable"
    " BESTHEA_USE_CUDA=\"${BESTHEA_USE_CUDA}\". Using auto-detection.")
    set(BESTHEA_USE_CUDA "AUTO")
  endif()
  
  set(BESTHEA_IS_USING_CUDA OFF)

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    cmake_policy(SET CMP0074 NEW)

    if("${BESTHEA_USE_CUDA}" STREQUAL "REQUIRE")
      find_package(CUDA REQUIRED)
    elseif("${BESTHEA_USE_CUDA}" STREQUAL "AUTO")
      find_package(CUDA QUIET)
    endif()

    if(CUDA_FOUND)
      set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})    
      enable_language(CUDA)

      # older nvcc does not support -forward-unknown-to-host-compiler at all
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2.89)
        set(BESTHEA_IS_USING_CUDA ON)
      else()
        if("${BESTHEA_USE_CUDA}" STREQUAL "REQUIRE")
          message(FATAL_ERROR "CUDA compiler is too old, besthea_cuda can be"
            " compiled only with nvcc 10.2.89 or higher")
        else()
          message(WARNING "CUDA compiler is too old, besthea_cuda can be"
            " compiled only with nvcc 10.2.89 or higher. CUDA will not be used")
        endif()
      endif()
    
      cmake_policy(SET CMP0104 NEW)
      set(CMAKE_CUDA_ARCHITECTURES 60-virtual)
    endif()
  else()
    if("${BESTHEA_USE_CUDA}" STREQUAL "REQUIRE")
      message(FATAL_ERROR "CMake >= 3.18 required to support besthea_cuda")
    endif()
  endif()
endmacro()
