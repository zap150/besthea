macro(check_insource)
  if("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    message(FATAL_ERROR
    "Remove CMakeCache.txt and CMakeFiles and use a dedicated build directory, "
    "e.g., mkdir build && cd build && cmake .. && make")
  endif()
endmacro()

macro(setup_compiler)

  # can't use add_compile_options, because cmake uses them for all langueges
  # and nvcc does not understand these intel-specific options
  #
  # using MY_ADDITIONAL_CXX_FLAGS helper variable to make passing 
  # the same options to CXX compiler and CUDA host compiler simpler

  set(MY_ADDITIONAL_CXX_FLAGS "")

  if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    message(STATUS "Using GNU ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3)
      message(FATAL_ERROR "GCC compiler is too old, besthea can be"
        " compiled only with g++ 8.3.0 or higher")
    endif()

    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -Wall -Wextra")
    # can't pass this to the cuda host compiler via -Xcompiler because of errors
    string(APPEND CMAKE_CXX_FLAGS " -pedantic-errors")

    # GNU cannot vectorise complicated loops
    #string(APPEND MY_ADDITIONAL_CXX_FLAGS " -fopt-info-omp-vec-optimized-missed")

  elseif (CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    message(STATUS "Using AppleClang ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
      message(FATAL_ERROR "Clang compiler is too old, besthea can be"
        " compiled only with AppleClang 12 or higher")
    endif()

    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -Wall -Wextra -pedantic-errors")
    # Clang cannot vectorise complicated loops
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -Wno-pass-failed")

  elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang
    AND NOT CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    message(STATUS "Using Clang ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
      message(FATAL_ERROR "Clang compiler is too old, besthea can be"
        " compiled only with clang++ 7 or higher")
    endif()

    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -Wall -Wextra -pedantic-errors")
    # Clang cannot vectorise complicated loops
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -Wno-pass-failed")
    #string(APPEND MY_ADDITIONAL_CXX_FLAGS
    #  " -Rpass=\"vect\" -Rpass-missed=\"vect\" -Rpass-analysis=\"vect\"")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
      # TODO: get rid of this warning (don't understand it)
      string(APPEND MY_ADDITIONAL_CXX_FLAGS " -Wno-dtor-name")
    endif()

  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    message(STATUS "Using Intel ${CMAKE_CXX_COMPILER_VERSION} toolchain")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0.1)
      message(FATAL_ERROR "Intel compiler is too old, besthea can be"
        " compiled only with icpc 19.0.1 or higher")
    endif()

    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -w3")
    #string(APPEND MY_ADDITIONAL_CXX_FLAGS " -qopt-report=5 -qopt-report-phase=vec")
    # zero used for undefined preprocessing identifier
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 193")
    # attribute appears more than once
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 2620")
    # parameter was never referenced
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 869")
    # declaration hides variable
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 1599")
    # value copied to temporary, reference to temporary used
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 383")
    # inlining inhibited by limit max-total-size
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 11074")
    # to get full report use -qopt-report=4 -qopt-report-phase ipo
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 11076")
    # specified as both a system and non-system include directory
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 2547")
    # unrecognised GCC pragma
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 2282")
    # floating-point equality and inequality comparisons
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 1572")
    # external function definition with no prior declaration
    string(APPEND MY_ADDITIONAL_CXX_FLAGS " -diag-disable 1418")

  else()
    message(FATAL_ERROR "Unknown C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)

  if (NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE RelWithDebInfo)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  string(APPEND CMAKE_CXX_FLAGS ${MY_ADDITIONAL_CXX_FLAGS})
  if(USE_CUDA)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler \"" ${MY_ADDITIONAL_CXX_FLAGS} "\"")
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
    set(MY_OMP_FLAG "-fopenmp")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(MY_OMP_FLAG "-qopenmp")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    set(MY_OMP_FLAG "-Xclang -fopenmp")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang AND NOT CMAKE_CXX_COMPILER_ID MATCHES AppleClang)
    set(MY_OMP_FLAG "-fopenmp")
  endif()
    
  string(APPEND CMAKE_CXX_FLAGS " " ${MY_OMP_FLAG})

  if(USE_CUDA)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler \"" ${MY_OMP_FLAG} "\"")
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
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

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

  if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    string(APPEND CMAKE_CUDA_FLAGS " -arch=compute_60")
  else()
    cmake_policy(SET CMP0104 NEW)
    set(CMAKE_CUDA_ARCHITECTURES 60-virtual)
  endif()
endmacro()