macro(check_insource)
  if("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    message(FATAL_ERROR
    "Remove CMakeCache.txt and CMakeFiles and use a dedicated build directory, "
    "e.g., mkdir build && cd build && cmake .. && make")
  endif()
endmacro()

macro(setup_compiler)
  if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3)
      message(FATAL_ERROR "GCC compiler is too old, besthea can be"
        " compiled only with g++ 8.3.0 or higher")
    endif()
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors")

  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0.1)
      message(FATAL_ERROR "Intel compiler is too old, besthea can be"
        " compiled only with icpc 19.0.1 or higher")
    endif()
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w3")
    # attribute appears more than once
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable 2620")
    # parameter was never referenced
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable 869")
    # declaration hides variable
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable 1599")
    # value copied to temporary, reference to temporary used
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable 383")
    # inlining inhibited by limit max-total-size
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable 11074")
    # to get full report use -qopt-report=4 -qopt-report-phase ipo
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable 11076")
    # specified as both a system and non-system include directory
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable 2547")

  else()
    message(FATAL_ERROR "Unknown C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)

  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
  endif()

  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3")

  if (NOT DEFINED DATA_WIDTH)
    set(DATA_WIDTH 8)
  endif()

  message(STATUS "Setting DATA_WIDTH to ${DATA_WIDTH}")

  add_compile_definitions(DATA_WIDTH=${DATA_WIDTH})
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
        link_libraries(stdc++fs)
      endif()
  else()
    message(STATUS "Found std::filesystem: libstdc++")
  endif()
endmacro()

# also adds flag to linker, but we want to link against iomp5
#macro(enable_OpenMP)
#  find_package(OpenMP REQUIRED)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endmacro()

macro(enable_OpenMP)
  if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    add_compile_options("-fopenmp")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    add_compile_options("-qopenmp")
  endif()
endmacro()

macro(enable_Boost)
  include_directories(SYSTEM third_party/boost/align/include)
  include_directories(SYSTEM third_party/boost/config/include)
  include_directories(SYSTEM third_party/boost/assert/include)
  include_directories(SYSTEM third_party/boost/static_assert/include)
  include_directories(SYSTEM third_party/boost/core/include)
endmacro()

macro(enable_Eigen)
  include_directories(SYSTEM third_party/eigen)
endmacro()

macro(enable_MKL)
  find_package(MKL REQUIRED)
  include_directories(SYSTEM ${MKL_INCLUDE_DIRS})
endmacro()

macro(enable_MPI)
  find_package(MPI REQUIRED)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endmacro()

macro(enable_catch)
  include_directories(SYSTEM third_party/catch)
endmacro()

macro(enable_Lyra)
  include_directories(SYSTEM third_party/Lyra/include)
endmacro()
