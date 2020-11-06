macro(enable_OpenMP_impl)
  find_package(OpenMP)

  if(OPENMP_FOUND)
    message(STATUS "Found OpenMP.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")

    if(NOT OpenMP_FOUND)
      # for compatibility with older cmake version
      set(OpenMP_FOUND TRUE)
    endif()
  else()
    message(STATUS "OpenMP not available. Your program will lack parallelism.")
  endif()
endmacro()

macro(enable_OpenMP)
  set(enable TRUE)

  if(BUILDSYS_DISABLE_OPENMP)
    set(enable FALSE)
  elseif(INTERNAL_BUILDSYS_DISABLE_OPENMP)
    set(enable FALSE)
  endif()

  if(enable)
    enable_OpenMP_impl()
  else()
    message(STATUS "Disabling OpenMP support because of BUILDSYS_DISABLE_OPENMP.")
  endif()
endmacro()

macro(enable_cpp17)
  if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic-errors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor -Wno-unused-function")
    #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")
    #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")
    #set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--as-needed")
  #elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang)
  #  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++ -Wno-conversion")
  #  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
  else()
    message(FATAL_ERROR "Unknown C++ compiler: ${CMAKE_CXX_COMPILER_ID}.")
  endif()
  message(STATUS "Enabled C++17 for C++ (${CMAKE_CXX_COMPILER_ID})")
endmacro()

macro(buildsys_library target)
  set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/lib)
  set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/lib)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin)

  set_target_properties(${target} PROPERTIES BUILDSYS_HOME_DIR
    ${CMAKE_CURRENT_LIST_DIR})
endmacro()
