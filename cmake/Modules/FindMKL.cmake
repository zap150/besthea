# a simple cmake script to locate Intel Math Kernel Library via MKLROOT

# Stage 1: find the root directory

set(MKLROOT_PATH $ENV{MKLROOT})

# Stage 2: find include path and libraries

if (MKLROOT_PATH)
  # includes
  set(EXPECT_MKL_INCPATH "${MKLROOT_PATH}/include")

  if (IS_DIRECTORY ${EXPECT_MKL_INCPATH})
    set(MKL_INCLUDE_DIRS ${EXPECT_MKL_INCPATH})
  endif (IS_DIRECTORY ${EXPECT_MKL_INCPATH})

  # libs
  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib")
  endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")

  if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	#if (CMAKE_SIZEOF_VOID_P MATCHES 8)
	  set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib/intel64")
	#else (CMAKE_SIZEOF_VOID_P MATCHES 8)
	#  set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib/ia32")
	#endif (CMAKE_SIZEOF_VOID_P MATCHES 8)
  endif (CMAKE_SYSTEM_NAME MATCHES "Linux")

  if (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})
	  set(MKL_LIBRARY_DIR ${EXPECT_MKL_LIBPATH})
  endif (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})

  string(REPLACE ":" ";" ICC_LIBRARY_DIR $ENV{LIBRARY_PATH})

  # find specific library files
  find_library(LIB_MKL_CORE NAMES mkl_core HINTS ${MKL_LIBRARY_DIR})
  if(BESTHEA_MKL_USE_TBB)
    find_library(LIB_MKL_TBB_THREAD NAMES mkl_tbb_thread
      HINTS ${MKL_LIBRARY_DIR})
    find_library(LIB_TBB NAMES tbb HINTS ${ICC_LIBRARY_DIR})
  else()
    find_library(LIB_MKL_INTEL_THREAD NAMES mkl_intel_thread
      HINTS ${MKL_LIBRARY_DIR})
  endif()
  find_library(LIB_MKL_INTEL_ILP64 NAMES mkl_intel_ilp64
    HINTS ${MKL_LIBRARY_DIR})
  find_library(LIB_IOMP5 NAMES iomp5 HINTS ${ICC_LIBRARY_DIR})
  find_library(LIB_PTHREAD NAMES pthread)

endif (MKLROOT_PATH)

if (CMAKE_SYSTEM_NAME MATCHES Linux AND CMAKE_CXX_COMPILER_ID MATCHES GNU)
  set(NO_AS_NEEDED -Wl,--no-as-needed)
endif (CMAKE_SYSTEM_NAME MATCHES Linux AND CMAKE_CXX_COMPILER_ID MATCHES GNU)

set(MKL_LIBRARIES
  ${NO_AS_NEEDED}
  ${LIB_MKL_CORE}
  ${LIB_MKL_INTEL_THREAD}
  ${LIB_MKL_INTEL_ILP64}
  ${LIB_IOMP5}
  ${LIB_PTHREAD})

# deal with QUIET and REQUIRED argument

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MKL DEFAULT_MSG
  MKL_LIBRARY_DIR
  LIB_MKL_CORE
  LIB_MKL_INTEL_THREAD
  LIB_MKL_INTEL_ILP64
  LIB_IOMP5
  LIB_PTHREAD
  MKL_INCLUDE_DIRS)

mark_as_advanced(
  LIB_MKL_CORE
  LIB_MKL_INTEL_THREAD
  LIB_MKL_INTEL_ILP64
  LIB_IOMP5
  LIB_PTHREAD
  MKL_INCLUDE_DIRS)

