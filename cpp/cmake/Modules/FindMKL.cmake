# a simple cmake script to locate Intel Math Kernel Library via MKLROOT

# Stage 1: find the root directory

set(MKLROOT_PATH $ENV{MKLROOT})

# Stage 2: find include path and libraries
	
if (MKLROOT_PATH)
  # root-path found
	
  set(EXPECT_MKL_INCPATH "${MKLROOT_PATH}/include")
	
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
	
  # set include
	
  if (IS_DIRECTORY ${EXPECT_MKL_INCPATH})
    set(MKL_INCLUDE_DIR ${EXPECT_MKL_INCPATH})
  endif (IS_DIRECTORY ${EXPECT_MKL_INCPATH})
	
  if (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})
	set(MKL_LIBRARY_DIR ${EXPECT_MKL_LIBPATH})
  endif (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})
	
  if (IS_DIRECTORY ${EXPECT_ICC_LIBPATH})
	set(ICC_LIBRARY_DIR ${EXPECT_ICC_LIBPATH})
  endif (IS_DIRECTORY ${EXPECT_ICC_LIBPATH})

  message($ENV{LIBRARY_PATH})
	
  # find specific library files
  find_library(LIB_MKL_CORE NAMES mkl_core HINTS ${MKL_LIBRARY_DIR})
  find_library(LIB_MKL_INTEL_THREAD NAMES mkl_intel_thread HINTS ${MKL_LIBRARY_DIR})
  find_library(LIB_MKL_INTEL_ILP64 NAMES mkl_intel_ilp64 HINTS ${MKL_LIBRARY_DIR})
  find_library(LIB_IOMP5 NAMES iomp5 HINTS $ENV{LIBRARY_PATH} ENV ${LIBRARY_PATH})
  find_library(LIB_PTHREAD NAMES pthread)	
	
endif (MKLROOT_PATH)

set(MKL_LIBRARIES
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
  MKL_INCLUDE_DIR)
    
mark_as_advanced(
  LIB_MKL_CORE 
  LIB_MKL_INTEL_THREAD 
  LIB_MKL_INTEL_ILP64 
  LIB_IOMP5 	
  LIB_PTHREAD 
  MKL_INCLUDE_DIR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_INT=long")
