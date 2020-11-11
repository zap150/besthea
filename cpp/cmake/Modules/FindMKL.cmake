# a simple cmake script to locate Intel Math Kernel Library

# This script looks for two places:
#	- the environment variable MKLROOT
#	- the directory /opt/intel/mkl


# Stage 1: find the root directory

set(MKLROOT_PATH $ENV{MKLROOT})

if (NOT MKLROOT_PATH)
	# try to find at /opt/intel/mkl
		
	if (EXISTS "/opt/intel/mkl")
		set(MKLROOT_PATH "/opt/intel/mkl")
	endif (EXISTS "/opt/intel/mkl")
endif (NOT MKLROOT_PATH)


# Stage 2: find include path and libraries
	
if (MKLROOT_PATH)
	# root-path found
	
	set(EXPECT_MKL_INCPATH "${MKLROOT_PATH}/include")
	
	# if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
	#     set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib")
	# endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
	
	# set(EXPECT_ICC_LIBPATH "$ENV{ICC_LIBPATH}")
	
	# if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	#     if (CMAKE_SIZEOF_VOID_P MATCHES 8)
	#         set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib/intel64")
	#     else (CMAKE_SIZEOF_VOID_P MATCHES 8)
	#         set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib/ia32")
	#     endif (CMAKE_SIZEOF_VOID_P MATCHES 8)
	# endif (CMAKE_SYSTEM_NAME MATCHES "Linux")

	if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
	    set(EXPECT_ICC_LIBPATH "${MKLROOT_PATH}/../compiler/lib")
	endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")

	if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	    set(EXPECT_ICC_LIBPATH "${MKLROOT_PATH}/../compiler/lib/intel64_lin")
	endif (CMAKE_SYSTEM_NAME MATCHES "Linux")

	set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib")
	
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
	
	# find specific library files
	find_library(LIB_MKL_CORE NAMES mkl_core HINTS ${MKL_LIBRARY_DIR})
	find_library(LIB_MKL_INTEL_THREAD NAMES mkl_intel_thread HINTS ${MKL_LIBRARY_DIR})
	find_library(LIB_MKL_INTEL_ILP64 NAMES mkl_intel_ilp64 HINTS ${MKL_LIBRARY_DIR})
	find_library(LIB_IOMP5 NAMES iomp5 HINTS ${ICC_LIBRARY_DIR})
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
    
mark_as_advanced(LIB_MKL_CORE LIB_MKL_INTEL_THREAD LIB_MKL_INTEL_ILP64 LIB_IOMP5 LIB_PTHREAD MKL_INCLUDE_DIR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_INT=long")
