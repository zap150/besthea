add_executable(run run.cpp)
target_link_libraries(run PUBLIC besthea)
target_link_libraries(run PUBLIC mkl_rt ${MPI_CXX_LIBRARIES})
#target_link_libraries(run PUBLIC mkl_core mkl_intel_thread mkl_intel_ilp64 iomp5 ${MPI_CXX_LIBRARIES})
