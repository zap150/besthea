add_executable(run run.cpp)
target_link_libraries(run PUBLIC besthea)
target_link_libraries(run PUBLIC mkl_rt ${MPI_CXX_LIBRARIES})
