add_executable(catch catch_mpi_main.cpp catch_test_pFMM)
target_link_libraries(catch PUBLIC besthea)
target_link_libraries(catch PUBLIC ${MKL_LIBRARIES} ${MPI_CXX_LIBRARIES})
