add_executable(neumann_onthefly_gpu neumann_onthefly_gpu.cpp)
target_include_directories(neumann_onthefly_gpu PUBLIC
  ${besthea_INCLUDE_DIRS})
target_link_libraries(neumann_onthefly_gpu PUBLIC besthea)
install(TARGETS neumann_onthefly_gpu
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
