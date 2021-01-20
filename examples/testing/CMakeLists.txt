add_executable(testing run.cpp)
target_include_directories(testing PUBLIC
  ${besthea_INCLUDE_DIRS})
target_include_directories(testing SYSTEM PUBLIC
  ${Lyra_INCLUDE_DIRS})
target_link_libraries(testing PUBLIC besthea)
install(TARGETS testing
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
