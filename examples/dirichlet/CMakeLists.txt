add_executable(dirichlet dirichlet.cpp)
target_include_directories(dirichlet PUBLIC
  ${besthea_INCLUDE_DIRS})
target_include_directories(dirichlet SYSTEM PUBLIC
  ${Lyra_INCLUDE_DIRS})
target_link_libraries(dirichlet PUBLIC besthea)
install(TARGETS dirichlet
  RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin")
