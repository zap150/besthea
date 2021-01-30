
add_executable(cuda_properties cuda_properties.cpp)

target_include_directories(cuda_properties PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda_properties PUBLIC ${CUDA_LIBRARIES})
