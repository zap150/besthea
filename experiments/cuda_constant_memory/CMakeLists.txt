
add_executable(cuda_constant_memory cuda_constant_memory.cu)

target_include_directories(cuda_constant_memory PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda_constant_memory PUBLIC ${CUDA_LIBRARIES})
