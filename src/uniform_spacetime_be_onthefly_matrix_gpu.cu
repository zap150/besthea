#include "besthea/uniform_spacetime_be_onthefly_matrix_gpu.h"

#include <iostream>



__global__ void hello_gpu_world_kernel(int number) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  printf("Hello %d world from block %3d thread %3d, tid %3d\n", number, blockIdx.x, threadIdx.x, tid);

}



template<>
void besthea::uniform_spacetime_be_onthefly_matrix_gpu<
  besthea::bem::spacetime_heat_sl_kernel_antiderivative,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >,
  besthea::bem::uniform_spacetime_be_space< besthea::bem::basis_tri_p0 >>::hello_gpu_world(int number) const {
  
  hello_gpu_world_kernel<<< 3, 5 >>>(number);

  cudaDeviceSynchronize(); // wait until completion

}



template<class kernel_type, class test_space_type, class trial_space_type>
void besthea::uniform_spacetime_be_onthefly_matrix_gpu<kernel_type, test_space_type, trial_space_type>::hello_gpu_world(int number) const {
  
  std::cout << "General method\n";

}


