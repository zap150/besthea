#include <cstdio>
#include <cuda_runtime.h>
#include "somefile.h"

namespace some_namespace
{
	constexpr int a = 42;
}

constexpr int b = 69;


__global__ void kernel()
{
	printf("bid %3d tid %3d a %3d b %3d c %3d d %3d\n", blockIdx.x, threadIdx.x, some_namespace::a, b, aesome_namespace_name::c, d);
}



int main()
{

	kernel<<< 2, 4 >>>();

	cudaDeviceSynchronize();
    

    return 0;
}
