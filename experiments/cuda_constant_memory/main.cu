#include <cstdio>
#include <cuda_runtime.h>


__constant__ __device__ int c_a;
__constant__ __device__ int c_b;


__global__ void kernel()
{
	printf("bid %3d tid %3d a %3d b %3d\n", blockIdx.x, threadIdx.x, c_a, c_b);
}



int main()
{
	int a = 3;
	int b = 5;

	cudaMemcpyToSymbol((const void*)&c_a, &a, sizeof(a));
	cudaMemcpyToSymbol(c_b, &b, sizeof(b));

	kernel<<< 2, 4 >>>();

	cudaDeviceSynchronize();
    

    return 0;
}
