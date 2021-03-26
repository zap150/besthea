#include <cstdio>

__global__ void muj_kernel()
{
    __shared__ int x[256];
    __shared__ int y[256];

    x[threadIdx.x] = 1;
    y[threadIdx.x] = 2;
    printf("%d\n", x[threadIdx.x] + y[threadIdx.x]);
}


int main()
{



    return 0;
}
