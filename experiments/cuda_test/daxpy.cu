#include "daxpy.h"


__global__ void d_daxpy(double alpha, double * d_x, double * d_y, long long count)
{
    long long index = blockIdx.x * blockDim.x + threadIdx.x;

    for (long long i = index; i < count; i += gridDim.x)
    {
        d_y[i] += alpha * d_x[i];
    }
}



void daxpy(double alpha, double * x, double * y, long long count)
{
    long long bytes = count * sizeof(*x);

    double *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);

    d_daxpy<<< 8, 256 >>>(alpha, d_x, d_y, count);

    cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}




