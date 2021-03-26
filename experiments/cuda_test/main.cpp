#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#include "daxpy.h"


int main(int argc, char ** argv)
{
    long long count = 13;

    if(argc > 1)
        count = atoll(argv[1]);

    printf("Count = %lld\n", count);


    double alpha = 7;
    double * x;
    double * y;
    long long bytes = count * sizeof(*x);
    cudaMallocHost(&x, bytes);
    cudaMallocHost(&y, bytes);

    // cudaError_t err = cudaGetLastError();
    // fprintf(stderr, "err %d: '%s'", err, cudaGetErrorString(err));

    for(long long i = 0; i < count; i++)
    {
        x[i] = i;
        y[i] = 100 * i;
    }
    for(long long i = 0; i < std::min(count, 16LL); i++)
        printf("%7.2f ", x[i]);
    printf("\n");
    for(long long i = 0; i < std::min(count, 16LL); i++)
        printf("%7.2f ", y[i]);
    printf("\n");

    daxpy(alpha, x, y, count);

    for(long long i = 0; i < std::min(count, 16LL); i++)
        printf("%7.2f ", y[i]);
    printf("\n");
    
    cudaFreeHost(x);
    cudaFreeHost(y);

    return 0;
}
