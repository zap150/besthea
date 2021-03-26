#include <cstdio>



__global__ void reduce(int * data) {

    __shared__ volatile int shmem[64]; // VOLATILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // there will be exactly 32 threads = 1 warp

    int tid = threadIdx.x;
    shmem[tid] = data[tid];
    shmem[tid + 32] = data[tid + 32];
    
    shmem[tid] += shmem[tid + 32];
    shmem[tid] += shmem[tid + 16];
    shmem[tid] += shmem[tid +  8];
    shmem[tid] += shmem[tid +  4];
    shmem[tid] += shmem[tid +  2];
    shmem[tid] += shmem[tid +  1];

    if(tid == 0)
        data[0] = shmem[0];
}


int main() {

    int size = 64;

    int * x;
    cudaMallocHost(&x, size * sizeof(*x));
    for(int i = 0; i < size; i++)
        x[i] = rand() % 100;
    
    // for(int i = 0; i < size; i++)
    //     printf("%2d:%2d\n", i, (int)x[i]);

    int resultCpu = 0;
    for(int i = 0; i < 64; i++)
    resultCpu += x[i];
    printf("CPU result: %d\n", (int)resultCpu);
    

    int * d_x;
    cudaMalloc(&d_x, size * sizeof(*d_x));
    cudaMemcpy(d_x, x, size * sizeof(*x), cudaMemcpyHostToDevice);
    reduce<<< 1, 32 >>>(d_x);
    cudaDeviceSynchronize();
    int resultGpu;
    cudaMemcpy(&resultGpu, d_x, sizeof(*x), cudaMemcpyDeviceToHost);
    printf("GPU result: %d\n", (int)resultGpu);


    cudaFree(d_x);
    cudaFreeHost(x);


    return 0;
}

