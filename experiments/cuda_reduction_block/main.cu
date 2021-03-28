#include <cstdio>
#include <cstdlib>
#include <besthea/time_measurer_cuda.h>

constexpr int tpb = 256;
constexpr int cacheline_size = 128;
constexpr int doubles_in_cacheline = cacheline_size / sizeof(double);



__host__ __device__ double calc_value(int row, int col) {
    return col;
}



__global__ void reduce_block_sum_classic(double * __restrict__ results, int width) {

    __shared__ volatile double shmem_vals[tpb];
    int tid = threadIdx.x;

    shmem_vals[threadIdx.x] = 0;
    __syncthreads();

    for(int col = threadIdx.x; col < width; col += blockDim.x)
    {
        shmem_vals[threadIdx.x] = calc_value(blockIdx.x, col);
        __syncthreads();

        // reduction start

        int thread_count = blockDim.x / 2;
        
        while(thread_count > 32) {
            if(tid < thread_count)
                shmem_vals[tid] += shmem_vals[tid + thread_count];
            __syncthreads();
            thread_count /= 2;
        }

        if(tid < 32) {
            shmem_vals[tid] += shmem_vals[tid + 32];
            shmem_vals[tid] += shmem_vals[tid + 16];
            shmem_vals[tid] += shmem_vals[tid +  8];
            shmem_vals[tid] += shmem_vals[tid +  4];
            shmem_vals[tid] += shmem_vals[tid +  2];
            shmem_vals[tid] += shmem_vals[tid +  1];

            if(tid == 0)
                results[doubles_in_cacheline * blockIdx.x] += shmem_vals[0];
        }

        __syncthreads();

        // reduction end
        
        shmem_vals[threadIdx.x] = 0;
    }
}



__global__ void reduce_block_sum_atomic(double * __restrict__ results, int width) {

    for(int col = threadIdx.x; col < width; col += blockDim.x)
    {
        double val = calc_value(blockIdx.x, col);

        // reduction start

        atomicAdd(&results[doubles_in_cacheline * blockIdx.x], val);

        // reduction end
    }
}



__global__ void reduce_block_sum_warpatomic(double * __restrict__ results, int width) {

    int lane = threadIdx.x % 32;
    int rounded_width = ((width - 1) / 32 + 1) * 32;

    for(int col = threadIdx.x; col < rounded_width; col += blockDim.x)
    {
        double val = (col < width) ? (calc_value(blockIdx.x, col)) : (0);

        // reduction start

        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val,  8);
        val += __shfl_down_sync(0xffffffff, val,  4);
        val += __shfl_down_sync(0xffffffff, val,  2);
        val += __shfl_down_sync(0xffffffff, val,  1);

        if(lane == 0)
            atomicAdd(&results[doubles_in_cacheline * blockIdx.x], val);

        // reduction end
    }
}



__global__ void reduce_block_sum_warpwarp(double * __restrict__ results, int width) {

    __shared__ volatile double warpResults[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int rounded_width = ((width - 1) / 32 + 1) * 32;

    if(warp == 0)
        warpResults[threadIdx.x] = 0;
    __syncthreads();

    for(int col = threadIdx.x; col < rounded_width; col += blockDim.x)
    {
        double val = (col < width) ? (calc_value(blockIdx.x, col)) : (0);

        // reduction start

        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val,  8);
        val += __shfl_down_sync(0xffffffff, val,  4);
        val += __shfl_down_sync(0xffffffff, val,  2);
        val += __shfl_down_sync(0xffffffff, val,  1);

        if(lane == 0)
            warpResults[warp] = val;
        __syncthreads();

        if(warp == 0) {
            val = warpResults[lane];

            val += __shfl_down_sync(0xffffffff, val, 16);
            val += __shfl_down_sync(0xffffffff, val,  8);
            val += __shfl_down_sync(0xffffffff, val,  4);
            val += __shfl_down_sync(0xffffffff, val,  2);
            val += __shfl_down_sync(0xffffffff, val,  1);

            if(lane == 0)
                results[doubles_in_cacheline * blockIdx.x] += val;
        }

        // reduction end

        if(threadIdx.x < 32)
            warpResults[threadIdx.x] = 0;
        __syncthreads();
    }
}




__global__ void print_results(double * __restrict__ results, int count) {

    if(blockIdx.x == 0 && threadIdx.x == 0) {
        count = min(count, 4);
        for(int i = 0; i < count; i++) {
            printf("Result row %3d: %10.3f\n", i, results[doubles_in_cacheline * i]);
        }
        printf("\n");
    }
}





int main(int argc, char ** argv) {

    if(argc <= 2) {
        fprintf(stderr, "Not enough arguments. Usage: ./program.x height width\n");
        return -1;
    }
    
    besthea::tools::time_measurer_cuda tm_classic, tm_atomic, tm_warpatomic, tm_warpwarp;

    int height = atoi(argv[1]);
    int width = atoi(argv[2]);

    double * d_results;
    cudaMalloc(&d_results, height * cacheline_size);
    
    // nonmeasured first kernel call
    reduce_block_sum_classic<<<height, tpb>>>(d_results, width);

    cudaMemset(d_results, 0, height * cacheline_size);
    tm_classic.start_submit();
    reduce_block_sum_classic<<<height, tpb>>>(d_results, width);
    tm_classic.stop_submit();
    print_results<<<1,1>>>(d_results, height);
    
    cudaMemset(d_results, 0, height * cacheline_size);
    tm_atomic.start_submit();
    reduce_block_sum_atomic<<<height, tpb>>>(d_results, width);
    tm_atomic.stop_submit();
    print_results<<<1,1>>>(d_results, height);
    
    cudaMemset(d_results, 0, height * cacheline_size);
    tm_warpatomic.start_submit();
    reduce_block_sum_warpatomic<<<height, tpb>>>(d_results, width);
    tm_warpatomic.stop_submit();
    print_results<<<1,1>>>(d_results, height);
    
    cudaMemset(d_results, 0, height * cacheline_size);
    tm_warpwarp.start_submit();
    reduce_block_sum_warpwarp<<<height, tpb>>>(d_results, width);
    tm_warpwarp.stop_submit();
    print_results<<<1,1>>>(d_results, height);


    cudaDeviceSynchronize();

    cudaFree(d_results);

    printf("Time classic:    %10.6f\n", tm_classic.get_time());
    printf("Time atomic:     %10.6f\n", tm_atomic.get_time());
    printf("Time warpatomic: %10.6f\n", tm_warpatomic.get_time());
    printf("Time warpwarp:   %10.6f\n", tm_warpwarp.get_time());

    return 0;
}

