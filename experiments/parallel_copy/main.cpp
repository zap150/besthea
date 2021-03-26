#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

#include <besthea/time_measurer.h>


int main(int argc, char ** argv)
{
    // running multiple std::copy in parallel does really speed up the copying

    if(argc <= 1)
    {
        printf("Not enough arguments, provide array size\n");
        return 1;
    }

    besthea::tools::time_measurer tm_alloc, tm_init, tm_copy, tm_delete;

    long long int power = atoll(argv[1]);
    long long int size = (1 << power);

    int * arr_in;
    int * arr_out;

    tm_alloc.start();
    arr_in = new int[size];
    arr_out = new int[size];
    tm_alloc.stop();

    tm_init.start();
    for(long long int i = 0; i < size; i++)
        arr_in[i] = i;
    tm_init.stop();
    
    for(long long int i = 0; i < std::min(16LL,size); i++)
        printf("%6d", arr_in[i]);
    printf("\n");

    tm_copy.start();
#pragma omp parallel
    {
        long long int tid = omp_get_thread_num();
        long long int threadCount = omp_get_num_threads();
        long long int myStart = (tid * size) / threadCount;
        long long int myEnd = ((tid+1) * size) / threadCount;

        std::copy(arr_in + myStart, arr_in + myEnd, arr_out + myStart);
    }
    tm_copy.stop();

    for(long long int i = 0; i < std::min(16LL,size); i++)
        printf("%6d", arr_out[i]);
    printf("\n");

    tm_delete.start();
    delete[] arr_in;
    delete[] arr_out;
    tm_delete.stop();

    printf("Time alloc:  %10.6f s\n", tm_alloc.get_time());
    printf("Time init:   %10.6f s\n", tm_init.get_time());
    printf("Time copy:   %10.6f s\n", tm_copy.get_time());
    printf("Time delete: %10.6f s\n", tm_delete.get_time());

    return 0;
}
