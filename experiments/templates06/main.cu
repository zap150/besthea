#include <cstdio>
/*
struct Neco
{
    int vek;
    float vaha;
};



template<typename T>
__global__ void ker(const Neco & n);

template<>
__global__ void ker<int>(const Neco & n)
{
    int num = n.vek + threadIdx.x + blockIdx.x;
    printf("Cislo: %d\n", num);
}

template<>
__global__ void ker<float>(const Neco & n)
{
    float num = n.vaha * threadIdx.x * blockIdx.x;
    printf("Cislo: %f\n", num);
}*/

int main()
{
    /*Neco n;
    n.vek = 23;
    n.vaha = 75;

    ker<int><<<2,3>>>(n);*/

    // it is nonsense for kernel parameters to be references...

    return 0;
}
