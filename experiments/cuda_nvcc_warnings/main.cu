#include <cstdio>
#include <cuda_runtime.h>



__global__ void kernel()
{
	double a = 3.14;
	long long int m = 7;
	long long int n = m * a;
	double b = 7.29;
	double c = n * b;
	printf("%f\n", c);

	double x = 5.743;
	int y = x;
	printf("%d\n", y);
}



int main()
{
	double a = 12.34;
	int b = a;
	printf("%d\n", b);

	kernel<<< 1, 1 >>>();

	cudaDeviceSynchronize();
    

    return 0;
}
