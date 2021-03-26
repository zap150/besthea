#include <cstdio>
#include <cuda_runtime.h>



__forceinline__  __host__ bool checkDeviceProperties()
{	
	cudaDeviceProp deviceProp;
	bool result = true;
	printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n");

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{	
		printf("There is no device supporting CUDA\n");
		result =  false;
	}

	int dev;
	for (dev = 0; dev < deviceCount; ++dev) 
	{
		cudaGetDeviceProperties(&deviceProp, dev);

		if (dev == 0) 
		{
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
			{
				printf("There is no device supporting CUDA.\n");
				result = false;
			}
			else if (deviceCount == 1)
				printf("There is 1 device supporting CUDA\n");
			else
				printf("There are %d devices supporting CUDA\n", deviceCount);
		}
		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	#if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		printf("  %-50s: %d.%d\n", "CUDA Driver Version", driverVersion/1000, driverVersion%100);
		
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  %-50s: %d.%d\n", "CUDA Runtime Version", runtimeVersion/1000, runtimeVersion%100);
	#endif
		printf("  %-50s: %d\n", "CUDA Capability Major revision number",	deviceProp.major);
		printf("  %-50s: %d\n", "CUDA Capability Minor revision number",	deviceProp.minor);
	#if CUDART_VERSION >= 2000
		printf("  %-50s: %d\n", "Number of multiprocessors",	deviceProp.multiProcessorCount);
		printf("  %-50s: %d\n", "Number of cores", 8 * deviceProp.multiProcessorCount);
	#endif
		printf("  %-50s: %u Mb\n", "Total amount of global memory", static_cast<unsigned int>(deviceProp.totalGlobalMem >> 20));
		printf("  %-50s: %llu bytes\n", "Total amount of constant memory", static_cast<long long unsigned int>(deviceProp.totalConstMem));
		printf("  %-50s: %llu bytes\n", "Total amount of shared memory per SM", static_cast<long long unsigned int>(deviceProp.sharedMemPerMultiprocessor));
		printf("  %-50s: %llu bytes\n", "Total amount of shared memory per block", static_cast<long long unsigned int>(deviceProp.sharedMemPerBlock));
		printf("  %-50s: %d\n", "Total number of registers available per SM", deviceProp.regsPerMultiprocessor);
		printf("  %-50s: %d\n", "Total number of registers available per block", deviceProp.regsPerBlock);
		printf("  %-50s: %d\n", "Warp size", deviceProp.warpSize);
		printf("  %-50s: %d\n", "Maximum number of threads per block", deviceProp.maxThreadsPerBlock);
		printf("  %-50s: %d x %d x %d\n", "Maximum sizes of each dimension of a block",
			   deviceProp.maxThreadsDim[0],
			   deviceProp.maxThreadsDim[1],
			   deviceProp.maxThreadsDim[2]);
		printf("  %-50s: %d x %d x %d\n", "Maximum sizes of each dimension of a grid", 
			   deviceProp.maxGridSize[0],
			   deviceProp.maxGridSize[1],
			   deviceProp.maxGridSize[2]);
		printf("  %-50s: %llu bytes\n", "Maximum memory pitch", static_cast<long long unsigned int>(deviceProp.memPitch));
		printf("  %-50s: %llu bytes\n", "Texture alignment", static_cast<long long unsigned int>(deviceProp.textureAlignment));
		printf("  %-50s: %.2f GHz\n", "Clock rate", deviceProp.clockRate * 1e-6f);
	#if CUDART_VERSION >= 2000
		printf("  %-50s: %s\n", "Concurrent copy and execution", deviceProp.deviceOverlap ? "Yes" : "No");
	#endif
	#if CUDART_VERSION >= 2020
		printf("  %-50s: %s\n", "Run time limit on kernels", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  %-50s: %s\n", "Integrated", deviceProp.integrated ? "Yes" : "No");
		printf("  %-50s: %s\n", "Support host page-locked memory mapping", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  %-50s: %s\n", "Compute mode", deviceProp.computeMode == cudaComputeModeDefault ?
																		"Default (multiple host threads can use this device simultaneously)" :
																		deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
																		deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
	#endif
	}
	printf("\nDevice Test PASSED -----------------------------------------------------\n\n");
	return result;
}



int main() {

    checkDeviceProperties();

    return 0;
}
