#pragma once
#include <cuda_runtime.h>

namespace problems
{
	__host__ __device__ double dejong5(double* args, int n);

	///\note This is only a 2d function
	__host__ __device__ double easom(double* args, int n);

	__host__ __device__ double michalewicz(double* args, int n, int m);
	__host__ __device__ double michalewicz2(double* args, int n);
}
